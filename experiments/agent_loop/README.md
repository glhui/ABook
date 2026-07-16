# 最小上下文 Runtime 与执行 Agent

这个实验展示类似 Codex 的最小纵向切片：Runtime 保存共享工作区上下文，并为
每个 Agent 保存独立的任务、Skill 和消息历史；执行 Agent 再通过专用文件工具和
受限 PowerShell 按需获取事实、修改文件并运行验证。

~~~text
工作区根目录与当前目录
        |
        +-- 根目录 AGENTS.md
        +-- 路径上的子目录 AGENTS.md
        `-- Skill manifest 目录
                    |
                    v
             WorkspaceContext  -------- 所有 Agent 共享
                    |
              ContextRuntime
             TaskState     \
     goal / plan / facts / validation
               /          \
      root AgentContext   worker AgentContext
      task / skill /      task / skill /
      message_history     message_history
               \          /
                模型指令与 user message
                    |
       +------------+-------------+----------------+-------------+
       |            |             |                |             |
   workspace     replace      PowerShell      select Skill   delegate
~~~

## 文件职责

~~~text
agent_loop/
├─ context.py          共享工作区上下文、AgentContext 和 ContextRuntime
├─ workspace_tools.py  文件工具、PowerShell 工具和统一重试边界
├─ runner.py           root 与子 Agent 共用的模型调用生命周期
├─ orchestration.py    Skill、父子会话和结构化交接工具
├─ agent_runtime.py    root Agent 定义和同步/异步调用入口
├─ main.py             模型配置与命令行入口
└─ skills/             可按需加载的 Skill
~~~

## 核心边界

`WorkspaceContextBuilder.build` 只构建 `WorkspaceContext`：

- 确保当前工作目录位于工作区根目录内。
- 从工作区根目录到当前工作目录依次加载适用的 AGENTS.md。
- 只读取所有 Skill manifest，不预加载每个 Skill 的完整正文。

WorkspaceContextBuilder 不调用模型、不枚举工作区文件、不读取业务文件，也不
执行工具。这样可以避免把可能截断的文件清单误当成完整事实。文件列表、正文、
搜索结果和 Git 状态均由执行 Agent 根据用户任务按需获取。

项目指令按作用域从宽到窄排列，更接近当前工作目录的 AGENTS.md 具有更高优先
级。`ContextRuntime.create_agent_context` 再为 root 或子 Agent 选择 Skill，
并保存该 Agent 自己的任务与消息历史。Runtime 只把共享工作区约束和当前 Skill
放入 instructions，并明确项目约束高于 Skill。可变 `TaskState`、证据目录和当前
请求通过 user message 发送；状态被标记为数据，不能覆盖固定指令或当前请求。

`AgentRunner.run_turn` 是 root、首次委派和继续子会话共用的唯一内部调用管线：
它依次检查压缩、构造 Runtime user message、设置请求与工具调用上限、调用模型，
并只在成功后写回消息历史和轮次。`run_agent` 只是 CLI 使用的同步薄适配。因此
root 和 worker 共享工作区事实，但不会混用消息历史。每个 Agent 的上下文窗口按
1,000,000 tokens 管理；历史估算达到
70%（700,000 tokens）时，Runtime 使用同一模型压缩较早消息，保留任务目标、
约束、决定、修改、验证和未决问题，并尽量保留最近一轮原始消息。模型提供 usage
时使用真实 token 数，离线模型没有 usage 时才使用字符数估算。压缩输出是结构化
检查点：Runtime 先校验并合并事实与未决事项，成功后才用摘要替换旧历史；未知
证据或工具结果中不存在的引用原文会触发模型重试，原历史不会提前丢弃。

例如，历史中曾读取 `config.py` 并得到 `evidence-8`，其工具结果出现
`ABOOK_MODEL`。`CompactionCheckpoint.facts` 可以保存“模型名称来自配置”并引用
该 ID 和逐字 quote `ABOOK_MODEL`；尚未验证的模型连通性则保存到
`unresolved_issues`。文件全文和重复对话仅由 `summary` 概括，不能被写成已验证
事实。

`TaskState` 保存当前任务的目标、计划、已完成步骤、重要事实、未决事项、完成
条件、修改文件、验证结果和总体状态。工作区工具为实际结果登记顺序
`evidence_id`，同时保存模型实际看到的有界结果文本。root Agent 通过
`update_task_state` 提交重要事实时，必须同时提供 ID 和结果中的逐字 `quote`；
Runtime 验证 quote 确实存在后，才将其合并为长期 `TaskFact`，后续状态更新不会
覆盖已有事实。未决事项可在问题解决后显式替换。文件修改和验证结果仍只由实际
工具调用记录。

## 执行工具

root Agent 当前具有十一个核心工具：

- `list_workspace_files`：递归列出工作区文件，跳过 `.git`、`.venv` 和缓存目录。
- `read_workspace_file`：读取 UTF-8 文件，超过上限时明确标记截断。
- `search_workspace_text`：搜索大小写不敏感的字面文本并返回文件和行号。
- `replace_workspace_text`：校验原文本出现次数后执行精确替换。
- `run_powershell_command`：运行一条受限的 PowerShell 或 Windows CLI 命令。
- `update_task_state`：更新计划、重要事实、完成条件和任务状态。
- `select_skill`：根据 manifest 目录按需加载另一个 Skill 的完整正文。
- `delegate_task`：选择固定模板，创建子 Agent 会话并返回 `session_id`。
- `continue_subagent`：把具体反馈和上一轮结构化交接交回同一个子 Agent。
- `list_subagents`：列出 root 创建的子会话、轮次和最新状态。
- `inspect_subagent`：读取一个子会话最近的完整结构化交接。

工具通过 `AgentDependencies` 同时获得 `ContextRuntime` 和当前 `AgentContext`。
工作区工具只读取共享路径；`select_skill` 只修改当前 Agent 的 Skill。所有路径都由宿主解析并验证，
不能通过 `..` 或符号链接越过工作区。结果数量、读取字符数和搜索文件数均有
上限。`.env` 不会被列出、读取或搜索。文本工具在首行返回 `evidence_id`，命令
工具在结构化 `CommandResult` 中返回该字段；证据记录还保留实际返回文本，供
`FactClaim` 的 quote 做精确子串校验。

精确替换成功后，Runtime 自动把相对路径加入 `TaskState.modified_files`。Python
测试、编译和 `pip check` 实际执行后，Runtime 自动记录命令、退出码和超时状态；
失败结果也会保留，供下一步修正，而不会被模型改写成成功。

路径不存在、越过工作区、替换次数不一致或命令被策略拒绝时，工具只抛出统一的
`RecoverableToolError`。`RetryToolset` 在工具执行边界将它转换为 PydanticAI
`ModelRetry`，PydanticAI 再生成 `RetryPromptPart` 并请求模型重新决策。具体工具
不依赖重试协议，也不需要重复编写异常转换代码。

精确替换工具具有副作用，Agent 指令和工具文档都要求只有用户明确要求修改文件
时才能调用。替换次数与预期不一致时会拒绝写入，避免模糊替换修改额外位置。

PowerShell 工具采用单命令白名单，只允许工作区查询、只读 Git 子命令、`rg`，
以及 Python 测试、编译和 `pip check`。它拒绝管道、重定向、变量展开、父目录、
绝对路径、安装命令和 Git 写操作，同时限制超时和输出长度。该策略用于展示
runtime 的命令边界；因此需要分支或工作区状态时，Agent 应通过该工具按需执行
只读 Git 命令，而不是依赖启动时快照。它并不等同于操作系统级沙箱。

## Skill 选择与子 Agent

共享上下文只包含所有 Skill 的名称和描述。每个 AgentContext 只加载自己当前
选择的 Skill；父 Agent 需要其他工作流时调用 `select_skill`，避免把所有 Skill
正文同时放入上下文。

父 Agent 可以通过 `delegate_task` 创建以下子 Agent：

- `explorer`：只读探索和证据收集。
- `worker`：执行用户已经明确授权的修改并验证。
- `reviewer`：只读审查实现、测试和风险。

子 Agent 与父 Agent 引用同一个 `WorkspaceContext`，但拥有独立的 `AgentContext`
和消息历史，工具集由模板决定。每个子上下文记录唯一 `parent_agent_id`；只有创建
会话的 root 可以查询或继续它。子 Agent 没有 Skill 选择或任何编排工具，不能创建、
查询或联系其他子 Agent；它的 Runtime 数据也只展示自己的证据，不展示其他子会话
交接。首次委派返回的 `session_id` 标识该子 Agent。

每轮交接包含 `completed`、`needs_follow_up` 或 `blocked` 状态，以及结构化摘要、
带逐字引用的事实、已解析证据、实际修改文件、实际验证结果、未决问题和建议下一
步。事实使用与 TaskState 相同的 ID + quote 校验；修改与验证由宿主根据本轮工具
调用补入，不能由子 Agent 声称。`completed` 不允许保留未决事项，其他状态必须
说明未决事项；Runtime 同时校验父子归属和连续轮次。父 Agent 可先使用查询工具
检查状态，再通过
`continue_subagent` 把具体反馈、上一轮状态、未决事项和建议动作交回原子 Agent。
会话只保存在当前 Runtime 内存中，不做跨进程恢复或子 Agent 间通信。

## 运行

实验读取仓库根目录 .env 中的 ABOOK_MODEL、ABOOK_API_KEY 和 ABOOK_BASE_URL：

~~~powershell
.\.venv\Scripts\python.exe -m experiments.agent_loop.main "解释当前项目"
~~~

从仓库根目录执行上述命令即可；`-m` 按 Python 模块启动实验，因此不需要输入
`experiments\agent_loop\main.py` 脚本路径。省略最后的请求参数时，程序会在终端
中提示输入首个请求。首轮完成后会持续显示 `You> ` 输入提示，并复用同一个 root
Agent 的消息历史和 Runtime 状态；输入 `/quit`、`/exit`、`quit` 或 `exit` 结束
会话。

也可以在 VS Code 中运行 `Debug Agent Loop`，并在集成终端中输入请求。

## 当前边界

当前父子 Agent 已能完成“发现—拆分—读取—修改—验证—审查”的最小闭环，但没有操作系统沙箱、
交互式审批、网络工具、任意 PowerShell 语法或 Git 写操作。生产级 runtime 仍需
在进程和文件系统层执行隔离，不能只依赖字符串策略。

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
 coordinator context    task AgentContext
 plan / skill /         work package / skill /
 message_history        message_history
               \          /
                模型指令与 user message
                    |
       +------------+-------------+----------------+-------------+
       |            |             |                |             |
   workspace     replace      PowerShell      select Skill   assign tasks
~~~

## 文件职责

~~~text
agent_loop/
├─ context.py          共享工作区上下文、AgentContext 和 ContextRuntime
├─ workspace_tools.py  文件工具、PowerShell 工具和统一重试边界
├─ runner.py           协调 Agent 与任务 Agent 共用的调用生命周期
├─ orchestration.py    Skill、任务分配和结构化交接工具
├─ agent_runtime.py    协调 Agent 定义和异步调用入口
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
级。`ContextRuntime.create_agent_context` 再为协调或任务 Agent 选择 Skill，
并保存该 Agent 自己的职责、任务与消息历史。Runtime 只把共享约束和当前 Skill
放入 instructions，并明确项目约束高于 Skill。可变 `TaskState`、证据目录和当前
请求通过 user message 发送；状态被标记为数据，不能覆盖固定指令或当前请求。

`AgentRunner.run_turn` 是协调 Agent 和任务 Agent 共用的唯一内部调用管线：
它依次检查压缩、构造 Runtime user message、设置请求与工具调用上限、调用模型，
并只在成功后写回消息历史和轮次。同一个 `AgentContext` 的调用由异步锁串行化，
因此后台完成事件和用户输入不会同时改写协调会话历史。协调 Agent 和 worker 共享
工作区事实，但不会混用消息历史。每个 Agent 的上下文窗口按
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
`evidence_id`，同时保存模型实际看到的有界结果文本。协调 Agent 通过
`update_task_state` 提交重要事实时，必须同时提供 ID 和结果中的逐字 `quote`；
Runtime 验证 quote 确实存在后，才将其合并为长期 `TaskFact`，后续状态更新不会
覆盖已有事实。未决事项可在问题解决后显式替换。文件修改和验证结果仍只由实际
工具调用记录。

## 执行工具

协调 Agent 当前具有十一个核心工具：

- `list_workspace_files`：递归列出工作区文件，跳过 `.git`、`.venv` 和缓存目录。
- `read_workspace_file`：读取 UTF-8 文件，超过上限时明确标记截断。
- `search_workspace_text`：搜索大小写不敏感的字面文本并返回文件和行号。
- `replace_workspace_text`：校验原文本出现次数后执行精确替换。
- `run_powershell_command`：运行一条受限的 PowerShell 或 Windows CLI 命令。
- `update_task_state`：更新计划、重要事实、完成条件和任务状态。
- `select_skill`：根据 manifest 目录按需加载另一个 Skill 的完整正文。
- `assign_tasks`：把 1 到 8 个工作包交给模板 Agent，并立即返回任务分配 ID。
- `send_task_feedback`：非阻塞地把修正反馈提交给原任务 Agent。
- `list_assignments`：列出任务分配、轮次和最新状态。
- `inspect_assignment`：读取指定任务最近一次完整结构化交接。

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

## Skill 与任务分配

共享上下文只包含所有 Skill 的名称和描述。每个 AgentContext 只加载自己当前
选择的 Skill；协调 Agent 需要其他工作流时调用 `select_skill`，避免把所有 Skill
正文同时放入上下文。

协调 Agent 像项目经理一样拆分目标，再通过 `assign_tasks` 把 1 到 8 个独立工作包
交给以下模板之一：

- `explorer`：只读探索和证据收集。
- `worker`：执行用户已经明确授权的修改并验证。
- `reviewer`：只读审查实现、测试和风险。

所有 Agent 引用同一个 `WorkspaceContext`，但每个 Agent 都有独立的
`AgentContext` 和消息历史。`role="coordinator"` 表示整体规划职责，`role="task"`
表示当前只负责一个工作包。角色只决定职责和工具范围。任务 Agent 不注册任务
分配工具，只能使用模板允许的工作区工具。`assign_tasks` 立即返回 `running` 状态
和 `assignment_id`，不会等待执行完成。

每轮交接包含 `completed`、`needs_follow_up` 或 `blocked` 状态，以及结构化摘要、
带逐字引用的事实、已解析证据、实际修改文件、实际验证结果、未决问题和建议下一
步。事实使用与 TaskState 相同的 ID + quote 校验；修改与验证由宿主根据本轮工具
调用补入，不能由模型自行声称。`completed` 不允许保留未决事项，其他状态必须
说明未决事项；Runtime 同时校验任务归属和连续轮次。协调 Agent 可检查交接，使用
`send_task_feedback` 把修正要求交回原执行 Agent，或用 `assign_tasks` 重新分配新的
工作包。任务分配只保存在当前 Runtime 内存中，不做跨进程恢复。

后台任务使用 CLI 的持续事件循环并发执行，不提供排队、取消或额外超时机制。
任一任务 Agent 完成或失败后，Runtime 保存状态并自动触发一轮协调调用；协调
Agent 根据最新交接更新计划，不必等待同批其他工作。并行 worker 必须修改不同
文件。Runtime 按 Agent 单独记录修改和验证，避免错误归属副作用。

## 运行

实验读取仓库根目录 .env 中的 ABOOK_MODEL、ABOOK_API_KEY 和 ABOOK_BASE_URL：

~~~powershell
.\.venv\Scripts\python.exe -m experiments.agent_loop.main "解释当前项目"
~~~

从仓库根目录执行上述命令即可；`-m` 按 Python 模块启动实验，因此不需要输入
`experiments\agent_loop\main.py` 脚本路径。省略最后的请求参数时，程序会在终端
中提示输入首个请求。首轮完成后会持续显示 `You> ` 输入提示，并复用同一个协调
Agent 的消息历史和 Runtime 状态；输入 `/quit`、`/exit`、`quit` 或 `exit` 结束
会话。

每次协调 Agent、任务 Agent 或历史压缩调用开始时，CLI 会立即输出带调用 ID 的
`started` 状态；调用成功或失败后输出对应的 `completed` 或 `failed` 状态。模型
最终回答仍在每轮完成后一次性输出，不启用逐 token 流式显示。CLI 使用持续异步
事件循环，因此等待终端输入时后台工作仍会运行；每个完成事件都会触发协调 Agent
重新规划并输出新的回答。并行调用状态可能交错，可通过调用 ID 和 Agent ID
区分。

也可以在 VS Code 中运行 `Debug Agent Loop`，并在集成终端中输入请求。

## 当前边界

当前协调与任务 Agent 已能完成“拆分—分配—执行—交接—重排”的最小闭环，但没有操作系统沙箱、
交互式审批、网络工具、任意 PowerShell 语法或 Git 写操作。生产级 runtime 仍需
在进程和文件系统层执行隔离，不能只依赖字符串策略。

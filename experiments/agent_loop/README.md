# 最小上下文 Runtime 与执行 Agent

这个实验展示类似 Codex 的最小纵向切片：Runtime 保存共享工作区上下文，并为
每个 Agent 保存独立的任务、Skill 和消息历史；执行 Agent 再通过专用文件工具和
受限 PowerShell 按需获取事实、修改文件并运行验证。

~~~text
工作区根目录与当前目录
        |
        +-- 根目录 AGENTS.md
        +-- 路径上的子目录 AGENTS.md
        +-- Git 分支与有界状态
        `-- Skill manifest 目录
                    |
                    v
             WorkspaceContext  -------- 所有 Agent 共享
                    |
              ContextRuntime
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

## 核心边界

`WorkspaceContextBuilder.build` 只构建 `WorkspaceContext`：

- 确保当前工作目录位于工作区根目录内。
- 从工作区根目录到当前工作目录依次加载适用的 AGENTS.md。
- 读取当前 Git 分支和最多 50 行工作区状态，不预加载完整 diff。
- 只读取所有 Skill manifest，不预加载每个 Skill 的完整正文。

WorkspaceContextBuilder 不调用模型、不枚举工作区文件、不读取业务文件，也不
执行工具。这样可以避免把可能截断的文件清单误当成完整事实。文件列表、正文和
搜索结果由执行 Agent 根据用户任务按需获取。

项目指令按作用域从宽到窄排列，更接近当前工作目录的 AGENTS.md 具有更高优先
级。`ContextRuntime.create_agent_context` 再为 root 或子 Agent 选择 Skill，
并保存该 Agent 自己的任务与消息历史。Runtime 在模型调用前把共享工作区指令和
当前 Agent 的 Skill 组合起来；用户任务仍作为独立的 user message 发送。

`run_agent` 从当前 `AgentContext.message_history` 读取历史，并在一轮结束后写回
`AgentRunResult.all_messages()`。因此 root 和 worker 共享工作区事实，但不会
混用消息历史。当前实验没有实现历史压缩。

## 执行工具

父执行 Agent 当前具有八个核心工具：

- `list_workspace_files`：递归列出工作区文件，跳过 `.git`、`.venv` 和缓存目录。
- `read_workspace_file`：读取 UTF-8 文件，超过上限时明确标记截断。
- `search_workspace_text`：搜索大小写不敏感的字面文本并返回文件和行号。
- `replace_workspace_text`：校验原文本出现次数后执行精确替换。
- `run_powershell_command`：运行一条受限的 PowerShell 或 Windows CLI 命令。
- `select_skill`：根据 manifest 目录按需加载另一个 Skill 的完整正文。
- `delegate_task`：选择固定模板，创建子 Agent 会话并返回 `session_id`。
- `continue_subagent`：把验证反馈或后续任务交回同一个子 Agent。

工具通过 `AgentDependencies` 同时获得 `ContextRuntime` 和当前 `AgentContext`。
工作区工具只读取共享路径；`select_skill` 只修改当前 Agent 的 Skill。所有路径都由宿主解析并验证，
不能通过 `..` 或符号链接越过工作区。结果数量、读取字符数和搜索文件数均有
上限。`.env` 不会被列出、读取或搜索。

路径不存在、越过工作区、替换次数不一致或命令被策略拒绝时，工具只抛出统一的
`RecoverableToolError`。`RetryToolset` 在工具执行边界将它转换为 PydanticAI
`ModelRetry`，PydanticAI 再生成 `RetryPromptPart` 并请求模型重新决策。具体工具
不依赖重试协议，也不需要重复编写异常转换代码。

精确替换工具具有副作用，Agent 指令和工具文档都要求只有用户明确要求修改文件
时才能调用。替换次数与预期不一致时会拒绝写入，避免模糊替换修改额外位置。

PowerShell 工具采用单命令白名单，只允许工作区查询、只读 Git 子命令、`rg`，
以及 Python 测试、编译和 `pip check`。它拒绝管道、重定向、变量展开、父目录、
绝对路径、安装命令和 Git 写操作，同时限制超时和输出长度。该策略用于展示
runtime 的命令边界，并不等同于操作系统级沙箱。

## Skill 选择与子 Agent

共享上下文只包含所有 Skill 的名称和描述。每个 AgentContext 只加载自己当前
选择的 Skill；父 Agent 需要其他工作流时调用 `select_skill`，避免把所有 Skill
正文同时放入上下文。

父 Agent 可以通过 `delegate_task` 创建以下子 Agent：

- `explorer`：只读探索和证据收集。
- `worker`：执行用户已经明确授权的修改并验证。
- `reviewer`：只读审查实现、测试和风险。

子 Agent 与父 Agent 引用同一个 `WorkspaceContext`，但拥有独立的 `AgentContext`
和消息历史，工具集由模板决定。子 Agent 没有 `select_skill`、`delegate_task`
和 `continue_subagent`，
因此不能继续创建孙 Agent。首次委派返回的 `session_id` 标识该子 Agent；验证
失败或需要补充修改时，父 Agent 可调用 `continue_subagent` 并显式传回原消息
历史。会话只保存在当前父 Agent 运行的内存中，不做持久化、并行调度或跨进程恢复。

## 运行

实验读取仓库根目录 .env 中的 ABOOK_MODEL、ABOOK_API_KEY 和 ABOOK_BASE_URL：

~~~powershell
.\.venv\Scripts\python.exe experiments\agent_loop\main.py "解释当前项目"
~~~

也可以在 VS Code 中运行 main.py，并在集成终端中输入请求。

## 当前边界

当前父子 Agent 已能完成“发现—拆分—读取—修改—验证—审查”的最小闭环，但没有操作系统沙箱、
交互式审批、网络工具、任意 PowerShell 语法或 Git 写操作。生产级 runtime 仍需
在进程和文件系统层执行隔离，不能只依赖字符串策略。

# 最小工具型 Agent Loop

这个实验使用 PydanticAI 展示一个有界的 reason、act、observe、continue 循环。
它不再把任务固定拆成 Thinker、Generator 和 Checker，而是由一个执行 Agent
根据当前上下文自行决定调用工具还是提交最终答案。

~~~text
用户请求和 Skill
       |
       v
执行 Agent 推理下一步
       |
       +------ 最终答案 ------> 结束
       |
       v
调用 inspect_text
       |
       v
宿主执行只读工具
       |
       v
工具结果进入同一次消息历史
       |
       +------> Agent 根据观察结果继续
~~~

PydanticAI 负责模型请求和工具调用之间的循环。宿主代码只负责注册工具、提供
调用预算和启动运行，因此不会再维护一套基于 PASS 或 FAIL 文本协议的循环。

## 当前工具

inspect_text 接收完整候选文本并返回以下结构化指标：

- total_characters：Python Unicode 字符数，包含空白和标点。
- non_whitespace_characters：排除 Unicode 空白后的字符数。
- line_count：统一换行符后的行数。
- paragraph_count：以一个或多个空白行为分隔的段落数。

工具只观察文本，不修改内容、不访问文件或网络，也不替模型评价文风。用户提出
字数、字符数、行数或段落数限制时，Agent 指令要求先调用工具检查完整候选文本，
不满足时继续修订和检查。

## 有界运行

默认最多允许 6 次模型请求和 4 次工具调用。预算由 UsageLimits 在 PydanticAI
运行时执行；达到上限时会抛出 UsageLimitExceeded，而不是静默返回未经验证的
结果。当前边界只控制调用次数，不控制 token 或费用，后续可按运行环境增加。

## Skill 加载

1. skill.json 保存用于发现 Skill 的元数据。
2. SKILL.md 保存提供给 Agent 的完整指令。
3. SkillRuntime 从固定目录读取并验证两者。
4. Skill 内容与 Agent 的循环和工具使用指令一起进入运行上下文。

## 运行

实验读取仓库根目录 .env 中的 ABOOK_MODEL、ABOOK_API_KEY 和 ABOOK_BASE_URL：

~~~powershell
.\.venv\Scripts\python.exe experiments\agent_loop\main.py "帮我生成一封200字的请帖"
~~~

也可以在 VS Code 中运行 main.py，并在集成终端的“请输入请求”后输入内容。
Debug Console 不支持 Python 标准输入，因此 launch.json 使用
integratedTerminal。

CLI 只打印最终答案。需要观察循环时，可检查 AgentRunResult.all_messages()；
其中依次包含模型的 ToolCallPart、宿主生成的 ToolReturnPart 和后续模型响应。
AgentRunResult.usage 记录模型请求数与工具调用数。

## 当前边界

这是 Codex 式循环的最小纵向切片，还没有提供文件读写、Shell、Git、测试、
MCP、审批、沙箱、可恢复线程或上下文压缩。下一步应优先增加一个受限的工作区
读取工具，再基于真实文件内容完成修改与验证闭环，而不是重新增加固定角色。

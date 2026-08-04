# Python 实验示例

## DeepSeek 工具对话

`pydantic_ai_deepseek_tool_chat.py` 使用 `.env` 的 `ABOOK_MODEL`、`ABOOK_API_KEY` 与 `ABOOK_BASE_URL`，通过 OpenAI 兼容接口连接 DeepSeek。它保留多轮消息历史，模型请求保持非流式；但每个工具调用和工具结果会在发生时立即打印，随后在下一次模型请求完成时打印模型回复。示例通过工具执行层注册了受工作区边界保护的 `read_file` 与 `bash`；示例只授权读取和 Bash，不授权写入或编辑。

可复用的轮次执行逻辑位于 `src/agent_runtime/run_turn.py`。调用方可通过 `on_response` 处理完整模型响应，通过 `on_event` 处理所有工具事件；后续增加事件展示逻辑时，无需复制 Agent 迭代和工具执行代码。

## 测试先行的多 Agent 协作

`python_code_test_agents.py` 演示三个预定义角色的手动协作：协调 Agent 先将需求拆为测试任务和编码任务；代码 Agent 先完成生产代码，测试 Agent 再完成 `tests/`，因此日志按角色顺序输出；代码 Agent 可在隔离项目中使用 Bash 执行 `python -m py_compile` 检查自己的源码，pytest 仍由宿主统一执行。若失败，宿主会将失败输出反馈给代码 Agent，并最多进行两次只修改生产代码的修复，然后以退出码给出最终结论。三个 Agent 只可访问项目根目录的 `tmp/` 共享隔离区，目录已被 Git 忽略，不会污染项目源码；示例会在文件修改和本地测试命令执行前请求一次显式确认。

```powershell
.\.venv\Scripts\python.exe examples\python_code_test_agents.py
```

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe examples\pydantic_ai_deepseek_tool_chat.py
```

可尝试输入：`现在几点？`、`请计算 (18 + 6) * 3`、`列出当前工作区的文件` 或 `查看 Git 状态`。工具在 Unix 使用 Bash，在 Windows 使用 PowerShell。输入 `/quit` 退出。

## LangGraph 编排的代码—测试—验证—修复

`langgraph_python_code_test_agents.py` 复用原有的协调、代码和测试 Agent，仅用 LangGraph
管理“任务拆分 → 代码实现 → 测试编写 → 宿主 pytest → 有界修复”的状态迁移。失败时图会在
`repair` 与 `validate` 节点之间最多循环两次；流程分支只依据用户确认和 pytest 真实退出码。

先安装示例依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install langgraph
```

再从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe examples\langgraph_python_code_test_agents.py
```

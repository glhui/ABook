# Python 实验示例

## DeepSeek 工具对话

`pydantic_ai_deepseek_tool_chat.py` 使用 `.env` 的 `ABOOK_MODEL`、`ABOOK_API_KEY` 与 `ABOOK_BASE_URL`，通过 OpenAI 兼容接口连接 DeepSeek。它保留多轮消息历史，模型请求保持非流式；但每个工具调用和工具结果会在发生时立即打印，随后在下一次模型请求完成时打印模型回复。示例通过工具执行层注册了受工作区边界保护的 `read_file` 与 `bash`；示例只授权读取和 Bash，不授权写入或编辑。

可复用的轮次执行逻辑位于 `src/agent_runtime/turns.py`。调用方可通过 `on_response` 处理完整模型响应，通过 `on_event` 处理所有工具事件；后续增加事件展示逻辑时，无需复制 Agent 迭代和工具执行代码。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe examples\pydantic_ai_deepseek_tool_chat.py
```

可尝试输入：`现在几点？`、`请计算 (18 + 6) * 3`、`列出当前工作区的文件` 或 `查看 Git 状态`。工具在 Unix 使用 Bash，在 Windows 使用 PowerShell。输入 `/quit` 退出。

## 测试先行的多 Agent 协作

示例在交互式终端中默认使用可折叠日志。Windows PowerShell 和 Unix 终端均可使用方向键或
`j`/`k` 选择日志，按 Enter 或空格展开/收起正文。可通过以下环境变量显式启用：

```powershell
$env:ABOOK_INTERACTIVE_LOGS = "1"
```

设置为 `0` 可关闭交互模式。

交互模式需要安装 `rich` 和 `prompt-toolkit`；非交互终端会自动回退到普通文本日志。

```powershell
.\.venv\Scripts\python.exe -m pip install rich prompt-toolkit
```

需要生成数学图片时，建议额外安装 `matplotlib`；宿主会执行可视化脚本，检查
`artifacts/visualization.png` 是否存在及其尺寸，并在终端支持时尝试内嵌预览，否则显示文件路径。

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
```

`python_code_test_agents.py` 演示三个预定义角色的手动协作：协调 Agent 先将需求拆为测试任务和编码任务；两个工作包可以并行，但示例刻意让代码 Agent 先完成生产代码、测试 Agent 再完成 `tests/`，因此日志按角色顺序输出。两者都必须在自己的节点使用 Bash 执行 `python -m py_compile`，确认各自文件可编译；pytest 仍由宿主统一执行。若 pytest 失败，工作流只回到代码编写节点，并最多重新编写两次生产代码，测试 Agent 不会再次运行或修改测试。三个 Agent 只可访问项目根目录的 `tmp/` 共享隔离区，目录已被 Git 忽略，不会污染项目源码；示例会在文件修改和本地测试命令执行前请求一次显式确认。

```powershell
.\.venv\Scripts\python.exe examples\python_code_test_agents.py
```

## LangGraph 编排的代码—测试与 Skill 路由

`langgraph_python_code_test_agents.py` 复用原有的协调、代码和测试 Agent，并用 LangGraph
管理“任务拆分 → Skill 路由 → 代码实现 → 测试编写 → 宿主 pytest”的状态迁移。协调器确定
源码路径、核心函数和需求后，工作流才从 `examples/skill_catalog/` 选择最多三个匹配的 Skill，
仅向代码 Agent 注入其正文；catalog 包含通用 Python、datetime、NumPy 和 SciPy Skill。失败时图会回到
`implement_code` 节点最多循环两次，不会重新执行 `write_tests`；流程分支只依据用户确认和 pytest 真实退出码。

先安装示例依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install langgraph
```

再从项目根目录运行：

```powershell
.\.venv\Scripts\python.exe examples\langgraph_python_code_test_agents.py
```

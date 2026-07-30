# Python 实验示例

## DeepSeek 工具对话

`pydantic_ai_deepseek_tool_chat.py` 使用 `.env` 的 `ABOOK_MODEL`、`ABOOK_API_KEY` 与 `ABOOK_BASE_URL`，通过 OpenAI 兼容接口连接 DeepSeek。它保留多轮消息历史，模型请求保持非流式；但每个工具调用和工具结果会在发生时立即打印，随后在下一次模型请求完成时打印模型回复。除时间和计算器外，示例还通过工具执行层注册了受工作区边界保护的 `read_file` 与 `bash`；示例只授权读取和 Bash，不授权写入或编辑。

可复用的轮次执行逻辑位于 `src/agent_runtime/run_turn.py`。调用方可通过 `on_response` 处理完整模型响应，通过 `on_event` 处理所有工具事件；后续增加事件展示逻辑时，无需复制 Agent 迭代和工具执行代码。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe examples\pydantic_ai_deepseek_tool_chat.py
```

可尝试输入：`现在几点？`、`请计算 (18 + 6) * 3`、`列出当前工作区的文件` 或 `查看 Git 状态`。工具在 Unix 使用 Bash，在 Windows 使用 PowerShell。输入 `/quit` 退出。

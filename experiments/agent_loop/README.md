# 最小 Agent Loop

这个实验只展示四个核心概念：

1. `skill.json` 只保存用于发现 Skill 的元数据。
2. `SKILL.md` 保存提供给 Agent 的完整指令。
3. `TaskState` 在多次模型调用之间保存计划、草稿、检查反馈和尝试次数。
4. 思考 Agent 运行一次；生成和检查 Agent 最多循环两次，防止无限循环。

```text
用户请求
   |
   v
加载 general/SKILL.md
   |
   v
思考 Agent
   |
   v
生成 Agent <-------+
   |                |
   v                |
检查 Agent --失败--+
   |
  通过或达到上限
   |
   v
TaskState
```

三个 Agent 是三个独立的 PydanticAI 调用，不共享消息历史。宿主只通过
`TaskState` 把必要结果交给下一阶段；这比复制完整对话更容易观察和控制。

实验直接读取仓库根目录 `.env` 中现有的 `ABOOK_MODEL`、`ABOOK_API_KEY` 和
`ABOOK_BASE_URL`：

```powershell
.\.venv\Scripts\python.exe experiments\agent_loop\main.py "解释什么是任务 DAG"
```

也可以直接在 VS Code 打开 `main.py`，点击右上角“运行 Python 文件”，然后在
集成终端的 `请输入请求：` 后输入问题。按 `F5` 并选择 `Debug Agent Loop` 会
使用 `.vscode/launch.json` 启动同一流程；配置必须使用 `integratedTerminal`，
因为 Debug Console 不支持 Python 的标准输入。

理解循环时，建议依次在以下语句设置断点：

- `state.plan = ...`：观察思考 Agent 的输出。
- `state.attempts += 1`：观察循环次数和上一次反馈。
- `state.draft = ...`：观察生成 Agent 的输出。
- `review = ...`：观察检查结果如何更新任务状态。

输出是完整的 JSON 任务状态。`approved` 为 `false` 表示达到两次生成上限后，
检查 Agent 仍未通过当前草稿。

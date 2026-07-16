# ABook Agent

ABook 是一个使用 [PydanticAI](https://ai.pydantic.dev/) 构建的命令行知识助理。它可以在多轮对话中检索本地知识，也可以在用户明确要求时保存新笔记。

当前版本以小而清晰的实现展示 PydanticAI 的核心能力：

- 使用 `Agent` 描述角色、行为边界和工具调用规则。
- 使用 `RunContext` 和依赖注入向工具提供知识库。
- 使用 Pydantic 模型验证写入的数据。
- 保留同一次 CLI 会话的消息历史，实现多轮对话。
- 通过模型 ID、API Key 和 `base_url` 调用 OpenAI 兼容服务。
- 使用本地 `TestModel` 完成不消耗 API 额度的自动化测试。

仓库顶层的 `experiments/agent_loop/` 还包含一个独立实验：确定性组织工作目录、
层级 `AGENTS.md` 和 Skill 目录作为共享工作区上下文，同时为每个 Agent 保存独立
Skill 与消息历史，并用结构化 TaskState 跟踪计划、修改文件和验证结果。Git 状态由
受限工具按需查询。父 Agent 可创建并继续子 Agent 会话，完成工作区发现、修改、
验证与审查。每个 Agent 的一百万 token 上下文达到 70% 时自动压缩旧历史；工具
结果使用证据 ID 和逐字摘录关联任务事实，压缩前会先生成结构化状态检查点。
固定项目指令与可变 Runtime 状态使用不同消息层级。该实验不属于
`src/abook_agent/` 产品包，不影响当前 CLI。

## 工作原理

```text
用户输入
   |
   v
命令行界面 (cli.py)
   |
   |  消息历史 + 当前问题 + 运行配置
   v
PydanticAI Agent (agent.py)
   |
   |  根据问题决定是否调用工具
   v
知识库工具 --------> JSON 存储 (knowledge.py)
   |
   v
模型组织最终回答
```

Agent 提供三个工具：

| 工具 | 行为 | 是否有副作用 |
| --- | --- | --- |
| `list_documents` | 列出全部文档标题和 ID | 否 |
| `search_documents` | 按关键词检索标题与正文 | 否 |
| `save_document` | 向本地知识库写入一条文档 | 是，仅在用户明确要求保存时调用 |

## 项目结构

```text
ABook/
|-- src/abook_agent/
|   |-- agent.py       # Agent、运行依赖和工具定义
|   |-- cli.py         # 多轮命令行交互入口
|   |-- config.py      # 环境变量配置与兼容模型构造
|   `-- knowledge.py   # 文档模型与 JSON 知识库
|-- tests/
|   |-- test_agent.py  # 存储和 Agent 离线测试
|   |-- test_config.py # 模型配置离线测试
|   `-- test_agent_loop.py # 上下文 Runtime 离线测试
|-- experiments/
|   `-- agent_loop/
|       |-- context.py         # 共享上下文与 Runtime 状态
|       |-- workspace_tools.py # 文件、PowerShell 与重试工具
|       |-- agent_runtime.py   # Agent 创建、Skill 与委派
|       `-- main.py            # 实验命令行入口
|-- .env.example       # 环境变量模板，不包含真实密钥
|-- AGENTS.md          # 项目编码与协作约束
|-- pyproject.toml     # 包信息和运行依赖
`-- requirements.txt  # 固定版本的直接依赖
```

## 环境要求

- Windows PowerShell
- Python 3.11 或更高版本
- 一个 OpenAI Chat Completions 兼容接口

项目已经在根目录创建了 `.venv`。在新的环境中，可使用以下命令重新创建：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` 只把项目以可编辑模式安装到当前 `.venv`，用于解析 `src`
目录中的包。它不会创建控制台命令，也不会修改系统 Python 或全局 PATH。
之后修改 `src/` 下的代码无需重复安装。

如果 PowerShell 禁止运行激活脚本，也可以不激活环境，直接使用虚拟环境中的可执行文件：

```powershell
.\.venv\Scripts\python.exe -m abook_agent.cli
```

## 配置模型

先复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

打开 `.env`，填写兼容接口提供的三个配置项：

```dotenv
ABOOK_MODEL=your-model-id
ABOOK_API_KEY=your-api-key
ABOOK_BASE_URL=https://your-endpoint.example.com/v1
```

ABook 始终通过 PydanticAI 的 `OpenAIChatModel` 和 `OpenAIProvider` 发送请求。
`ABOOK_MODEL` 直接填写接口提供的模型 ID，不要添加 `openai:` 等供应商前缀。
三个配置项都是必填项，缺失时程序会在启动阶段列出缺失的变量。

配置项说明：

| 变量 | 说明 |
| --- | --- |
| `ABOOK_MODEL` | 兼容接口提供的原始模型 ID |
| `ABOOK_API_KEY` | 兼容接口的访问密钥 |
| `ABOOK_BASE_URL` | API 根地址，通常以 `/v1` 结尾 |

`.env` 已被 `.gitignore` 忽略，不应提交到版本库。`.env.example` 只能存放明显的占位值。

## 运行 Agent

激活虚拟环境后通过模块入口运行：

```powershell
python -m abook_agent.cli
```

输入 `/quit`、`/exit`，或按 `Ctrl+C` 可结束会话。

### 使用案例

保存信息时应明确表达写入意图：

```text
You: 请记住，生产环境部署在上海区域。
ABook: 已保存生产环境部署区域的信息。
```

之后可以在同一次或新的会话中查询：

```text
You: 生产环境部署在哪里？
ABook: 根据知识库中的记录，生产环境部署在上海区域。
```

也可以要求查看当前文档：

```text
You: 列出知识库里的所有文档。
```

自然语言回答由所配置的模型生成，具体措辞可能不同。Agent 只有在工具返回相应内容后，才应声称信息来自知识库。

## 知识库存储

知识默认写入 `data/knowledge.json`。每条记录都经过 Pydantic 验证，结构如下：

```json
{
  "id": "自动生成的唯一标识",
  "title": "长度为 1 到 200 的标题",
  "content": "长度为 1 到 20000 的正文"
}
```

当前检索实现适合本地小型知识库：查询会被拆分为不区分大小写的词项，按照词项在标题和正文中的出现次数排序。当数据量明显增长时，应将 `KnowledgeStore` 替换为数据库或带索引的检索服务，而不是继续扩大 JSON 文件。

当前 JSON 存储不适合多个进程同时写入。不要同时启动多个会修改同一知识库文件的 Agent 实例。

## 运行测试

测试使用 PydanticAI 的 `TestModel`，不会访问外部模型，也不需要 API Key：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

完整的本地验证命令为：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
```

## 开发约束

修改代码前请阅读 [AGENTS.md](AGENTS.md)。其中定义了模块边界、类型标注、详实注释、PydanticAI 工具、安全和测试要求。

开发时需要特别遵守以下规则：

- 模型名称和密钥只能通过配置传入，不能硬编码进 Agent。
- Agent 工具通过 `AgentDependencies` 获取存储实例。
- 写入工具只有在用户明确授权时才能调用。
- 新增行为必须提供不依赖外部 API 的测试。
- 修改实现时同步更新相关注释、配置示例和 README。

## 常见问题

### 提示缺少 API Key

确认 `.env` 位于项目启动目录，并设置了非空的 `ABOOK_MODEL`、
`ABOOK_API_KEY` 和 `ABOOK_BASE_URL`。

### 提示找不到 `abook_agent` 模块

确认项目已安装到当前 `.venv`。可以重新执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

安装只发生在项目虚拟环境中。完成后使用
`.\.venv\Scripts\python.exe -m abook_agent.cli` 启动。

### 没有找到知识库数据

知识库固定存储在启动目录下的 `data/knowledge.json`。请确认运行命令时位于项目根目录。

### Agent 没有检索到语义相近的内容

当前实现是关键词匹配，不是向量语义检索。查询中需要包含文档标题或正文中实际出现的关键词。

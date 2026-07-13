# ABook 项目开发约束

本文件适用于仓库根目录及其所有子目录。任何开发者或编码 Agent 在修改代码前，都应先阅读并遵守本文件。若子目录存在更具体的 `AGENTS.md`，则子目录文件可补充或覆盖这里的规则。

## 1. 项目目标与技术栈

ABook 是一个基于 PydanticAI 的本地知识助理。目前使用以下技术：

- Python 3.11 或更高版本。
- PydanticAI 2.x，负责 Agent、工具调用和模型接入。
- Pydantic 2.x，负责数据验证和结构化模型。
- `python-dotenv`，负责加载本地环境变量。
- `unittest`，负责不依赖外部 API 的自动化测试。
- JSON 文件，作为当前阶段的本地知识存储。

开发时优先延续现有架构。除非确实解决了当前需求，不要引入 Web 框架、数据库、任务队列或新的依赖注入框架。

## 2. 目录职责

```text
src/abook_agent/
  agent.py       # Agent 定义、依赖类型和可供模型调用的工具
  cli.py         # 命令行交互入口，不放置领域逻辑
  config.py      # 环境变量和运行配置
  knowledge.py   # 文档模型与知识库存取逻辑
tests/           # 与 src 结构对应的离线测试
```

新增代码时应放入职责最接近的模块。CLI、模型接口和存储层之间必须保持清晰边界：

- CLI 不直接读写知识库文件。
- Agent 工具通过 `AgentDependencies` 获取依赖，不创建隐藏的全局存储实例。
- 存储层不依赖 PydanticAI，也不调用语言模型。
- 配置统一从 `Settings` 获取，不在业务代码中散落读取环境变量。

## 3. Python 编码规范

### 3.1 基本规则

- 遵循 PEP 8，使用 4 个空格缩进。
- 文件、变量和函数使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。
- 新增的公共函数、方法和类必须提供完整类型标注。
- 优先使用现代 Python 类型语法，例如 `list[str]`、`str | None`。
- 函数应保持单一职责。出现复杂分支、重复逻辑或难以命名的大段代码时，应拆分函数。
- 使用 `pathlib.Path` 处理路径，不手工拼接路径分隔符。
- 使用标准 JSON API 或 Pydantic 模型处理结构化数据，不用字符串拼接生成或解析 JSON。
- 默认使用 UTF-8 读写文本；涉及中文内容时必须显式指定 `encoding="utf-8"`。
- 不捕获异常后静默忽略。应在合适的边界转换为清楚、可操作的错误信息。
- 不提交密钥、令牌、真实 `.env`、运行数据、缓存或虚拟环境。

### 3.2 命名要求

名称应表达业务含义，避免 `data`、`item`、`obj`、`tmp` 等缺少上下文的名称。在非常短且语义明显的局部作用域中可以使用 `item`。

推荐：

```python
matching_documents = store.search(query)
model_name = settings.model
```

不推荐：

```python
d = store.search(q)
x = settings.model
```

## 4. 注释与文档规范

本项目要求提供详实、准确且可维护的注释。注释的重点是说明“为什么这样设计”“有哪些约束”“调用者需要注意什么”，而不是逐行翻译代码。

### 4.1 必须添加注释或文档字符串的场景

- 模块承担了不容易从文件名判断的职责。
- 公共类、公共函数、Agent 工具和非直观的私有函数。
- Pydantic 字段存在单位、范围、格式或兼容性约束。
- 算法包含排序、评分、截断、重试、并发或状态转换。
- 代码为规避第三方库限制、兼容性问题或安全问题而采用特殊写法。
- 异常处理只允许在特定边界进行，或存在预期的降级行为。
- 修改数据、调用外部服务或产生其他副作用的代码。

公共函数的文档字符串应根据需要包含：用途、参数含义、返回结果、副作用、可能抛出的异常。简单属性或一眼可知的函数无需机械地列出所有栏目。

推荐：

```python
def search(self, query: str, limit: int = 5) -> list[Document]:
    """按关键词相关度检索本地文档。

    查询会被拆分为不区分大小写的词项，每次出现都会增加相关度。
    结果按相关度降序返回，最多返回 ``limit`` 条。该实现适用于小型
    本地知识库；数据规模扩大后应替换为索引检索，而不是继续增加遍历逻辑。

    Args:
        query: 用户输入的自然语言查询；空白查询不匹配任何文档。
        limit: 最大结果数，调用方必须保证其为正数。

    Returns:
        按相关度排序的文档列表。
    """
```

不推荐：

```python
# 搜索文档
def search(query: str) -> list[Document]:
    # 遍历文档
    for document in documents:
        ...
```

后者只是复述代码，没有解释匹配语义、排序规则或适用边界。

### 4.2 注释维护要求

- 修改实现时同步修改相关注释、文档字符串、README 和示例。
- 不保留已经失效的注释或被注释掉的旧代码；版本历史应由版本控制保存。
- TODO 必须说明待解决的问题和完成条件，不能只写 `TODO: fix`。
- 不在注释中记录密钥、内部地址或其他敏感信息。
- 注释可以使用中文；代码标识符、协议字段和第三方 API 原名保持英文。

## 5. PydanticAI 约束

### 5.1 Agent 定义

- Agent 的系统指令应明确角色、工具使用条件和事实边界。
- 不得指示模型伪造工具结果或把模型常识描述成知识库内容。
- 模型 ID、API Key 和 Base URL 由运行配置传入，不硬编码在 Agent 定义中，以便测试替换为 `TestModel`。
- 多轮对话必须显式传递消息历史，避免看似多轮、实际无上下文的交互。
- 需要结构化结果时使用 Pydantic 模型作为输出类型，不自行解析模型生成的 JSON 字符串。

### 5.2 工具定义

每个 Agent 工具都必须：

- 使用清晰、具体的函数名称和文档字符串，使模型能正确判断调用时机。
- 对参数设置合理的长度、数量或范围限制。
- 通过 `RunContext[AgentDependencies]` 获取依赖。
- 返回稳定、容易被模型理解的结果；空结果和错误结果应有明确语义。
- 明确区分只读工具和有副作用的工具。
- 只有用户明确授权时，才允许模型调用写入、删除、发送或其他有副作用的工具。

示例：

```python
@abook_agent.tool
def search_documents(
    ctx: RunContext[AgentDependencies], query: str, limit: int = 5
) -> str:
    """检索知识库；仅在回答需要引用已存储信息时调用。

    ``limit`` 会在工具边界限制到 1 到 10，防止模型请求过多内容，
    同时控制进入模型上下文的文本量。
    """
    bounded_limit = max(1, min(limit, 10))
    matches = ctx.deps.store.search(query, bounded_limit)
    if not matches:
        return "No matching documents found."
    return "\n\n".join(document.content for document in matches)
```

有副作用的工具必须在 Agent 指令和工具文档中同时写明授权条件：

```python
@abook_agent.tool
def save_document(
    ctx: RunContext[AgentDependencies], title: str, content: str
) -> str:
    """仅当用户明确要求记住或保存内容时，才将文档写入知识库。"""
    ...
```

## 6. 数据与安全约束

- 所有外部输入在进入存储层前必须经过 Pydantic 或等效的显式验证。
- 文件路径来自配置时，应避免允许不受控输入覆盖任意文件。
- 修改持久化格式时必须考虑旧数据兼容性，并增加迁移或兼容读取测试。
- Agent 输出不应被视为可信数据；执行命令、写文件或调用外部接口前必须再次验证。
- 日志和异常信息不得泄漏 API Key、完整认证头或用户隐私数据。
- `.env.example` 只使用明显的占位值，不放入可用凭据。
- 对删除、覆盖、批量修改等不可逆操作，应先获得用户明确确认。

## 7. 测试规范

任何行为修改都必须配套测试。测试应覆盖正常路径以及与改动相关的边界或失败路径。

- Agent 测试默认使用 PydanticAI `TestModel`，不得依赖真实 API Key 或产生费用。
- 文件存储测试使用 `tempfile.TemporaryDirectory`，不得污染仓库的 `data/`。
- 测试名称描述行为和预期结果，例如 `test_empty_query_returns_no_documents`。
- 测试必须可重复、互相隔离，不依赖执行顺序、当前日期或外部网络。
- 修复缺陷时先添加能够复现缺陷的测试，再修复实现。
- 不用降低断言强度、跳过测试或捕获所有异常的方式掩盖失败。

示例：

```python
def test_agent_can_call_list_documents_tool(self) -> None:
    """验证工具注册和依赖注入，而不访问真实模型接口。"""
    with tempfile.TemporaryDirectory() as directory:
        dependencies = AgentDependencies(
            store=KnowledgeStore(Path(directory) / "knowledge.json")
        )
        model = TestModel(call_tools=["list_documents"])

        result = abook_agent.run_sync(
            "List the notes", deps=dependencies, model=model
        )

        self.assertIn("knowledge base is empty", result.output)
```

提交修改前至少运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
```

## 8. 依赖与配置管理

- 运行依赖声明在 `pyproject.toml`；`requirements.txt` 保留直接依赖的可复现版本。
- 新增依赖前，先确认标准库或现有依赖不能合理解决问题。
- 新增环境变量时，同步更新 `Settings`、`.env.example` 和 README。
- 运行配置只包含 `ABOOK_MODEL`、`ABOOK_API_KEY` 和 `ABOOK_BASE_URL`。
- 不在模块导入阶段验证 API Key 或连接外部服务，以保证离线测试和工具导入可用。

## 9. 变更原则

- 保持改动范围与需求一致，不顺手重构无关模块。
- 优先复用现有模式，只有确实降低复杂度或消除有意义的重复时才新增抽象。
- 修改公共行为或数据格式时，更新测试、类型、注释和用户文档。
- 保留工作区中不属于当前任务的修改，不执行破坏性的 Git 或文件操作。
- 完成后说明改动内容、验证命令和未验证的风险；不要声称未实际运行的检查已经通过。

## 10. 完成标准

一项代码任务只有同时满足以下条件才算完成：

1. 实现符合需求并保持模块职责清晰。
2. 公共接口、关键算法、约束与副作用具有详实且准确的注释。
3. 已添加或更新与风险相匹配的离线测试。
4. 测试、语法编译和依赖检查通过。
5. 配置示例与用户文档和实现保持一致。
6. 未引入密钥、运行数据、缓存或无关文件。

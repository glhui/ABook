"""定义 ABook Agent、运行依赖以及模型可调用的知识库工具。

Agent 本身不绑定具体模型，也不持有全局存储。模型和依赖在每次运行时注入，
从而允许 CLI 使用 OpenAI 兼容接口，同时让测试使用完全离线的 ``TestModel``。
"""

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from .knowledge import KnowledgeStore


@dataclass
class AgentDependencies:
    """向一次 Agent 运行及其工具显式提供的外部依赖。

    Attributes:
        store: 本次运行访问的知识库。由调用方决定实际文件位置，因此工具
            无需依赖全局状态，测试也可以使用临时目录。
    """

    store: KnowledgeStore


# 模型刻意留到 run_sync/run 时传入。若在这里绑定真实模型，模块导入和
# 离线测试将隐式依赖接口配置，不利于测试以及切换兼容服务。
abook_agent = Agent(
    deps_type=AgentDependencies,
    name="abook_agent",
    instructions=(
        "You are ABook, a concise knowledge assistant. Use the knowledge tools "
        "when the user asks about stored information. Never claim that a fact is "
        "in the knowledge base unless a tool result contains it. You may save a "
        "note only when the user explicitly asks you to remember or store it."
    ),
)


@abook_agent.tool
def list_documents(ctx: RunContext[AgentDependencies]) -> str:
    """列出知识库中的全部文档标题和 ID。

    这是只读工具，适合在用户询问“有哪些笔记”时调用。返回内容不包含正文，
    避免仅为展示目录就把全部知识库加载到模型上下文中。
    """
    documents = ctx.deps.store.list()
    if not documents:
        return "The knowledge base is empty."
    return "\n".join(f"{item.id}: {item.title}" for item in documents)


@abook_agent.tool
def search_documents(
    ctx: RunContext[AgentDependencies], query: str, limit: int = 5
) -> str:
    """按查询词检索已存储文档，并向模型返回标题、ID 和正文。

    Args:
        ctx: PydanticAI 注入的运行上下文，其中包含当前知识库。
        query: 用于匹配标题和正文的自然语言查询。
        limit: 期望的最大结果数。工具边界会将它限制在 1 到 10，避免模型
            请求过多正文并无意中占满上下文窗口。

    Returns:
        供模型引用的纯文本检索结果；没有匹配项时返回明确的空结果消息。
    """
    bounded_limit = max(1, min(limit, 10))
    matches = ctx.deps.store.search(query, bounded_limit)
    if not matches:
        return "No matching documents found."
    return "\n\n".join(
        f"[{item.title}] (id: {item.id})\n{item.content}" for item in matches
    )


@abook_agent.tool
def save_document(
    ctx: RunContext[AgentDependencies], title: str, content: str
) -> str:
    """仅在用户明确要求记住或保存内容时写入一条文档。

    这是有副作用的工具。调用会修改当前知识库文件；系统指令明确禁止模型
    根据含糊陈述自行保存。首尾空白在验证前移除，空标题或正文会由
    ``Document`` 的 Pydantic 约束拒绝。
    """
    document = ctx.deps.store.add(title=title.strip(), content=content.strip())
    return f"Saved '{document.title}' with id {document.id}."

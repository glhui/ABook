"""提供经过数据验证的文档模型和 JSON 知识库存储。"""

import json
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """知识库中的单条文档。

    标题和正文长度限制既用于拒绝无效数据，也用于控制单个工具结果可能占用
    的模型上下文。ID 在首次创建时生成，读取已有数据时则保留文件中的值。
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=20_000)


class KnowledgeStore:
    """面向单进程、小型知识库的 JSON 文件存储。

    每次操作都会读取整个文件，写入也会重写完整文档列表。这让格式直观且
    易于替换，但不适合大型数据集或多个进程并发写入。数据规模增长后应保留
    此类的公开语义，并将实现替换为数据库或检索索引。
    """

    def __init__(self, path: Path) -> None:
        """绑定知识库路径，但不立即创建或读取文件。"""
        self.path = path

    def list(self) -> list[Document]:
        """读取并验证全部文档。

        文件尚不存在表示知识库为空，不视为错误。已存在但不是合法 JSON，
        或记录不符合 ``Document`` 约束时，解析/验证异常会传递给调用方，避免
        在数据损坏时悄悄返回不完整结果。
        """
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [Document.model_validate(item) for item in payload]

    def search(self, query: str, limit: int = 5) -> list[Document]:
        """使用不区分大小写的词频对文档进行简单相关度排序。

        查询按空白拆分；每个词项在标题和正文中每出现一次便获得一分。结果
        按总分降序排列，并截取前 ``limit`` 项。此算法有意保持轻量，不提供
        分词、模糊匹配或向量语义检索能力。

        Args:
            query: 用户查询；空字符串或纯空白查询返回空列表。
            limit: 最大结果数。Agent 工具会把外部输入限制在 1 到 10。

        Returns:
            仅包含至少一个查询词项的文档，按相关度从高到低排列。
        """
        terms = query.casefold().split()
        if not terms:
            return []

        def score(document: Document) -> int:
            # 标题与正文使用同一权重。这里选择可解释的词频评分，以保持当前
            # JSON 实现简单；若将来标题需要更高权重，应同步更新测试和 README。
            text = f"{document.title}\n{document.content}".casefold()
            return sum(text.count(term) for term in terms)

        matches = ((score(document), document) for document in self.list())
        ranked = sorted(
            (item for item in matches if item[0] > 0),
            key=lambda item: item[0],
            reverse=True,
        )
        return [document for _, document in ranked[:limit]]

    def add(self, title: str, content: str) -> Document:
        """验证并追加一条文档，然后持久化整个知识库。

        父目录会按需创建。该操作会修改磁盘文件，且没有跨进程锁，因此调用方
        不应让多个实例并发写入同一路径。Pydantic 验证失败或文件写入失败时，
        异常会传递给上层，由 CLI 作为本次 Agent 运行错误展示。

        Returns:
            包含自动生成 ID 的已保存文档。
        """
        document = Document(title=title, content=content)
        documents = self.list()
        documents.append(document)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                [item.model_dump() for item in documents],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return document

"""提供 ABook 的同步、多轮命令行入口。"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.messages import ModelMessage

from .agent import AgentDependencies, abook_agent
from .config import Settings, create_model
from .knowledge import KnowledgeStore


KNOWLEDGE_PATH = Path("data/knowledge.json")


def run_chat() -> None:
    """加载配置并运行交互会话，直到用户退出或输入流结束。

    消息历史仅保存在当前进程内，退出后不会持久化；知识库写入则由工具立即
    保存到 JSON 文件。模型或工具异常在 CLI 边界转换为可见错误，使单次失败
    不会直接终止整个会话。
    """
    load_dotenv()
    settings = Settings.from_env()
    deps = AgentDependencies(store=KnowledgeStore(KNOWLEDGE_PATH))
    model = create_model(settings)
    history: list[ModelMessage] = []

    print(f"ABook agent ({settings.model}). \n Type /quit to exit.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.casefold() in {"/quit", "/exit"}:
            # casefold() 提供更激进的小写转换（如德语 ß->ss），适合无大小写匹配的字符串比对
            break

        try:
            result = abook_agent.run_sync(
                prompt,
                deps=deps,
                model=model,
                message_history=history,
            )
        except Exception as exc:
            # CLI 是最外层交互边界，需要展示接口、网络、验证和存储错误并
            # 允许用户继续输入。领域层不捕获这些异常，以免丢失具体失败原因。
            print(f"Agent error: {exc}")
            continue

        print(f"ABook: {result.output}")
        # all_messages 同时包含本轮用户消息、工具调用/结果及模型回复。只有
        # 成功运行后才替换历史，失败的半成品消息不会污染后续对话上下文。
        history = result.all_messages()


def main() -> None:
    """控制台脚本和 ``python -m`` 共用的无参数入口。"""
    run_chat()


if __name__ == "__main__":
    main()

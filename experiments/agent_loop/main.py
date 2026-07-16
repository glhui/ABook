"""运行最小上下文 Runtime 与工具型 Agent 的命令行入口。"""

import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .agent_runtime import create_agent, run_agent
from .context import ContextRuntime, TaskState, WorkspaceContextBuilder


def create_model() -> OpenAIChatModel:
    """使用项目 .env 中的 OpenAI 兼容配置创建模型。"""
    load_dotenv()
    provider = OpenAIProvider(
        base_url=os.environ["ABOOK_BASE_URL"],
        api_key=os.environ["ABOOK_API_KEY"],
    )
    return OpenAIChatModel(os.environ["ABOOK_MODEL"], provider=provider)


def main() -> None:
    """组织上下文，运行带工作区工具的执行 Agent，并打印回答。"""
    request = " ".join(sys.argv[1:]).strip()
    if not request:
        request = input("请输入请求：").strip()
    if not request:
        raise SystemExit("请求不能为空")

    experiment_root = Path(__file__).parent
    workspace_context = WorkspaceContextBuilder(
        workspace_root=Path.cwd(),
        skills_root=experiment_root / "skills",
    ).build()
    runtime = ContextRuntime(workspace_context, TaskState(goal=request))
    root_context = runtime.create_agent_context("root", request, "general")
    result = run_agent(create_agent(create_model()), runtime, root_context)
    print(result.output)


if __name__ == "__main__":
    main()

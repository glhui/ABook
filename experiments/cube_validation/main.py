"""运行二阶魔方复原的手动编排 Agent 流。"""

import asyncio
from pathlib import Path
import sys

from experiments.agent_loop.core.context import ContextRuntime, TaskState, WorkspaceContextBuilder
from experiments.agent_loop.core.main import create_model

from .flow import CubeValidationFlow


def main() -> None:
    """读取打乱公式，运行求解、验证和汇总三个显式步骤。"""
    scramble = " ".join(sys.argv[1:]).strip() or "R U R' U'"
    experiment_root = Path(__file__).parent.parent
    workspace_context = WorkspaceContextBuilder(
        workspace_root=Path.cwd(),
        skills_root=experiment_root / "agent_loop" / "skills",
    ).build()
    runtime = ContextRuntime(
        workspace_context,
        TaskState(goal=f"验证二阶魔方复原：{scramble}"),
    )
    model = create_model()
    result = asyncio.run(CubeValidationFlow(model, runtime, scramble).run())
    print(f"Candidate: {result.candidate}")
    print(f"Deterministic verdict: {result.deterministic_verdict}")
    print(f"Verification: {result.verification}")
    print(f"Conclusion: {result.conclusion}")


if __name__ == "__main__":
    main()

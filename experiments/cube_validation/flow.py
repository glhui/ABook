"""不用 Workflow 抽象的二阶魔方手动 Agent 编排。"""

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.toolsets import FunctionToolset

from experiments.agent_loop.core.context import AgentContext, AgentDependencies, ContextRuntime
from experiments.agent_loop.core.runner import AgentTurnResult

from .cube import CubeState, find_solution, verify_solution


@dataclass(frozen=True)
class CubeValidationResult:
    """三个独立角色的输出，保留候选和验证证据以便审计。"""

    candidate: str
    deterministic_verdict: str
    verification: str
    conclusion: str


class CubeToolsets:
    """按能力组合创建通用魔方工具集，集中管理工具编排而非复制工具列表。"""

    def __init__(self, scramble: str) -> None:
        self.scramble = scramble

    def create(self, *, can_search: bool, can_verify: bool) -> FunctionToolset[AgentDependencies]:
        """为角色提供只读状态工具，以及可选搜索或验证能力。"""
        def inspect_state() -> str:
            state = CubeState().apply(self.scramble)
            return f"scramble={self.scramble}\npermutation={state.permutation}\norientation={state.orientation}"

        tools = [inspect_state]
        if can_search:
            def search_solution(max_depth: int = 7) -> str:
                solution = find_solution(self.scramble, max_depth)
                return solution if solution is not None else "NO_SOLUTION_WITHIN_DEPTH"
            tools.append(search_solution)
        if can_verify:
            def check_solution(solution: str) -> str:
                return "SOLVED" if verify_solution(self.scramble, solution) else "NOT_SOLVED"
            tools.append(check_solution)
        return FunctionToolset[AgentDependencies](tools=tools)


class CubeValidationFlow:
    """显式串联求解、验证与汇总 Agent，不依赖 ``experiments.workflow``。"""

    def __init__(self, model: Model, runtime: ContextRuntime, scramble: str) -> None:
        CubeState().apply(scramble)
        self.runtime = runtime
        self.scramble = scramble
        toolsets = CubeToolsets(scramble)
        self.solver = Agent(model, deps_type=AgentDependencies, toolsets=[toolsets.create(can_search=True, can_verify=True)], instructions="给出二阶魔方复原候选。先检查状态；可搜索并验证。最终只输出动作记号。")
        self.verifier = Agent(model, deps_type=AgentDependencies, toolsets=[toolsets.create(can_search=False, can_verify=True)], instructions="独立验证候选动作。必须调用 check_solution，最终说明 SOLVED 或 NOT_SOLVED。")
        self.reporter = Agent(model, deps_type=AgentDependencies, instructions="根据给定候选和验证结果，简洁报告是否已验证复原；验证失败时不要声称成功。")

    async def run(self) -> CubeValidationResult:
        """以 Python 明确传递步骤产物，三个 Agent 不共享消息历史。"""
        solver_context = self.runtime.create_agent_context("cube-solver", "求解二阶魔方", "general", role="task")
        candidate_turn: AgentTurnResult[str] = await self.runtime.get_agent_runner().run_turn(self.solver, self.runtime, solver_context, f"打乱公式：{self.scramble}")
        try:
            deterministic_verdict = (
                "SOLVED"
                if verify_solution(self.scramble, candidate_turn.output)
                else "NOT_SOLVED"
            )
        except ValueError as error:
            deterministic_verdict = f"INVALID_NOTATION: {error}"
        verifier_context = self.runtime.create_agent_context("cube-verifier", "验证二阶魔方候选", "general", role="task")
        verification_turn: AgentTurnResult[str] = await self.runtime.get_agent_runner().run_turn(self.verifier, self.runtime, verifier_context, f"打乱公式：{self.scramble}\n候选动作：{candidate_turn.output}\n宿主确定性判定：{deterministic_verdict}")
        reporter_context = self.runtime.create_agent_context("cube-reporter", "汇总验证结论", "general", role="task")
        conclusion_turn: AgentTurnResult[str] = await self.runtime.get_agent_runner().run_turn(self.reporter, self.runtime, reporter_context, f"打乱：{self.scramble}\n候选：{candidate_turn.output}\n确定性判定：{deterministic_verdict}\nAgent 验证：{verification_turn.output}")
        return CubeValidationResult(candidate_turn.output, deterministic_verdict, verification_turn.output, conclusion_turn.output)

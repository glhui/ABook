"""测试最小上下文 Runtime，不访问真实模型接口。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import tempfile
import unittest

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import ToolDefinition

from experiments.agent_loop.agent_runtime import (
    TaskAssignmentRequest,
    assign_tasks,
    cancel_assignment,
    create_coordinator_agent,
    inspect_assignment,
    list_assignments,
    run_coordinator_turn,
    select_skill,
    send_task_feedback,
    update_task_state,
)
from experiments.agent_loop.context import (
    AgentCallEvent,
    AgentContext,
    AgentDependencies,
    AssignmentCompletionEvent,
    AssignmentSession,
    ContextWindowPolicy,
    ContextRuntime,
    EvidenceQuoteClaim,
    FactClaim,
    ProjectInstruction,
    SkillMetadata,
    TaskState,
    WorkspaceContext,
    WorkspaceContextBuilder,
)
from experiments.agent_loop.conversation import ConversationSession
from experiments.agent_loop.main import (
    DeepSeekThinkingChatModel,
    run_conversation,
)
from experiments.agent_loop.persistence import RuntimeStateStore
from experiments.agent_loop.runner import AgentCallLimits, AgentRunner
from experiments.agent_loop.scheduler import AssignmentScheduler
from experiments.agent_loop.workspace_tools import (
    RecoverableToolError,
    WorkspaceTextEdit,
    apply_workspace_edits,
    list_workspace_files,
    read_workspace_file,
    replace_workspace_text,
    run_python_validation,
    run_powershell_command,
    search_workspace_text,
    write_workspace_file,
)


def create_test_runtime(
    workspace_root: Path,
    task: str = "测试任务",
    skill_content: str = "根据工具证据完成任务。",
) -> tuple[ContextRuntime, AgentContext]:
    """创建带 general Skill 的最小 Runtime 和协调 AgentContext。"""
    resolved_workspace_root = workspace_root.resolve()
    skills_root = resolved_workspace_root / "skills"
    general_skill = skills_root / "general"
    general_skill.mkdir(parents=True, exist_ok=True)
    (general_skill / "skill.json").write_text(
        '{"name":"general","description":"通用测试 Skill"}',
        encoding="utf-8",
    )
    (general_skill / "SKILL.md").write_text(
        skill_content, encoding="utf-8"
    )
    workspace_context = WorkspaceContext(
        workspace_root=resolved_workspace_root.as_posix(),
        working_directory=resolved_workspace_root.as_posix(),
        skills_root=skills_root.as_posix(),
        project_instructions=(),
        available_skills=(
            SkillMetadata(
                name="general", description="通用测试 Skill"
            ),
        ),
    )
    runtime = ContextRuntime(workspace_context, TaskState(goal=task))
    return runtime, runtime.create_agent_context(
        "coordinator", task, "general", role="coordinator"
    )


def run_test_agent(
    agent: Agent[AgentDependencies, str],
    runtime: ContextRuntime,
    agent_context: AgentContext,
    request: str | None = None,
) -> AgentRunResult[str]:
    """为不启动后台任务的测试同步执行一轮异步 Agent 调用。"""
    return asyncio.run(
        run_coordinator_turn(agent, runtime, agent_context, request)
    ).raw_result


class WorkspaceContextBuilderTests(unittest.TestCase):
    """验证共享工作区上下文的确定性组织。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name)
        self.skills_root = self.workspace_root / "skills"
        general_skill = self.skills_root / "general"
        general_skill.mkdir(parents=True)
        (general_skill / "skill.json").write_text(
            '{"name":"general","description":"通用测试 Skill"}',
            encoding="utf-8",
        )
        (general_skill / "SKILL.md").write_text(
            "只根据已提供的上下文回答。",
            encoding="utf-8",
        )
        review_skill = self.skills_root / "review"
        review_skill.mkdir()
        (review_skill / "skill.json").write_text(
            '{"name":"review","description":"审查代码"}',
            encoding="utf-8",
        )
        (review_skill / "SKILL.md").write_text(
            "审查当前修改。", encoding="utf-8"
        )

    def test_build_loads_only_applicable_agents_files_in_scope_order(
        self,
    ) -> None:
        """只加载通往工作目录的指令，并保持根目录到子目录的顺序。"""
        (self.workspace_root / "AGENTS.md").write_text(
            "根目录约束", encoding="utf-8"
        )
        source_directory = self.workspace_root / "src" / "feature"
        source_directory.mkdir(parents=True)
        (self.workspace_root / "src" / "AGENTS.md").write_text(
            "src 约束", encoding="utf-8"
        )
        unrelated_directory = self.workspace_root / "docs"
        unrelated_directory.mkdir()
        (unrelated_directory / "AGENTS.md").write_text(
            "不适用的约束", encoding="utf-8"
        )

        workspace_context = WorkspaceContextBuilder(
            self.workspace_root, self.skills_root
        ).build(
            working_directory=source_directory,
        )

        self.assertEqual(
            ["AGENTS.md", "src/AGENTS.md"],
            [
                instruction.path
                for instruction in workspace_context.project_instructions
            ],
        )
        self.assertNotIn(
            "不适用的约束", workspace_context.render_instructions()
        )
        self.assertNotIn(
            "只根据已提供的上下文回答。",
            workspace_context.render_instructions(),
        )
        self.assertEqual(
            ["general", "review"],
            [
                metadata.name
                for metadata in workspace_context.available_skills
            ],
        )
        self.assertNotIn("Git 快照", workspace_context.render_instructions())

    def test_context_does_not_preload_workspace_files(self) -> None:
        """业务文件必须留给工具按需读取，不能进入初始上下文。"""
        (self.workspace_root / "secret.txt").write_text(
            "不应预加载的正文", encoding="utf-8"
        )

        workspace_context = WorkspaceContextBuilder(
            self.workspace_root, self.skills_root
        ).build()

        rendered_context = workspace_context.render_instructions()
        self.assertNotIn("secret.txt", rendered_context)
        self.assertNotIn("不应预加载的正文", rendered_context)

    def test_runtime_persistence_serializes_concurrent_tool_threads(
        self,
    ) -> None:
        """并发工具线程保存快照时不会争用同一个临时文件。"""
        workspace_context = WorkspaceContextBuilder(
            self.workspace_root, self.skills_root
        ).build()
        runtime = ContextRuntime(
            workspace_context, TaskState(goal="并发持久化")
        )
        coordinator = runtime.create_agent_context(
            "coordinator", "并发持久化", "general", role="coordinator"
        )
        state_store = RuntimeStateStore(
            self.workspace_root / ".abook" / "runtime-state.json"
        )
        runtime.persistence_handler = state_store.save

        def register(index: int) -> None:
            runtime.register_evidence(
                "file_read",
                f"file-{index}.txt",
                "concurrent test",
                f"content-{index}",
                coordinator.agent_id,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(register, range(20)))

        restored = state_store.load(workspace_context)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(20, len(restored.evidence_records))
        self.assertEqual(
            [],
            list((self.workspace_root / ".abook").glob("*.tmp")),
        )

    def test_runtime_rejects_blank_task_and_external_directory(self) -> None:
        """Agent 任务和共享工作目录分别在所属边界完成校验。"""
        builder = WorkspaceContextBuilder(
            self.workspace_root, self.skills_root
        )

        runtime = ContextRuntime(
            builder.build(), TaskState(goal="分析模块")
        )
        with self.assertRaisesRegex(ValueError, "任务不能为空"):
            runtime.create_agent_context("coordinator", "  ", "general")
        with self.assertRaisesRegex(ValueError, "协调 Agent 上下文不存在"):
            runtime.create_agent_context(
                "worker-1",
                "执行任务",
                "general",
                role="task",
                coordinator_id="missing",
            )

        with tempfile.TemporaryDirectory() as external_directory:
            with self.assertRaisesRegex(ValueError, "工作目录必须"):
                builder.build(
                    working_directory=Path(external_directory),
                )

    def test_task_agent_state_hides_other_agent_evidence(self) -> None:
        """任务 Agent 只看到自己的证据，协调 Agent 可汇总全部证据。"""
        runtime = ContextRuntime(
            WorkspaceContextBuilder(
                self.workspace_root, self.skills_root
            ).build(),
            TaskState(goal="整体任务"),
        )
        coordinator_context = runtime.create_agent_context(
            "coordinator", "协调任务", "general", role="coordinator"
        )
        first_task_context = runtime.create_agent_context(
            "explorer-1",
            "任务一",
            "general",
            role="task",
            coordinator_id="coordinator",
        )
        second_task_context = runtime.create_agent_context(
            "explorer-2",
            "任务二",
            "general",
            role="task",
            coordinator_id="coordinator",
        )
        runtime.register_evidence(
            "file_read",
            "first.txt",
            "lines 1-2",
            "first",
            first_task_context.agent_id,
        )
        runtime.register_evidence(
            "file_read",
            "second.txt",
            "lines 1-2",
            "second",
            second_task_context.agent_id,
        )

        first_prompt = runtime.build_user_prompt(first_task_context, "继续")
        coordinator_prompt = runtime.build_user_prompt(
            coordinator_context, "汇总"
        )

        self.assertIn("first.txt", first_prompt)
        self.assertNotIn("second.txt", first_prompt)
        self.assertIn("first.txt", coordinator_prompt)
        self.assertIn("second.txt", coordinator_prompt)

    def test_runtime_event_channels_support_multiple_subscribers(self) -> None:
        """日志和宿主可同时订阅事件，取消一方不会覆盖另一方。"""
        runtime = ContextRuntime(
            WorkspaceContextBuilder(
                self.workspace_root, self.skills_root
            ).build(),
            TaskState(goal="广播事件"),
        )
        first_events: list[AgentCallEvent] = []
        second_events: list[AgentCallEvent] = []
        unsubscribe_first = runtime.call_events.subscribe(first_events.append)
        runtime.call_events.subscribe(second_events.append)
        event = AgentCallEvent(
            call_id=1,
            agent_id="coordinator",
            kind="agent",
            phase="started",
            turn_index=1,
        )

        runtime.emit_call_event(event)
        unsubscribe_first()
        runtime.emit_call_event(event)

        self.assertEqual([event], first_events)
        self.assertEqual([event, event], second_events)


class AgentRuntimeTests(unittest.TestCase):
    """验证执行 Agent 的工具注册与上下文边界。"""

    def test_deepseek_thinking_model_omits_tool_choice(self) -> None:
        """DeepSeek 思考模式保留函数工具，但不发送不兼容的 tool_choice。"""
        model = DeepSeekThinkingChatModel(
            "deepseek-v4-flash",
            provider=OpenAIProvider(
                base_url="https://api.deepseek.example/v1",
                api_key="test-key",
            ),
        )
        request_parameters = ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name="read_workspace_file",
                    description="读取文件",
                    parameters_json_schema={
                        "type": "object",
                        "properties": {},
                    },
                )
            ]
        )

        tools, tool_choice = model._get_tool_choice(
            {}, request_parameters
        )

        self.assertEqual(1, len(tools))
        self.assertIsNone(tool_choice)

    def test_agent_registers_core_workspace_tools(self) -> None:
        """模型收到工作区和编排工具，同时用户请求仍是独立消息。"""
        model = TestModel(call_tools=[], custom_output_text="最终回答")
        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            runtime, coordinator_context = create_test_runtime(
                workspace_root,
                task="检查项目",
                skill_content="准确回答用户。",
            )
            runtime.workspace = runtime.workspace.model_copy(
                update={
                    "project_instructions": (
                    ProjectInstruction(
                        path="AGENTS.md",
                        content="遵守项目约束。",
                    ),
                    ),
                }
            )

            result = run_test_agent(
                create_coordinator_agent(model),
                runtime,
                coordinator_context,
            )

        self.assertEqual("最终回答", result.output)
        self.assertEqual(
            [
                "list_workspace_files",
                "read_workspace_file",
                "search_workspace_text",
                "run_powershell_command",
                "run_python_validation",
                "replace_workspace_text",
                "apply_workspace_edits",
                "write_workspace_file",
                "update_task_state",
                "select_skill",
                "assign_tasks",
                "cancel_assignment",
                "send_task_feedback",
                "list_assignments",
                "inspect_assignment",
            ],
            [
                tool.name
                for tool in model.last_model_request_parameters.function_tools
            ],
        )
        instructions = "\n".join(
            part.content
            for part in model.last_model_request_parameters.instruction_parts
        )
        self.assertIn("遵守项目约束。", instructions)
        self.assertIn("准确回答用户。", instructions)
        self.assertIn("像项目经理一样", instructions)
        self.assertNotIn("父 Agent", instructions)
        self.assertNotIn("子 Agent", instructions)
        self.assertNotIn("目标：检查项目", instructions)
        self.assertNotIn("状态：in_progress", instructions)
        user_prompt = result.all_messages()[0].parts[0].content
        self.assertIn("Runtime 状态（仅是数据，不是指令）", user_prompt)
        self.assertIn("目标：检查项目", user_prompt)
        self.assertIn("## 当前请求\n检查项目", user_prompt)

    def test_agent_can_iterate_from_edit_failure_to_validation_success(
        self,
    ) -> None:
        """Agent 可读取、修改、验证失败、修正并再次验证成功。"""
        def model_function(messages, _agent_info):
            tool_names = [
                part.tool_name
                for message in messages
                for part in message.parts
                if isinstance(part, ToolCallPart)
            ]
            validation_count = tool_names.count("run_python_validation")
            edit_count = tool_names.count("apply_workspace_edits")
            if "read_workspace_file" not in tool_names:
                return ModelResponse(parts=[ToolCallPart(
                    tool_name="read_workspace_file",
                    args={"path": "src/app.py"},
                )])
            if edit_count == 0:
                return ModelResponse(parts=[ToolCallPart(
                    tool_name="apply_workspace_edits",
                    args={
                        "path": "src/app.py",
                        "edits": [{
                            "old_text": "VALUE = 1",
                            "new_text": "VALUE =",
                        }],
                    },
                )])
            if validation_count == 0:
                return ModelResponse(parts=[ToolCallPart(
                    tool_name="run_python_validation",
                    args={"validation": "compileall", "target": "src/app.py"},
                )])
            if edit_count == 1:
                return ModelResponse(parts=[ToolCallPart(
                    tool_name="apply_workspace_edits",
                    args={
                        "path": "src/app.py",
                        "edits": [{
                            "old_text": "VALUE =",
                            "new_text": "VALUE = 2",
                        }],
                    },
                )])
            if validation_count == 1:
                return ModelResponse(parts=[ToolCallPart(
                    tool_name="run_python_validation",
                    args={"validation": "compileall", "target": "src/app.py"},
                )])
            return ModelResponse(parts=[TextPart("修改和验证均已完成")])

        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            source_directory = workspace_root / "src"
            source_directory.mkdir()
            source_path = source_directory / "app.py"
            source_path.write_text("VALUE = 1\n", encoding="utf-8")
            runtime, coordinator_context = create_test_runtime(
                workspace_root, task="修改数值并验证"
            )

            result = run_test_agent(
                create_coordinator_agent(FunctionModel(model_function)),
                runtime,
                coordinator_context,
            )
            final_content = source_path.read_text(encoding="utf-8")

        self.assertEqual("修改和验证均已完成", result.output)
        self.assertEqual("VALUE = 2\n", final_content)
        self.assertEqual(
            [1, 0],
            [
                validation.exit_code
                for validation in runtime.task_state.validation_results
            ],
        )
        self.assertEqual(2, runtime.modification_revisions_by_agent["coordinator"])
        self.assertEqual(2, runtime.validated_revisions_by_agent["coordinator"])

    def test_agent_context_keeps_history_between_runs(self) -> None:
        """AgentContext 保存第一轮历史，并在第二轮自动传回模型。"""
        responses = ["第一轮回答", "第二轮回答"]

        def model_function(messages, _agent_info):
            return ModelResponse(parts=[TextPart(responses.pop(0))])

        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            runtime, coordinator_context = create_test_runtime(
                workspace_root,
                task="第一轮",
                skill_content="准确回答用户。",
            )
            agent = create_coordinator_agent(FunctionModel(model_function))
            first_result = run_test_agent(
                agent, runtime, coordinator_context
            )
            second_result = run_test_agent(
                agent,
                runtime,
                coordinator_context,
                request="第二轮",
            )

        message_contents = [
            part.content
            for message in second_result.all_messages()
            for part in message.parts
            if hasattr(part, "content")
        ]
        self.assertTrue(
            any("## 当前请求\n第一轮" in content for content in message_contents)
        )
        self.assertIn("第一轮回答", message_contents)
        self.assertTrue(
            any("## 当前请求\n第二轮" in content for content in message_contents)
        )
        self.assertEqual(
            second_result.all_messages(), coordinator_context.message_history
        )
        self.assertEqual(2, coordinator_context.turn_count)

    def test_conversation_reuses_coordinator_context_until_exit_command(self) -> None:
        """终端连续输入复用同一历史，并且退出命令不会进入模型消息。"""
        requests = iter(["第二轮", "/quit"])
        outputs: list[str] = []
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory), task="第一轮"
            )
            run_conversation(
                create_coordinator_agent(
                    TestModel(call_tools=[], custom_output_text="持续回答")
                ),
                runtime,
                coordinator_context,
                "第一轮",
                input_fn=lambda _prompt: next(requests),
                output_fn=outputs.append,
            )

        serialized_history = ModelMessagesTypeAdapter.dump_json(
            coordinator_context.message_history
        ).decode()
        self.assertEqual(
            [
                "Call[1] coordinator agent turn 1 started",
                "Call[1] coordinator agent turn 1 completed "
                "(requests=1; tool_calls=0)",
                "Assistant> 持续回答",
                "Call[2] coordinator agent turn 2 started",
                "Call[2] coordinator agent turn 2 completed "
                "(requests=1; tool_calls=0)",
                "Assistant> 持续回答",
            ],
            outputs,
        )
        self.assertEqual(2, coordinator_context.turn_count)
        self.assertIn("## 当前请求\\n第一轮", serialized_history)
        self.assertIn("## 当前请求\\n第二轮", serialized_history)
        self.assertNotIn("/quit", serialized_history)

    def test_assignment_completion_resumes_coordinator(self) -> None:
        """任务完成后，CLI 事件循环自动调用协调 Agent 处理交接。"""
        auto_response_written = Event()
        outputs: list[str] = []

        async def model_function(messages, agent_info):
            if agent_info.output_tools:
                await asyncio.sleep(0.02)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=agent_info.output_tools[0].name,
                            args={
                                "status": "completed",
                                "summary": "后台检查完成",
                                "evidence_ids": [],
                                "unresolved_issues": [],
                                "recommended_next_actions": [],
                            },
                        )
                    ]
                )
            contents = [
                part.content
                for message in messages
                for part in message.parts
                if hasattr(part, "content")
                and isinstance(part.content, str)
            ]
            if any("Runtime 任务完成事件" in text for text in contents):
                return ModelResponse(parts=[TextPart("已处理后台交接")])
            if any(
                getattr(part, "tool_name", None) == "assign_tasks"
                for message in messages
                for part in message.parts
            ):
                return ModelResponse(parts=[TextPart("协调任务继续执行")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="assign_tasks",
                        args={
                            "tasks": [
                                {
                                    "template": "explorer",
                                    "task": "检查后台任务",
                                },
                                {
                                    "template": "reviewer",
                                    "task": "并行复核后台任务",
                                }
                            ]
                        },
                    )
                ]
            )

        def output_fn(message: str) -> None:
            outputs.append(message)
            if message == "Assistant> 已处理后台交接":
                auto_response_written.set()

        def input_fn(_prompt: str) -> str:
            if not auto_response_written.wait(timeout=2):
                raise AssertionError("协调 Agent 未在任务完成后自动续跑")
            return "/quit"

        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory), task="启动后台检查"
            )
            run_conversation(
                create_coordinator_agent(FunctionModel(model_function)),
                runtime,
                coordinator_context,
                "启动后台检查",
                input_fn=input_fn,
                output_fn=output_fn,
            )

        self.assertIn("Assistant> 协调任务继续执行", outputs)
        self.assertIn("Assistant> 已处理后台交接", outputs)
        self.assertLess(
            outputs.index("Assistant> 协调任务继续执行"),
            outputs.index("Assistant> 已处理后台交接"),
        )
        self.assertEqual(2, coordinator_context.turn_count)
        self.assertEqual(
            "completed", runtime.assignments["explorer-1"].status
        )
        self.assertEqual(
            "completed", runtime.assignments["reviewer-1"].status
        )
        self.assertEqual([], runtime.pending_completion_events)

    def test_async_agent_turn_returns_uniform_runtime_metadata(self) -> None:
        """异步公开入口返回输出、usage、压缩标记和统一轮次。"""
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory), task="异步调用"
            )
            turn = asyncio.run(
                run_coordinator_turn(
                    create_coordinator_agent(
                        TestModel(
                            call_tools=[], custom_output_text="异步完成"
                        )
                    ),
                    runtime,
                    coordinator_context,
                )
            )

        self.assertEqual("异步完成", turn.output)
        self.assertEqual(1, turn.turn_index)
        self.assertFalse(turn.compacted)
        self.assertEqual(0, turn.compaction_requests)
        self.assertEqual(turn.messages, coordinator_context.message_history)

    def test_agent_runner_enforces_request_limit(self) -> None:
        """协调 Agent 和任务 Agent 共用 Runner 的模型请求上限。"""
        def model_function(_messages, _agent_info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="list_workspace_files",
                        args={"directory": ".", "limit": 10},
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory), task="持续调用工具"
            )
            runner = AgentRunner(
                AgentCallLimits(request_limit=1, tool_calls_limit=10)
            )
            call_events = []
            runtime.call_events.subscribe(call_events.append)

            with self.assertRaises(UsageLimitExceeded):
                asyncio.run(
                    runner.run_turn(
                        create_coordinator_agent(FunctionModel(model_function)),
                        runtime,
                        coordinator_context,
                        coordinator_context.task,
                    )
                )

        self.assertEqual(0, coordinator_context.turn_count)
        self.assertEqual([], coordinator_context.message_history)
        self.assertEqual(
            ["started", "failed"],
            [event.phase for event in call_events],
        )
        self.assertIn("UsageLimitExceeded", call_events[-1].detail)

    def test_runtime_uses_injected_agent_runner(self) -> None:
        """每个 Runtime 可拥有独立 Runner，不依赖模块级共享单例。"""
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, _coordinator_context = create_test_runtime(
                Path(workspace_directory)
            )
            runner = AgentRunner(
                AgentCallLimits(request_limit=2, tool_calls_limit=3)
            )
            runtime.agent_runner = runner

            self.assertIs(runner, runtime.get_agent_runner())

    def test_assignment_followup_retries_before_retaining_event(self) -> None:
        """协调自动续跑使用有界重试，并在瞬时失败恢复后返回结果。"""
        attempts = 0

        def model_function(_messages, _agent_info):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("temporary coordinator failure")
            return ModelResponse(parts=[TextPart("已恢复")])

        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory)
            )
            session = ConversationSession(
                create_coordinator_agent(FunctionModel(model_function)),
                runtime,
                coordinator_context,
                input_fn=lambda _prompt: "/quit",
                output_fn=lambda _message: None,
            )
            event = AssignmentCompletionEvent(
                assignment_id="worker-1",
                coordinator_id=coordinator_context.agent_id,
                status="completed",
            )

            turn = asyncio.run(session._resume_coordinator([event]))

        self.assertIsNotNone(turn)
        assert turn is not None
        self.assertEqual("已恢复", turn.output)
        self.assertEqual(3, attempts)

    def test_context_compacts_at_seventy_percent_of_one_million_tokens(
        self,
    ) -> None:
        """默认阈值为 700,000，测试小窗口达到同一比例时会自动摘要。"""
        default_policy = ContextWindowPolicy()
        self.assertEqual(1_000_000, default_policy.window_tokens)
        self.assertEqual(
            700_000, default_policy.compaction_threshold_tokens
        )

        normal_responses = ["第一轮回答", "第二轮回答"]

        def model_function(messages, _agent_info):
            contents = [
                part.content
                for message in messages
                for part in message.parts
                if hasattr(part, "content")
                and isinstance(part.content, str)
            ]
            if any("压缩以上历史" in content for content in contents):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=_agent_info.output_tools[0].name,
                            args={
                                "summary": "保留目标和关键结论",
                                "facts": [
                                    {
                                        "statement": "已确认上下文规则",
                                        "citations": [
                                            {
                                                "evidence_id": "evidence-1",
                                                "quote": "上下文规则",
                                            }
                                        ],
                                    }
                                ],
                                "unresolved_issues": ["继续第二轮"],
                            },
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart(normal_responses.pop(0))]
            )

        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory), task="第一轮"
            )
            agent = create_coordinator_agent(FunctionModel(model_function))
            run_test_agent(agent, runtime, coordinator_context)
            runtime.register_evidence(
                "file_read",
                "context.py",
                "characters 1-20 of 20",
                "上下文规则",
                coordinator_context.agent_id,
            )
            runtime.context_window = ContextWindowPolicy(
                window_tokens=20, compaction_ratio=0.70
            )
            call_events = []
            runtime.call_events.subscribe(call_events.append)

            turn = asyncio.run(
                AgentRunner().run_turn(
                    agent, runtime, coordinator_context, request="第二轮"
                )
            )
            result = turn.raw_result

        serialized_history = ModelMessagesTypeAdapter.dump_json(
            result.all_messages()
        ).decode()
        self.assertEqual(1, coordinator_context.compaction_count)
        self.assertTrue(turn.compacted)
        self.assertEqual(1, turn.compaction_requests)
        self.assertEqual(
            "保留目标和关键结论", coordinator_context.conversation_summary
        )
        self.assertIn("Runtime 压缩历史摘要", serialized_history)
        self.assertNotIn("第一轮回答", serialized_history)
        self.assertIn("第二轮回答", serialized_history)
        self.assertEqual(
            "已确认上下文规则",
            runtime.task_state.important_facts[0].statement,
        )
        self.assertEqual(["继续第二轮"], runtime.task_state.unresolved_issues)
        self.assertEqual(
            [
                ("agent", "started"),
                ("compaction", "started"),
                ("compaction", "completed"),
                ("agent", "completed"),
            ],
            [(event.kind, event.phase) for event in call_events],
        )

    def test_next_run_renders_updated_agent_skill(self) -> None:
        """Skill 变化保存在 AgentContext，并在下一次运行时重新渲染。"""
        model = TestModel(call_tools=[], custom_output_text="完成")
        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            runtime, coordinator_context = create_test_runtime(workspace_root)
            review_skill = Path(runtime.workspace.skills_root) / "review"
            review_skill.mkdir()
            (review_skill / "skill.json").write_text(
                '{"name":"review","description":"审查修改"}',
                encoding="utf-8",
            )
            (review_skill / "SKILL.md").write_text(
                "只审查当前修改。", encoding="utf-8"
            )
            agent = create_coordinator_agent(model)
            run_test_agent(agent, runtime, coordinator_context)

            select_skill(
                SimpleNamespace(
                    deps=AgentDependencies(runtime, coordinator_context)
                ),
                "review",
            )
            run_test_agent(
                agent,
                runtime,
                coordinator_context,
                request="审查修改",
            )

        instructions = "\n".join(
            part.content
            for part in model.last_model_request_parameters.instruction_parts
        )
        self.assertIn("只审查当前修改。", instructions)
        self.assertNotIn("根据工具证据完成任务。", instructions)

    def test_update_task_state_changes_only_model_owned_fields(self) -> None:
        """协调 Agent 可更新计划和事实，Runtime 记录字段保持独立。"""
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory), task="实现 TaskState"
            )
            tool_context = SimpleNamespace(
                deps=AgentDependencies(runtime, coordinator_context)
            )
            evidence = runtime.register_evidence(
                "file_read",
                "context.py",
                "lines 1-20",
                "class TaskState: Runtime state",
                coordinator_context.agent_id,
            )

            rendered_state = update_task_state(
                tool_context,
                plan=["定义状态", "运行测试"],
                completed_steps=["分析现状"],
                important_facts=[
                    FactClaim(
                        statement="TaskState 属于 Runtime",
                        citations=[
                            EvidenceQuoteClaim(
                                evidence_id=evidence.evidence_id,
                                quote="class TaskState",
                            )
                        ],
                    )
                ],
                completion_criteria=["相关测试通过"],
            )
            update_task_state(
                tool_context,
                important_facts=[
                    FactClaim(
                        statement="TaskState 属于 Runtime",
                        citations=[
                            EvidenceQuoteClaim(
                                evidence_id=evidence.evidence_id,
                                quote="class TaskState",
                            )
                        ],
                    )
                ],
                unresolved_issues=["补充验证"],
            )
            update_task_state(tool_context, unresolved_issues=[])

        self.assertEqual(
            ["定义状态", "运行测试"], runtime.task_state.plan
        )
        self.assertEqual([], runtime.task_state.modified_files)
        self.assertEqual([], runtime.task_state.validation_results)
        self.assertEqual(1, len(runtime.task_state.important_facts))
        self.assertEqual([], runtime.task_state.unresolved_issues)
        self.assertIn("TaskState 属于 Runtime", rendered_state)
        self.assertIn("evidence-1", rendered_state)
        self.assertIn("context.py (lines 1-20)", rendered_state)

    def test_update_task_state_rejects_unknown_evidence(self) -> None:
        """模型不能把不存在的工具结果登记为已证实事实。"""
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory)
            )
            tool_context = SimpleNamespace(
                deps=AgentDependencies(runtime, coordinator_context)
            )

            with self.assertRaisesRegex(
                RecoverableToolError, "未知证据 ID"
            ):
                update_task_state(
                    tool_context,
                    important_facts=[
                        FactClaim(
                            statement="未经工具证实的事实",
                            citations=[
                                EvidenceQuoteClaim(
                                    evidence_id="evidence-99",
                                    quote="不存在",
                                )
                            ],
                        )
                    ],
                )

    def test_update_task_state_rejects_quote_absent_from_evidence(self) -> None:
        """有效 ID 不能支持工具结果中没有出现的原文。"""
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory)
            )
            evidence = runtime.register_evidence(
                "file_read",
                "auth.py",
                "characters 1-30 of 30",
                "API_KEY = settings.api_key",
                coordinator_context.agent_id,
            )

            with self.assertRaisesRegex(
                RecoverableToolError, "不存在引用原文"
            ):
                update_task_state(
                    SimpleNamespace(
                        deps=AgentDependencies(runtime, coordinator_context)
                    ),
                    important_facts=[
                        FactClaim(
                            statement="认证使用 OAuth",
                            citations=[
                                EvidenceQuoteClaim(
                                    evidence_id=evidence.evidence_id,
                                    quote="OAuth",
                                )
                            ],
                        )
                    ],
                )

    def test_recoverable_tool_error_returns_to_model_for_new_decision(
        self,
    ) -> None:
        """工具路径错误产生 RetryPromptPart，并触发第二次模型请求。"""
        def model_function(messages, _agent_info):
            has_retry = any(
                isinstance(part, RetryPromptPart)
                for message in messages
                for part in message.parts
            )
            if has_retry:
                return ModelResponse(parts=[TextPart("已根据错误重新决策")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_workspace_file",
                        args={
                            "path": "missing.txt",
                            "max_characters": 100,
                        },
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            runtime, coordinator_context = create_test_runtime(
                workspace_root,
                task="读取文件",
                skill_content="根据工具结果回答。",
            )
            result = run_test_agent(
                create_coordinator_agent(FunctionModel(model_function)),
                runtime,
                coordinator_context,
            )

        retry_parts = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        ]
        self.assertEqual("已根据错误重新决策", result.output)
        self.assertEqual(1, len(retry_parts))
        self.assertIn("missing.txt", retry_parts[0].content)
        self.assertEqual(2, result.usage.requests)

    def test_read_file_accepts_file_path_compatibility_field(self) -> None:
        """常见的 file_path 字段可读取文件，不消耗参数校验重试次数。"""
        model_call_count = 0

        def model_function(_messages, _agent_info):
            nonlocal model_call_count
            model_call_count += 1
            if model_call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="read_workspace_file",
                            args={
                                "file_path": "note.txt",
                                "max_characters": 100,
                            },
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("已读取兼容路径")])

        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            (workspace_root / "note.txt").write_text(
                "兼容字段内容", encoding="utf-8"
            )
            runtime, coordinator_context = create_test_runtime(workspace_root)
            result = run_test_agent(
                create_coordinator_agent(FunctionModel(model_function)),
                runtime,
                coordinator_context,
            )

        self.assertEqual("已读取兼容路径", result.output)
        self.assertEqual(2, result.usage.requests)
        self.assertEqual(1, result.usage.tool_calls)

    def test_missing_read_path_is_a_recoverable_tool_error(self) -> None:
        """漏传路径时返回明确反馈，避免直接耗尽工具参数校验重试。"""
        def model_function(messages, _agent_info):
            has_retry = any(
                isinstance(part, RetryPromptPart)
                for message in messages
                for part in message.parts
            )
            if has_retry:
                return ModelResponse(parts=[TextPart("已补充读取参数")])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="read_workspace_file", args={}
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory).resolve()
            )
            result = run_test_agent(
                create_coordinator_agent(FunctionModel(model_function)),
                runtime,
                coordinator_context,
            )

        retry_parts = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        ]
        self.assertEqual("已补充读取参数", result.output)
        self.assertEqual(1, len(retry_parts))
        self.assertIn("必须提供 path", retry_parts[0].content)

    def test_rejected_powershell_command_returns_to_model_without_retry(
        self,
    ) -> None:
        """安全策略拒绝命令后，模型可改用其他决策而不耗尽工具重试。"""
        model_call_count = 0

        def model_function(_messages, _agent_info):
            nonlocal model_call_count
            model_call_count += 1
            if model_call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="run_powershell_command",
                            args={"command": "Get-ChildItem | Measure-Object"},
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart("已改用安全工具")])

        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, coordinator_context = create_test_runtime(
                Path(workspace_directory).resolve()
            )
            result = run_test_agent(
                create_coordinator_agent(FunctionModel(model_function)),
                runtime,
                coordinator_context,
            )

        retry_parts = [
            part
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, RetryPromptPart)
        ]
        self.assertEqual("已改用安全工具", result.output)
        self.assertEqual(1, result.usage.tool_calls)
        self.assertEqual([], retry_parts)


class WorkspaceToolTests(unittest.TestCase):
    """验证文件和 PowerShell 工具的工作区边界。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name) / "workspace"
        source_directory = self.workspace_root / "src"
        source_directory.mkdir(parents=True)
        (source_directory / "app.py").write_text(
            "first line\nNeedle value\nlast line\n",
            encoding="utf-8",
        )
        (self.workspace_root / ".env").write_text(
            "API_KEY=secret", encoding="utf-8"
        )
        virtual_environment = self.workspace_root / ".venv"
        virtual_environment.mkdir()
        (virtual_environment / "ignored.txt").write_text(
            "Needle", encoding="utf-8"
        )
        runtime, agent_context = create_test_runtime(self.workspace_root)
        self.tool_context = SimpleNamespace(
            deps=AgentDependencies(runtime, agent_context)
        )

    def test_list_read_and_search_use_real_workspace_files(self) -> None:
        """列表、读取和搜索共享同一个受限工作区依赖。"""
        listing = list_workspace_files(self.tool_context, ".", 20)
        content = read_workspace_file(
            self.tool_context, "src/app.py", max_characters=10
        )
        matches = search_workspace_text(
            self.tool_context, "needle", ".", 20
        )

        self.assertIn("src/app.py", listing)
        self.assertNotIn(".venv/ignored.txt", listing)
        self.assertNotIn(".env", listing)
        self.assertIn("truncated after 10 characters", content)
        self.assertIn("src/app.py:2:Needle value", matches)
        self.assertNotIn("ignored.txt", matches)
        self.assertIn("[evidence_id=evidence-1]", listing)
        self.assertIn("[evidence_id=evidence-2]", content)
        self.assertIn("[evidence_id=evidence-3]", matches)
        evidence = self.tool_context.deps.runtime.evidence_records
        self.assertEqual("src/app.py", evidence["evidence-2"].source)

    def test_tools_report_recoverable_path_errors(self) -> None:
        """路径错误使用统一异常，交给 RetryToolset 转换。"""
        with self.assertRaisesRegex(RecoverableToolError, "工作区根目录内"):
            list_workspace_files(self.tool_context, "..", 20)

        with self.assertRaisesRegex(RecoverableToolError, "敏感文件"):
            read_workspace_file(self.tool_context, ".env")

    def test_read_workspace_file_supports_line_ranges(self) -> None:
        """大文件可按包含首尾行的范围读取，并记录实际范围。"""
        content = read_workspace_file(
            self.tool_context,
            "src/app.py",
            start_line=2,
            end_line=2,
        )

        self.assertIn("Needle value", content)
        self.assertNotIn("first line", content)
        evidence = self.tool_context.deps.runtime.evidence_records[
            "evidence-1"
        ]
        self.assertIn("lines 2-2 of 3", evidence.detail)

    def test_replace_workspace_text_requires_exact_occurrence_count(
        self,
    ) -> None:
        """精确替换成功写入，计数不一致时保持文件不变。"""
        result = replace_workspace_text(
            self.tool_context,
            "src/app.py",
            "Needle value",
            "Changed value",
        )

        source_path = self.workspace_root / "src" / "app.py"
        self.assertIn("Replaced 1 occurrence", result)
        self.assertIn(
            "Changed value", source_path.read_text(encoding="utf-8")
        )
        self.assertEqual(
            ["src/app.py"],
            self.tool_context.deps.runtime.task_state.modified_files,
        )

        with self.assertRaisesRegex(RecoverableToolError, "出现次数"):
            replace_workspace_text(
                self.tool_context,
                "src/app.py",
                "Changed value",
                "Unexpected value",
                expected_replacements=2,
            )
        self.assertNotIn(
            "Unexpected value", source_path.read_text(encoding="utf-8")
        )

    def test_apply_workspace_edits_is_atomic(self) -> None:
        """多段编辑任一项不匹配时不写入已经通过的前置编辑。"""
        source_path = self.workspace_root / "src" / "app.py"
        original_content = source_path.read_text(encoding="utf-8")

        with self.assertRaisesRegex(RecoverableToolError, "第 2 项"):
            apply_workspace_edits(
                self.tool_context,
                "src/app.py",
                [
                    WorkspaceTextEdit(
                        old_text="first line", new_text="changed first"
                    ),
                    WorkspaceTextEdit(
                        old_text="missing", new_text="never written"
                    ),
                ],
            )

        self.assertEqual(
            original_content, source_path.read_text(encoding="utf-8")
        )
        git_directory = self.workspace_root / ".git"
        git_directory.mkdir()
        (git_directory / "config").write_text(
            "protected", encoding="utf-8"
        )
        with self.assertRaisesRegex(RecoverableToolError, "Runtime 管理"):
            apply_workspace_edits(
                self.tool_context,
                ".git/config",
                [WorkspaceTextEdit(
                    old_text="protected", new_text="changed"
                )],
            )

    def test_write_workspace_file_creates_without_overwriting(self) -> None:
        """新文件工具可创建父目录，但拒绝覆盖和敏感路径。"""
        result = write_workspace_file(
            self.tool_context,
            "tests/test_created.py",
            "VALUE = 1\n",
            create_parent_directories=True,
        )

        created_path = self.workspace_root / "tests" / "test_created.py"
        self.assertIn("Created tests/test_created.py", result)
        self.assertEqual("VALUE = 1\n", created_path.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(RecoverableToolError, "已经存在"):
            write_workspace_file(
                self.tool_context, "tests/test_created.py", "VALUE = 2\n"
            )
        with self.assertRaisesRegex(RecoverableToolError, "敏感"):
            write_workspace_file(self.tool_context, ".env", "secret")

    def test_structured_python_validation_records_result(self) -> None:
        """结构化验证不经过 shell，并把真实结果写入 TaskState。"""
        valid_source = self.workspace_root / "src" / "valid.py"
        valid_source.write_text("VALUE = 1\n", encoding="utf-8")

        result = run_python_validation(
            self.tool_context,
            "compileall",
            target="src/valid.py",
        )

        self.assertEqual(0, result.exit_code)
        self.assertFalse(result.timed_out)
        validations = self.tool_context.deps.runtime.task_state.validation_results
        self.assertEqual(1, len(validations))
        self.assertEqual(
            "python -m compileall -q src/valid.py",
            validations[0].command,
        )

    def test_powershell_tool_runs_allowed_command_in_workspace(self) -> None:
        """允许的单命令返回退出码和输出，并使用指定工作目录。"""
        result = run_powershell_command(
            self.tool_context,
            "Get-Location",
            working_directory="src",
        )

        self.assertEqual(0, result.exit_code)
        self.assertFalse(result.timed_out)
        self.assertIn("src", result.stdout.casefold())
        self.assertEqual("evidence-1", result.evidence_id)

    def test_validation_command_updates_task_state(self) -> None:
        """实际执行的编译命令由宿主记录到 TaskState。"""
        result = run_powershell_command(
            self.tool_context,
            "python -m compileall -q src",
        )

        validations = (
            self.tool_context.deps.runtime.task_state.validation_results
        )
        self.assertNotEqual(0, result.exit_code)
        self.assertEqual(1, len(validations))
        self.assertEqual("python -m compileall -q src", validations[0].command)
        self.assertEqual(result.exit_code, validations[0].exit_code)
        self.assertFalse(validations[0].timed_out)

    def test_powershell_tool_rejects_mutating_and_compound_commands(
        self,
    ) -> None:
        """删除命令和 PowerShell 管道在执行前被策略拒绝。"""
        with self.assertRaisesRegex(RecoverableToolError, "不允许执行命令"):
            run_powershell_command(
                self.tool_context, "Remove-Item src/app.py"
            )

        with self.assertRaisesRegex(RecoverableToolError, "复合语法"):
            run_powershell_command(
                self.tool_context,
                "Get-ChildItem | Measure-Object",
            )

        with self.assertRaisesRegex(RecoverableToolError, "敏感配置文件"):
            run_powershell_command(
                self.tool_context,
                "Get-Item .env",
            )


class TaskCoordinationTests(unittest.TestCase):
    """验证协调 Agent 能分配并跟进固定模板的任务。"""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.workspace_root = Path(self.temporary_directory.name).resolve()
        self.skills_root = self.workspace_root / "skills"
        general_skill = self.skills_root / "general"
        general_skill.mkdir(parents=True)
        (general_skill / "skill.json").write_text(
            '{"name":"general","description":"通用 Skill"}',
            encoding="utf-8",
        )
        (general_skill / "SKILL.md").write_text(
            "根据工具证据完成任务。", encoding="utf-8"
        )
        workspace_context = WorkspaceContextBuilder(
            self.workspace_root, self.skills_root
        ).build()
        self.runtime = ContextRuntime(
            workspace_context, TaskState(goal="整体任务")
        )
        self.coordinator_context = self.runtime.create_agent_context(
            "coordinator", "协调任务", "general", role="coordinator"
        )
        self.dependencies = AgentDependencies(
            self.runtime,
            self.coordinator_context,
        )

    def test_select_skill_loads_requested_content(self) -> None:
        """协调 Agent 只按需加载选中 Skill 的完整正文。"""
        tool_context = SimpleNamespace(deps=self.dependencies)

        selected_skill = select_skill(tool_context, "general")

        self.assertIn("# Skill: general", selected_skill)
        self.assertIn("根据工具证据完成任务。", selected_skill)
        self.assertEqual(
            "general", self.coordinator_context.skill.metadata.name
        )

    def test_task_agent_cannot_manage_assignments(self) -> None:
        """任务职责上下文不能绕过工具注册边界分配其他工作。"""
        task_context = self.runtime.create_agent_context(
            "worker-manual",
            "执行工作包",
            "general",
            role="task",
            coordinator_id=self.coordinator_context.agent_id,
        )
        with self.assertRaisesRegex(
            RecoverableToolError, "没有任务协调职责"
        ):
            asyncio.run(
                assign_tasks(
                    SimpleNamespace(
                        deps=AgentDependencies(self.runtime, task_context),
                        model=TestModel(call_tools=[]),
                    ),
                    tasks=[
                        TaskAssignmentRequest(
                            template="explorer", task="越权分配"
                        )
                    ],
                )
            )

    def test_coordinator_assigns_read_only_explorer(self) -> None:
        """任务分配立即返回，explorer 只获得只读工作区工具。"""
        task_model = TestModel(
            call_tools=[],
            custom_output_args={
                "status": "completed",
                "summary": "工作包完成",
                "evidence_ids": [],
                "unresolved_issues": [],
                "recommended_next_actions": [],
            },
        )
        tool_context = SimpleNamespace(
            deps=self.dependencies,
            model=task_model,
        )

        async def run_scenario():
            receipts = await assign_tasks(
                tool_context,
                tasks=[
                    TaskAssignmentRequest(
                        template="explorer",
                        task="定位相关文件",
                        skill_id="general",
                    )
                ],
            )
            receipt = receipts[0]
            self.assertEqual("queued", receipt.status)
            self.assertEqual([], self.runtime.assignment_history)
            assignment = self.runtime.assignments[receipt.assignment_id]
            self.assertIsNotNone(assignment.background_task)
            await assignment.background_task
            return receipt, inspect_assignment(
                SimpleNamespace(deps=self.dependencies),
                receipt.assignment_id,
            )

        receipt, result = asyncio.run(run_scenario())

        task_tool_names = [
            tool.name
            for tool in task_model.last_model_request_parameters.function_tools
        ]
        self.assertEqual("explorer", result.template)
        self.assertEqual("explorer-1", receipt.assignment_id)
        self.assertEqual("工作包完成", result.summary)
        self.assertEqual("completed", result.status)
        self.assertEqual(1, result.turn_index)
        self.assertEqual((), result.evidence)
        self.assertEqual((), result.modified_files)
        self.assertEqual((), result.validation_results)
        self.assertEqual((), result.unresolved_issues)
        self.assertEqual((), result.recommended_next_actions)
        task_context = self.runtime.agent_contexts[result.agent_id]
        self.assertIsNot(task_context, self.coordinator_context)
        self.assertEqual([], self.coordinator_context.message_history)
        self.assertTrue(task_context.message_history)
        self.assertEqual("task", task_context.role)
        self.assertEqual("coordinator", task_context.coordinator_id)
        self.assertEqual(1, len(self.runtime.assignment_history))
        self.assertEqual(
            [
                "list_workspace_files",
                "read_workspace_file",
                "search_workspace_text",
                "run_powershell_command",
                "run_python_validation",
            ],
            task_tool_names,
        )
        self.assertNotIn("assign_tasks", task_tool_names)
        self.assertNotIn("send_task_feedback", task_tool_names)
        self.assertNotIn("list_assignments", task_tool_names)
        self.assertNotIn("inspect_assignment", task_tool_names)

    def test_worker_must_validate_latest_file_revision(self) -> None:
        """worker 修改后尝试直接完成时，会被要求验证最新修订。"""
        def task_model_function(messages, agent_info):
            tool_names = [
                part.tool_name
                for message in messages
                for part in message.parts
                if isinstance(part, ToolCallPart)
            ]
            retry_messages = [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, RetryPromptPart)
            ]
            if "write_workspace_file" not in tool_names:
                return ModelResponse(parts=[ToolCallPart(
                    tool_name="write_workspace_file",
                    args={
                        "path": "src/generated.py",
                        "content": "VALUE = 1\n",
                        "create_parent_directories": True,
                    },
                )])
            if retry_messages and "run_python_validation" not in tool_names:
                return ModelResponse(parts=[ToolCallPart(
                    tool_name="run_python_validation",
                    args={
                        "validation": "compileall",
                        "target": "src/generated.py",
                    },
                )])
            return ModelResponse(parts=[ToolCallPart(
                tool_name=agent_info.output_tools[0].name,
                args={"status": "completed", "summary": "实现已通过验证"},
            )])

        async def run_scenario():
            receipt = (await assign_tasks(
                SimpleNamespace(
                    deps=self.dependencies,
                    model=FunctionModel(task_model_function),
                ),
                tasks=[TaskAssignmentRequest(
                    template="worker", task="创建并验证模块"
                )],
            ))[0]
            assignment = self.runtime.assignments[receipt.assignment_id]
            await assignment.background_task
            return inspect_assignment(
                SimpleNamespace(deps=self.dependencies),
                receipt.assignment_id,
            )

        result = asyncio.run(run_scenario())

        self.assertEqual("completed", result.status)
        self.assertEqual(1, len(result.validation_results))
        self.assertEqual(0, result.validation_results[0].exit_code)
        self.assertEqual(
            1, self.runtime.modification_revisions_by_agent[result.agent_id]
        )
        self.assertEqual(
            1, self.runtime.validated_revisions_by_agent[result.agent_id]
        )

    def test_coordinator_assigns_independent_tasks_without_waiting(self) -> None:
        """统一分配工具即时返回，工作包随后在后台并行完成。"""
        active_calls = 0
        peak_active_calls = 0

        async def task_model_function(_messages, agent_info):
            nonlocal active_calls, peak_active_calls
            active_calls += 1
            peak_active_calls = max(peak_active_calls, active_calls)
            await asyncio.sleep(0.02)
            active_calls -= 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=agent_info.output_tools[0].name,
                        args={
                            "status": "completed",
                            "summary": "并行完成",
                            "evidence_ids": [],
                            "unresolved_issues": [],
                            "recommended_next_actions": [],
                        },
                    )
                ]
            )

        call_events = []
        self.runtime.call_events.subscribe(call_events.append)
        async def run_scenario():
            receipts = await assign_tasks(
                SimpleNamespace(
                    deps=self.dependencies,
                    model=FunctionModel(task_model_function),
                ),
                tasks=[
                    TaskAssignmentRequest(
                        template="explorer", task="检查模块 A"
                    ),
                    TaskAssignmentRequest(
                        template="reviewer", task="检查模块 B"
                    ),
                ],
            )
            self.assertEqual(0, active_calls)
            self.assertEqual(
                ["queued", "queued"],
                [receipt.status for receipt in receipts],
            )
            await asyncio.gather(
                *(
                    self.runtime.assignments[
                        receipt.assignment_id
                    ].background_task
                    for receipt in receipts
                )
            )
            return receipts

        receipts = asyncio.run(run_scenario())

        self.assertEqual(2, peak_active_calls)
        self.assertEqual(
            ["explorer-1", "reviewer-1"],
            [receipt.assignment_id for receipt in receipts],
        )
        self.assertEqual(
            ["started", "started", "completed", "completed"],
            [event.phase for event in call_events],
        )

    def test_coordinator_sends_feedback_to_same_agent_with_history(
        self,
    ) -> None:
        """验证失败后，协调 Agent 可把反馈交回原 worker。"""
        received_messages = []

        def task_model_function(messages, _agent_info):
            received_messages.append(messages)
            first_turn = len(received_messages) == 1
            response = "初次实现" if first_turn else "修正实现"
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=_agent_info.output_tools[0].name,
                        args={
                            "status": (
                                "needs_follow_up"
                                if first_turn
                                else "completed"
                            ),
                            "summary": response,
                            "evidence_ids": [],
                            "unresolved_issues": (
                                ["测试失败"] if first_turn else []
                            ),
                            "recommended_next_actions": (
                                ["提供失败输出"] if first_turn else []
                            ),
                        },
                    )
                ]
            )

        tool_context = SimpleNamespace(
            deps=self.dependencies,
            model=FunctionModel(task_model_function),
        )
        async def run_scenario():
            first_receipt = (
                await assign_tasks(
                    tool_context,
                    tasks=[
                        TaskAssignmentRequest(
                            template="worker",
                            task="实现功能",
                            skill_id="general",
                        )
                    ],
                )
            )[0]
            assignment = self.runtime.assignments[first_receipt.assignment_id]
            await assignment.background_task
            first_result = inspect_assignment(
                SimpleNamespace(deps=self.dependencies),
                first_receipt.assignment_id,
            )
            second_receipt = await send_task_feedback(
                tool_context,
                assignment_id=first_receipt.assignment_id,
                feedback="测试失败，请根据错误修正",
            )
            self.assertEqual("queued", second_receipt.status)
            await assignment.background_task
            second_result = inspect_assignment(
                SimpleNamespace(deps=self.dependencies),
                first_receipt.assignment_id,
            )
            return first_result, second_result

        first_result, second_result = asyncio.run(run_scenario())

        second_run_history = ModelMessagesTypeAdapter.dump_json(
            received_messages[1]
        ).decode()
        self.assertEqual(
            first_result.assignment_id, second_result.assignment_id
        )
        self.assertEqual("needs_follow_up", first_result.status)
        self.assertEqual(("测试失败",), first_result.unresolved_issues)
        self.assertEqual("修正实现", second_result.summary)
        self.assertEqual("completed", second_result.status)
        self.assertEqual(2, second_result.turn_index)
        self.assertIn("实现功能", second_run_history)
        self.assertIn("初次实现", second_run_history)
        self.assertIn("测试失败，请根据错误修正", second_run_history)
        self.assertIn("上一轮结构化交接", second_run_history)
        self.assertIn("needs_follow_up", second_run_history)
        self.assertIn("提供失败输出", second_run_history)
        snapshots = list_assignments(
            SimpleNamespace(deps=self.dependencies)
        )
        self.assertEqual(1, len(snapshots))
        self.assertEqual("completed", snapshots[0].status)
        self.assertEqual(2, snapshots[0].turn_count)
        self.assertEqual(
            second_result,
            inspect_assignment(
                SimpleNamespace(deps=self.dependencies),
                first_result.assignment_id,
            ),
        )

    def test_task_handoff_resolves_tool_evidence(self) -> None:
        """协调 Agent 收到结构化摘要和解析后的证据。"""
        (self.workspace_root / "note.txt").write_text(
            "关键结论", encoding="utf-8"
        )

        def task_model_function(messages, agent_info):
            tool_results = [
                part.content
                for message in messages
                for part in message.parts
                if hasattr(part, "tool_name")
                and part.tool_name == "read_workspace_file"
                and hasattr(part, "content")
            ]
            if not tool_results:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="read_workspace_file",
                            args={"path": "note.txt"},
                        )
                    ]
                )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=agent_info.output_tools[0].name,
                        args={
                            "status": "completed",
                            "summary": "已确认关键结论",
                            "facts": [
                                {
                                    "statement": "note.txt 包含关键结论",
                                    "citations": [
                                        {
                                            "evidence_id": "evidence-1",
                                            "quote": "关键结论",
                                        }
                                    ],
                                }
                            ],
                            "evidence_ids": ["evidence-1"],
                            "unresolved_issues": [],
                            "recommended_next_actions": [],
                        },
                    )
                ]
            )

        async def run_scenario():
            receipt = (
                await assign_tasks(
                    SimpleNamespace(
                        deps=self.dependencies,
                        model=FunctionModel(task_model_function),
                    ),
                    tasks=[
                        TaskAssignmentRequest(
                            template="explorer", task="读取结论"
                        )
                    ],
                )
            )[0]
            await self.runtime.assignments[
                receipt.assignment_id
            ].background_task
            return inspect_assignment(
                SimpleNamespace(deps=self.dependencies),
                receipt.assignment_id,
            )

        result = asyncio.run(run_scenario())

        self.assertEqual("已确认关键结论", result.summary)
        self.assertEqual(1, len(result.evidence))
        self.assertEqual("note.txt", result.evidence[0].source)
        self.assertEqual(1, len(result.facts))
        self.assertEqual(
            "note.txt 包含关键结论", result.facts[0].statement
        )

    def test_unknown_assignment_is_recoverable(self) -> None:
        """错误分配 ID 可返回协调模型重新安排。"""
        tool_context = SimpleNamespace(deps=self.dependencies)

        with self.assertRaisesRegex(
            RecoverableToolError, "未知任务分配"
        ):
            asyncio.run(
                send_task_feedback(
                    tool_context,
                    assignment_id="worker-99",
                    feedback="继续修改",
                )
            )

    def test_unknown_agent_skill_is_recoverable_before_start(self) -> None:
        """整批 Skill 在创建任何后台会话前完成校验。"""
        tool_context = SimpleNamespace(
            deps=self.dependencies,
            model=TestModel(call_tools=[]),
        )

        with self.assertRaisesRegex(RecoverableToolError, "无法加载"):
            asyncio.run(
                assign_tasks(
                    tool_context,
                    tasks=[
                        TaskAssignmentRequest(
                            template="explorer",
                            task="执行任务",
                            skill_id="missing",
                        )
                    ],
                )
            )
        self.assertEqual({}, self.runtime.assignments)

    def test_background_assignment_failure_is_recorded_and_notified(self) -> None:
        """后台异常会被记录并形成 failed 任务事件。"""
        completion_events = []

        async def completion_handler(event):
            completion_events.append(event)

        async def failing_model(_messages, _agent_info):
            raise RuntimeError("task model failed")

        async def run_scenario():
            self.runtime.assignment_events.subscribe(completion_handler)
            receipt = (
                await assign_tasks(
                    SimpleNamespace(
                        deps=self.dependencies,
                        model=FunctionModel(failing_model),
                    ),
                    tasks=[
                        TaskAssignmentRequest(
                            template="explorer", task="触发失败"
                        )
                    ],
                )
            )[0]
            assignment = self.runtime.assignments[receipt.assignment_id]
            await assignment.background_task
            return receipt, assignment

        receipt, assignment = asyncio.run(run_scenario())

        self.assertEqual("failed", assignment.status)
        self.assertIn("task model failed", assignment.error)
        self.assertEqual("failed", completion_events[0].status)
        with self.assertRaisesRegex(RecoverableToolError, "执行失败"):
            inspect_assignment(
                SimpleNamespace(deps=self.dependencies),
                receipt.assignment_id,
            )

    def test_scheduler_honors_concurrency_limit_and_priority(self) -> None:
        """单并发调度按优先级启动同一批次的任务。"""
        self.runtime.max_concurrent_assignments = 1
        start_order: list[str] = []
        active_calls = 0
        peak_active_calls = 0

        async def task_model_function(messages, agent_info):
            nonlocal active_calls, peak_active_calls
            serialized = ModelMessagesTypeAdapter.dump_json(messages).decode()
            start_order.append("high" if "高优先级" in serialized else "low")
            active_calls += 1
            peak_active_calls = max(peak_active_calls, active_calls)
            await asyncio.sleep(0.01)
            active_calls -= 1
            return ModelResponse(parts=[ToolCallPart(
                tool_name=agent_info.output_tools[0].name,
                args={"status": "completed", "summary": "完成"},
            )])

        async def run_scenario() -> None:
            receipts = await assign_tasks(
                SimpleNamespace(
                    deps=self.dependencies,
                    model=FunctionModel(task_model_function),
                ),
                tasks=[
                    TaskAssignmentRequest(
                        template="explorer", task="低优先级", priority=-10
                    ),
                    TaskAssignmentRequest(
                        template="reviewer", task="高优先级", priority=10
                    ),
                ],
            )
            await asyncio.gather(*[
                self.runtime.assignments[receipt.assignment_id].background_task
                for receipt in receipts
            ])

        asyncio.run(run_scenario())
        self.assertEqual(1, peak_active_calls)
        self.assertEqual(["high", "low"], start_order)

    def test_scheduler_retries_failure_and_releases_dependency(self) -> None:
        """失败任务按策略重试，成功后再释放依赖任务。"""
        attempts = 0
        execution_order: list[str] = []

        async def task_model_function(messages, agent_info):
            nonlocal attempts
            serialized = ModelMessagesTypeAdapter.dump_json(messages).decode()
            if "先执行" in serialized:
                attempts += 1
                execution_order.append(f"first-{attempts}")
                if attempts == 1:
                    raise RuntimeError("temporary failure")
            else:
                execution_order.append("dependent")
            return ModelResponse(parts=[ToolCallPart(
                tool_name=agent_info.output_tools[0].name,
                args={"status": "completed", "summary": "完成"},
            )])

        async def run_scenario() -> None:
            tool_context = SimpleNamespace(
                deps=self.dependencies,
                model=FunctionModel(task_model_function),
            )
            first = (await assign_tasks(
                tool_context,
                tasks=[TaskAssignmentRequest(
                    template="explorer", task="先执行", max_attempts=2
                )],
            ))[0]
            dependent = (await assign_tasks(
                tool_context,
                tasks=[TaskAssignmentRequest(
                    template="reviewer",
                    task="依赖后执行",
                    depends_on=[first.assignment_id],
                )],
            ))[0]
            await asyncio.gather(
                self.runtime.assignments[first.assignment_id].background_task,
                self.runtime.assignments[dependent.assignment_id].background_task,
            )

        asyncio.run(run_scenario())
        self.assertEqual(["first-1", "first-2", "dependent"], execution_order)
        self.assertEqual("completed", self.runtime.assignments["reviewer-1"].status)

    def test_scheduler_blocks_missing_restored_dependency(self) -> None:
        """损坏快照中的未知依赖会变为 blocked，而不会抛出 KeyError。"""
        completion_events: list[AssignmentCompletionEvent] = []

        async def completion_handler(event: AssignmentCompletionEvent) -> None:
            completion_events.append(event)

        async def executor(
            assignment: AssignmentSession, request: str
        ) -> AssignmentCompletionEvent:
            self.fail("缺少依赖的任务不应进入执行器")

        async def run_scenario() -> AssignmentSession:
            assignment = AssignmentSession(
                assignment_id="reviewer-1",
                coordinator_id=self.coordinator_context.agent_id,
                agent_id="reviewer-1",
                template="reviewer",
                agent=None,
                depends_on=("missing-1",),
            )
            self.runtime.assignments[assignment.assignment_id] = assignment
            self.runtime.assignment_events.subscribe(completion_handler)
            scheduler = AssignmentScheduler(self.runtime, executor)
            scheduler.enqueue(assignment, "检查恢复任务")
            assert assignment.background_task is not None
            await assignment.background_task
            return assignment

        assignment = asyncio.run(run_scenario())

        self.assertEqual("blocked", assignment.status)
        self.assertIn("依赖任务不存在", assignment.error)
        self.assertEqual("blocked", completion_events[0].status)

    def test_batch_task_keys_schedule_mixed_serial_and_parallel_work(self) -> None:
        """同批别名可表达前置阶段与其后的并行实现、测试分支。"""
        started_tasks: list[str] = []

        async def task_model_function(messages, agent_info):
            serialized = ModelMessagesTypeAdapter.dump_json(messages).decode()
            task_name = next(
                name
                for name in ("需求分析", "测试设计", "代码实现", "测试编写")
                if name in serialized
            )
            started_tasks.append(task_name)
            if task_name in {"需求分析", "测试设计"}:
                await asyncio.sleep(0.01)
            return ModelResponse(parts=[ToolCallPart(
                tool_name=agent_info.output_tools[0].name,
                args={"status": "completed", "summary": "完成"},
            )])

        async def run_scenario() -> tuple:
            receipts = await assign_tasks(
                SimpleNamespace(
                    deps=self.dependencies,
                    model=FunctionModel(task_model_function),
                ),
                tasks=[
                    TaskAssignmentRequest(
                        template="explorer", task="需求分析", task_key="analysis"
                    ),
                    TaskAssignmentRequest(
                        template="reviewer", task="测试设计", task_key="test_design"
                    ),
                    TaskAssignmentRequest(
                        template="worker",
                        task="代码实现",
                        task_key="implementation",
                        depends_on=["analysis", "test_design"],
                    ),
                    TaskAssignmentRequest(
                        template="worker",
                        task="测试编写",
                        depends_on=["analysis", "test_design"],
                    ),
                ],
            )
            await asyncio.gather(*[
                self.runtime.assignments[receipt.assignment_id].background_task
                for receipt in receipts
            ])
            return receipts

        receipts = asyncio.run(run_scenario())

        self.assertEqual({"需求分析", "测试设计"}, set(started_tasks[:2]))
        self.assertEqual({"代码实现", "测试编写"}, set(started_tasks[2:]))
        self.assertTrue(all(
            self.runtime.assignments[receipt.assignment_id].status == "completed"
            for receipt in receipts
        ))
        code_assignment = self.runtime.assignments[receipts[2].assignment_id]
        self.assertEqual(
            (receipts[0].assignment_id, receipts[1].assignment_id),
            code_assignment.depends_on,
        )

    def test_batch_task_keys_reject_cyclic_dependencies_before_enqueue(self) -> None:
        """循环别名不会创建永远无法被调度器释放的后台任务。"""
        tool_context = SimpleNamespace(
            deps=self.dependencies,
            model=TestModel(call_tools=[]),
        )

        with self.assertRaisesRegex(RecoverableToolError, "循环"):
            asyncio.run(assign_tasks(
                tool_context,
                tasks=[
                    TaskAssignmentRequest(
                        template="explorer",
                        task="任务 A",
                        task_key="task_a",
                        depends_on=["task_b"],
                    ),
                    TaskAssignmentRequest(
                        template="reviewer",
                        task="任务 B",
                        task_key="task_b",
                        depends_on=["task_a"],
                    ),
                ],
            ))
        self.assertEqual({}, self.runtime.assignments)

    def test_runtime_snapshot_recovers_interrupted_assignment(self) -> None:
        """快照保留历史和调度元数据，并把中断运行转换为排队。"""
        state_store = RuntimeStateStore(self.workspace_root / "state.json")
        self.runtime.persistence_handler = state_store.save

        async def run_scenario() -> str:
            receipt = (await assign_tasks(
                SimpleNamespace(deps=self.dependencies, model=TestModel(call_tools=[])),
                tasks=[TaskAssignmentRequest(
                    template="explorer",
                    task="可恢复任务",
                    priority=7,
                    max_attempts=3,
                )],
            ))[0]
            await self.runtime.assignments[receipt.assignment_id].background_task
            return receipt.assignment_id

        assignment_id = asyncio.run(run_scenario())
        assignment = self.runtime.assignments[assignment_id]
        assignment.status = "running"
        assignment.pending_request = "恢复后继续"
        assignment.attempts = 1
        self.runtime.modification_revisions_by_agent[assignment.agent_id] = 2
        self.runtime.validated_revisions_by_agent[assignment.agent_id] = 1
        state_store.save(self.runtime)

        restored = state_store.load(self.runtime.workspace)

        self.assertIsNotNone(restored)
        assert restored is not None
        recovered = restored.assignments[assignment_id]
        self.assertEqual("queued", recovered.status)
        self.assertEqual(7, recovered.priority)
        self.assertEqual(3, recovered.max_attempts)
        self.assertEqual(1, recovered.attempts)
        self.assertIsNone(recovered.agent)
        self.assertTrue(restored.agent_contexts[assignment_id].message_history)
        self.assertEqual(1, len(restored.pending_completion_events))
        self.assertEqual(
            2,
            restored.modification_revisions_by_agent[assignment.agent_id],
        )
        self.assertEqual(
            1,
            restored.validated_revisions_by_agent[assignment.agent_id],
        )

    def test_coordinator_can_cancel_queued_assignment(self) -> None:
        """取消排队任务会完成其等待句柄，且不会启动对应模型调用。"""
        self.runtime.max_concurrent_assignments = 1
        release_first = asyncio.Event()
        started_tasks: list[str] = []

        async def task_model_function(messages, agent_info):
            serialized = ModelMessagesTypeAdapter.dump_json(messages).decode()
            task_name = "first" if "占用并发位" in serialized else "second"
            started_tasks.append(task_name)
            if task_name == "first":
                await release_first.wait()
            return ModelResponse(parts=[ToolCallPart(
                tool_name=agent_info.output_tools[0].name,
                args={"status": "completed", "summary": "完成"},
            )])

        async def run_scenario() -> None:
            tool_context = SimpleNamespace(
                deps=self.dependencies,
                model=FunctionModel(task_model_function),
            )
            receipts = await assign_tasks(
                tool_context,
                tasks=[
                    TaskAssignmentRequest(
                        template="explorer", task="占用并发位"
                    ),
                    TaskAssignmentRequest(
                        template="reviewer", task="等待后取消"
                    ),
                ],
            )
            await asyncio.sleep(0)
            await cancel_assignment(
                tool_context, receipts[1].assignment_id, "不再需要复核"
            )
            await self.runtime.assignments[
                receipts[1].assignment_id
            ].background_task
            release_first.set()
            await self.runtime.assignments[
                receipts[0].assignment_id
            ].background_task

        asyncio.run(run_scenario())
        self.assertEqual(["first"], started_tasks)
        cancelled = self.runtime.assignments["reviewer-1"]
        self.assertEqual("cancelled", cancelled.status)
        self.assertEqual("不再需要复核", cancelled.error)


if __name__ == "__main__":
    unittest.main()

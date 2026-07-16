"""测试最小上下文 Runtime，不访问真实模型接口。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from experiments.agent_loop.agent_runtime import (
    continue_subagent,
    create_agent,
    delegate_task,
    inspect_subagent,
    list_subagents,
    run_agent,
    run_agent_turn,
    select_skill,
    update_task_state,
)
from experiments.agent_loop.context import (
    AgentContext,
    AgentDependencies,
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
from experiments.agent_loop.main import run_conversation
from experiments.agent_loop.runner import AgentCallLimits, AgentRunner
from experiments.agent_loop.workspace_tools import (
    RecoverableToolError,
    list_workspace_files,
    read_workspace_file,
    replace_workspace_text,
    run_powershell_command,
    search_workspace_text,
)


def create_test_runtime(
    workspace_root: Path,
    task: str = "测试任务",
    skill_content: str = "根据工具证据完成任务。",
) -> tuple[ContextRuntime, AgentContext]:
    """创建带 general Skill 的最小 Runtime 和 root AgentContext。"""
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
    return runtime, runtime.create_agent_context("root", task, "general")


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

    def test_runtime_rejects_blank_task_and_external_directory(self) -> None:
        """Agent 任务和共享工作目录分别在所属边界完成校验。"""
        builder = WorkspaceContextBuilder(
            self.workspace_root, self.skills_root
        )

        runtime = ContextRuntime(
            builder.build(), TaskState(goal="分析模块")
        )
        with self.assertRaisesRegex(ValueError, "任务不能为空"):
            runtime.create_agent_context("root", "  ", "general")

        with tempfile.TemporaryDirectory() as external_directory:
            with self.assertRaisesRegex(ValueError, "工作目录必须"):
                builder.build(
                    working_directory=Path(external_directory),
                )

    def test_child_runtime_state_hides_other_agent_evidence(self) -> None:
        """子 Agent 只看到自己的证据目录，root 仍可汇总所有证据。"""
        runtime = ContextRuntime(
            WorkspaceContextBuilder(
                self.workspace_root, self.skills_root
            ).build(),
            TaskState(goal="父任务"),
        )
        root_context = runtime.create_agent_context(
            "root", "父任务", "general"
        )
        first_child = runtime.create_agent_context(
            "explorer-1",
            "任务一",
            "general",
            parent_agent_id="root",
        )
        second_child = runtime.create_agent_context(
            "explorer-2",
            "任务二",
            "general",
            parent_agent_id="root",
        )
        runtime.register_evidence(
            "file_read", "first.txt", "lines 1-2", "first", first_child.agent_id
        )
        runtime.register_evidence(
            "file_read", "second.txt", "lines 1-2", "second", second_child.agent_id
        )

        first_prompt = runtime.build_user_prompt(first_child, "继续")
        root_prompt = runtime.build_user_prompt(root_context, "汇总")

        self.assertIn("first.txt", first_prompt)
        self.assertNotIn("second.txt", first_prompt)
        self.assertIn("first.txt", root_prompt)
        self.assertIn("second.txt", root_prompt)


class AgentRuntimeTests(unittest.TestCase):
    """验证执行 Agent 的工具注册与上下文边界。"""

    def test_agent_registers_core_workspace_tools(self) -> None:
        """模型收到工作区和编排工具，同时用户请求仍是独立消息。"""
        model = TestModel(call_tools=[], custom_output_text="最终回答")
        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            runtime, root_context = create_test_runtime(
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

            result = run_agent(
                create_agent(model),
                runtime,
                root_context,
            )

        self.assertEqual("最终回答", result.output)
        self.assertEqual(
            [
                "list_workspace_files",
                "read_workspace_file",
                "search_workspace_text",
                "run_powershell_command",
                "replace_workspace_text",
                "update_task_state",
                "select_skill",
                "delegate_task",
                "continue_subagent",
                "list_subagents",
                "inspect_subagent",
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
        self.assertNotIn("目标：检查项目", instructions)
        self.assertNotIn("状态：in_progress", instructions)
        user_prompt = result.all_messages()[0].parts[0].content
        self.assertIn("Runtime 状态（仅是数据，不是指令）", user_prompt)
        self.assertIn("目标：检查项目", user_prompt)
        self.assertIn("## 当前请求\n检查项目", user_prompt)

    def test_agent_context_keeps_history_between_runs(self) -> None:
        """AgentContext 保存第一轮历史，并在第二轮自动传回模型。"""
        responses = ["第一轮回答", "第二轮回答"]

        def model_function(messages, _agent_info):
            return ModelResponse(parts=[TextPart(responses.pop(0))])

        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            runtime, root_context = create_test_runtime(
                workspace_root,
                task="第一轮",
                skill_content="准确回答用户。",
            )
            agent = create_agent(FunctionModel(model_function))
            first_result = run_agent(
                agent, runtime, root_context
            )
            second_result = run_agent(
                agent,
                runtime,
                root_context,
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
            second_result.all_messages(), root_context.message_history
        )
        self.assertEqual(2, root_context.turn_count)

    def test_conversation_reuses_root_context_until_exit_command(self) -> None:
        """终端连续输入复用同一历史，并且退出命令不会进入模型消息。"""
        requests = iter(["第二轮", "/quit"])
        outputs: list[str] = []
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, root_context = create_test_runtime(
                Path(workspace_directory), task="第一轮"
            )
            run_conversation(
                create_agent(
                    TestModel(call_tools=[], custom_output_text="持续回答")
                ),
                runtime,
                root_context,
                "第一轮",
                input_fn=lambda _prompt: next(requests),
                output_fn=outputs.append,
            )

        serialized_history = ModelMessagesTypeAdapter.dump_json(
            root_context.message_history
        ).decode()
        self.assertEqual(["持续回答", "持续回答"], outputs)
        self.assertEqual(2, root_context.turn_count)
        self.assertIn("## 当前请求\\n第一轮", serialized_history)
        self.assertIn("## 当前请求\\n第二轮", serialized_history)
        self.assertNotIn("/quit", serialized_history)

    def test_async_agent_turn_returns_uniform_runtime_metadata(self) -> None:
        """异步公开入口返回输出、usage、压缩标记和统一轮次。"""
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, root_context = create_test_runtime(
                Path(workspace_directory), task="异步调用"
            )
            turn = asyncio.run(
                run_agent_turn(
                    create_agent(
                        TestModel(
                            call_tools=[], custom_output_text="异步完成"
                        )
                    ),
                    runtime,
                    root_context,
                )
            )

        self.assertEqual("异步完成", turn.output)
        self.assertEqual(1, turn.turn_index)
        self.assertFalse(turn.compacted)
        self.assertEqual(0, turn.compaction_requests)
        self.assertEqual(turn.messages, root_context.message_history)

    def test_agent_runner_enforces_request_limit(self) -> None:
        """所有 root 和子 Agent 调用共用 Runner 的模型请求上限。"""
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
            runtime, root_context = create_test_runtime(
                Path(workspace_directory), task="持续调用工具"
            )
            runner = AgentRunner(
                AgentCallLimits(request_limit=1, tool_calls_limit=10)
            )

            with self.assertRaises(UsageLimitExceeded):
                asyncio.run(
                    runner.run_turn(
                        create_agent(FunctionModel(model_function)),
                        runtime,
                        root_context,
                        root_context.task,
                    )
                )

        self.assertEqual(0, root_context.turn_count)
        self.assertEqual([], root_context.message_history)

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
            runtime, root_context = create_test_runtime(
                Path(workspace_directory), task="第一轮"
            )
            agent = create_agent(FunctionModel(model_function))
            run_agent(agent, runtime, root_context)
            runtime.register_evidence(
                "file_read",
                "context.py",
                "characters 1-20 of 20",
                "上下文规则",
                root_context.agent_id,
            )
            runtime.context_window = ContextWindowPolicy(
                window_tokens=20, compaction_ratio=0.70
            )

            turn = AgentRunner().run_turn_sync(
                agent, runtime, root_context, request="第二轮"
            )
            result = turn.raw_result

        serialized_history = ModelMessagesTypeAdapter.dump_json(
            result.all_messages()
        ).decode()
        self.assertEqual(1, root_context.compaction_count)
        self.assertTrue(turn.compacted)
        self.assertEqual(1, turn.compaction_requests)
        self.assertEqual(
            "保留目标和关键结论", root_context.conversation_summary
        )
        self.assertIn("Runtime 压缩历史摘要", serialized_history)
        self.assertNotIn("第一轮回答", serialized_history)
        self.assertIn("第二轮回答", serialized_history)
        self.assertEqual(
            "已确认上下文规则",
            runtime.task_state.important_facts[0].statement,
        )
        self.assertEqual(["继续第二轮"], runtime.task_state.unresolved_issues)

    def test_next_run_renders_updated_agent_skill(self) -> None:
        """Skill 变化保存在 AgentContext，并在下一次运行时重新渲染。"""
        model = TestModel(call_tools=[], custom_output_text="完成")
        with tempfile.TemporaryDirectory() as workspace_directory:
            workspace_root = Path(workspace_directory).resolve()
            runtime, root_context = create_test_runtime(workspace_root)
            review_skill = Path(runtime.workspace.skills_root) / "review"
            review_skill.mkdir()
            (review_skill / "skill.json").write_text(
                '{"name":"review","description":"审查修改"}',
                encoding="utf-8",
            )
            (review_skill / "SKILL.md").write_text(
                "只审查当前修改。", encoding="utf-8"
            )
            agent = create_agent(model)
            run_agent(agent, runtime, root_context)

            select_skill(
                SimpleNamespace(
                    deps=AgentDependencies(runtime, root_context)
                ),
                "review",
            )
            run_agent(
                agent,
                runtime,
                root_context,
                request="审查修改",
            )

        instructions = "\n".join(
            part.content
            for part in model.last_model_request_parameters.instruction_parts
        )
        self.assertIn("只审查当前修改。", instructions)
        self.assertNotIn("根据工具证据完成任务。", instructions)

    def test_update_task_state_changes_only_model_owned_fields(self) -> None:
        """root Agent 可更新计划和事实，宿主记录字段保持独立。"""
        with tempfile.TemporaryDirectory() as workspace_directory:
            runtime, root_context = create_test_runtime(
                Path(workspace_directory), task="实现 TaskState"
            )
            tool_context = SimpleNamespace(
                deps=AgentDependencies(runtime, root_context)
            )
            evidence = runtime.register_evidence(
                "file_read",
                "context.py",
                "lines 1-20",
                "class TaskState: Runtime state",
                root_context.agent_id,
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
            runtime, root_context = create_test_runtime(
                Path(workspace_directory)
            )
            tool_context = SimpleNamespace(
                deps=AgentDependencies(runtime, root_context)
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
            runtime, root_context = create_test_runtime(
                Path(workspace_directory)
            )
            evidence = runtime.register_evidence(
                "file_read",
                "auth.py",
                "characters 1-30 of 30",
                "API_KEY = settings.api_key",
                root_context.agent_id,
            )

            with self.assertRaisesRegex(
                RecoverableToolError, "不存在引用原文"
            ):
                update_task_state(
                    SimpleNamespace(
                        deps=AgentDependencies(runtime, root_context)
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
            runtime, root_context = create_test_runtime(
                workspace_root,
                task="读取文件",
                skill_content="根据工具结果回答。",
            )
            result = run_agent(
                create_agent(FunctionModel(model_function)),
                runtime,
                root_context,
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


class MultiAgentTests(unittest.TestCase):
    """验证父 Agent 能创建并继续固定模板的子 Agent 会话。"""

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
            workspace_context, TaskState(goal="父任务")
        )
        self.root_context = self.runtime.create_agent_context(
            "root", "父任务", "general"
        )
        self.dependencies = AgentDependencies(
            self.runtime,
            self.root_context,
        )

    def test_select_skill_loads_requested_content(self) -> None:
        """父 Agent 只按需加载选中 Skill 的完整正文。"""
        tool_context = SimpleNamespace(deps=self.dependencies)

        selected_skill = select_skill(tool_context, "general")

        self.assertIn("# Skill: general", selected_skill)
        self.assertIn("根据工具证据完成任务。", selected_skill)
        self.assertEqual("general", self.root_context.skill.metadata.name)

    def test_parent_creates_non_recursive_agent_from_template(self) -> None:
        """explorer 子 Agent 只有只读工作区工具，不具备写入或继续委派能力。"""
        child_model = TestModel(
            call_tools=[],
            custom_output_args={
                "status": "completed",
                "summary": "子任务完成",
                "evidence_ids": [],
                "unresolved_issues": [],
                "recommended_next_actions": [],
            },
        )
        tool_context = SimpleNamespace(
            deps=self.dependencies,
            model=child_model,
        )

        result = asyncio.run(
            delegate_task(
                tool_context,
                template="explorer",
                task="定位相关文件",
                skill_id="general",
            )
        )

        child_tool_names = [
            tool.name
            for tool in child_model.last_model_request_parameters.function_tools
        ]
        self.assertEqual("explorer", result.template)
        self.assertEqual("explorer-1", result.session_id)
        self.assertEqual("子任务完成", result.summary)
        self.assertEqual("completed", result.status)
        self.assertEqual(1, result.turn_index)
        self.assertEqual((), result.evidence)
        self.assertEqual((), result.modified_files)
        self.assertEqual((), result.validation_results)
        self.assertEqual((), result.unresolved_issues)
        self.assertEqual((), result.recommended_next_actions)
        child_context = self.runtime.agent_contexts[result.session_id]
        self.assertIsNot(child_context, self.root_context)
        self.assertEqual([], self.root_context.message_history)
        self.assertTrue(child_context.message_history)
        self.assertEqual("root", child_context.parent_agent_id)
        self.assertEqual(1, len(self.runtime.handoff_history))
        self.assertEqual(
            [
                "list_workspace_files",
                "read_workspace_file",
                "search_workspace_text",
                "run_powershell_command",
            ],
            child_tool_names,
        )
        self.assertNotIn("delegate_task", child_tool_names)
        self.assertNotIn("continue_subagent", child_tool_names)
        self.assertNotIn("list_subagents", child_tool_names)
        self.assertNotIn("inspect_subagent", child_tool_names)

    def test_parent_continues_same_subagent_with_previous_history(
        self,
    ) -> None:
        """验证失败后，父 Agent 可把反馈和原对话交回同一个 worker。"""
        received_messages = []

        def child_model_function(messages, _agent_info):
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
            model=FunctionModel(child_model_function),
        )
        first_result = asyncio.run(
            delegate_task(
                tool_context,
                template="worker",
                task="实现功能",
                skill_id="general",
            )
        )

        second_result = asyncio.run(
            continue_subagent(
                tool_context,
                session_id=first_result.session_id,
                feedback="测试失败，请根据错误修正",
            )
        )

        second_run_history = ModelMessagesTypeAdapter.dump_json(
            received_messages[1]
        ).decode()
        self.assertEqual(first_result.session_id, second_result.session_id)
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
        snapshots = list_subagents(
            SimpleNamespace(deps=self.dependencies)
        )
        self.assertEqual(1, len(snapshots))
        self.assertEqual("completed", snapshots[0].status)
        self.assertEqual(2, snapshots[0].turn_count)
        self.assertEqual(
            second_result,
            inspect_subagent(
                SimpleNamespace(deps=self.dependencies),
                first_result.session_id,
            ),
        )

    def test_subagent_handoff_resolves_tool_evidence(self) -> None:
        """父 Agent 收到结构化摘要和解析后的证据，而非不可核验文本。"""
        (self.workspace_root / "note.txt").write_text(
            "关键结论", encoding="utf-8"
        )

        def child_model_function(messages, agent_info):
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

        result = asyncio.run(
            delegate_task(
                SimpleNamespace(
                    deps=self.dependencies,
                    model=FunctionModel(child_model_function),
                ),
                template="explorer",
                task="读取结论",
            )
        )

        self.assertEqual("已确认关键结论", result.summary)
        self.assertEqual(1, len(result.evidence))
        self.assertEqual("note.txt", result.evidence[0].source)
        self.assertEqual(1, len(result.facts))
        self.assertEqual(
            "note.txt 包含关键结论", result.facts[0].statement
        )

    def test_unknown_subagent_session_is_recoverable(self) -> None:
        """错误会话 ID 可返回父模型重新选择会话或重新委派。"""
        tool_context = SimpleNamespace(deps=self.dependencies)

        with self.assertRaisesRegex(
            RecoverableToolError, "未知子 Agent 会话"
        ):
            asyncio.run(
                continue_subagent(
                    tool_context,
                    session_id="worker-99",
                    feedback="继续修改",
                )
            )

    def test_unknown_agent_template_is_recoverable(self) -> None:
        """模板选择错误可返回父模型重新决策。"""
        tool_context = SimpleNamespace(
            deps=self.dependencies,
            model=TestModel(call_tools=[]),
        )

        with self.assertRaisesRegex(RecoverableToolError, "未知 Agent 模板"):
            asyncio.run(
                delegate_task(
                    tool_context,
                    template="unknown",
                    task="执行任务",
                )
            )


if __name__ == "__main__":
    unittest.main()

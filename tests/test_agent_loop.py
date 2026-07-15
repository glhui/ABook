"""测试最小上下文 Runtime，不访问真实模型接口。"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from experiments.agent_loop.main import (
    AgentContext,
    AgentDependencies,
    ContextRuntime,
    continue_subagent,
    delegate_task,
    ProjectInstruction,
    RecoverableToolError,
    RepositoryContext,
    SkillMetadata,
    WorkspaceContext,
    WorkspaceContextBuilder,
    create_agent,
    list_workspace_files,
    read_workspace_file,
    replace_workspace_text,
    run_agent,
    run_powershell_command,
    select_skill,
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
        repository=RepositoryContext(
            is_repository=False,
            branch=None,
            status_lines=(),
            status_truncated=False,
        ),
        available_skills=(
            SkillMetadata(
                name="general", description="通用测试 Skill"
            ),
        ),
    )
    runtime = ContextRuntime(workspace_context)
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
        self.assertFalse(workspace_context.repository.is_repository)

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

        runtime = ContextRuntime(builder.build())
        with self.assertRaisesRegex(ValueError, "任务不能为空"):
            runtime.create_agent_context("root", "  ", "general")

        with tempfile.TemporaryDirectory() as external_directory:
            with self.assertRaisesRegex(ValueError, "工作目录必须"):
                builder.build(
                    working_directory=Path(external_directory),
                )


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
                "select_skill",
                "delegate_task",
                "continue_subagent",
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
        self.assertNotIn("检查项目", instructions)
        self.assertEqual("检查项目", result.all_messages()[0].parts[0].content)

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
        self.assertIn("第一轮", message_contents)
        self.assertIn("第一轮回答", message_contents)
        self.assertIn("第二轮", message_contents)
        self.assertEqual(
            second_result.all_messages(), root_context.message_history
        )

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
        self.runtime = ContextRuntime(workspace_context)
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
            call_tools=[], custom_output_text="子任务完成"
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
        self.assertEqual("子任务完成", result.output)
        child_context = self.runtime.agent_contexts[result.session_id]
        self.assertIsNot(child_context, self.root_context)
        self.assertEqual([], self.root_context.message_history)
        self.assertTrue(child_context.message_history)
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

    def test_parent_continues_same_subagent_with_previous_history(
        self,
    ) -> None:
        """验证失败后，父 Agent 可把反馈和原对话交回同一个 worker。"""
        received_messages = []

        def child_model_function(messages, _agent_info):
            received_messages.append(messages)
            response = "初次实现" if len(received_messages) == 1 else "修正实现"
            return ModelResponse(parts=[TextPart(response)])

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
                task="测试失败，请根据错误修正",
            )
        )

        second_run_contents = [
            part.content
            for message in received_messages[1]
            for part in message.parts
            if hasattr(part, "content")
        ]
        self.assertEqual(first_result.session_id, second_result.session_id)
        self.assertEqual("修正实现", second_result.output)
        self.assertIn("实现功能", second_run_contents)
        self.assertIn("初次实现", second_run_contents)
        self.assertIn("测试失败，请根据错误修正", second_run_contents)

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
                    task="继续修改",
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

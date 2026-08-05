import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from skill_loading import (
    SkillCatalog,
    SkillSelectionPolicy,
    SkillSelector,
    SkillTaskContext,
    collect_python_import_names,
    render_skill_instructions,
)


# 验证 Skill catalog 只加载 manifest，并按稳定信号和预算加载少量正文。
class SkillLoadingTests(unittest.TestCase):
    # 路径信号应选中仓库 Skill，import 信号应选中第三方包 Skill。
    def test_selects_repository_and_package_skills_from_distinct_signals(self: "SkillLoadingTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_skill(
                root,
                "agent-profiles",
                {
                    "name": "agent-profiles",
                    "version": "1.0.0",
                    "kind": "repository",
                    "description": "Agent profile repository rules.",
                    "roles": ["python_code"],
                    "path_patterns": ["src/agent_profiles/**/*.py"],
                    "priority": 90,
                    "estimated_tokens": 300,
                },
                "保持 Agent 工具权限边界。",
            )
            self._write_skill(
                root,
                "pydantic-ai",
                {
                    "name": "pydantic-ai",
                    "version": "1.0.0",
                    "kind": "package",
                    "description": "Pydantic AI package usage.",
                    "roles": ["python_code"],
                    "import_names": ["pydantic_ai"],
                    "estimated_tokens": 300,
                },
                "使用结构化 Agent 输出。",
            )

            selected = SkillSelector(SkillCatalog.discover(root)).select(
                SkillTaskContext(
                    task="调整 Agent 创建逻辑",
                    role="python_code",
                    target_paths=("src/agent_profiles/python_code.py",),
                    import_names=("pydantic_ai.models",),
                )
            )

            self.assertEqual({skill.name for skill in selected}, {"agent-profiles", "pydantic-ai"})
            self.assertTrue(any("目标路径匹配" in reason for reason in selected[0].reasons))
            self.assertIn("Skill: agent-profiles", render_skill_instructions(selected))

    # 无路由信号的 Skill 即使 priority 很高也不能进入候选。
    def test_does_not_select_skill_from_priority_alone(self: "SkillLoadingTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_skill(
                root,
                "unrelated",
                {
                    "name": "unrelated",
                    "version": "1.0.0",
                    "kind": "workflow",
                    "description": "Unrelated workflow.",
                    "roles": ["python_code"],
                    "triggers": ["database migration"],
                    "priority": 100,
                    "estimated_tokens": 100,
                },
                "迁移数据库。",
            )

            selected = SkillSelector(SkillCatalog.discover(root)).select(
                SkillTaskContext(task="实现字符串解析", role="python_code")
            )

            self.assertEqual(selected, ())

    # 数量与 token 预算是防止全量加载的硬边界。
    def test_applies_count_and_token_limits_before_loading_content(self: "SkillLoadingTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(3):
                self._write_skill(
                    root,
                    f"skill-{index}",
                    {
                        "name": f"skill-{index}",
                        "version": "1.0.0",
                        "kind": "workflow",
                        "description": f"Workflow {index}.",
                        "roles": ["python_code"],
                        "triggers": ["共同关键词"],
                        "priority": 100 - index,
                        "estimated_tokens": 400,
                    },
                    f"Skill {index}",
                )

            selected = SkillSelector(
                SkillCatalog.discover(root),
                SkillSelectionPolicy(max_skills=2, max_tokens=500),
            ).select(SkillTaskContext(task="共同关键词", role="python_code"))

            self.assertEqual([skill.name for skill in selected], ["skill-0"])

    # 未选中的正文使用无效 UTF-8 也不影响发现和其他 Skill 的选择，证明 catalog 不会预加载正文。
    def test_does_not_read_unselected_skill_content(self: "SkillLoadingTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_skill(
                root,
                "selected",
                {
                    "name": "selected",
                    "version": "1.0.0",
                    "kind": "workflow",
                    "description": "Selected workflow.",
                    "triggers": ["命中"],
                    "estimated_tokens": 100,
                },
                "有效正文",
            )
            invalid_directory = root / "unselected"
            invalid_directory.mkdir()
            (invalid_directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": "unselected",
                        "version": "1.0.0",
                        "kind": "workflow",
                        "description": "Unselected workflow.",
                        "triggers": ["不会匹配"],
                        "estimated_tokens": 100,
                    }
                ),
                encoding="utf-8",
            )
            (invalid_directory / "SKILL.md").write_bytes(b"\xff\xfe")

            selected = SkillSelector(SkillCatalog.discover(root)).select(
                SkillTaskContext(task="命中", role="python_code")
            )

            self.assertEqual([skill.name for skill in selected], ["selected"])

    # 显式选择仍需通过角色硬过滤，防止把不适用规则注入 Agent。
    def test_rejects_explicit_skill_for_incompatible_role(self: "SkillLoadingTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._write_skill(
                root,
                "test-only",
                {
                    "name": "test-only",
                    "version": "1.0.0",
                    "kind": "workflow",
                    "description": "Test-only workflow.",
                    "roles": ["python_test"],
                    "estimated_tokens": 100,
                },
                "只用于测试。",
            )

            with self.assertRaisesRegex(ValueError, "不适用于角色"):
                SkillSelector(SkillCatalog.discover(root)).select(
                    SkillTaskContext(
                        task="实现功能",
                        role="python_code",
                        explicit_skills=("test-only",),
                    )
                )

    # 现有源码的 import 可转换为稳定包名，直接作为包 Skill 路由信号。
    def test_collects_top_level_python_import_names(self: "SkillLoadingTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "module.py"
            source_path.write_text(
                "import pydantic_ai.models\nfrom pathlib import Path\nfrom pydantic import BaseModel\n",
                encoding="utf-8",
            )

            import_names = collect_python_import_names((source_path,))

            self.assertEqual(import_names, ("pathlib", "pydantic", "pydantic_ai"))

    # 创建测试 catalog 中的一个完整 Skill 目录。
    def _write_skill(
        self: "SkillLoadingTests",
        root: Path,
        directory_name: str,
        manifest: dict[str, object],
        content: str,
    ) -> None:
        skill_directory = root / directory_name
        skill_directory.mkdir()
        (skill_directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        (skill_directory / "SKILL.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

from pathlib import Path

from skill_loading.models import SkillDescriptor, SkillManifest


MANIFEST_FILE_NAME = "manifest.json"
SKILL_FILE_NAME = "SKILL.md"


# 仅发现 Skill manifest，并在最终选中后读取对应正文。
class SkillCatalog:
    # 建立按名称索引，确保后续选择不需要再次扫描文件系统。
    def __init__(self: "SkillCatalog", descriptors: tuple[SkillDescriptor, ...]) -> None:
        self._descriptors = {descriptor.manifest.name: descriptor for descriptor in descriptors}
        if len(self._descriptors) != len(descriptors):
            raise ValueError("Skill 名称必须在 catalog 中唯一。")

    # 返回仅含 manifest 和路径的稳定快照。
    @property
    def descriptors(self: "SkillCatalog") -> tuple[SkillDescriptor, ...]:
        return tuple(self._descriptors.values())

    # 递归发现 manifest；缺失正文的目录会立即失败，避免运行中选到不完整 Skill。
    @classmethod
    def discover(cls, root_directory: Path) -> "SkillCatalog":
        if not root_directory.is_dir():
            raise ValueError(f"Skill 目录不存在：{root_directory}")
        descriptors: list[SkillDescriptor] = []
        for manifest_path in sorted(root_directory.rglob(MANIFEST_FILE_NAME)):
            skill_directory = manifest_path.parent
            content_path = skill_directory / SKILL_FILE_NAME
            if not content_path.is_file():
                raise ValueError(f"Skill 缺少 {SKILL_FILE_NAME}：{skill_directory}")
            manifest = SkillManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            descriptors.append(
                SkillDescriptor(
                    manifest=manifest,
                    directory=skill_directory.resolve(),
                    content_path=content_path.resolve(),
                )
            )
        return cls(tuple(descriptors))

    # 按唯一名称读取 descriptor，不存在时给出明确错误。
    def get(self: "SkillCatalog", name: str) -> SkillDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as error:
            raise ValueError(f"未知 Skill：{name}") from error

    # 只为已完成路由的 Skill 读取正文，并拒绝空内容。
    def load_content(self: "SkillCatalog", name: str) -> str:
        descriptor = self.get(name)
        content = descriptor.content_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Skill 正文不能为空：{descriptor.content_path}")
        return content

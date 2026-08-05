from dataclasses import dataclass
from fnmatch import fnmatch

from skill_loading.catalog import SkillCatalog
from skill_loading.models import SelectedSkill, SkillDescriptor, SkillSelectionPolicy, SkillTaskContext


EXPLICIT_SKILL_SCORE = 10_000
PATH_MATCH_SCORE = 400
IMPORT_MATCH_SCORE = 350
SYMBOL_MATCH_SCORE = 250
TRIGGER_MATCH_SCORE = 100


# 保存尚未读取正文的候选结果。
@dataclass(frozen=True)
class _SkillCandidate:
    descriptor: SkillDescriptor
    score: int
    reasons: tuple[str, ...]


# 根据角色、任务、路径和代码线索选择少量 Skill，再加载最终正文。
class SkillSelector:
    # 保存 catalog 与本次选择预算；不在构造阶段读取任何 Skill 正文。
    def __init__(
        self: "SkillSelector",
        catalog: SkillCatalog,
        policy: SkillSelectionPolicy | None = None,
    ) -> None:
        self._catalog = catalog
        self._policy = policy or SkillSelectionPolicy()

    # 先完成硬过滤与 manifest 打分，预算确定后才读取正文。
    def select(self: "SkillSelector", context: SkillTaskContext) -> tuple[SelectedSkill, ...]:
        self._validate_explicit_skills(context)
        candidates = self._create_candidates(context)
        selected_candidates = self._apply_limits(candidates, context.role)
        return tuple(self._load_selected(candidate) for candidate in selected_candidates)

    # 显式名称拼写错误或角色不兼容时直接报告，避免静默退化为错误上下文。
    def _validate_explicit_skills(self: "SkillSelector", context: SkillTaskContext) -> None:
        for name in context.explicit_skills:
            descriptor = self._catalog.get(name)
            roles = descriptor.manifest.roles
            if roles and context.role not in roles:
                raise ValueError(f"Skill {name} 不适用于角色 {context.role}。")

    # 只有出现至少一个实际路由信号的 manifest 才成为候选。
    def _create_candidates(self: "SkillSelector", context: SkillTaskContext) -> tuple[_SkillCandidate, ...]:
        candidates: list[_SkillCandidate] = []
        for descriptor in self._catalog.descriptors:
            manifest = descriptor.manifest
            if manifest.roles and context.role not in manifest.roles:
                continue
            reasons, signal_score = _match_manifest(descriptor, context)
            if not reasons:
                continue
            candidates.append(
                _SkillCandidate(
                    descriptor=descriptor,
                    score=signal_score + manifest.priority,
                    reasons=tuple(reasons),
                )
            )
        return tuple(sorted(candidates, key=lambda item: (-item.score, item.descriptor.manifest.name)))

    # 按得分选择主 Skill，并在同一数量与 token 预算内补充一层显式依赖。
    def _apply_limits(
        self: "SkillSelector",
        candidates: tuple[_SkillCandidate, ...],
        role: str,
    ) -> tuple[_SkillCandidate, ...]:
        selected: list[_SkillCandidate] = []
        selected_names: set[str] = set()
        used_tokens = 0
        for candidate in candidates:
            expanded = self._expand_candidate(candidate, role)
            new_candidates = [item for item in expanded if item.descriptor.manifest.name not in selected_names]
            new_tokens = sum(item.descriptor.manifest.estimated_tokens for item in new_candidates)
            if len(selected) + len(new_candidates) > self._policy.max_skills:
                continue
            if used_tokens + new_tokens > self._policy.max_tokens:
                continue
            if any(_conflicts(item, selected) for item in new_candidates):
                continue
            selected.extend(new_candidates)
            selected_names.update(item.descriptor.manifest.name for item in new_candidates)
            used_tokens += new_tokens
        return tuple(selected)

    # 依赖只由 manifest 名称解析，并受深度上限约束，防止递归加载整个 catalog。
    def _expand_candidate(self: "SkillSelector", candidate: _SkillCandidate, role: str) -> tuple[_SkillCandidate, ...]:
        expanded: list[_SkillCandidate] = []
        visiting: set[str] = set()

        # 先加入依赖再加入主 Skill，使最终 instructions 的基础知识位于具体知识之前。
        def visit(current: _SkillCandidate, depth: int) -> None:
            name = current.descriptor.manifest.name
            if name in visiting:
                raise ValueError(f"Skill 依赖存在循环：{name}")
            visiting.add(name)
            if depth < self._policy.max_dependency_depth:
                for dependency_name in current.descriptor.manifest.depends_on:
                    dependency = self._catalog.get(dependency_name)
                    if dependency.manifest.roles and role not in dependency.manifest.roles:
                        raise ValueError(f"Skill 依赖 {dependency_name} 不适用于角色 {role}。")
                    visit(
                        _SkillCandidate(dependency, current.score, (f"由 {name} 依赖",)),
                        depth + 1,
                    )
            visiting.remove(name)
            if all(item.descriptor.manifest.name != name for item in expanded):
                expanded.append(current)

        visit(candidate, 0)
        return tuple(expanded)

    # 将最终选择转换为携带正文和选择证据的公共结果。
    def _load_selected(self: "SkillSelector", candidate: _SkillCandidate) -> SelectedSkill:
        manifest = candidate.descriptor.manifest
        return SelectedSkill(
            name=manifest.name,
            version=manifest.version,
            kind=manifest.kind,
            score=candidate.score,
            reasons=candidate.reasons,
            estimated_tokens=manifest.estimated_tokens,
            content=self._catalog.load_content(manifest.name),
        )


# 计算一个 manifest 的可解释匹配信号；priority 不能单独让 Skill 入选。
def _match_manifest(descriptor: SkillDescriptor, context: SkillTaskContext) -> tuple[list[str], int]:
    manifest = descriptor.manifest
    reasons: list[str] = []
    score = 0
    if manifest.name in context.explicit_skills:
        reasons.append("用户显式指定")
        score += EXPLICIT_SKILL_SCORE
    matched_paths = [path for path in context.target_paths if _matches_any_path(path, manifest.path_patterns)]
    if matched_paths:
        reasons.append(f"目标路径匹配：{matched_paths[0]}")
        score += PATH_MATCH_SCORE
    normalized_imports = {_normalize_import(name) for name in context.import_names}
    matched_imports = [name for name in manifest.import_names if _normalize_import(name) in normalized_imports]
    if matched_imports:
        reasons.append(f"导入匹配：{matched_imports[0]}")
        score += IMPORT_MATCH_SCORE
    normalized_symbols = {symbol.casefold() for symbol in context.symbols}
    matched_symbols = [symbol for symbol in manifest.symbols if symbol.casefold() in normalized_symbols]
    if matched_symbols:
        reasons.append(f"符号匹配：{matched_symbols[0]}")
        score += SYMBOL_MATCH_SCORE
    task = context.task.casefold()
    matched_triggers = [trigger for trigger in manifest.triggers if trigger.casefold() in task]
    if matched_triggers:
        reasons.append(f"任务关键词匹配：{matched_triggers[0]}")
        score += TRIGGER_MATCH_SCORE
    return reasons, score


# 同时兼容 ** 匹配零层或多层目录的常用写法。
def _matches_any_path(path: str, patterns: tuple[str, ...]) -> bool:
    normalized_path = path.replace("\\", "/").lstrip("./")
    for pattern in patterns:
        normalized_pattern = pattern.replace("\\", "/").lstrip("./")
        if fnmatch(normalized_path, normalized_pattern):
            return True
        if "/**/" in normalized_pattern and fnmatch(normalized_path, normalized_pattern.replace("/**/", "/")):
            return True
    return False


# 包 Skill 以顶层 import 名作为稳定路由键，兼容 package.submodule。
def _normalize_import(name: str) -> str:
    return name.strip().split(".", maxsplit=1)[0].replace("-", "_").casefold()


# 任一方向声明冲突都视为不可共同加载。
def _conflicts(candidate: _SkillCandidate, selected: list[_SkillCandidate]) -> bool:
    candidate_manifest = candidate.descriptor.manifest
    for selected_candidate in selected:
        selected_manifest = selected_candidate.descriptor.manifest
        if selected_manifest.name in candidate_manifest.conflicts_with:
            return True
        if candidate_manifest.name in selected_manifest.conflicts_with:
            return True
    return False


# 以固定优先级声明包装正文，使项目约束始终高于按需 Skill。
def render_skill_instructions(skills: tuple[SelectedSkill, ...]) -> str:
    if not skills:
        return ""
    sections = [
        "以下 Skill 仅用于当前任务。用户要求、AGENTS.md 和角色约束高于 Skill；发生冲突时遵循更高优先级约束。"
    ]
    for skill in skills:
        sections.append(f"## Skill: {skill.name} {skill.version}\n\n{skill.content}")
    return "\n\n".join(sections)

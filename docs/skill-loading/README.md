# Skill 按需加载

`skill_loading` 将 Skill 分成轻量路由信息与完整正文。`SkillCatalog.discover` 启动时
只读取每个目录的 `manifest.json`，不会读取 `SKILL.md`；`SkillSelector` 根据当前
任务完成硬过滤、打分和预算截断后，才读取最终选中的正文。

## 目录约定

```text
skills/
├─ packages/
│  └─ pydantic-ai/
│     ├─ manifest.json
│     └─ SKILL.md
└─ repository/
   └─ agent-profiles/
      ├─ manifest.json
      └─ SKILL.md
```

包 Skill 用 `import_names`、`symbols` 和任务关键词识别第三方包；仓库 Skill 用
`path_patterns` 识别本项目的模块边界。Skill 正文用于说明“在该领域如何工作”，
当前接口、实现与测试事实仍应从仓库文件中读取，避免 Skill 成为容易过期的源码副本。

## 选择顺序

路由按以下步骤执行：

1. 根据 `roles` 过滤不适用于当前 Agent 的 Skill。
2. 使用显式名称、目标路径、import、符号和任务关键词产生候选；`priority` 只用于候选排序，不能单独触发加载。
3. 按得分从高到低处理候选，并解析受深度限制的 `depends_on`。
4. 应用 `conflicts_with`、`max_skills` 和 `max_tokens` 硬限制。
5. 读取最终选中的 `SKILL.md`，并记录每个 Skill 的选择原因。

默认最多加载三个 Skill，总声明预算为 6,000 tokens，依赖深度最多一层。调用方可以
通过 `SkillSelectionPolicy` 收紧这些限制。

## 调用示例

```python
from pathlib import Path

from skill_loading import (
    SkillCatalog,
    SkillSelector,
    SkillTaskContext,
    collect_python_import_names,
    render_skill_instructions,
)

source_path = Path("src/agent_profiles/python_code.py")
catalog = SkillCatalog.discover(Path("skills"))
selected = SkillSelector(catalog).select(
    SkillTaskContext(
        task="调整 Python 代码 Agent 的创建逻辑",
        role="python_code",
        target_paths=(source_path.as_posix(),),
        import_names=collect_python_import_names((source_path,)),
        symbols=("create_python_code_agent",),
    )
)
skill_instructions = render_skill_instructions(selected)
```

`skill_instructions` 可以传给 `create_python_code_agent`。固定角色说明、项目
`AGENTS.md` 和用户任务始终高于 Skill 内容。

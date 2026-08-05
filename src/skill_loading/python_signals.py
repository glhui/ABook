import ast
from pathlib import Path


# 从已有 Python 文件提取顶层 import 名，供包 Skill 路由使用。
def collect_python_import_names(paths: tuple[Path, ...]) -> tuple[str, ...]:
    import_names: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                import_names.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                import_names.add(node.module.split(".", maxsplit=1)[0])
    return tuple(sorted(import_names))

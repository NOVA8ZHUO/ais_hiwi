#!/usr/bin/env python3
"""Extract detector core modules from the validated notebooks.

The source notebooks remain untouched. This script copies imports, constants,
function definitions, and class definitions into importable Python modules while
skipping notebook top-level execution cells that capture cameras, save files, or
display figures.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent

BAD_TARGETS = {
    "DETECTION_RESULTS",
    "DEBUG_IMAGES",
    "FINAL_LIVE_DETECTION_RESULTS",
    "FINAL_LIVE_RESULT_SUMMARY",
    "DETECTION_RESULTS_CAMERA_TABLE_OFFLINE",
    "DETECTION_RESULTS_CAMERA_TABLE_MANUAL",
    "LAST_LIVE_DETECTION_RESULTS",
    "LAST_LIVE_COLOR_BGR",
    "LAST_LIVE_ANNOTATED",
    "LAST_LIVE_DEBUG_VIEWS",
    "LAST_LIVE_TIMESTAMP_SEC",
    "FINAL_LIVE_FRAME_AVAILABLE",
    "color_image_bgr",
    "depth_image_m",
    "intrinsics",
    "result_dict",
    "annotated_image",
    "debug_images",
    "saved_paths",
}

SKIP_IMPORT_ROOTS = {"matplotlib", "pandas"}


def _source_from_notebook(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    blocks: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            blocks.append("".join(cell.get("source", [])))
    return "\n\n".join(blocks)


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _is_upper_constant_assignment(node: ast.AST) -> bool:
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False

    names: set[str] = set()
    for target in targets:
        names.update(_target_names(target))
    if not names or names & BAD_TARGETS:
        return False
    return all(name.isupper() for name in names)


def _is_import_try(node: ast.Try) -> bool:
    return any(isinstance(item, (ast.Import, ast.ImportFrom)) for item in node.body)


def _import_root(alias_name: str) -> str:
    return alias_name.split(".", 1)[0]


def _is_skipped_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return all(_import_root(alias.name) in SKIP_IMPORT_ROOTS for alias in node.names)
    if isinstance(node, ast.ImportFrom) and node.module:
        return _import_root(node.module) in SKIP_IMPORT_ROOTS
    return False


def _selected_source_blocks(source: str) -> list[str]:
    tree = ast.parse(source)
    selected: list[str] = []
    for node in tree.body:
        include = False
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            include = False
        elif _is_skipped_import(node):
            include = False
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(source, node) or ""
            include = "pd." not in segment and "display(" not in segment
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            include = True
        elif isinstance(node, ast.Try):
            include = _is_import_try(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            include = _is_upper_constant_assignment(node)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and not selected
        ):
            include = True

        if include:
            segment = ast.get_source_segment(source, node)
            if segment:
                selected.append(segment.rstrip())
    return selected


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "global_rgb_part_detection.ipynb").is_file()
            and (candidate / "local_rgbd_grasp_refinement.ipynb").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find global_rgb_part_detection.ipynb and "
        "local_rgbd_grasp_refinement.ipynb in this directory or its parents."
    )


def extract(notebook_name: str, output_name: str) -> Path:
    source = _source_from_notebook(find_project_root() / notebook_name)
    blocks = _selected_source_blocks(source)
    output = OUT_DIR / output_name
    header = (
        "#!/usr/bin/env python3\n"
        f"# Generated from {notebook_name} by extract_notebook_cores.py.\n"
        "# Do not edit the notebook to update this file; rerun the extractor.\n\n"
        "from __future__ import annotations\n\n"
    )
    body = "\n\n\n".join(blocks)
    output.write_text(header + body + "\n", encoding="utf-8")
    return output


def main() -> None:
    outputs = [
        extract("global_rgb_part_detection.ipynb", "global_rgb_part_detector_core.py"),
        extract("local_rgbd_grasp_refinement.ipynb", "local_rgbd_grasp_refiner_core.py"),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

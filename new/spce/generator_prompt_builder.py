#!/usr/bin/env python3
"""Build a generator-writing prompt for a given ARC-AGI-2 task.

Usage:
    python generator_prompt_builder.py <task_id>
        Legacy: just build the prompt.

    python generator_prompt_builder.py <task_id> --prepare
        Move any generate_input code from the solver file to
        generators/{task_id}.py (renamed to avoid collisions),
        then build the prompt.

Reads:
    - BARC/generator_prompt.md              (template)
    - BARC/tmp/{task_id}_spec.yaml          (human-reviewed spec)
    - BARC/arc-agi-2-eval-annotations-merge.json  (annotation)

Writes:
    - BARC/tmp/generator_prompt_{task_id}.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent  # generators/
ROOT = BASE.parent  # project root
SOLVER_DIR = ROOT.parent / "new_code" / "solvers" / "arc2"
SPEC_DIR = ROOT / "spce_dis"
ARC_UTILS_FILE = ROOT.parent / "new_code" / "solvers" / "arc_utils.py"


def _solver_path(task_id: str) -> Path:
    return SOLVER_DIR / f"{task_id}.py"


def _eval_path(task_id: str) -> Path:
    candidates = [
        ROOT / "data" / "evaluation" / f"{task_id}.json",
        ROOT.parent / "NVARC" / "ARC-data" / "ARC-AGI-2" / "data" / "evaluation" / f"{task_id}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    sys.exit(
        "Eval data not found for "
        f"{task_id}. Tried: {', '.join(str(p) for p in candidates)}"
    )


def _arc_utils_path() -> Path:
    if ARC_UTILS_FILE.exists():
        return ARC_UTILS_FILE
    sys.exit(f"arc_utils.py not found: {ARC_UTILS_FILE}")


def _gen_file(task_id: str) -> Path:
    return BASE / "arc2" / f"{task_id}.py"


def _find_main_end(solver_lines: list[str]) -> int:
    """Return the 0-based index of the last line belonging to main().

    After the 'def main(' line, main() continues as long as lines are
    blank or indented. The first non-blank, non-indented line marks
    the end of main().
    """
    in_main = False
    last = 0
    for i, line in enumerate(solver_lines):
        if line.startswith("def main("):
            in_main = True
            last = i
            continue
        if in_main:
            if line.strip() == "" or line[0] in (" ", "\t"):
                last = i
            else:
                break  # any top-level code/import/comment = not part of main()
    return last


def _next_version(gen_file: Path) -> int:
    """Determine the next version number from existing functions in gen_file."""
    if not gen_file.exists():
        return 1
    nums = [int(m.group(1))
            for m in re.finditer(r"def generate_input_v(\d+)_", gen_file.read_text())]
    return max(nums) + 1 if nums else 1


def _gen_file_header(task_id: str) -> str:
    solver_path = _solver_path(task_id)
    return (
        "from __future__ import annotations\n"
        "import numpy as np\n"
        "from arc_utils import colors_present, components_mask, get_bg\n"
        "import importlib.util\n"
        f'_solver_path = r"{solver_path}"\n'
        f'_spec = importlib.util.spec_from_file_location("arc2_{task_id}", _solver_path)\n'
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "assert _spec and _spec.loader\n"
        "_spec.loader.exec_module(_mod)\n"
        "main = _mod.main\n"
    )


# ── prepare ───────────────────────────────────────────────────────────

def prepare(task_id: str) -> None:
    solver = _solver_path(task_id)
    solver_lines = solver.read_text().splitlines(keepends=True)
    main_end = _find_main_end(solver_lines)
    new_code = "".join(solver_lines[main_end + 1:]).strip()

    if new_code:
        gen_file = _gen_file(task_id)
        gen_file.parent.mkdir(parents=True, exist_ok=True)

        v = _next_version(gen_file)
        label = f"v{v}_subagent"

        renamed = new_code.replace(
            "def generate_input()", f"def generate_input_{label}()", 1
        )

        if not gen_file.exists():
            gen_file.write_text(_gen_file_header(task_id) + "\n\n")

        with open(gen_file, "a") as f:
            f.write(f"\n\n# === {label} ===\n\n{renamed}\n")
        print(f"Moved generator code → {gen_file} as generate_input_{label}()")

        clean = "".join(solver_lines[: main_end + 1])
        if not clean.endswith("\n"):
            clean += "\n"
        solver.write_text(clean)
        print(f"Cleaned {solver} → {main_end + 1} lines")
    else:
        print(f"Solver {solver} is already clean (no generator code)")

    prompt = build_prompt(task_id)
    out_path = ROOT / "tmp" / f"generator_prompt_{task_id}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt)
    print(f"Wrote {len(prompt)} chars to {out_path}")


# ── prompt building ───────────────────────────────────────────────────

def build_prompt(task_id: str) -> str:
    template = (BASE / "generator_prompt.md").read_text()

    spec_path = SPEC_DIR / f"{task_id}_spec.yaml"
    if not spec_path.exists():
        sys.exit(f"Spec not found: {spec_path}")
    spec_yaml = spec_path.read_text().strip()

    ann_path = ROOT / "arc-agi-2-eval-annotations-merge.json"
    with open(ann_path) as f:
        annotations = json.load(f)
    if task_id not in annotations:
        sys.exit(f"Annotation not found for {task_id}")

    solver_path = _solver_path(task_id)
    eval_path = _eval_path(task_id)
    arc_utils_path = _arc_utils_path()

    return template.format(
        task_id=task_id,
        task_module=task_id,
        spec_yaml=spec_yaml,
        solver_path=solver_path,
        eval_path=eval_path,
        arc_utils_path=arc_utils_path,
    )


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        sys.exit(
            "Usage:\n"
            "  python generator_prompt_builder.py <task_id> --prepare\n"
            "  python generator_prompt_builder.py <task_id>   (build prompt only)"
        )

    task_id = sys.argv[1]

    if "--prepare" in sys.argv[2:]:
        prepare(task_id)
    else:
        prompt = build_prompt(task_id)
        out_path = ROOT / "tmp" / f"generator_prompt_{task_id}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(prompt)
        print(f"Wrote {len(prompt)} chars to {out_path}")


if __name__ == "__main__":
    main()

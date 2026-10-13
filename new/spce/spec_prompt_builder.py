#!/usr/bin/env python3
"""Build a spec-extraction prompt for a given ARC-AGI-2 task.

Usage:
    python spec_prompt_builder.py <task_id>

Reads:
    - spce/spec_prompt.md                         (template)
    - arc-agi-2-eval-annotations-merge.json       (annotation)

Writes:
    - tmp/spec_prompt_{task_id}.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent  # spce/
ROOT = BASE.parent  # project root


def _prepare_solver(task_id: str) -> None:
    """Archive any generate_input code from solver file, then clean it."""
    from generator_prompt_builder import (
        _solver_path, _find_main_end, _gen_file, _next_version, _gen_file_header,
    )
    solver = _solver_path(task_id)
    if not solver.exists():
        return
    solver_lines = solver.read_text().splitlines(keepends=True)
    main_end = _find_main_end(solver_lines)
    new_code = "".join(solver_lines[main_end + 1:]).strip()
    if not new_code:
        print(f"Solver {solver} is already clean")
        return

    # Archive generator code
    gen_file = _gen_file(task_id)
    gen_file.parent.mkdir(parents=True, exist_ok=True)
    v = _next_version(gen_file)
    label = f"v{v}_subagent"
    renamed = new_code.replace("def generate_input()", f"def generate_input_{label}()", 1)
    if not gen_file.exists():
        gen_file.write_text(_gen_file_header(task_id) + "\n\n")
    with open(gen_file, "a") as f:
        f.write(f"\n\n# === {label} ===\n\n{renamed}\n")
    print(f"Archived generator code → {gen_file} as generate_input_{label}()")

    # Clean solver
    clean = "".join(solver_lines[:main_end + 1])
    if not clean.endswith("\n"):
        clean += "\n"
    solver.write_text(clean)
    print(f"Cleaned {solver} → {main_end + 1} lines")


def build_prompt(task_id: str) -> str:
    from generator_prompt_builder import _eval_path, _solver_path

    template = (BASE / "spec_prompt.md").read_text()

    ann_path = ROOT / "arc-agi-2-eval-annotations-merge.json"
    with open(ann_path) as f:
        annotations = json.load(f)
    if task_id not in annotations:
        sys.exit(f"Annotation not found for {task_id}")
    annotation_json = json.dumps(annotations[task_id], indent=2)

    solver_path = str(_solver_path(task_id))
    eval_path = str(_eval_path(task_id))

    prompt = template.format(
        task_id=task_id,
        solver_path=solver_path,
        eval_path=eval_path,
        annotation_json=annotation_json,
    )
    return prompt


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python spec_prompt_builder.py <task_id>")
    task_id = sys.argv[1]

    _prepare_solver(task_id)
    prompt = build_prompt(task_id)

    out_path = ROOT / "tmp" / f"spec_prompt_{task_id}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(prompt)
    print(f"Wrote {len(prompt)} chars to {out_path}")


if __name__ == "__main__":
    main()

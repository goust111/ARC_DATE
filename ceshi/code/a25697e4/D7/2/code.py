# Variant of a25697e4
# Dimension: D7 — 变换规则：对接后工具处理 (transform)
# Change: 工具从原始位置移除（恢复为背景色），仅在对接位置显示；输出 = 框架 + 对接后的工具... → 仅保留plug，handle消失
# Design analysis:
# - The visible rule change is shifted from "工具从原始位置移除（恢复为背景色），仅在对接位置显示；输出 = 框架 + 对接后的工具" to "仅保留plug，handle消失".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 对接后只有plug部分显示在凹槽中，handle部分完全消失（变为背景色）。输出只显示框架和嵌入的plug色块。简化输出但解题者仍需理解完整的匹配过程。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D7_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D7_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

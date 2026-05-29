# Variant of a25697e4
# Dimension: D16 — 符号编码：变换指示 (symbol_instructions)
# Change: 无显式符号编码，变换（旋转/翻转）完全由形状匹配隐式确定... → 工具内部有方向标记
# Design analysis:
# - The visible rule change is shifted from "无显式符号编码，变换（旋转/翻转）完全由形状匹配隐式确定" to "工具内部有方向标记".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 工具内含一个特殊颜色的单像素标记（如红色），标记在plug中的相对位置指示对接后该像素应位于凹槽的哪个角落，从而唯一确定旋转/翻转。解题者需从标记推断完整变换。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D16_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D16_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

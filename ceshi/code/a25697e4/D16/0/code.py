# Variant of a25697e4
# Dimension: D16 — 符号编码：变换指示 (symbol_instructions)
# Change: 无显式符号编码，变换（旋转/翻转）完全由形状匹配隐式确定... → 彩色标记像素指示旋转方向
# Design analysis:
# - The visible rule change is shifted from "无显式符号编码，变换（旋转/翻转）完全由形状匹配隐式确定" to "彩色标记像素指示旋转方向".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 在框架附近放置1-3个特殊颜色的标记像素，标记数量指示工具需旋转90°的次数（1个=90°, 2个=180°, 3个=270°）。解题者需先解码标记再执行变换。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D16_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D16_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

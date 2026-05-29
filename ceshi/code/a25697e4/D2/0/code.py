# Variant of a25697e4
# Dimension: D2 — 匹配条件：旋转/翻转允许范围 (match_cond)
# Change: 允许完整D4群变换（4种旋转 × 2种翻转 = 最多8种朝向）来对齐工具plug与凹槽... → 仅允许旋转，禁止翻转
# Design analysis:
# - The visible rule change is shifted from "允许完整D4群变换（4种旋转 × 2种翻转 = 最多8种朝向）来对齐工具plug与凹槽" to "仅允许旋转，禁止翻转".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 工具只能通过0°/90°/180°/270°旋转来匹配凹槽，不允许翻转。这意味着凹槽设计必须保证存在某个旋转角度可精确匹配，且颜色交换规则永远不触发。解题者需判断手性方向，推理难度因排除翻转而略降。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D2_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D2_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

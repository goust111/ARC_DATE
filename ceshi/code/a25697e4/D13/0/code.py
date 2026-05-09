# Variant of a25697e4
# Dimension: D13 — 配对判定：凹槽面积与工具面积的关系 (match_cond)
# Change: 工具的某一色区域像素数恰好等于框架凹槽面积，以此确定配对和plug/handle角色... → handle面积匹配凹槽
# Design analysis:
# - The visible rule change is shifted from "工具的某一色区域像素数恰好等于框架凹槽面积，以此确定配对和plug/handle角色" to "handle面积匹配凹槽".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 反转当前规则——handle（而非plug）的面积等于凹槽面积，handle嵌入凹槽而plug突出。视觉效果反转，'填入'和'突出'的颜色角色互换。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D13_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D13_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

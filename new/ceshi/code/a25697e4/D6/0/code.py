# Variant of a25697e4
# Dimension: D6 — 变换规则：颜色交换条件 (transform)
# Change: 当对接变换使用了翻转（reflection）且plug与handle的像素数不等时，交换工具的两种颜色... → 始终交换颜色
# Design analysis:
# - The visible rule change is shifted from "当对接变换使用了翻转（reflection）且plug与handle的像素数不等时，交换工具的两种颜色" to "始终交换颜色".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 无论是否使用翻转，对接后工具的两种颜色始终互换。规则简化为无条件交换。解题者不需要判断翻转与否，但需始终记得交换，推理模式从条件判断变为固定操作。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D6_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D6_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

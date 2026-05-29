# Variant of a25697e4
# Dimension: D10 — 网格输出模式 (grid_shape)
# Change: 输出网格尺寸与输入完全相同（18-30行 × 25-30列），仅内容变化... → 紧凑裁剪输出
# Design analysis:
# - The visible rule change is shifted from "输出网格尺寸与输入完全相同（18-30行 × 25-30列），仅内容变化" to "紧凑裁剪输出".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 输出仅保留对接后的框架+工具区域，裁剪掉所有多余背景。每个框架对接结果单独裁剪输出或整体紧凑裁剪。输出网格显著小于输入。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D10_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D10_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

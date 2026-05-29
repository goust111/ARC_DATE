# Variant of a25697e4
# Dimension: D7 — 变换规则：对接后工具处理 (transform)
# Change: 工具从原始位置移除（恢复为背景色），仅在对接位置显示；输出 = 框架 + 对接后的工具... → 原位留下轨迹
# Design analysis:
# - The visible rule change is shifted from "工具从原始位置移除（恢复为背景色），仅在对接位置显示；输出 = 框架 + 对接后的工具" to "原位留下轨迹".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 工具从原始位置移除后，原位留下单色轨迹（如灰色或第三种颜色的影子），指示工具曾经的位置。增加视觉信息，输出更复杂。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D7_1", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D7_1"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

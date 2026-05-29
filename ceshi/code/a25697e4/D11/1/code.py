# Variant of a25697e4
# Dimension: D11 — 多步序列：对接顺序 (multi_step)
# Change: 各工具独立对接，顺序不影响结果（每对独立处理）... → 链式对接
# Design analysis:
# - The visible rule change is shifted from "各工具独立对接，顺序不影响结果（每对独立处理）" to "链式对接".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 第一个工具对接到框架后，其handle部分形成新的凹槽供第二个工具对接，形成嵌套结构。类似16b78196中的链式放置。解题者需识别迭代对接的终止条件。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D11_1", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D11_1"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

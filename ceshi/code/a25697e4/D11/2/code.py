# Variant of a25697e4
# Dimension: D11 — 多步序列：对接顺序 (multi_step)
# Change: 各工具独立对接，顺序不影响结果（每对独立处理）... → 先对接再移动
# Design analysis:
# - The visible rule change is shifted from "各工具独立对接，顺序不影响结果（每对独立处理）" to "先对接再移动".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 所有工具先对接到各自框架，然后整体执行第二步操作（如所有对接后的框架整体向某方向平移或居中）。两步操作解题者需串联理解。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D11_2", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D11_2"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

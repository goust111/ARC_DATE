# Variant of a25697e4
# Dimension: D15 — 工具放置距离约束 (meta)
# Change: 工具放置在远离目标框架的位置（Manhattan距离>=4），不接触任何边缘... → 工具放置在对面边缘
# Design analysis:
# - The visible rule change is shifted from "工具放置在远离目标框架的位置（Manhattan距离>=4），不接触任何边缘" to "工具放置在对面边缘".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 工具必须放置在框架所在边缘的对面边缘上（如框架在左边则工具在右边边缘）。工具本身接触边缘，解题者需区分框架组件和工具组件。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D15_1", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D15_1"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

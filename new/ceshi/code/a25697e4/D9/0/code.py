# Variant of a25697e4
# Dimension: D9 — 基数关系：工具-框架配对 (meta)
# Change: 1:1配对，每个工具恰好对应一个框架，1-2对... → 多工具对一框架（many:1）
# Design analysis:
# - The visible rule change is shifted from "1:1配对，每个工具恰好对应一个框架，1-2对" to "多工具对一框架（many:1）".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 一个大框架有多个凹槽（2-3个），需要多个不同工具分别填入不同凹槽。解题者需为每个凹槽找到对应工具并分别对接。增加组合推理复杂度。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D9_0", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D9_0"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

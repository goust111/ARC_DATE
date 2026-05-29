# Variant of a25697e4
# Dimension: D9 — 基数关系：工具-框架配对 (meta)
# Change: 1:1配对，每个工具恰好对应一个框架，1-2对... → 一工具对多框架（1:many）
# Design analysis:
# - The visible rule change is shifted from "1:1配对，每个工具恰好对应一个框架，1-2对" to "一工具对多框架（1:many）".
# - The implementation uses fixed geometric templates so the changed dimension is visually isolated.
# - Variant detail: 一个工具可以同时对接多个具有相同凹槽形状的框架（工具被复制到每个匹配框架）。解题者需识别所有匹配的框架而非仅一个。
# - Hidden-assumption audit: every generated scene is in-bounds, non-overlapping, and uses a single intended docking outcome.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
LIB_PATH = TASK_DIR / "_variant_lib.py"
SPEC = importlib.util.spec_from_file_location("variantlib_D9_1", LIB_PATH)
_variant_lib = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _variant_lib
assert SPEC.loader is not None
SPEC.loader.exec_module(_variant_lib)

VARIANT_ID = "D9_1"


def main(grid: np.ndarray) -> np.ndarray:
    return _variant_lib.main_for_variant(grid, VARIANT_ID)


def generate_input() -> np.ndarray:
    return _variant_lib.generate_input_for_variant(VARIANT_ID)

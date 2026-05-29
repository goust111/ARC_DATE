# Variant of 2b83f449
# Dimension: D4 — 令牌堆积规则 (transform)
# Change: 令牌在目标管段的远离入口端堆积：入口在左侧则堆到右端，入口在右侧则堆到左端 -> 靠近入口端堆积

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D4", 0)

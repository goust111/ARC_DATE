# Variant of 2b83f449
# Dimension: D5 — 令牌来源位置 (src)
# Change: 令牌位于每个非末行管道行的左右两端(col 0 和 col W-1)，每行恰好2个令牌 -> 仅单侧有令牌

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).resolve().parents[2] / "_variant_lib.py"
_SPEC = importlib.util.spec_from_file_location("_variant_lib_2b83f449", _LIB_PATH)
_LIB = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_LIB)
main, generate_input = _LIB.build_functions("D5", 0)

# Variant of 71e489b6
# Dimension: D11 — 多步处理序列 (multi_step)
# Change: 单次检测 -> 迭代检测与标记
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D11/0")

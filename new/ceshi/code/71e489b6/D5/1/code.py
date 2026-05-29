# Variant of 71e489b6
# Dimension: D5 — 光晕颜色规则 (transform)
# Change: 单一橙色 -> 按缺陷大小分色
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D5/1")

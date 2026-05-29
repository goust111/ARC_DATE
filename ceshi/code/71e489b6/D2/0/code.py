# Variant of 71e489b6
# Dimension: D2 — 缺陷检测条件 (src)
# Change: 小缺陷<=3 -> 仅检测单像素缺陷
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D2/0")

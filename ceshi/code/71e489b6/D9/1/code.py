# Variant of 71e489b6
# Dimension: D9 — 输入中的干扰项类型 (src)
# Change: 单一蓝毛刺 -> 多色干扰毛刺
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D9/1")

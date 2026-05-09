# Variant of 65b59efc
# Dimension: D10 — 分隔线：分隔符的样式和结构 (meta)
# Change: 虚线分隔 -> 实线分隔
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D10/0")

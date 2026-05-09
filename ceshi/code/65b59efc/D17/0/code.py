# Variant of 65b59efc
# Dimension: D17 — 背景色与分隔色的选择 (meta)
# Change: 固定背景/分隔色 -> 每例变化
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D17/0")

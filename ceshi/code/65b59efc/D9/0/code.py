# Variant of 65b59efc
# Dimension: D9 — 标记放置：tile列分配规则 (meta)
# Change: 同列统一色 -> 同列可混合多色
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D9/0")

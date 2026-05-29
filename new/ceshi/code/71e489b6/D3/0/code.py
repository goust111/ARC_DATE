# Variant of 71e489b6
# Dimension: D3 — 毛刺吸收机制 (transform)
# Change: 毛刺吸收 -> 毛刺保留并标记
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D3/0")

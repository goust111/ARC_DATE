# Variant of 71e489b6
# Dimension: D7 — 光晕的覆盖与叠加规则 (transform)
# Change: 任意覆写 -> 仅覆写蓝色像素
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D7/2")

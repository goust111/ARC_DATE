# Variant of 71e489b6
# Dimension: D13 — 连通性定义 (meta)
# Change: 4连通 -> 8连通分量检测
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D13/0")

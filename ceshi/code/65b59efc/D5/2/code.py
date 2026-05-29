# Variant of 65b59efc
# Dimension: D5 — 模板印章的重绘方式 (transform)
# Change: 正常重绘 -> 颜色交换
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D5/2")

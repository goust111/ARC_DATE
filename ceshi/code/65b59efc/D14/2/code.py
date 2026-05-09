# Variant of 65b59efc
# Dimension: D14 — 数值约束：模板数量K (meta)
# Change: 固定范围K -> K随示例变化
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D14/2")

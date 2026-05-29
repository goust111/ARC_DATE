# Variant of 65b59efc
# Dimension: D12 — 模板印章的几何变换 (transform)
# Change: 固定朝向 -> 按输出行旋转
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D12/0")

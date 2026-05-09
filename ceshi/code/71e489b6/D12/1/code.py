# Variant of 71e489b6
# Dimension: D12 — 光晕与缺陷像素的保留规则 (transform)
# Change: 缺陷中心保黑 -> 中心第三色
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import make_variant

main, generate_input = make_variant("D12/1")

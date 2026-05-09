# Variant of 36a08778
# Dimension: D2 — 架子碰撞后的扩散方式 (transform)
# Change: 双侧溢出... → 穿透架子继续下落（架子有缺口）
#
# 设计要点：
# - 允许“单孔架子”：由两个水平段组成，中间空 1 格。
# - 水流命中这类架子时不再从两端溢出，而是沿架子上方流向那个孔，再从孔位继续单路下落。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _variant_common import build_variant_functions

main, generate_input = build_variant_functions("D2_1")

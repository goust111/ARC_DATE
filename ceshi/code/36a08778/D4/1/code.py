# Variant of 36a08778
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: 连续轨迹（continuous_trail），所有经过的背景像素都涂为水色（品红）... → 颜色随深度变化
#
# 设计要点：
# - 根流保持品红；第一次分流后的子流改用下一种颜色；更深层继续按固定调色板递变。
# - 颜色取决于“分支深度”，不是绝对高度，因此同层不同分支会共享颜色。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _variant_common import build_variant_functions

main, generate_input = build_variant_functions("D4_1")

# Variant of 36a08778
# Dimension: D7 — 终止条件 (dst)
# Change: 到达底边才停止... → 碰到排水口像素停止
#
# 设计要点：
# - 输入中额外加入少量排水口颜色；当水流下一步将进入排水口时，当前分支立即终止。
# - 排水口本身保持原色，以便解题者从截断位置反推出“吸收停止”的规则。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _variant_common import build_variant_functions

main, generate_input = build_variant_functions("D7_0")

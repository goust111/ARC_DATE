# Combo variant of 36a08778 (index 32)
# combo_spec: D7/0+D8/0+D14/3
# Name: 颜色停止+架子溶解+入水方向溢出
#
# 设计分析摘要：
# - 保持原题“顶部水源 + 水平架子 + 瀑布传播”的核心对象语义不变。
# - 具体组合规则由共享引擎按 combo_spec 打开：包括对角线下落、单侧/交替/入水侧溢出、
#   缺口穿透、深度变色、固定扩散、单像素水源、排水口停止、距离/层数限制、架子溶解、
#   网格扩展、多水源排序、孔洞架子和四色系统等。
# - generate_input() 直接为该 combo 构造能触发关键行为的输入，并由 main() 做自校验。

from __future__ import annotations

import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from _combo_common import build_combo_functions

main, generate_input = build_combo_functions("D7/0+D8/0+D14/3")

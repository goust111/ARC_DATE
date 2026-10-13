已将完整实现写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/7.py)。`main()` 按题面做灰线分区、指令区/图案区识别、按颜色与相邻区域配对，并根据 Red 的相对位置执行翻转或旋转后居中替换；`generate_input()` 会随机生成 4 到 6 对对象，保证四种指令分支都出现，且颜色、位置、形状、网格尺寸都有变化。

已完成自检：100 次随机样例验证通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
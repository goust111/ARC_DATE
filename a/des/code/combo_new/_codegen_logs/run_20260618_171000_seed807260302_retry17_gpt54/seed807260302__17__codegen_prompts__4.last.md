已将实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/4.py)。`main()` 会从左半区暴力评估候选两色棋盘相位，提取 4 连通遮挡块，并把被遮挡的棋盘内容按原相对位置印到右半区，再加上 8 邻域外扩一像素的彩色边框；`generate_input()` 会随机化网格尺寸、棋盘相位、颜色、遮挡物数量/尺寸/位置，并用 oracle 校验生成样例。

已按要求完成自检：100 次生成与求解通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
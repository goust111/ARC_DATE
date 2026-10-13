已将实现写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/15/5.py)。`main()` 会通用识别带单像素缺口的灰色矩形房间，读取内部标记线的数量和方向，并按“平行实线 / 垂直虚线”的规则向外发射射线；`generate_input()` 随机化了网格尺寸、房间数量、位置、缺口侧、标记方向、颜色和是否被其他房间截断，避免重叠与退化样例。

已按要求完成自检：100 次 `generate_input()` / `main()` 通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
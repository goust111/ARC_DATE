已将实现写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/11/2.py)。文件包含自包含的 `main(grid)` 和 `generate_input()`：`main()` 用单色 4 连通分量识别目标与 3 像素 L 箭头，并把目标形状按箭头反方向投射到对应边界；`generate_input()` 随机化网格尺寸、对象数量、颜色、方向和不规则连通形状，同时预留输出副本位置，并用 `main(inp) == expected` 反查验收。

自检已通过：100 次采样得到 `variants 100`，且 `py_compile` 成功。验证产生的 `__pycache__` 也已清理。
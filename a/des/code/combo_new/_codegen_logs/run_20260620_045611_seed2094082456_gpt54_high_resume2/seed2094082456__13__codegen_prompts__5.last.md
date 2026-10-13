代码已写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/5.py)。实现是自包含的：`main(grid)` 按题面规则读取顶端图例、按 8 连通分组 `grey` 轮廓、统计像素数并在各自 bbox 内填充被 4 连通封闭的黑色内部区域；`generate_input()` 会随机化网格尺寸、图例顺序、对象个数、轮廓模板、朝向和位置，并保证对象之间不会误连。

题面要求的自检已通过：100 次生成得到 `variants 100`，`py_compile` 通过，且已清理验证产生的 `__pycache__`。
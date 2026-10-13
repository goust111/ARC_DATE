已按要求把代码写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/10/6.py)。`main()` 会先识别角落竖直图例，再把每个多色对象拆成单色层，按层之间的外到内邻接关系恢复深度并按图例重着色；`generate_input()` 会随机化网格尺寸、图例角落、对象数量、层数、中心形状和对象颜色，并通过重试避免重叠与退化样例。

自检已完成并通过：100 次 `generate_input()`/`main()` 验证通过，`variants 100`，`py_compile` 通过，验证产生的 `__pycache__` 也已清理。
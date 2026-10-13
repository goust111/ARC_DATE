已将实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/16/4.py)。`main(grid)` 按每种颜色的可见包围盒恢复原始实心矩形，再按面积升序水平排布；`generate_input()` 会随机生成带层叠遮挡的多矩形样例，并显式过滤退化情况。

已运行你要求的自检与编译检查，结果通过：100 次随机样例全部成功，`variants 100`，并且已清理生成的 `__pycache__`。
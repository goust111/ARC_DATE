已按要求把代码写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/2.py)。`main(grid)` 会提取所有单色实心奇数尺寸矩形，按面积从大到小以最大矩形中心重绘；`generate_input()` 会随机化网格尺寸、矩形数量、长宽比例、颜色和散放位置，并保证对象不接触、面积唯一、输出始终在 `30x30` 以内。

已完成自检：100 次随机样例生成与求解通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
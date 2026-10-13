代码已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/6.py)。实现是自包含的：`main(grid)` 会识别实心矩形与箭头连通块，解析箭头方向和线段长度，再按规则裁剪对应矩形并清除箭头；`generate_input()` 会随机化网格尺寸、对象数量、方向、裁剪分支、箭头长度、位置和颜色，并用求解器反验生成结果。

自检已按要求执行通过：100 次随机样例验证得到 `variants 100`，`python -m py_compile` 也通过；验证产生的 `__pycache__` 已清理。
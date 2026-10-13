代码已写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/11/1.py)。`main(grid)` 按边缘纵向图例读取 1..5 的填充色映射，把 `grey` 作为围栏，查找所有不接触边界的非灰色封闭区域，统计其中 `非背景、非 grey` 的颜色种类数，并整块重绘为对应图例色。`generate_input()` 会随机化网格尺寸、图例左右位置、区域数量、围栏凹口形状和被困颜色种类数，且带重试校验避免退化样例。

自检已通过：100 次生成得到 `variants 100`，`python -m py_compile` 通过，验证产生的 `__pycache__` 已清理。
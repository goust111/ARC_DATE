已将实现写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/7.py)。`main()` 会定位右下角 3xN 色盘、按旋转/翻转等价对对象做形状配对，再按“先横后竖”补出混色连线；`generate_input()` 会随机生成 2 或 3 对对象，随机化网格尺寸、形状、颜色、图例顺序、相对布局，并用 `main()` 做回验筛掉退化样例。

已按要求完成自检：100 次随机样例通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
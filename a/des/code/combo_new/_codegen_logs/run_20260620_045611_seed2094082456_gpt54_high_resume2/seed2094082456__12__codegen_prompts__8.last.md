代码已写入 [8.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/8.py)。`main()` 会先追踪边界起始的 Green 简单路径，只检查当前前进方向上的门户，再按同色配对做链式传送并沿原方向继续延伸；`generate_input()` 会随机化网格尺寸、链长、朝向、起点边界侧、门户位置和未使用门户对，并在内部用 `main()` 回验。

自检已执行并通过：100 次生成/求解循环得到 `variants 100`，`py_compile` 通过，验证产生的 `__pycache__` 已清理。
代码已写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/1.py)。`main()` 会识别 Grey 方框、穷举对齐碎片并取并集恢复原型，再按底部 Red 单像素 2x2 标记解码旋转并输出；`generate_input()` 会随机化对象数量、内图案尺寸、形状、碎片子集、输入朝向、目标朝向和布局，并在生成期用 `main()` 反验，避免退化样例。

自检已按要求执行并通过：100 次随机样例 `variants 100`，尺寸约束通过，`python -m py_compile` 通过，验证产生的 `__pycache__` 已清理。
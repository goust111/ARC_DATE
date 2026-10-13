代码已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/16/6.py)。`main()` 按“同色 4 连通碎片 -> 8 邻接分组 -> 按组大小旋转 -> bbox 中心对齐回贴”的规则实现；`generate_input()` 会随机化网格尺寸、组数、组大小、碎片形状、颜色和位置，并用 `main()` 反验生成结果。

已按要求完成自检：
- 100 次随机样例通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed2094082456/16/6.py` 通过
- 验证产生的 `__pycache__` 已清理
已将实现写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/9/1.py)。

`main(grid)` 会识别所有矩形轮廓，统计各自内部同色标记数量，并用标记群相对矩形中心的主偏移方向决定刺突朝向；`generate_input()` 会随机化网格尺寸、对象数量、位置、框大小、标记颜色、方向和长度，并预留输出保留区，避免对象或刺突冲突。

自检已通过：
- 100 次随机样例生成与求解通过
- `variants 100`
- `python -m py_compile` 通过
- 验证产生的 `__pycache__` 已清理
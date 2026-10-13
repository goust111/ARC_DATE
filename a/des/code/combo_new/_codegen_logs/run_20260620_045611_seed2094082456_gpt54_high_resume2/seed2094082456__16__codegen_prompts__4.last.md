已按要求把完整实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/16/4.py)。

`main()` 是确定性规则实现：识别实心矩形、匹配唯一橙色 V 箭头、按箭头方向只剥离最外一层并把露出的边改成灰色；`generate_input()` 会随机化网格尺寸、矩形数量、颜色、位置、尺寸和方向，并用结构约束避免对象/箭头冲突或越界。

已完成自检：
- 100 次随机生成与求解通过，`variants 100`
- `python -m py_compile` 通过
- 验证产生的 `__pycache__` 已清理
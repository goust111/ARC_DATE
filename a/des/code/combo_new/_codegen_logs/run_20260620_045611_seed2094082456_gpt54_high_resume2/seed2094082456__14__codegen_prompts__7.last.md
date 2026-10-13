已将实现写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/7.py)。`main()` 按分隔线切区、统计每区红色计数器像素数、裁出模板最小 bbox 并做整数缩放；`generate_input()` 会随机化布局、模板形状/朝向、颜色、计数值、单元格尺寸和放置位置，并用求解器回放校验生成结果。

已按你的自检要求执行，结果通过：100 次生成得到 `variants 100`，`py_compile` 通过，并且已清理 `__pycache__`。
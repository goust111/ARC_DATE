已将完整实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/9/4.py:1)。`main()` 按矩形边框对象解析内部种子、根据种子相对中心的偏置反向投影，并按种子像素数做整数放大；`generate_input()` 随机化了网格尺寸、对象数量、颜色、方向、种子形状与对象尺寸，并在放置时预留阴影区域来避免重叠和歧义。

已执行你要求的自检：
- 100 次随机样例检查通过，`variants 100`
- `python -m py_compile` 通过
- 已清理验证生成的 `__pycache__`
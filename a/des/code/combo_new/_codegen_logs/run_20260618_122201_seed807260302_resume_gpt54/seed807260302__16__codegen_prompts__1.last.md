已将实现写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/16/1.py)。

`main()` 会检测底部调色条、识别封闭矩形边框、统计内部非黑像素数并按第 `N` 个调色条颜色重绘边框，同时清空内部色块。`generate_input()` 会随机化网格尺寸、边框数量与尺寸、内部计数、位置和调色条长度，并通过重试避免重叠与歧义。

已按要求完成自检：
- 100 次 `generate_input()` / `main()` 验证通过
- `variants 100`
- `python -m py_compile` 通过
- 已清理 `__pycache__`
已实现并写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/16/1.py)。

`main(grid)` 通过检测矩形环、读取内部唯一标记、先做内部填充，再按标记最近边方向平移整段围栏轮廓生成仅边框阴影。`generate_input()` 会随机化网格尺寸、对象数量、矩形大小、位置、标记颜色、四个方向，并通过预留原框和可见阴影区域来避免重叠和退化。

已按要求完成自检：
- 100 次 `generate_input()` / `main()` 运行通过
- `variants 100`
- `python -m py_compile` 通过
- 已清理验证产生的 `__pycache__`
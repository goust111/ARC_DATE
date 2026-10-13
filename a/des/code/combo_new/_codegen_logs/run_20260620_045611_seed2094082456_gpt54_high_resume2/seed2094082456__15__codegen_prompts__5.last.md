代码已写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/5.py)。

实现内容是自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`：`main()` 会解析左侧字典中的 Maroon 转向符号，提取右侧 Azure 路径从远端到 Orange 终点的 `L/R` 拐弯序列，并把匹配到的 3x3 或 4x4 图案贴到 Orange 方块左上角锚点；`generate_input()` 会随机化字典项数量、路径数量、转向分支、图案样式、网格尺寸和放置位置，并用 oracle 自验筛掉退化样例。

已按要求完成验证：
- `100` 次自检通过
- `variants 100`
- `python -m py_compile` 通过
- 已清理验证产生的 `__pycache__`
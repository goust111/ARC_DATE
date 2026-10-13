已按要求把完整实现写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/16/2.py)。

实现包含 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`。`main()` 按颜色识别唯一的同色 `L` 形组件，定位拐点和原始水平/垂直方向，再用同色孤立像素在拐点同行/同列的计数来重绘两条射线；`generate_input()` 随机化了网格大小、对象数量、颜色、方向、原始臂长和三种计数分支，并在生成时独立构造期望输出做反校验。

已执行并通过你给的自检要求：
- `100` 次随机样例验证通过，`variants 100`
- `python -m py_compile` 通过
- 验证产生的 `__pycache__` 已清理
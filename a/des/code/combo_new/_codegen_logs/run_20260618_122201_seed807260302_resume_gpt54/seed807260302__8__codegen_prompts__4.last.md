已将完整实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/4.py)。文件内包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，`main()` 带了按要求分四部分组织的 docstring；生成器会随机覆盖箭头数量、颜色、方向、是否传送、以及边界停止/灰墙停止两类分支，并用几何约束避免重叠和歧义。

已按要求完成自检：100 次随机生成与求解全部通过，`variants 100`；`python -m py_compile` 也通过；验证产生的 `__pycache__` 已清理。
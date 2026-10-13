实现已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/6.py)。

代码是自包含的，包含 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`。`main()` 通过枚举每种颜色对应的十字模板中心、根据遮挡关系建立层序图并做唯一拓扑排序来恢复从底到顶的顺序；`generate_input()` 随机化网格尺寸、十字数量、位置、颜色和重叠方式，并只接受能被通用求解器唯一还原的样例。

已按要求完成验证：
- 100 次自检通过，`variants 100`
- `python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/6.py"` 通过
- 验证产生的 `__pycache__` 已清理
代码已写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed168776016/17/4.py)。文件包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，实现了按同色锚点配对、由相邻灰标记决定 L 形走向，并在生成器里约束了非重叠、非歧义和分支随机化。

自检结果：`python` 100 次随机样例验证通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
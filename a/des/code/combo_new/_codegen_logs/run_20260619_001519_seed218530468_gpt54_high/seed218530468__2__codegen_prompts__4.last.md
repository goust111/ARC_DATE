目标文件写入到 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed218530468/2/4.py)。其中包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，生成器会随机混合矩形环与带凹口的非矩形环，并保证珠子、轨道、内部 maroon 标记的约束成立。

自检结果：按你给的 100 次脚本验证通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
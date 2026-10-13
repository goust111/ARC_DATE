已将实现写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed402918419/17/5.py)。文件包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，按题意实现了基于顶边/左边刻度尺窗口的整对象筛选，并在生成器里构造了保留对象、跨界对象和窗口外对象。

自检已通过：100 次随机样例验证通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
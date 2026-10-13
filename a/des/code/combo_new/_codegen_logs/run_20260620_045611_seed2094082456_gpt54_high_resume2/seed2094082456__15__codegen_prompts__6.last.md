已将完整实现写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/6.py)。文件包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`；`main()` 用回溯恢复被遮挡矩形并按面积重排，`generate_input()` 会随机生成 3 到 4 个重叠矩形样例，并在生成期用求解器做一致性校验。

已按 prompt 跑完自检：`100` 次生成/求解通过，`variants 100`，`python -m py_compile` 通过，验证产生的 `__pycache__` 也已清理。
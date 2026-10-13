已将实现写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/16/3.py)。文件包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`：`main()` 会识别整行整列分隔线、切分等尺寸分区、提取左上种子，并按左右对称 / 上下对称 / 无轴对称三类选择横向铺砌、纵向铺砌或双向铺砌；`generate_input()` 会随机生成 2x3 或 3x2 分区、平衡三种分支、避免 bbox 缩退和零扩展，并用 `main()` 回验。

已按要求完成自检与清理：100 次采样通过，`variants 100`，`py_compile` 通过，验证产生的 `__pycache__` 已删除。
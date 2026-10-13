已将实现写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/2.py)。代码是自包含的，包含 `main(grid: np.ndarray) -> np.ndarray`、`generate_input() -> np.ndarray`，并按题意实现了“颜色匹配锚点 + 枚举碎片内对齐像素 + 唯一合法平移”的求解逻辑；生成器则先构造完整模板，再只保留锚点并把碎片散放到框外，同时验证唯一性和可解性。

已完成自检：100 次随机样例全部通过，`variants 100`；`python -m py_compile` 也通过；验证产生的 `__pycache__` 已清理。
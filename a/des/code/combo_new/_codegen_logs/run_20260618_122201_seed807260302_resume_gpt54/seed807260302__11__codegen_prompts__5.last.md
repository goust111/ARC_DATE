代码已写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/11/5.py)。

实现包含 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`。`main()` 按题面做矩形框检测、唯一种子定位、按最近内壁距离填充同色实心方块；`generate_input()` 随机化网格尺寸、框数量、框尺寸、位置、种子颜色和种子半径层级，并在返回前调用 `main()` 做结构自验。

自检已完成：
- 100 次随机样例验证通过，`variants 100`
- `python -m py_compile` 通过
- 验证产生的 `__pycache__` 已清理
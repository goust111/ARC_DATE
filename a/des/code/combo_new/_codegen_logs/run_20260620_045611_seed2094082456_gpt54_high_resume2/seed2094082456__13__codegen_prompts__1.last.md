代码已写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/1.py)。

实现内容包括 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，`main()` 带了按要求的四段式 docstring。生成器会随机化网格尺寸、图例段数量、对象数量、位置、颜色、孔洞方向和外轮廓，并用 `main()` 做回验避免退化样例。

自检已完成并通过：
- 100 次生成/求解校验通过
- 变体数 `variants 100`
- `python -m py_compile` 通过
- 验证产生的 `__pycache__` 已清理
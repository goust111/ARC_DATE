代码已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/6.py)。

实现包含 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，求解器按题意做了箭头识别、方向配对、半区裁剪和叠加；生成器会随机化方向组合、印章尺寸、图案细节、箭头锚点和整体布局，并在生成时用 `main()` 反验样例有效性。

自检已通过：
- 100 次随机样例检查通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/6.py` 通过
- 已清理验证产生的 `__pycache__`
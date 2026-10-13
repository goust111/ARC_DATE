代码已写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/15/7.py)。实现是自包含的，包含 `main(grid)` 和 `generate_input()`；`main()` 会识别 1-3 层同色嵌套空心矩形框架，提取最内层图案，并按层数做逐级 `2x2` 缩小后复制到层间左上角和最外层右下外侧，且只覆盖 `black`，不会破坏框架。

自检已完成并通过：
- 100 次随机样例验证通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed807260302/15/7.py` 通过
- 验证产生的 `__pycache__` 已清理
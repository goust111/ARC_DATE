已将实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/4.py)。

`main()` 会先把同色矩形核心与单像素桥臂拆开，再按桥的颜色匹配端推断源/目标，并按桥长把目标矩形靠桥一侧的行或列重绘。`generate_input()` 随机生成 1 到 3 组桥接对象，覆盖水平/垂直、双向传递、整块覆盖/部分覆盖，并用独立 oracle 校验生成样例和 `main()` 一致。

已执行并通过：
- 100 次自检：`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/4.py`

验证产生的 `__pycache__` 已清理。
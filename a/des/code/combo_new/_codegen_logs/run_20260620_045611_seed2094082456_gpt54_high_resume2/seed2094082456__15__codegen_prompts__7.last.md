已将实现写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/7.py:1)。`main()` 按封闭区域规则读取每个 `Magenta` 指针的穿墙颜色源并填充区域；`generate_input()` 随机化了网格尺寸、房间数量、主干朝向、房间尺寸、左右/上下分布、指针方向、颜色和额外灰色枝杈，并在生成阶段用构造出的期望输出加 `main()` 回验。

自检已通过：100 次采样得到 `variants 100`，`python -m py_compile` 也通过，验证产生的 `__pycache__` 已清理。
已将实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/13/4.py)。`main()` 按顶行/左列刻度解析同色坐标、生成节点并按首个顶行出现顺序覆盖绘制 L 形折线；`generate_input()` 随机化了网格尺寸、颜色数、每色节点数，以及有/无重叠两类样例，并用重试避免退化输入。

验证已执行并通过：100 次随机自检得到 `variants 100`，`py_compile` 成功，目标目录下的 `__pycache__` 也已清理。
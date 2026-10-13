代码已写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/16/7.py:1)。实现是自包含的：`main()` 会识别边缘颜色条、提取每条由 `blue/red/grey` 构成的单路径、从蓝端点按路径顺序遍历，并在遇到红色关卡时切换到颜色条中的下一个颜色；`generate_input()` 会随机化网格尺寸、颜色序列长度、路径数量、路径形状、关卡数量和颜色条上下边位置，并用 `main()` 反校验生成结果。

按要求完成了自检：100 次样例生成得到 `variants 100`，`py_compile` 通过，之后已清理验证产生的 `__pycache__`。
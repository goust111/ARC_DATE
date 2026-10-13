代码已写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/7.py)。`main()` 按题意忽略底部色带、用上方每种非黑非灰颜色的可见像素恢复矩形 bbox，按面积升序重着色并清空底行；`generate_input()` 随机生成 2 到 4 个面积互异、位置不重叠、四角保留可见的带灰噪声矩形，并用显式期望输出校验生成结果。

自检已完成并通过：100 次随机样例 `variants 100`，`python -m py_compile` 通过，且已清理本题生成的 `__pycache__` 条目。
代码已写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/11/4.py)。`main(grid)` 会识别整行/整列镜线、提取非线条小连通块，并按最近镜线做轴对称复制；`generate_input()` 会随机采样横线版和竖线版、不同网格尺寸、线数、对象数、形状和位置，并在生成阶段过滤掉等距、越界、粘连和重叠样例。

已按要求完成自检：100 次随机样例验证通过，`variants 100`；`python -m py_compile` 通过；`__pycache__` 已清理。
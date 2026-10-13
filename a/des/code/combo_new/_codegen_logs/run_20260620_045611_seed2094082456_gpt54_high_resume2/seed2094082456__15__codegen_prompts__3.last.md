已将实现写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/3.py)。`main(grid)` 按水平条纹切分区域并执行底部堆叠重排；`generate_input()` 随机化了条纹数量、颜色顺序、区域高度、网格宽度、各区颜色层数和散点分布，并显式约束了同色同列不重复、区域高度足够容纳颜色层。

已执行你要求的自检命令。结果是 `variants 100`，通过了 100 次随机生成与求解检查，也通过了 `python -m py_compile`，并清理了验证产生的 `__pycache__`。
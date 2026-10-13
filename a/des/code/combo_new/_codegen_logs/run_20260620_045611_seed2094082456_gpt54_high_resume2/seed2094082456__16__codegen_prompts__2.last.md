已实现并写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/16/2.py)。`main(grid)` 按最大模板组件、内部矩形缺口、外部矩形碎片、尺寸匹配与行/列对齐来完成填洞并清除原碎片；`generate_input()` 随机化网格尺寸、模板尺寸、缺口数量/尺寸、碎片颜色与水平/垂直分支，并用 `main()` 做 oracle 自验。

自检已通过：100 次随机生成得到 `variants 100`，`py_compile` 通过，`__pycache__` 已清理。
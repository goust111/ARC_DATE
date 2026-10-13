已将实现写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/15/2.py)。`main()` 按对象颜色组件识别带孔对象，统计封闭孔洞数，并用孔内标记色对对象外侧背景做对应层数的 4 邻域扩张；`generate_input()` 用重试采样生成不同网格尺寸、对象数量、孔洞数量、孔洞布局和外轮廓 protrusion 的样例，并混入无孔对象。

已执行用户要求的自检与编译检查，`100` 次生成得到 `variants 100`，`py_compile` 通过，验证后已清理目标目录下的 `__pycache__`。
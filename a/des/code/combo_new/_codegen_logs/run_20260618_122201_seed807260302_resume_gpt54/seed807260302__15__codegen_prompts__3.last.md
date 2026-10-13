已将实现写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/15/3.py)。

`main()` 按颜色提取两个 4 连通碎片，枚举相对平移，找出能无重叠补成实心轴对齐矩形的唯一尺寸，再按面积升序把填满的矩形底对齐排到输出。`generate_input()` 随机采样矩形数量、尺寸、颜色和纵/横向不规则切分轮廓，把两块互补碎片打散放置，并用 `main()` 反验只保留合法样例。

自检已完成：
```text
variants 100
```
`python -m py_compile` 通过，验证产生的 `__pycache__` 已清理。
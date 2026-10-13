已将实现写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/9/3.py)。`main()` 按“去掉阀门后的灰色 4 连通段”做分段，并根据相邻阀门颜色集合决定是否重着色；`generate_input()` 会随机生成 1 到 3 条不相交的自避路径，随机化网格尺寸、路径形状、阀门数量和颜色，同时强制每个样例都包含单端段、同色双端段、异色双端段三类情况。

已完成验证：
- 题目要求的 100 次自检通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed807260302/9/3.py` 通过
- 额外做了 50 次结构检查
- 已清理验证产生的 `__pycache__`
已将实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/4.py)。`main()` 按红墙分割出的 4 连通区域建邻接图，再把每个区域的邻居数映射成从墙向内的分层填充深度；`generate_input()` 用随机连通粗网格分区渲染成闭合红墙网络，保证每个区域恰好一个唯一样色种子，并同时出现“部分填充”和“空间不足被填满”的情况。

自检已完成并通过：`SourceFileLoader` 100 轮生成/求解通过，得到 `variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
已将实现写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/5.py)。`main()` 按可见原色的 bbox 恢复矩形色块，统计块内 orange 种子数后重绘同心环；`generate_input()` 用递归矩形切分生成无背景、紧密拼接的矩形棋盘，并随机覆盖 `N=1..4`、不同块数/尺寸以及“小块省略内层”的情况。

验证已完成：
- 题目要求的 100 次自检通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/5.py` 通过
- 额外抽样 200 次检查到 `N=1..4` 都在出现，且截断分支有稳定覆盖
- 已清理目标目录下的 `__pycache__`
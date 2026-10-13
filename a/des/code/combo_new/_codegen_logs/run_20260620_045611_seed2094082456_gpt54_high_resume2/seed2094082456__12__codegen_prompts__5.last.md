已将实现写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/5.py)。

`main()` 会从四个角的 3x3 图例按颜色提取模板，并处理“中心有色”和“中心为黑时改用最近锚点”两种对齐方式；`generate_input()` 随机化网格尺寸、角落模板、颜色和种子布局，并用重试过滤退化/冲突样例，保证四种颜色都实际出现且两类锚点分支持续覆盖。

已完成你要求的验证：
- 100 次 `generate_input()` / `main()` 自检通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed2094082456/12/5.py` 通过
- 验证产生的 `__pycache__` 已清理
已按要求实现并写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/15/6.py)。

实现内容是单文件自包含的 `main(grid)` 和 `generate_input()`：`main()` 按“同色成对端点 + 路径上是否有黄色干扰标记”决定画红线或绿线；`generate_input()` 随机化网格尺寸、对数、颜色、方向、线长和额外干扰点，并用重试避免线段交叉、对象粘连和歧义输入，同时保证红/绿两个分支都会出现。

已执行自检：
- 100 次随机生成与求解通过
- `variants 100`
- `python -m py_compile` 通过
- 已清理验证产生的 `__pycache__`
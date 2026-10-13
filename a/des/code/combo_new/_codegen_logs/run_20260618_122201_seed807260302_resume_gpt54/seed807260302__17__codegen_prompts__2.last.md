实现已写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/2.py)。

`main()` 会按颜色找两块碎片、恢复完整矩形，并根据 `R > C` 或 `R <= C` 分别做按列橙色交替或按行天蓝色交替；`generate_input()` 随机生成 2-3 对同色碎片，保证平移拼合唯一、分支覆盖、位置不重叠，并保留右下角图例。

已按要求完成自检：
- `100` 次生成/求解通过，`variants 100`
- `python -m py_compile` 通过
- 验证产生的 `__pycache__` 已清理
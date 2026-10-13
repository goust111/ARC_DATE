已将实现写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/12/3.py)。

`main()` 现在会识别角落灰框模板、按像素对间距排序并映射模板颜色，再按间距长度向外绘制射线；`generate_input()` 会随机化网格尺寸、模板角与方向、像素对数量、方向、间距和位置，并用回验避免退化样例。

已执行题目要求的自检：
- 100 次随机生成与求解通过，`variants 100`
- `python -m py_compile` 通过
- 验证产生的 `__pycache__` 已清理
代码已写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/3.py)。

`main()` 不是按固定模板识别，而是把整张图的非黑前景解析成一棵树，找出唯一合法的“source -> maroon”主路径，把其余彩色叶子当作门，并沿路径模拟颜色覆盖传播。`generate_input()` 会随机化网格尺寸、路径长度与转折、门数量与位置、源色和门色，并用 `main()` 反验样例合法性。

已执行自检：
- `100` 次随机样例检查通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/3.py` 通过
- 已清理验证产生的 `__pycache__`
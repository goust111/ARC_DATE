已将实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/17/4.py)。

`main()` 会解析左列和顶行的颜色段及其连续长度，并在黑色画布上从外到内绘制嵌套矩形环；`generate_input()` 会随机化环数、颜色、横纵厚度和中心剩余尺寸，并用 `main()` 做生成期反验，保证样例有效且不越界。

验证已完成：
- 100 次随机自检通过，`variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed2094082456/17/4.py` 通过
- 已清理验证产生的 `__pycache__`
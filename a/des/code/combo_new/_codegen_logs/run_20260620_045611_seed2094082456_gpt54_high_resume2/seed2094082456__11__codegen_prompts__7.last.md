代码已写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/11/7.py)。`main()` 会识别单像素厚矩形房间、读取内部唯一种子、用房间内部面积开方得到边长，并用边框颜色在种子位置重绘实心正方形；`generate_input()` 随机化了网格尺寸、房间数量、边长、房间长宽因子分解、种子位置和颜色分配，并在生成后用 `main()` 回验。

按你给的自检命令已通过：100 次采样得到 `variants 100`，`python -m py_compile` 也通过，验证产生的 `__pycache__` 已清理。
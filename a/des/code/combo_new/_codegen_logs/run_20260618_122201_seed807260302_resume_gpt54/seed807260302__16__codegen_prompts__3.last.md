实现已写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/16/3.py)。`main(grid)` 会按提示列读取顺序、按颜色合并同框碎片并识别缺口端口，再重建并裁剪链式输出；`generate_input()` 会随机化框数、颜色、尺寸、方向、提示角落和散布位置，并用 `main()` 反验生成结果。

自检已执行并通过：100 次随机样例验证得到 `variants 100`，`python -m py_compile` 通过，验证产生的 `__pycache__` 已清理。
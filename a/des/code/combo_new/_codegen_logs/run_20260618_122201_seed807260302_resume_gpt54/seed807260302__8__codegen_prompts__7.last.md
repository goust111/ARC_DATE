代码已写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/7.py)。`main()` 会按三边矩形容器的开口方向识别局部“重力”，把容器内部同色散点重新压到底边形成紧密矩形块；`generate_input()` 会随机生成 2 到 4 个不重叠容器，覆盖四个开口方向、不同尺寸、不同像素数量，并规避输入输出相同的退化样例。

已执行自检：
- 100 次 `generate_input()` / `main()` 验证通过
- 变体数 `variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/7.py` 通过
- 已清理目标目录下的 `__pycache__`
已将完整实现写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/14/7.py)。`main(grid)` 会识别灰色分区线、定位带单个边框标记的面板、按标记所在边整体平移内部图案，并用标记色填充腾空轨迹；`generate_input()` 会随机生成不同面板尺寸、行列数、图案、颜色、标记数量和方向的样例，并用 `main()` 反验避免退化输入。

已按要求运行自检并通过：
- 100 次随机样例验证通过，`variants 100`
- `python -m py_compile` 通过
- 已清理验证产生的 `__pycache__` 目录
已将实现写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/1.py)。

`main(grid)` 会识别唯一灰色矩形框，按框中心划分四象限，并按“同色总数 = 枝长”的规则向外生长两条分支；分支在穿过框和原始种子时会跳过这些保护像素，保证框和种子保持不变。`generate_input()` 随机化了网格尺寸、框尺寸与位置、颜色数、每色计数、种子象限分布，并在生成时逐个验证枝条可完整长出、不会互相冲突、且都会穿出框外。

已执行并通过你要求的自检：
- 100 次 `generate_input()` / `main()` 验证通过，`variants 100`
- `python -m py_compile` 通过
- 已清理验证产生的 `__pycache__`
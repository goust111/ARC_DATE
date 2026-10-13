代码已写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/2.py:1)。`main(grid)` 会识别底部灰框和其下方的路径色键，重建“路径色像素 + 被路标覆盖的像素”组成的完整简单路径，按起点到终点遍历并把路标颜色序列填回对应框内；`generate_input()` 会随机生成 2 到 4 条路径、不同折线形状、不同路标数量/颜色、不同框顺序，并用 `main()` 做 oracle 校验。

已按要求完成自检：100 次随机样例通过，`variants 100`；`python -m py_compile` 通过；生成的 `__pycache__` 已清理。
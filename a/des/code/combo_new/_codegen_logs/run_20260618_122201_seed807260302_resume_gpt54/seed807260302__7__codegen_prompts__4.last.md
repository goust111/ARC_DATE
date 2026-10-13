代码已写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/7/4.py)。`main(grid)` 会解析每个橙色开口矩形、提取内部唯一种子块、按缺口方向旋转后平铺整个内部；`generate_input()` 会随机化对象数量、布局、框尺寸、缺口方向、种子大小、颜色图案和种子偏移，并过滤掉旋转不可区分的退化种子。

按要求完成了自检：100 次随机生成全部通过，`variants 100`；`py_compile` 通过；验证后已清理目标目录下的 `__pycache__`。
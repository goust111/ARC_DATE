代码已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/11/6.py)。实现是自包含的：`main(grid)` 用 4 连通组件识别 5 像素 T 形箭头，按边框同色匹配目标边，只让朝向一致的箭头移动，并按“离目标近的先占位”处理冲突；`generate_input()` 通过拒绝采样随机化网格大小、边框颜色、箭头数量、颜色、朝向和是否发生冲突。

已按要求完成自检：100 次 `generate_input()` / `main()` 验证通过，`variants 100`，`python -m py_compile` 通过，并已清理验证产生的 `__pycache__`。
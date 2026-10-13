已将实现写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/16/7.py)。

`main()` 会按 4 连通分量解析每条 `green -> red` 简单路径，提取主体颜色序列、做相邻去重，再沿起终点对齐方向重画成直线。`generate_input()` 采用反向构造：先确定压缩后的颜色序列，再用搜索生成合法的弯折 induced path，并随机化路径数量、方向、正反向、重复长度、网格尺寸和放置位置，最后用 `main()` 做 oracle 校验。

已按要求完成自检：
- 100 次 `generate_input()` / `main()` 验证通过
- `variants 100`
- `python -m py_compile` 通过
- 已清理生成的 `__pycache__`
代码已写入 [8.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/10/8.py:1)。

`main()` 在 [8.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/10/8.py:70) 里按题意完成了“模板行首次出现位置决定左循环移位”的解码；`generate_input()` 在 [8.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/10/8.py:247) 里随机化了网格尺寸、模板序列结构、灰色控制行数量、标记左右侧位置和重复颜色使用，并带输入有效性校验。`main()` 还额外兼容了题面里那个“模板行与控制列交点留黑”的冲突写法。

已执行你要求的自检：
- 100 次 `generate_input()` / `main()` 验证通过
- 变体数：`variants 100`
- `python -m py_compile` 通过
- 已清理验证产生的 `__pycache__`
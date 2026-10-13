实现已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/13/6.py)。

`main()` 按房间解析 maroon 分隔线后做两阶段射线模拟：先直射，再对撞到 grey 墙的射线做一次顺时针反弹；`generate_input()` 会随机生成 2 到 6 个房间，覆盖直达边界和撞墙反弹两类分支，并在生成阶段调用 `main()` 做一致性过滤。

已完成自检：
- 100 次 `generate_input()` / `main()` 验证通过
- `variants 100`
- `python -m py_compile` 通过
- 已清理 `__pycache__`
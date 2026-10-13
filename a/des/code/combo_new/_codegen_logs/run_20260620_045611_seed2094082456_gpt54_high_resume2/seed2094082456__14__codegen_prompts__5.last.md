已将实现写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/5.py)。

`main()` 实现了箭头识别、灰色对角镜面反射、射线追踪和终点复制；`generate_input()` 用场景组合生成随机样例，覆盖了 1-3 个箭头、1-3 个镜面、两种镜面朝向、直射/反射/双反射，以及边界停止和被其他箭头阻挡停止。

自检已运行并通过：
- 100 次 `generate_input()` / `main()` 验证通过
- `variants 100`
- `python -m py_compile` 通过
- 已清理 `__pycache__`
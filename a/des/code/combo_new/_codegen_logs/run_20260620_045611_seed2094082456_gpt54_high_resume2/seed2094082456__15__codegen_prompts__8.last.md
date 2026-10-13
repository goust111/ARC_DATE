代码已写入 [8.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/8.py)。`main()` 按“同边框色配对 + Azure/Orange 判定源/接收器 + 在两者中心通道画交替条纹”的规则实现；`generate_input()` 随机化了网格尺寸、对数、颜色、水平/垂直方向、源所在侧和间距，并用占位约束避免重叠或连线冲突。

已按要求完成自检：100 次生成/求解通过，`variants 100`；`python -m py_compile` 通过；生成的 `__pycache__` 已清理。
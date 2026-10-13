已将完整实现写入 [4.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/11/4.py)。文件包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`，`main()` 按角落图例解析交叉点度数并追踪直线段做双色分段重绘，`generate_input()` 随机生成主十字干线加对角折线分支的网络，稳定覆盖边界段、同色端点段、异色端点段三类规则分支。

已按要求完成自检：100 次随机生成/求解通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
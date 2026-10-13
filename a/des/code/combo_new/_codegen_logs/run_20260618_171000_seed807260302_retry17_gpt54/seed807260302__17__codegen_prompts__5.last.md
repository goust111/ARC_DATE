已将实现写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/5.py)。`main()` 会按单色连通分量识别空心矩形框、用严格包含关系计算嵌套深度，并只对叶子容器里的种子对象做 `depth * 90°` 顺时针旋转后居中回填；`generate_input()` 则随机生成带种子嵌套链和空框补丁，拼装到不超过 `30x30` 的画布上，并用独立构造的期望输出校验 `main()`。

已完成你要求的自检：100 次生成/求解通过，`variants 100`；`python -m py_compile` 通过；验证产生的 `__pycache__` 已清理。
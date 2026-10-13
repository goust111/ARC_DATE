已完成实现并写入目标文件：

- `main` 与 `generate_input` 已写入：`/home/lixinyu/snap/a/des/code/eval_new/seed202524619/3/5.py`

自检结果：

- 运行命令中的 100 轮生成/变换检查通过
- 所有输入/输出网格均为 `np.ndarray` 且尺寸均 `<= 30x30`
- 输入变体去重数：`variants 100`（≥20）
- `py_compile` 通过
- 已清理验证生成的 `__pycache__` (`/home/lixinyu/snap/a/des/code/eval_new/seed202524619/3/__pycache__`)
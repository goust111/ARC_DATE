代码已写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/3.py:1)。`main()` 按题面规则里的明确步骤实现为“对镜像对应的差异像素做原始颜色交换”，`generate_input()` 会随机生成双面板镜像底稿，并均衡注入 `two_colors`、`left_only`、`right_only` 三类差异，保证尺寸、分隔线、颜色数和非退化性约束。

已运行自检与编译检查，结果通过：`variants 100`，`py_compile` 通过。目标目录下没有残留 `__pycache__`。唯一需要说明的是，题面里“交换后两面板变为完美镜像对称”与前面的逐像素交换步骤在几何上互相冲突，所以实现遵循了更具体的逐步规则。
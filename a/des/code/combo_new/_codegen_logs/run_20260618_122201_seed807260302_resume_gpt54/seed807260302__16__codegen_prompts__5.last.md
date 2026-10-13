代码已写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/16/5.py)。`main()` 按中心十字、边框和背景做结构识别，再对四个象限里的种子像素生成朝对应角落外扩的 L 形枝条；`generate_input()` 会随机化网格尺寸、四象限颜色和每象限种子数量/位置，并用独立构造的期望输出回验 `main()`。

自检已完成并通过：100 次随机样例得到 `variants 100`，满足 `>=20` 的要求；`python -m py_compile` 也通过；目标目录下的 `__pycache__` 已清理。
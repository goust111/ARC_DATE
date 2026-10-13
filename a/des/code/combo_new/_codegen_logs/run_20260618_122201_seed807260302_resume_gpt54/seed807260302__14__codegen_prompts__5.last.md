实现已写入 [5.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/14/5.py)。

`main()` 按 grey 闭合轮廓逐个分离 interior/outside，并分别实现了红色外环、蓝色角点对角生长、绿色四个极值点双像素短线；`generate_input()` 会随机生成 3 到 4 个不同形状并保证三种种子分支覆盖、对象不接触、外扩不越界。

自检已执行并通过：`variants 100`，`python -m py_compile` 通过，`__pycache__` 已清理。
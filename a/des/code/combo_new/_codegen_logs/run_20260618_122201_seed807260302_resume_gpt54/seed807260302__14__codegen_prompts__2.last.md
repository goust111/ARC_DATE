代码已写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/14/2.py)。`main()` 做了通用的箭头解析和按轮次同步传播，`generate_input()` 用参数化 bundle 随机拼装样例，并用 `main()` 反验候选输入。

这版实现采用了题面里唯一可见且可判定的方向约定：水平 2 格箭头向右，垂直 2 格箭头向下；首波发射者取“在原始网格里不被其他箭头直线命中”的箭头。生成器会随机混合普通转向链、三段链和抢先阻断链，覆盖激活、染色、遇黑停止和后续波次阻断这些分支。

已执行并通过：
- 100 次自检，结果 `variants 100`
- `python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed807260302/14/2.py`

`__pycache__` 已清理。
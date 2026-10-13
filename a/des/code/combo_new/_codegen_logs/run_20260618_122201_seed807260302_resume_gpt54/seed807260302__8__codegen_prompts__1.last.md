代码已写入 [1.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/8/1.py)。

实现是自包含的，`main(grid)` 会识别破口围栏、解析 `maroon` 模板条、按旋转等价匹配内部形状，并依据缺口方向把对应模板盖印到围栏外侧。`generate_input()` 会随机化网格尺寸、模板数、模板形状、模板角落位置、围栏数量、颜色、内部旋转和缺口方向，并用 `main()` 回验生成结果。

已完成自检：
```text
variants 100
```
`python -m py_compile` 也已通过，并且已清理验证产生的 `__pycache__`。
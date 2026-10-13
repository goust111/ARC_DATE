已完成实现，代码写入 [7.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/7.py)。

`main()` 实现了按平移归一化轮廓配对、读取底部调色板索引差、再按 `G=1/2/>=2` 绘制不同折线样式的规则；`generate_input()` 会随机化网格尺寸、调色板长度与顺序、对象数量、形状、颜色配对、位置和线段方向，并保证三种分支都稳定出现且输入不退化。

已按要求执行自检并通过：
```bash
python - <<'PY'
...
print("variants", len(sigs))
...
PY
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/17/7.py"
```

结果里 `variants 100`，`py_compile` 通过，验证产生的 `__pycache__` 已清理。
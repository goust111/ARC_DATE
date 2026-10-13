代码已写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/2.py)。

实现包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`：`main()` 会解析上半部图例模板、下半部种子和灰色框，并按种子对齐后的周期平铺规则填充框内；`generate_input()` 会随机化网格尺寸、模板族、颜色、框数量、框尺寸、开口位置和颜色复用，并用重试避免退化输入。

已按要求完成自检：
```bash
python - <<'PY'
...
PY
```
结果：`variants 100`

以及：
```bash
python -m py_compile /home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/2.py
```

验证后已清理目标目录下的 `__pycache__`。
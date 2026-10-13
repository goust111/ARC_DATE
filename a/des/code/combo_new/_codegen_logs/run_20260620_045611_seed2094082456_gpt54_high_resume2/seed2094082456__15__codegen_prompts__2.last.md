代码已写入 [2.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/2.py)。

实现包含自包含的 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`。`main()` 会识别灰色矩形模板、模板内唯一彩色锚点、以及框外带内孔的单色碎片；先按颜色配对，再按“孔洞像素对齐锚点且完整落入模板内部”的唯一平移位置完成回填。`generate_input()` 会随机化网格尺寸、2/3 个模板、模板尺寸、锚点位置、碎片形状、孔洞大小和散落位置，并在生成阶段调用 `main()` 做一致性过滤，避免退化样例。

已按要求完成自检：
```bash
python - <<'PY'
...
PY
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/15/2.py"
```
结果 `variants 100`，编译通过，且已清理 `__pycache__`。
代码已写入 [6.py](/home/lixinyu/snap/a/des/code/combo_new/seed807260302/14/6.py)。里面包含自包含的 `main(grid)` 和 `generate_input()`：`main()` 按灰色矩形底板、内部孔洞、外部彩色碎片做旋转形状匹配并回填；`generate_input()` 随机化网格尺寸、孔洞数量、孔洞形状、孔洞位置、碎片颜色、碎片旋转和散放位置，并用 `main()` 反向回验。

已按要求完成自检：
```bash
python - <<'PY'
...
print("variants", len(sigs))
...
PY
# 输出: variants 100

python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed807260302/14/6.py"
```

验证后已清理目标目录下的 `__pycache__`。
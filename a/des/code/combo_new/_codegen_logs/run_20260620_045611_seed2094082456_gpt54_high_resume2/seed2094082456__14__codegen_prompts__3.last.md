代码已写入 [3.py](/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/3.py)。`main()` 会先识别并裁掉右侧绿/蓝图例区，再按颜色做端点对齐补线，最后按闭合与否分别做整框填充或仅保留路径；`generate_input()` 会随机生成至少一个闭合对象和一个非闭合对象，并用求解器结果反校验样例合法性。

已按你要求完成自检：
```bash
python - <<'PY'
...
PY
python -m py_compile "/home/lixinyu/snap/a/des/code/combo_new/seed2094082456/14/3.py"
```
结果通过，`variants 100`，并且已清理验证产生的 `__pycache__`。
已按要求落地到目标文件：

- [4.py](/home/lixinyu/snap/a/des/code/eval_new/seed558458950/4/4.py)

自检结果（按给定命令）：

- `100` 次随机生成通过 `generate_input()` 与 `main()` 组合调用，未触发断言。
- 输入/输出尺寸均满足 `<= 30`（`max(shape) <= 30`）。
- 生成变体数为 `variants 100`（≥ 20）。
- `python -m py_compile ...` 通过。
- 已清理 `__pycache__`。
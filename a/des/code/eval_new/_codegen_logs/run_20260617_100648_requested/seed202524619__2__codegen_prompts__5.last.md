已完成：目标文件已写入并通过自检命令。

- 目标文件：`/home/lixinyu/snap/a/des/code/eval_new/seed202524619/2/5.py`
- 自检结果：
  - `100` 次 `generate_input()` 运行后，输入形状均 ≤ `30x30` 且每次均得到不同签名（`variants 100`）
  - `py_compile` 通过（无语法错误）
  - 自检脚本最后的断言均通过
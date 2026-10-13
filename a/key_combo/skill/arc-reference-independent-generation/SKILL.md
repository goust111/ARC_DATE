---
name: arc-reference-independent-generation
description: 按给定 ARC 参考题文件生成创新题目。Use when the user asks to generate ARC creative puzzles from one or more key_combo reference Markdown files, especially when outputs must be independent per reference file and stored under /home/lixinyu/snap/a/key_combo/des/2.
---

# ARC 参考题独立生成创新题

## 用途

根据用户给定的一个或多个参考题 `.md` 文件，按顺序为每个文件启动一个独立 subagent，并把指定命令传给 subagent。

## 流程

1. 获取参考题文件路径列表；如果用户给的是目录，列出其中需要处理的 `.md` 文件并按文件名排序。
2. 并行处理参考题文件。根据使用情况并行启动多个 subagent。
3. 对当前参考题文件，取 basename 作为输出文件名，例如 `001.md`。
4. 输出路径固定为 `/home/lixinyu/snap/a/key_combo/des/fable/2/<basename>`。
5. 给每个 subagent 传入下面的三行命令，只替换参考题路径和输出路径。

## Subagent 命令模板

```text
参考{REFERENCE_FILE}
  依照/home/lixinyu/snap/a/key_combo/skill/arc-reference-independent-generation/标准.md中的步骤创新，存储为{OUTPUT_FILE}，注意不能查看上述两个文件外的任意文件
```


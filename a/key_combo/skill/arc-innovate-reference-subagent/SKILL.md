---
name: arc-innovate-reference-subagent
description: 根据给定 ARC 参考题 Markdown 文件启动 subagent 生成创新题。Use when reference puzzles are already packaged as Markdown files like /home/lixinyu/snap/a/key_combo/select/1569447623/009.md and should be passed directly to subagents instead of sampling or extracting from the JSON index.
---

# ARC 参考题路径出题

## 执行流程

### 第一步：获取参考题路径

获取用户给定的参考题 `.md` 文件路径。若用户给的是目录，列出其中需要处理的 `.md` 文件并按文件名排序。

### 第二步：确定输出路径

为每个参考题文件取 basename 作为输出文件名，例如：

```text
/home/lixinyu/snap/a/key_combo/select/1569447623/001.md
-> /home/lixinyu/snap/a/key_combo/des/opus/2/001.md
```

输出目录固定为 `/home/lixinyu/snap/a/key_combo/des/opus/2`。如果输出文件已存在，先询问用户是否覆盖。

### 第三步：使用 subagent 生成

每个参考题文件单独启动一个 subagent。不要把多个参考题文件交给同一个 subagent。

传给 subagent 的命令只替换参考题路径和输出路径，其他说明保持一致。subagent 自己读取参考题文件和创新方法论，并完成生成与保存。

### 第四步：收尾汇总

完成后汇总：

- 已处理的参考题文件；
- 写入的输出文件；
- 跳过或失败的文件及原因。

## Subagent 命令模板

```text
参考{REFERENCE_FILE}
  依照/home/lixinyu/snap/a/as/ceshi/skill
    /arc_creative_prompt.md中的步骤创新，存储为{OUTPUT_FILE}，只读取上述两个文件，其余文件禁止读取
```

## 约束

- 不要读取 `/home/lixinyu/snap/a/code/arc1_full_plus_arc2_train_index.json` 来随机选题。
- 不要在主对话里先汇总或重建原语池；参考题相关信息让 subagent 直接查看当前参考题文件。
- 每个参考题文件对应一次独立 subagent 任务。
- 输出文件名必须使用参考题文件名，例如 `001.md`、`009.md`。

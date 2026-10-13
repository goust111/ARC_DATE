---
name: arc-key-combo
description: 从预选参考题组提取原语，按创新方法论生成多逻辑组合ARC新题（新版提示词）
---

# ARC 创新题目生成流程（key_combo · 新版）

## 关键文件
- **参考题目录**：`/home/lixinyu/snap/a/key_combo/select/`（已分组，每组 20 道，文件名 001.md ~ 051.md）
- **创新方法论**：`/home/lixinyu/snap/a/key_combo/skill/new/arc_creative_prompt.md`
- **输出目录**：`/home/lixinyu/snap/a/key_combo/des/{序号}_new/`（如使用 001.md 则输出到 `001_new/`）

## 执行流程

### 第一步：选取参考题组并提取原语池

确定本轮使用的参考题组文件：

```bash
ls /home/lixinyu/snap/a/key_combo/select/*.md | sort -V
```

选择一个尚未处理过的组文件（如 `001.md`），读取其中全部 20 道参考题的 puzzle_id、docstring 和 tags。

### 第二步：汇总原语池并理解参考题逻辑

1. 从每道参考题的 `tags` 字段提取四类原语（tags 已按 `operation:`、`object:`、`relation:`、`condition:` 前缀分类），去重汇总得到本轮**基础原语池**。
2. 结合 docstring 记录每道参考题中哪些原语**共现**（已有组合），新题需要避开这些已有组合。
3. 结合 docstring 理解每道参考题的核心推理逻辑：原语间的依赖关系、耦合方式。

### 第三步：读取创新方法论并生成新题

读取 `/home/lixinyu/snap/a/key_combo/skill/new/arc_creative_prompt.md`，将原语池和参考题的 docstring 一并传给方法论作为参考，按照其中的方法论逐步执行（设计逻辑组合、验证不可跳步等）。

生成数量**不设上限**。

### 第四步：保存输出

根据所选参考题组的序号创建输出目录，如使用 `001.md` 则输出到 `/home/lixinyu/snap/a/key_combo/des/001_new/`：

```bash
mkdir -p /home/lixinyu/snap/a/key_combo/des/001_new
```

按创新方法论中的输出格式写入该目录下的 `roundN.md`。如该目录下已有 round 文件，则递增编号：

```bash
ls /home/lixinyu/snap/a/key_combo/des/001_new/round*.md 2>/dev/null | sort -V | tail -1
```

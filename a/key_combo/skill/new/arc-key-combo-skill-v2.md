---
name: arc-key-combo-v2
description: 从预选参考题组提取原语，按抽签驱动方法论（v2）生成多逻辑组合ARC新题
---

# ARC 创新题目生成流程（key_combo · v2 抽签驱动）

## 关键文件
- **参考题目录**：`/home/lixinyu/snap/a/key_combo/select/`（已分组，每组 20 道，文件名 001.md ~ 051.md）
- **创新方法论（v2）**：`/home/lixinyu/snap/a/key_combo/skill/new/arc_creative_prompt_v2.md`
- **输出目录**：`/home/lixinyu/snap/a/key_combo/des/{序号}_new/`（如使用 001.md 则输出到 `001_new/`）

## 执行流程

### 第一步：选取参考题组并提取原语池

确定本轮使用的参考题组文件：

```bash
ls /home/lixinyu/snap/a/key_combo/select/*.md | sort -V
```

选择一个组文件（如 `001.md`），读取其中全部 20 道参考题的 puzzle_id、docstring 和 tags。

### 第二步：汇总原语池并编号

1. 从每道参考题的 `tags` 字段提取四类原语（tags 已按 `operation:`、`object:`、`relation:`、`condition:` 前缀分类），去重汇总得到本轮**基础原语池**，并将四类混编为一个**带序号的总列表**（1 ~ N），供抽签使用。
2. 结合 docstring 记录每道参考题中哪些原语**共现**（已有组合），新题需要避开这些已有组合。
3. 结合 docstring 理解每道参考题的核心推理逻辑：原语间的依赖关系、耦合方式。

### 第三步：读取 v2 方法论并按抽签流程生成新题

读取 `/home/lixinyu/snap/a/key_combo/skill/new/arc_creative_prompt_v2.md`，将编号原语池和参考题的 docstring 一并作为参考，严格按其中的**抽签驱动流程**执行：

1. 每道题从一次真随机抽签开始（用 `shuf` 抽原语序号，不要自选）：

   ```bash
   shuf -i 1-N -n 3   # N 为原语池大小
   ```

2. 对抽中的原语组做关系发现，写不出非平凡关系就丢弃重抽（记录丢弃次数和原因）。
3. 关系成立才展开成完整题目。
4. 全部题目完成后统一做骨架查重与验证（对照本轮及同组输出目录下**所有已有 round 文件**，含 v1 产出）。

生成数量**不设上限**，废稿多是正常的。

### 第四步：保存输出

根据所选参考题组的序号创建输出目录，如使用 `001.md` 则输出到 `/home/lixinyu/snap/a/key_combo/des/001_new/`：

```bash
mkdir -p /home/lixinyu/snap/a/key_combo/des/001_new
```

按 v2 方法论中的输出格式（含抽签记录、核心关系、抽签统计）写入该目录下的 `roundN.md`。如该目录下已有 round 文件（含 v1 产出），则递增编号：

```bash
ls /home/lixinyu/snap/a/key_combo/des/001_new/round*.md 2>/dev/null | sort -V | tail -1
```

---
name: gen-combo-propose
description: 为 ARC-AGI-2 任务从维度分析中自由组合多个 dim/alt，生成融合变体描述并写入 YAML。用法：/gen-combo-propose <task_id> [task_id2 ...]，组合数量与构成由模型自行决定，最后将筛选出来的变体写入 YAML 文件。
---

## 任务
对每个 `task_id`，从 `ceshi/yaml/dim_analysis_v9_{task_id}.yaml` 自由提出若干「多 dim/alt 组合」的融合变体，然后将融合描述写入 YAML 文件。本 skill 仅负责组合提议、筛选、确认和落盘，**不涉及代码生成**。

## 参数
- `task_id`：任务 ID（必填，可多个任务）
- 不接受任何形式的内联组合参数。提议阶段输出的组合方案写入 YAML
- 组合数量由模型在系统遍历后决定，宁多勿少；每个维度至少参与一个组合

## 数据约定

- **combo_spec 字符串**：参与组合的 `dim_id/alt_index` 按 `dim_id` 自然序排序后用 `+` 拼接，如 `D1/0+D5/2+D14/0`。用于标记该组合由哪几个变体融合而来，注意序号从0开始，也就是D1/0...3,D从1开始，即D1...D4。
- **index**：每个组合的序号（从 1 开始），后续 `/gen-combo-code` 阶段以此命名文件路径
- **des_combo YAML 文件**：`/home/lixinyu/snap/new/ceshi/des_combo/{task_id}.yaml`

## 流程

### 步骤一：描述融合筛选 → `.claude/agents/combo-proposer.md`

对每个 `task_id` 启动一个 combo-proposer subagent（多任务并行）进行组合提议与筛选。

**启动 subagent**：传入 `task_id`，subagent 按 `combo-proposer.md` 中的流程自行读取 YAML、solver、spec，系统遍历维度对并提议组合
   - 单任务：前台启动
   - 多任务：全部用 `run_in_background: true` 并行
   - 启动时

### 步骤二：写入 YAML 文件

将筛选出来的结果写入最终组合列表：
```
mkdir -p /home/lixinyu/snap/new/ceshi/des_combo
```
若某任务没有任何组合通过筛选，留空并注明原因。

每个任务写一个 YAML 文件 `/home/lixinyu/snap/new/ceshi/des_combo/{task_id}.yaml`，格式如下：

```yaml
task_id: "2c181942"
combos:
  - index: 1
    combo_spec: "D1/0+D5/2"
    name: "旋转匹配+颜色映射"
    description: "输入中的精灵需要先旋转至与框架洞匹配的朝向，再按洞的边框颜色进行重新着色后填入。solver 需要同时判断旋转角度和颜色对应关系。"
    existing_code:
      - "D1/0"
    reason: "旋转判断和颜色映射协同改变了匹配+变换的推理链，且两者在视觉上可区分。"

  - index: 2
    combo_spec: "D3/1+D7/0+D14/0"
    name: "多步级联+逆序处理+网格裁剪"
    description: "..."
    existing_code: []
    reason: "..."
```

YAML 字段说明：
| 字段 | 说明 |
|------|------|
| task_id | 任务 ID |
| combos | 组合列表 |
| combos[].index | 序号（从 1 开始） |
| combos[].combo_spec | 组合规格字符串，如 `D1/0+D5/2` |
| combos[].name | 组合名称（简短中文描述） |
| combos[].description | 组合后的完整描述（融合后的 input→output 规则） |
| combos[].existing_code | 组成项中已有 `code.py` 的列表，无则为空列表 |
| combos[].reason | 推荐原因 |

## 输出
- YAML 文件：`ceshi/des_combo/{task_id}.yaml`
- 输入来源：
  1. `ceshi/yaml/dim_analysis_v9_{task_id}.yaml`（必备，全量）
  2. `arc2_solver_select/{task_id}.py`（必备，骨架）
  3. `spce_dis/{task_id}_spec.yaml`（可选，原题代码生成时的参考文件，用于理解任务设计细节和验证可行性）
  4. `ceshi/code/{task_id}/{dim_id}/{alt_index}/code.py`（命中即记录，未命中跳过）

## 模式示例
- **单任务**：`/gen-combo-propose 2c181942` → 直接对该任务自由组合，确认后写入 YAML
- **多任务**：`/gen-combo-propose 2c181942 67e490f4` → 并行提议、统一展示，确认后分别写入 YAML
- **后续步骤**：YAML 写入完成后，可运行 `/gen-combo-code <task_id>` 进入代码生成阶段

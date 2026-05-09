---
name: gen-dim-analysis
description: 为 ARC-AGI-2 任务进行维度分析并推荐创新变体。用法：/gen-dim-analysis <task_id> [task_id2 ...]，推荐数量由模型根据筛选标准自行决定。
---

## 任务
对指定任务执行维度分析，生成创新变体候选方案，筛选后存入txt文件。

## 参数
- `task_id`：任务 ID（必填，可多个，如 `2c181942 67e490f4`）

## 模型要求
- 本 skill 中所有 subagent 都必须**显式固定**使用 `Opus 4.6`
- 启动 subagent 时必须显式传入 `model: opus-4.6`
- 不要使用 `opus` 这类会漂移到新版本的别名，也不要使用 Auto、Premium 或继承默认模型

## 步骤

### 阶段一：维度分析（per task，多任务并行）

对每个 `task_id`：

1. 清理旧版文件：
   - **默认**：直接删除 `ceshi/yaml/dim_analysis_*_{task_id}.yaml` 和 `ceshi/tem1/dim_analysis_prompt_{task_id}.txt`
   - **仅当用户明确要求备份时**，才 `mv` 到 `/tmp/dim_backups/`

2. 调用 prompt builder 生成提示词：
   ```
   cd /home/lixinyu/snap/new
   python dim_analysis_prompt_builder.py <task_id>
   ```

3. 启动 subagent 执行维度分析（提示词很长，用 subagent 隔离上下文）：
   > 读取 `/home/lixinyu/snap/new/ceshi/tem1/dim_analysis_prompt_{task_id}.txt`，按提示词指令执行分析，将 YAML 写入 `/home/lixinyu/snap/new/ceshi/yaml/dim_analysis_v9_{task_id}.yaml`
   > 启动时显式设置 `model: opus-4.6`

   单任务前台启动，多任务用 `run_in_background: true` 并行。

### 阶段二：筛选（per task，分析完成后各自启动）

每个任务的分析 subagent 完成后，立即启动筛选 subagent：

4. 启动筛选 subagent：
   > 读取 `/home/lixinyu/snap/new/ceshi/yaml/dim_analysis_v9_{task_id}.yaml`。
   > 如果存在 `/home/lixinyu/snap/new/spce_dis/{task_id}_spec.yaml`，也一并读取——spec 是原题代码生成时的参考文件，可用于理解原始任务的完整设计细节，并验证替代方案的可行性。
   > 同时读取 solver 代码 `/home/lixinyu/snap/new/arc2_solver_select/{task_id}.py`，对照推理链判断创新有效性。
   > 启动时显式设置 `model: opus-4.6`
   >
   > 从所有维度×替代方案中筛选**全部**符合标准的题目，数量由模型自行判断——只要满足下述全部标准就纳入推荐，不预设上限也不为凑数放宽标准；若没有任何题目通过，允许返回空列表并说明原因。筛选标准：
   > - **有效创新**：该变体必须改变解题者的推理过程（排序规则、匹配条件、变换逻辑、输出构造方式等），而非仅改变视觉外观。
   > - **可行性**：变体符合ARC题目要求，且对照 spec 中的数值约束（网格尺寸、对象数量/尺寸、颜色数等），该变体在物理上可实现。不可行的方案应排除。
   > - **多样性**：该变体的 generator 能生成多种不同的输入输出网格组合（随机参数变化时产出视觉上有差异的样本），而非总是生成几乎相同的网格。
   > - **人类可视性**：凡是依赖颜色ID等仅对程序可见对人类不可见的信号的方案一律排除（注意人类看到的不是数字矩阵，是可视化后的图像）。
   > - **题目适配性**：该变体在该题上应该合理、自然且可成立。
   >
   > 返回推荐结果：每条包含维度 ID、Alt 索引、名称、描述摘要、推荐原因。

### 阶段三：汇总、确认与落盘（parent 执行）

收集所有筛选 subagent 返回的推荐结果，按任务将最终题号列表写入文件目录 `/home/lixinyu/snap/new/ceshi/des`：
   - 先确保目录存在：
     ```
     mkdir -p /home/lixinyu/snap/new/ceshi/des
     ```
   - 每个任务写一个文件：`/home/lixinyu/snap/new/ceshi/des/{task_id}.txt`
   - 文件内容仅保留该任务最终确认的创新题号，每行一个，格式固定为 `dim_id/alt_index`

   示例：
   ```
   /home/lixinyu/snap/new/ceshi/des/2c181942.txt
   D1/0
   D5/1
   D14/0
   ```
若某任务没有任何变体通过筛选标准，对应表格留空并注明原因。
## 输出
- YAML 文件：每个任务一个，`ceshi/yaml/dim_analysis_v9_{task_id}.yaml`
- 筛选后的题号列表文件：`ceshi/des/{task_id}.txt`
- 文件内容示例：
  ```
  D1/0
  D5/1
  D14/0
  ```

## 模式示例
- **单任务**：`/gen-dim-analysis 2c181942`
- **多任务**：`/gen-dim-analysis 2c181942 67e490f4` → 并行分析，统一展示推荐，确认后写入 `ceshi/des/`

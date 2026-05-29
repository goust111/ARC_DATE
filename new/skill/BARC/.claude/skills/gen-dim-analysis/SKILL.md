---
name: gen-dim-analysis
description: 为 ARC-AGI-2 任务进行维度分析并推荐创新变体。用法：/gen-dim-analysis <task_id> [task_id2 ...]，推荐数量由模型根据筛选标准自行决定。
---

## 任务
对指定任务执行维度分析，生成创新变体候选方案，筛选后存入txt文件。

## 参数
- `task_id`：任务 ID（必填，可多个，如 `2c181942 67e490f4`）
- 启动 subagent 时不需要传入model参数，默认继承父模型

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

   单任务前台启动，多任务用 `run_in_background: true` 并行。

### 阶段二：筛选（per task，分析完成后各自启动）

每个任务的分析 subagent 完成后，立即启动筛选 subagent：

4. 启动筛选 subagent：
   启动时，请将以下指令作为 subagent 的 `prompt` 传入（请替换其中的 `{task_id}`）：

   """
   请执行 ARC 变体筛选任务。你需要：
   1. 读取 `/home/lixinyu/snap/new/ceshi/yaml/dim_analysis_v9_{task_id}.yaml`。
   2. 检查并读取 `/home/lixinyu/snap/new/spce_dis/{task_id}_spec.yaml`（如果存在），作为验证替代方案可行性的参考。
   3. 读取 solver 代码 `/home/lixinyu/snap/new/arc2_solver_select/{task_id}.py`，以理解原题的推理逻辑，但评估创新性时应从人类解题者的角度判断推理过程是否改变，而非判断原 solver 代码是否兼容。

   请从 YAML 的所有维度×替代方案中筛选出**全部**符合标准的变体。数量由你自行判断，只要满足下述全部标准就纳入推荐，不预设上限，也不为凑数放宽标准；若没有任何题目通过，请返回空列表并说明原因。

   筛选标准：
   - **有效创新**：该变体必须改变解题者的推理过程（排序规则、匹配条件、变换逻辑、输出构造方式等），而非仅改变视觉外观。注意：一些较为复杂的输出构造方式的改变也属于有效创新。
   - **可行性**：变体在 ARC 框架下可实现（网格不超过30×30、颜色不超过10种等）。参考 spec理解原题的设计约束范围，spec 中的具体参数（如图案类型数、对象数量、网格精确尺寸等）是原题的参数，变体可以合理调整，但要说明调整方向。
   - **多样性**：该变体的 generator 能生成多种不同的输入输出网格组合（随机参数变化时产出视觉上有差异的样本），而非总是生成几乎相同的网格。
   - **人类可视性**：变体规则必须能被人类从样例中推断出来。禁止将颜色 ID 作为数值参与排序、算术或其他隐藏运算。禁止输出逻辑中的变换规则包含随机逻辑（如"随机选色""随机选位置"作为规则的一部分）。人类看到的是图像而非数字矩阵，所有变化必须有可见的结构性线索。
   - **题目适配性**：变体可以改变推理链中的某个关键步骤（如匹配条件从颜色改为形状），只要整体任务结构（源对象→变换→输出）保持连贯即可。不要求变体与原题在每个方面都相似。


   读取 solver 代码以理解原题的推理逻辑，但评估创新性时应从人类解题者的角度判断推理过程是否改变，而非判断原 solver 代码是否兼容。

   返回格式：
   请直接在回复中返回推荐结果，每条包含：维度 ID、Alt 索引、名称、描述摘要、推荐原因。
   """
### 阶段三：汇总与落盘（parent 执行）

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
若某任务没有任何变体通过筛选标准，在最后汇总输出阶段注明原因。
## 输出
- YAML 文件：每个任务一个，`ceshi/yaml/dim_analysis_v9_{task_id}.yaml`
- 筛选后的题号列表文件：`ceshi/des/{task_id}.txt`
- 文件内容示例：
  ```
  D1/0
  D5/1
  D14/0
  ```
  注意：`dim_id/alt_index`中alt_index 从 0 开始计数；dim_id 编号从 D1 开始！

## 模式示例
- **单任务**：`/gen-dim-analysis 2c181942`
- **多任务**：`/gen-dim-analysis 2c181942 67e490f4` → 并行分析，统一展示推荐，确认后写入 `ceshi/des/`

---
name: gen-combo-code
description: 为已确认的融合组合生成实现代码，从 YAML 读取组合信息，通过 Codex CLI (gpt-5.4) 并行生成代码并审查。用法：/gen-combo-code <task_id> [task_id2 ...]
---

## 任务
对用户已确认的融合组合，生成融合实现代码（包含 `main()` 和 `generate_input()`），测试验证，输出汇总报告。组合列表从 `/home/lixinyu/snap/new/ceshi/des_combo/{task_id}.yaml` 读取，调用方只需传入 `task_id`。

## 参数
- `task_id`：任务 ID（必填，可多个任务）
- 不接受任何形式的内联组合参数。

## 后端
- 通过 Bash 调用 `codex exec`，不使用 Claude subagent
- 不允许退化为本地实现，即只能通过 `exec` 实现，并且进行六项审查，禁止变为本地生成且不审查
- 将组合提示词的**完整内容原样**作为 prompt 传给 Codex，不可摘要或改写
- 为避免 trusted directory / Git repo 检查阻塞执行，始终显式添加 `--skip-git-repo-check`
- 调用形式：
  ```
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "<完整提示词>"
  ```

## 数据约定

- **combo_spec**：`dim_id/alt_index` 按 `dim_id` 自然序排列，`+` 拼接，如 `D1/0+D5/2+D14/0`。用于标记该组合由哪几个变体融合而来
- **index**：YAML 中每个组合的序号（从 1 开始），用于文件路径命名
- **des_combo YAML 文件**：`/home/lixinyu/snap/new/ceshi/des_combo/{task_id}.yaml`
- **代码路径**：`ceshi/code_combo/{task_id}/{index}/code.py`
- **提示词路径**：`ceshi/tem3/combo_prompt_{task_id}_{index}.txt`

## 步骤

### 阶段零：读取 YAML（per task）

对每个传入的 `task_id`：

1. 读取 `/home/lixinyu/snap/new/ceshi/des_combo/{task_id}.yaml`
2. 解析 YAML，从 `combos` 列表中提取每个组合的 `index` 和 `combo_spec`

### 阶段一：生成组合提示词（per combo）

对 YAML 中每个组合（以 `index` 和 `combo_spec` 为核心参数）：

1. 清理旧版文件：
   - **默认**：直接删除 `ceshi/tem3/combo_prompt_{task_id}_{index}.txt` 和 `ceshi/code_combo/{task_id}/{index}/code.py`
   - **仅当用户明确要求备份时**，才 `mv` 到 `/tmp/combo_backups/`
2. 确保目录存在：
   ```
   mkdir -p /home/lixinyu/snap/new/ceshi/tem3
   ```
3. 调用 prompt builder 生成组合提示词：
   ```
   cd /home/lixinyu/snap/new
   python combo_prompt_builder.py <task_id> <index>
   ```

### 阶段二：生成组合代码（Codex 并行）

为每个组合通过 Bash 调用 Codex：

1. 读取 `/home/lixinyu/snap/new/ceshi/tem3/combo_prompt_{task_id}_{index}.txt` 的完整内容
2. 如果提示词中包含 Claude 专属表述，做等价替换
3. 通过 `codex exec` 执行：
   ```
   codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "<完整提示词>"
   ```

并行策略：
- **单组合**：前台执行
- **多组合**：全部在后台并行启动

### 阶段三：审查修复（等所有 Codex 任务完成后，不可跳过）

不得跳过六项审查。

对每个组合，读取 `ceshi/code_combo/{task_id}/{index}/code.py` 和对应提示词，审查：

1. **融合一致性**：`main()` 是否真正同时实现了所有组成项的"改为"行为？有没有偏向某一项而忽略其他？组成项之间应有交互的部分是否实现？
2. **ARC 可解性**：变体规则必须能被人类从 2-4 组 input→output 样例中(颜色变换和逻辑变化)，基于显著且可见的结构线索稳定推断出来，不依赖颜色值参与任何变换、排序，不依赖复杂的乘除、标签值或其他不易推断的隐藏规则，不允许出现随机逻辑。
3. **未命中项的实现质量**：对于没有现成参考代码的组成项，其变换逻辑是否被正确实现？
4. **可视化整洁度**：即使逻辑正确，如果可视化后网格看上去乱七八糟——同一网格区域多种元素覆盖叠加、不同对象或区域无法分辨——也不可接受。需确保每个网格单元的语义清晰、视觉可读。
5. **颜色变化可推断性**：如果融合涉及颜色映射或变化，颜色种类不宜过多（<=5）且必须有足够的结构性线索（如位置、形状、分区等）让人类能推断出颜色变化规则。避免出现大量颜色变化却无任何提示的情况，否则人类无法从样例中归纳出规律。
6. **分支覆盖率**：若题目有多个分支，每个分支的覆盖率是否均衡合理，是否都有覆盖到

若审查发现问题，parent 直接编辑 code.py，无需走 codex，重新测试直至通过，若修改五次代码仍然无法修改至符合变体描述，可放弃该变体，但要在最终汇总输出中说明，并将该变体有关文件夹删除。

#### 输出汇总表格

| 任务 ID | index | combo_spec | 组合名称 | 已有代码命中（existing_code） | 测试结果 | 状态 |
|---------|-------|------------|----------|--------------|----------|------|
| 2c181942 | 1 | D1/0+D5/2 | xxx | D1/0 | 47/50 | ✓ |
| 2c181942 | 2 | D3/1+D7/0+D14/0 | xxx | （无） | 50/50 | ✓（审查后修复） |

## 输出
- 组合代码：`ceshi/code_combo/{task_id}/{index}/code.py`
- 组合提示词（中间产物）：`ceshi/tem3/combo_prompt_{task_id}_{index}.txt`
- 输入来源：`ceshi/des_combo/{task_id}.yaml`
- 参考来源（由 prompt builder 自动注入）：`spce_dis/{task_id}_spec.yaml`（可选，原题代码生成时的参考文件，用于理解任务设计细节和约束验证）

## 模式示例
- **单任务**：`/gen-combo-code 2c181942` → 从 YAML 读取全部组合，用 Codex 并行实现
- **多任务**：`/gen-combo-code 2c181942 67e490f4` → 分别读取各任务的 YAML，对所有组合并行启动 Codex
- **前置步骤**：需先运行 `/gen-combo-propose <task_id>` 生成 YAML 文件

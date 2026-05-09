---
name: gen-variant-code
description: 为已确认的 ARC-AGI-2 创新变体生成实现代码，支持 Claude subagent (c) 和 OpenAI Codex CLI (o) 两种后端。用法：/gen-variant-code <task_id> [task_id2 ...] [--backend=c|o]，从 ceshi/des/{task_id}.txt 读取该任务的全部已确认创新题号并并行生成实现。
---

## 任务
对用户确认的创新变体题目，生成变体实现代码（包含 `main()` 和 `generate_input()`），测试验证，输出汇总报告。支持 Claude subagent (c) 和 OpenAI Codex CLI (o) 两种后端。变体列表从 `/home/lixinyu/snap/new/ceshi/des/{task_id}.txt` 读取，调用方只需传入 `task_id`。

## 参数
- `task_id`：任务 ID（必填，可多个任务）
- 不接受任何形式的内联题号参数。该 skill 始终从 `/home/lixinyu/snap/new/ceshi/des/{task_id}.txt` 读取该任务的全部已确认创新题号
- `--backend=c|o`：c=Claude subagent, o=OpenAI Codex CLI（可选，默认 c）
- 文件每行一个题号，格式固定为 `dim_id/alt_index`，例如：
  ```
  D1/0
  D5/1
  D14/0
  ```

## 后端

### OpenAI Codex CLI (o)
- 通过 Bash 调用 `codex exec`
- 将 `variant_prompt_{task_id}_{dim_id}_alt{alt_index}.txt` 的**完整内容原样**作为 prompt 传给 Codex，不可摘要或改写
- 调用形式：
  ```
  codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "<完整提示词>"
  ```
- 默认保留 `--skip-git-repo-check`，因为该 workflow 可能在非 Git 工作目录下执行；不要省略这个参数

## 步骤

### 阶段零：读取确认结果（parent 执行，per task）

对每个传入的 `task_id`：

1. 读取 `/home/lixinyu/snap/new/ceshi/des/{task_id}.txt`
2. 将文件中每一行解析为一个 `dim_id/alt_index`，作为该任务要实现的全部创新变体列表
3. 若文件不存在、为空、或任一行不符合 `D<number>/<alt_index>` 形式，立即报错并停止该任务，不要猜测或补全；提示用户先运行 `/gen-dim-analysis` 生成对应文件

### 阶段一：生成提示词（parent 执行，per 变体）

对每个 `task_id` 在阶段零解析得到的每个 `dim_id/alt_index`：

1. 清理旧版变体文件：
   - **默认**：直接删除 `ceshi/tem2/variant_prompt_{task_id}_{dim_id}_alt{alt_index}.txt` 和 `ceshi/code/{task_id}/{dim_id}/{alt_index}/code.py`
   - **仅当用户明确要求备份时**，才 `mv` 到 `/tmp/variant_backups/`

2. 确保 `ceshi/tem2/` 目录存在：
   ```
   mkdir -p /home/lixinyu/snap/new/ceshi/tem2
   ```

3. 调用 prompt builder 生成变体提示词（需显式指定输出路径到 `ceshi/tem2/`）：
   ```
   cd /home/lixinyu/snap/new
   python variant_prompt_builder.py <task_id> <dim_id> <alt_index> ceshi/tem2/variant_prompt_<task_id>_<dim_id>_alt<alt_index>.txt
   ```

### 阶段二：生成变体代码（按后端执行，所有变体并行）

4. 为每个变体启动对应后端执行：
   > OpenAI 后端：读取同一提示词文件全文，按“后端”章节中的规则改写掉 Claude 专属执行表述后，通过 `codex exec --skip-git-repo-check` 执行。提示词中其余分析、实现、测试要求保持不变。

   - **单变体**：前台启动
   - **多变体**：
     - OpenAI 后端：全部在后台并行启动 `codex exec`

### 阶段三：parent 侧审查（必执行阶段；等所有后端任务完成后自动进入，严禁漏掉、跳过、弱化或用“已生成代码”代替审查；必须严格按 5、6、7 顺序执行，此过程由模型审查，不需要存储相关信息）

**强制要求：**
- 阶段二结束后，**必须**进入阶段三，不能直接结束任务，不能直接向用户汇报“已完成/已生成/可交付”。
- **第 5 步是阶段三的入口和硬性门禁**。对每个变体完成第 5 步审查前，不得进入第 6 步或第 7 步，不得给该变体标记完成。
- 即使后端声称“已测试通过”或代码看起来合理，parent 也**仍然必须**亲自执行第 5 步审查；不得把后端自检结果视为对第 5 步的替代。
- 若存在多个变体，必须**逐个变体**完成第 5 步，不能抽查，不能只审查失败样例，不能只看汇总日志。

5. 对每个变体，读取 `code.py` 和对应的提示词，审查以下问题：
   - **此步必须逐个变体、逐项核对，不能省略任何一项；这是是否可交付的最终人工审查。**
   - **变体描述一致性**：`main()` 中的新逻辑是否真正实现了提示词中"改为"所描述的行为？有没有偷换概念或过度简化？
   - **ARC 可解性**：变体规则必须能被人类从 2-4 组 input→output 样例中(颜色变换和逻辑变化)，基于显著且可见的结构线索稳定推断出来，**禁止颜色值参与任何颜色变换、排序**，不依赖复杂的乘除或复杂数学知识，不依赖标签值或其他不易推断的隐藏规则，不允许出现随机逻辑。
   - **分支覆盖率**：若题目有多个分支，每个分支的覆盖率是否均衡合理，是否都有覆盖到
   - **可视化整洁度**：即使逻辑正确，如果可视化后网格看上去乱七八糟——同一网格区域多种元素覆盖叠加、不同对象或区域无法分辨——也不可接受。需确保每个网格单元的语义清晰、视觉可读。
   - **颜色变化可推断性**：如果变体涉及颜色映射或变化，颜色种类不宜过多且必须有足够的结构性线索（如位置、形状、分区等）让人类能推断出颜色变化规则。避免出现大量颜色变化却无任何提示的情况，否则人类无法从样例中归纳出规律。

6. 审查发现问题的变体，parent 直接修改 `code.py` 修复，重新测试后**再次回到第 5 步重审**，直至通过；若无法将代码修改至符合变体描述，可放弃该变体。

7. 仅当某个变体已经完成第 5 步审查，并在需要时完成第 6 步修复与回审后，才可将其写入汇总表格。输出汇总表格（注意需要输出是否全部通过测审查）：

   | 任务 ID | 维度 | Alt | 变体名称 | 测试结果 | 状态 |
   |---------|------|-----|----------|----------|------|
   | 2c181942 | D1 | 0 | xxx | 48/50 | ✓ |
   | 2c181942 | D5 | 1 | xxx | 50/50 | ✓（审查后修复） |

## 输出
- 变体代码：`ceshi/code/{task_id}/{dim_id}/{alt_index}/code.py`
- 变体提示词（中间产物）：`ceshi/tem2/variant_prompt_{task_id}_{dim_id}_alt{alt_index}.txt`
- 输入来源：`ceshi/des/{task_id}.txt`
- 参考来源（由 prompt builder 自动注入）：`spce_dis/{task_id}_spec.yaml`（可选，原题代码生成时的参考文件，用于理解任务设计细节和约束验证）

## 模式示例
- **单任务（Claude）**：`/gen-variant-code 2c181942` → 从 `ceshi/des/2c181942.txt` 读取全部创新题号，并用 Claude subagent 并行实现
- **单任务（OpenAI）**：`/gen-variant-code 2c181942 --backend=o` → 从 `ceshi/des/2c181942.txt` 读取全部创新题号，并用 Codex CLI 并行实现
- **多任务**：`/gen-variant-code 2c181942 67e490f4 --backend=c` → 分别读取 `ceshi/des/2c181942.txt` 与 `ceshi/des/67e490f4.txt`，对所有题号并行启动 Claude subagent

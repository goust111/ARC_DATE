---
name: codex-generation-supervisor
description: 调度并检查 Codex 批量生成 ARC 题目代码。Use when Claude Code receives many task IDs and must plan Codex parallelism, invoke gen-combo-code or gen-variant-code, monitor failures and quota limits, enforce required review steps, and produce a final progress summary.
---

# Codex Generation Supervisor

## 任务

当用户给出一批题号并要求生成代码时，由 Claude Code 负责总体调度和检查；实际生成由 Codex 执行。Claude Code 必须合理规划并发、启动生成、持续监测、处理异常、检查 Codex 是否完整执行指定 skill 的全部步骤，并在全部完成或达到额度限制时输出汇总。

## 输入参数

- `task_ids`：用户给出的题号列表，可包含多个题号，例如 40 个。
- `mode`：生成类型。
  - `code`：调用 `/home/lixinyu/snap/new/skill/BARC/.claude/skills/gen-variant-code`
  - `combo`：调用 `/home/lixinyu/snap/new/skill/BARC/.claude/skills/gen-combo-code`
- `backend`：Codex 后端配置，可选。默认使用 `o`。

如果用户没有明确给出 `mode`，先询问用户使用 `code` 还是 `combo`，不要猜测。

## 固定工作目录

所有生成和监测操作都在以下目录执行：

```bash
cd /home/lixinyu/snap/new
```

不要在其他目录启动 Codex。启动前确认当前路径是 `/home/lixinyu/snap/new`。

## 调度规划

1. 统计 `task_ids` 数量，并把任务拆成可监测的小批次。
2. 根据任务数量、Codex 额度、机器负载和历史失败情况决定并发数。
3. 如果题号很多，不要一次性无约束启动全部 Codex；优先分批启动，完成一批、检查一批、再继续下一批。
4. 可使用 Claude subagent 辅助计算批次划分、并发数量和进度汇总，但最终调度决策由 Claude Code 负责。
5. 每个 Codex 任务都要记录：
   - 题号
   - `mode`
   - `backend`
   - 启动时间
   - 当前状态：`pending`、`running`、`reviewing`、`needs-fix`、`completed`、`failed`、`quota-blocked`
   - 输出位置或日志位置
   - 异常和处理记录

如果任务数 `>= 3`，启动前必须创建批量进度文件：

```bash
/home/lixinyu/snap/new/tem/batch_gen_<YYYYMMDD_HHMM>.md
```

批量进度文件记录每个 `task_id × backend` 的状态，初始为 `pending`。每收到一个完成通知，立即更新对应条目并报告当前进度，例如 `35/80 done`。

## 执行模型选择

生成侧优先使用 `codex exec`。

并发、额度、失败恢复、漏审补做由 Claude Code 层统一管理。需要估算批次、汇总失败原因或判断下一批并发数时，可使用 Claude subagent 辅助计算。

## Codex 后端参数

| backend | MODEL | EFFORT |
|---|---|---|
| `o` | `gpt-5.4` | `high` |
| `ox` | `gpt-5.4` | `xhigh` |
| `o5` | `gpt-5.5` | `high` |
| `o5x` | `gpt-5.5` | `xhigh` |

启动 `codex exec` 时必须根据 `backend` 映射设置 model 和 reasoning effort。用户未指定时使用 `o`。

## 运行载体选择

默认由 Claude Code 直接启动并监控多个 `codex exec` 进程，分别保存日志和状态。

只有在以下情况才使用 tmux：

- 预计单批运行很久，Claude Code 会话或终端可能中断。
- 需要保留可手动查看的长期会话。
- 并发任务较多，必须把每个 Codex 的输出隔离到独立 pane/window。

如果使用 tmux，仍然必须由 Claude Code 维护任务状态表并检查日志。不要因为放入 tmux 就减少监测、审查或额度检查。

## 后台启动规则

并行启动多个 Codex 时，每个 Codex 必须使用独立的 Bash 工具调用，并设置 `run_in_background: true`。不要在同一个 shell 脚本里用 `&` 分叉多个 Codex。

每个后台调用只负责一个可追踪的 Codex 执行单元。具体生成步骤完全遵循 `mode` 对应的生成 skill，不在本 skill 里重新定义生成流程。

## 监测循环

Claude Code 必须持续监测每个 Codex 的执行情况，直到任务完成、失败或被额度阻塞。

Codex 负责按被调用 skill 完成生成、测试、审查和必要修复。Claude Code 不在 Codex 进程完成前用产物文件判断完成，也不提前做汇总或补审查判断。必须等独立后台调用的完成通知返回，确认该 Codex 完全退出后，才核对日志、产物和审查证据。

每轮监测至少检查：

1. Codex 进程是否仍在运行。
2. 日志是否持续输出。
3. 是否出现异常、报错、超时、卡死、权限问题、模型额度限制或上下文问题。
4. 收到完成通知后，是否已经生成预期文件。
5. 是否进入并完成对应生成 skill 要求的审查阶段。
6. Codex 是否只是声称完成，但实际漏掉测试、审查或修复步骤。

如果 Codex 仍在运行但长时间停止输出，不要直接判定完成。先要求当前 Codex 继续监测并输出当前进度；若仍无进展，再根据日志判断是否需要重启或转为故障处理。

## 批量状态规则

批量进度只能由完成通知驱动：

- 仅在收到 Claude Agent 完成通知或 Codex `run_in_background` 完成通知时，才把对应条目从 `pending` 改为 `done` 或 `failed`。
- 不要用文件系统查询，例如 `ls`、`wc -l`、`stat`，来判定任务完成状态。
- 不要因为 attempt 文件、提示词文件或 `code.py` 存在就判定完成；这些文件可能是中间版本。
- 每收到一个通知，立即更新 batch 文件并报告当前进度。

汇总前置条件：

- batch 文件中所有条目都已经是 `done` 或 `failed`，才可以开始汇总。
- 汇总时的完成、失败和中断统计以 batch 文件为准。
- 不要另起文件系统扫描提前汇总。

## 故障处理

当某个 Codex 任务异常时，按以下顺序处理：

1. 读取该任务日志，判断失败类型。
2. 如果是暂时性问题或输出中断，尝试让 Codex 继续执行并补充进度。
3. 如果是进程退出但任务未完成，重启该题号的生成任务。
4. 如果是已生成代码但审查或测试缺失，要求当前 Codex 补做缺失步骤；不要重新生成已完成部分，除非代码不可用。
5. 如果是提示词、输入文件或路径缺失，记录为 `failed`，说明缺失文件和题号。
6. 如果是额度限制，进入“额度限制处理”。

每次重启或补做都要记录原因和结果。不要无限重试；同一题号连续失败多次后，标记为 `failed` 并在汇总中说明。

## 必查项

Claude Code 必须检查 Codex 是否完全按照被调用 skill 的要求执行。

### code 模式

检查 `gen-variant-code` 中要求的生成、测试和 parent 侧审查是否完成。特别注意：

- 是否使用 Codex 后端，而不是本地手写或其他后端。
- 是否读取完整提示词。
- 是否生成对应 `code.py`。
- 是否完成提示词要求的审查步骤。
- 是否存在漏审、只看汇总、不逐题检查、未测试却声称完成的情况。

### combo 模式

检查 `gen-combo-code` 中要求的生成、测试和六项审查是否完成。特别注意：

- 是否读取 YAML 中的组合信息。
- 是否为每个组合生成对应提示词和 `code.py`。
- 是否完成六项审查，不能只生成代码后直接结束。
- 是否检查融合一致性、ARC 可解性和其他必需审查项。

如果发现 Codex 漏掉任何必查项，必须要求它补做；补做完成前，该题号不能标记为 `completed`。

## 额度限制处理

需要持续关注 Codex 的 5 小时额度限制和一周额度限制。

如果接近或达到额度限制：

1. 停止启动新的 Codex 任务。
2. 允许已接近完成的任务在不违反额度限制的前提下收尾。
3. 记录已完成题号。
4. 记录正在进行但未完成的题号，并说明当前阶段。
5. 记录尚未开始的题号。
6. 输出当前处理到哪个题号、哪些题号完成、哪些题号中断。
7. 如果额度已经完全阻塞，立即进入最终汇总，不要继续重试。

额度相关信息必须清楚写入汇总，方便用户之后从中断位置继续。

## 最终汇总

当所有题号处理完，或因为额度/故障必须停止时，输出汇总。汇总必须包含：

- 本次 `mode`：`code` 或 `combo`
- 本次 `backend`
- 总题号数
- 已完成题号
- 进行到一半但未完成的题号及当前阶段
- 未开始题号
- 失败题号及失败原因
- 额度状态：正常、接近 5 小时限制、达到 5 小时限制、接近周限制、达到周限制
- 是否存在被要求补做审查的题号
- 可继续执行的下一批题号建议

## 输出模板

```markdown
## Codex 生成监测汇总

- 模式：code|combo
- 总数：N
- 已完成：...
- 未完成但已开始：...
- 未开始：...
- 失败：...
- 额度状态：...
- 补做审查：...

## 当前进度

已处理到：<题号或批次>

## 后续建议

下一批建议处理：...
```

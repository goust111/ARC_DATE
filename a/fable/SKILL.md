---
name: arc-prompt-fable
description: 用 multi 模板从 ARC_code（eval+train 全量）生成一轮 ARC 题目描述提示词：每个 seed 把 350 个候选题打乱后按 20 题一组切分成 17 个 prompt，再用有限并发的子代理逐个执行并保存结果。
---

# ARC 提示词全覆盖单轮生成（fable 版）

## 适用场景

Claude Code、Codex、Cursor 或其他代码代理都可以按下面流程执行；如果当前环境有"子代理/并行任务"能力，就按有限并发执行 prompt，否则按顺序执行。

## 参数约定

本 skill 的来源、模板、分组方式全部固定，唯一可选参数是 seed：

- 输出根目录统一为 `/home/lixinyu/snap/a/code/fable/prompt`。
- 来源固定为 `--source both`（eval + train 全量候选，约 350 题）。
- 提示词模板固定为 `/home/lixinyu/snap/a/code/fable/prompt_multi_layer_zh.md`（多层结构版，含取材绑定与五项审查要求）。
- 采样策略固定为默认的 `shuffle_partition`：把全部候选题打乱一次，按 20 题一组切分，组内和组间不重复——一轮内 17 个 prompt 的参考题互不重叠，350 个候选生成 `floor(350 / 20) = 17` 个 prompt，覆盖 340 题，本轮剩余 10 题跳过；换 seed 会重新打乱。
- 用户指定 seed：作为 `--rng_seed`。同一个 seed 生成完全相同的 17 组分组。
- 用户未指定 seed：随机生成一个，并追加写入输出根目录下的 `round_seeds.txt`（若已存在则保留原内容继续追加），方便复现。
- 一次最多启动的子代理数量由 `MAX_AGENTS` 控制，MAX_AGENTS 等于 6。

## 总体流程

1. 确定本轮 `SEED`（用户指定或随机生成）。
2. 运行提示词生成脚本，得到一轮 17 个 prompt 文件夹。
3. 每个文件夹必须包含：
   - `prompt.md`：给模型执行的完整提示词。
   - `des.md`：保存模型生成的新题描述。
   - `ids.txt`：保存本组抽取的 20 个参考题号。
4. 读取本轮所有 `prompt.md`。
5. 使用当前工具环境中的子代理/并行任务机制，按 `MAX_AGENTS` 限制分批执行 prompt。
6. 将每个执行结果完整写入对应文件夹的 `des.md`。
7. 最后检查所有生成目录中的 `des.md` 均非空。

## 第一步：生成 prompt

```bash
OUT_ROOT=/home/lixinyu/snap/a/code/fable/prompt
TEMPLATE=/home/lixinyu/snap/a/code/fable/prompt_multi_layer_zh.md

mkdir -p "$OUT_ROOT"
# 用户指定了 seed 就直接 SEED=<值>；否则随机生成并记录
SEED=$(python -c "import random; print(random.randrange(1, 2_147_483_647))")
printf "%s\n" "$SEED" >> "$OUT_ROOT/round_seeds.txt"

python /home/lixinyu/snap/a/code/generate_prompt_from_arc_code_comments.py \
  --source both \
  --num_descriptions 20 \
  --num_generations 5 \
  --rng_seed "$SEED" \
  --run_name "seed${SEED}" \
  --prompt_template "$TEMPLATE" \
  --out_dir "$OUT_ROOT"
```

不传 `--batch_size`，脚本在 `shuffle_partition` 下自动取满 17 组。

输出位置示例：

```text
/home/lixinyu/snap/a/code/fable/prompt/seed1/1/
/home/lixinyu/snap/a/code/fable/prompt/seed1/2/
...
/home/lixinyu/snap/a/code/fable/prompt/seed1/17/
```

## 第二步：执行本轮 prompt

需要执行的文件是：

```text
/home/lixinyu/snap/a/code/fable/prompt/seed<SEED>/*/prompt.md
```

如果当前环境支持子代理并行，按 `MAX_AGENTS=6` 分批执行：先启动前 6 个 prompt，完成并写入 `des.md` 后再执行下一批。不要一次性启动过多 agent。

给每个子任务或并行代理的指令统一使用：

```text
请严格执行下面的 ARC 题目描述生成提示词，注意生成中文描述。

[粘贴对应 prompt.md 的完整内容]
```

每个任务完成后，将完整输出写入对应文件夹的 `des.md`。注意：md 文件中只记录新题描述（`<new_puzzle>` 块），不记录其他内容，中文记录。

## 最终检查

回复用户前检查：

- 本轮 17 个文件夹（`1` 到 `17`）都存在。
- 所有 `prompt.md` 都非空。
- 所有 `ids.txt` 都包含 20 个题号，且 17 个 `ids.txt` 之间题号互不重复。
- 所有 `des.md` 都包含模型输出，且每道题带有 `source` 字段。
- 不应该生成 `.jsonl` manifest。

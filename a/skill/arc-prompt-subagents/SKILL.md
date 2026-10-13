---
name: arc-prompt-subagents
description: 从 ARC_code 注释批量生成 ARC 题目描述提示词，可用 -a 控制是否加入 annotation；每个 seed 会把候选题打乱后按 20 题一组切分，再用有限并发的子代理逐个执行并保存结果。
---

# ARC 提示词批量生成

## 适用场景

Claude Code、Codex、Cursor 或其他代码代理都可以按下面流程执行；如果当前环境有“子代理/并行任务”能力，就按有限并发执行 prompt，否则按顺序执行。

## 参数约定

- 输出根目录统一为 `/home/lixinyu/snap/a/des`。
- 用户传入 `-a`：使用 annotation。
- 用户不传 `-a`：默认不使用 annotation。
- 用户指定 `eval`、`train` 或 `both`：作为 `--source`；未指定时默认 `both`。
- 用户指定分类 `1`-`5`：作为 `--type_id`，只从该类别候选题中生成 prompt；未指定时从全部类别中生成。
- 用户指定 seed：作为 `--rng_seed`。同一个 seed 和同一个候选题池会生成相同分组；不同 seed 会生成不同打乱顺序。
- 用户指定轮数但没有指定 seed 列表：自动随机生成对应数量的不重复 seed，并保存到输出根目录下的 `round_seeds.txt`，注意若txt已存在，继续写入，保留原内容，方便复现。
- 默认采样策略是 `shuffle_partition`：每个 seed 先把全部候选题打乱一次，再按 20 个题一组切分，组内和组间不重复；不足 20 个的尾部题目本轮跳过。
- 如果候选题是 332 个，默认会生成 `floor(332 / 20) = 16` 个 prompt，覆盖 320 个题，跳过本轮最后 12 个题；下一个 seed 会重新打乱。
- 一次最多启动的子代理数量由 `MAX_AGENTS` 控制，MAX_AGENTS等于6。

分类编号对应：

```text
1 = move_along_path
2 = transform_in_place
3 = move_to_match
4 = symmetry/pattern
5 = move_in_order
```

## 总体流程

1. 根据用户指定的 seed 或轮数准备 `SEEDS`。
2. 对每个 `SEED` 运行提示词生成脚本，生成一轮 prompt 文件夹。
3. 每个文件夹必须包含：
   - `prompt.md`：给模型执行的完整提示词。
   - `des.md`：保存模型生成的新题描述。
   - `ids.txt`：保存本次抽取的 20 个参考题号。
4. 读取每轮所有 `prompt.md`。
5. 使用 Claude Code、Codex 或当前工具环境中的子代理/并行任务机制，按 `MAX_AGENTS` 限制分批执行 prompt。
6. 将每个执行结果完整写入对应文件夹的 `des.md`。
7. 最后检查所有生成目录中的 `des.md` 均非空。

## 第一步：生成 prompt

将 `SOURCE` 替换为 `eval`、`train` 或 `both`；如果用户没有指定，使用 `both`。

如果用户指定分类，将 `TYPE_ARGS` 设置为 `--type_id <编号>`；如果用户没有指定分类，设置为空字符串。

如果用户指定具体 seed 列表，将 `SEEDS` 设置为这些 seed，例如 `SEEDS="1 2 3"`。

如果用户只指定轮数，将 `NUM_ROUNDS` 设置为轮数，并自动随机生成不重复 seed。

`RUN_NAME` 建议包含 seed，例如 `seed1`、`seed2`。每一轮会创建一个 `RUN_NAME` 目录，并在里面用 `1`、`2`、... 表示本轮不同 prompt。

### 不使用 annotation

```bash
SOURCE=both
NUM_ROUNDS=3
TYPE_ARGS=""
OUT_ROOT=/home/lixinyu/snap/a/des

mkdir -p "$OUT_ROOT"
SEEDS=$(python - "$NUM_ROUNDS" <<'PY'
import random
import sys

num_rounds = int(sys.argv[1])
print(" ".join(map(str, random.sample(range(1, 2_147_483_647), num_rounds))))
PY
)
printf "%s\n" $SEEDS > "$OUT_ROOT/round_seeds.txt"

for SEED in $SEEDS; do
  RUN_NAME="seed${SEED}"
  python /home/lixinyu/snap/a/code/generate_prompt_from_arc_code_comments.py \
    --source "$SOURCE" \
    --num_descriptions 20 \
    --num_generations 5 \
    --rng_seed "$SEED" \
    --run_name "$RUN_NAME" \
    $TYPE_ARGS
done
```

输出位置示例：

```text
/home/lixinyu/snap/a/des/seed1/1/
/home/lixinyu/snap/a/des/seed1/2/
...
```

### 使用 annotation（用户传 `-a`）

```bash
SOURCE=both
NUM_ROUNDS=3
TYPE_ARGS=""
OUT_ROOT=/home/lixinyu/snap/a/des

mkdir -p "$OUT_ROOT"
SEEDS=$(python - "$NUM_ROUNDS" <<'PY'
import random
import sys

num_rounds = int(sys.argv[1])
print(" ".join(map(str, random.sample(range(1, 2_147_483_647), num_rounds))))
PY
)
printf "%s\n" $SEEDS > "$OUT_ROOT/round_seeds.txt"

for SEED in $SEEDS; do
  RUN_NAME="seed${SEED}"
  python /home/lixinyu/snap/a/code/generate_prompt_from_arc_code_comments.py \
    --source "$SOURCE" \
    --num_descriptions 20 \
    --num_generations 5 \
    --rng_seed "$SEED" \
    --run_name "$RUN_NAME" \
    -a \
    $TYPE_ARGS
done
```

输出位置示例：

```text
/home/lixinyu/snap/a/des/seed1/1/
/home/lixinyu/snap/a/des/seed1/2/
...
```

## 第二步：执行本轮 prompt

输出根目录为 `/home/lixinyu/snap/a/des`。

需要执行的文件是所有待执行轮次中的：

```text
<输出根目录>/<RUN_NAME>/*/prompt.md
```

如果当前环境支持子代理并行，按 `MAX_AGENTS` 分批执行。例如 `MAX_AGENTS=6` 时，先执行前三个 prompt，完成并写入 `des.md` 后再执行下一批。不要一次性启动过多 agent。

给每个 Claude Code/Codex 子任务或并行代理的指令统一使用：

```text
请严格执行下面的 ARC 题目描述生成提示词，注意生成中文描述。

[粘贴对应 prompt.md 的完整内容]
```

每个任务完成后，将完整输出写入对应的，注意：md文件中只记录新题描述，不记录其他，中文记录：

```text
<输出根目录>/<RUN_NAME>/*/des.md
```

## 最终检查

回复用户前检查：

- 所有待执行轮次的文件夹都存在。
- 所有待执行轮次的 `prompt.md` 都非空。
- 所有待执行轮次的 `ids.txt` 都包含 20 个题号。
- 所有待执行轮次的 `des.md` 都包含模型输出。
- 不应该生成 `.jsonl` manifest。

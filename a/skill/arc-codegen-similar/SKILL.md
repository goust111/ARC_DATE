---
name: arc-codegen-similar
description: 从新题 des.md 生成 ARC 代码实现 prompts，先用 annotations/docstring/代码特征做相似机制重排选择参考题，再让代码代理实现 main() 和 generate_input()，并由 parent 做多轮随机验证与审查。
---

# ARC 相似机制代码生成

## 适用场景

当用户已经有新题描述目录，例如：

```text
/home/lixinyu/snap/a/input_no/opus/1/des.md
/home/lixinyu/snap/a/input_no/opus/2/des.md
```

并希望根据每个 `<new_puzzle index="...">` 自动生成对应 Python 代码文件：

```text
/home/lixinyu/snap/a/input_no/opus/1/1.py
/home/lixinyu/snap/a/input_no/opus/1/2.py
...
```

使用本 skill。它不是随机抽 20 个参考题，而是对每个新题单独选择机制相似参考题。

## 核心思路

1. 读取新题 `des.md`。
2. 解析每个 `<new_puzzle index="N">...</new_puzzle>`。
3. 从 `/home/lixinyu/snap/new/ARC_code` 收集旧题 `main()` docstring 和代码静态特征。
4. 从 `/home/lixinyu/snap/new/annotations` 读取结构化 annotation。
5. 对新题和旧题抽取机制标签：
   - `object:*`：对象结构，如 closed region、frame、path、template、marker。
   - `operation:*`：核心操作，如 flood fill、ray cast、shape match、rotate、translate、count、recolor。
   - `relation:*`：空间关系，如 inside、adjacent、nearest、aligned、between。
   - `output:*`：输出形态，如 same size、crop、single row。
   - `code:*`：代码特征，如 BFS、bbox、connected components、rotation、ray loop、random retry。
6. 按机制相似度、文本相似度、代码特征相似度综合打分，并用 MMR 减少重复参考题。
7. 为每道新题生成一个代码实现 prompt，prompt 中包含：
   - 新题描述。
   - 相似参考题路径、docstring、annotation、相似度细节。
   - 必读参考代码路径。
   - 每个参考题对应的输入生成 spec 路径和完整 YAML 内容。
   - `main()`/`generate_input()` 实现约束。
   - 随机多样性和反退化验证命令。

为避免每次重复扫描 `ARC_code`，可先构建参考索引：

```bash
python /home/lixinyu/snap/a/code/build_arc_reference_index.py \
  --source both \
  --out /home/lixinyu/snap/a/code/arc_reference_index.json
```

后续 prompt builder 通过 `--reference_index /home/lixinyu/snap/a/code/arc_reference_index.json` 直接读取该索引。

## 参数约定

- `DES_PATHS`：一个或多个 `des.md` 文件路径，由用户传入；不要在 skill 或脚本中固定死。
- `ROOT`：可选，包含若干新题文件夹的根目录；只有用户按根目录批量处理时才使用。
- `FOLDERS`：要处理的子文件夹，例如 `1 2`。
- `SOURCE`：参考题来源，`eval`、`train` 或 `both`，默认 `eval`。
- `SPEC_DIR`：参考题输入生成 spec 根目录，默认 `/home/lixinyu/snap/new/spce_dis`；目录内应为 `<dataset>/<puzzle_id>_spec.yaml`。spec 在相似题选定后才按 puzzle id 映射，不参与相似度计算；存在时注入完整 YAML，缺失时在 prompt 中明确标记。
- `TOP_K`：每道新题注入的相似参考题数量，默认 `8`。
- `MAX_AGENTS`：并行代码代理数量，默认 `5`；机器负载较高时降到 `3`，资源充足且任务较轻时可升到 `6`。
- `OUT_DIR`：可选，用户指定的产物存储根目录。传入后，所有生成产物（代码、`codegen_prompts/`、`similarity_reports/`、`similar_ids/`、manifest）写到 `<OUT_DIR>/<puzzle_dir 文件夹名>/` 下；不传则就地写在每个 des.md 旁边（旧行为）。输入 des.md 始终保持原位不动。
- 目标代码路径默认为 `<out_base>/<index>.py`，其中 `out_base` = 传了 `--out_dir` 时为 `<OUT_DIR>/<文件夹名>`，否则为 `<puzzle_dir>`。
- 如果历史流程必须无扩展名，可在脚本中传 `--code_suffix ''`，但默认使用 `.py`。

## 阶段一：生成相似参考 prompt

示例：由调用者显式传入新题描述路径，处理 `opus/1` 和 `opus/2`。

```bash
DES_PATHS="/home/lixinyu/snap/a/input_no/opus/1/des.md /home/lixinyu/snap/a/input_no/opus/2/des.md"
SOURCE=eval
TOP_K=8
OUT_DIR=/home/lixinyu/snap/a/output_no/opus   # 用户指定的产物存储根目录；省略 --out_dir 则就地写在 des.md 旁

python /home/lixinyu/snap/a/code/build_arc_codegen_prompts_with_similarity.py \
  $(printf -- ' --des_path %s' $DES_PATHS) \
  --reference_index /home/lixinyu/snap/a/code/arc_reference_index.json \
  --spec_dir /home/lixinyu/snap/new/spce_dis \
  --source "$SOURCE" \
  --top_k "$TOP_K" \
  --candidate_pool 40 \
  --mmr_lambda 0.18 \
  --out_dir "$OUT_DIR"
```

如果用户明确要求按根目录批量，也可以使用：

```bash
python /home/lixinyu/snap/a/code/build_arc_codegen_prompts_with_similarity.py \
  --root /home/lixinyu/snap/a/input_no/opus \
  --folders 1 2 \
  --reference_index /home/lixinyu/snap/a/code/arc_reference_index.json \
  --spec_dir /home/lixinyu/snap/new/spce_dis
```

每个新题目录会在 `out_base` 下生成（`out_base` = 传了 `--out_dir` 时为 `<OUT_DIR>/<文件夹名>`，否则为输入 `<puzzle_dir>`）：

```text
<out_base>/<index>.py
<out_base>/codegen_prompts/<index>.md
<out_base>/similarity_reports/<index>.json
<out_base>/similar_ids/<index>.txt
<out_base>/codegen_manifest.json
```

其中：

- `codegen_prompts/<index>.md`：交给代码代理执行的完整 prompt。
- `similarity_reports/<index>.json`：相似度细节和重排结果。
- `similar_ids/<index>.txt`：该新题选中的参考题 id。

## 阶段二：执行代码生成 prompt

读取所有（位于 `out_base` 下，即传了 `--out_dir` 时在 `<OUT_DIR>/<文件夹名>/codegen_prompts/`，否则在 des.md 旁）：

```text
<out_base>/codegen_prompts/*.md
```

对每个 prompt，启动代码代理执行。执行指令必须是：

```text
请严格执行下面的 ARC 新题代码生成提示词。必须把代码写入 prompt 指定的目标路径，并运行 prompt 里的自检命令。

[粘贴 codegen_prompts/<index>.md 的完整内容]
```

代码代理必须同时阅读每个参考题块中的 `<generation_spec>`。它只用于借鉴输入构造策略、参数范围、分支覆盖与拒绝条件；新题 Markdown 描述是唯一规则来源，参考 spec 不得覆盖或改写新题规则。

执行每个代码生成 prompt 时，还必须遵守以下语义一致性硬约束：

```text
1. 输出必须唯一可判定
   输入中的线索必须足以唯一确定目标对象、目标操作和结果。
   不允许依赖含糊解释，或让同一可见线索在不同语境下产生不同答案。

2. 组合步骤必须不可替代
   题目描述中的关键步骤都应对结果产生必要影响。
   不能让某一步只是表面存在，实际可被更简单的捷径完全替代。

3. 实现必须遵守题面定义的参照系
   所有判断、匹配、距离、对称、排序、分区等，都必须使用题面指定的对象范围和坐标参照。
   不得为了实现方便换成另一套参照系。

4. 生成样例要覆盖反例
   生成器要主动产生能排除错误捷径的样例。
   不只生成最容易通过的标准形态，还要覆盖边界情况、分支情况和易混淆情况。

5. 验证要检查语义而不只是自洽
   不能只验证 main(generate_input()) 能闭环。
   还要检查实现是否真的满足题面规则、关键依赖、唯一性和非退化性。
```

如果使用 Codex CLI，可用：

```bash
codex exec --skip-git-repo-check -c model='"gpt-5.4"' -c reasoning.effort='"high"' "$(cat <prompt_path>)"
```

多题时按 `MAX_AGENTS=5` 有限并发执行；不要一次性启动过多代理。

## 阶段三：parent 侧强制审查

代码代理完成后，parent 必须逐个题亲自检查，不能只相信代理输出。

随机验证之前，parent 必须先抽查每个 `main()` 的核心逻辑，判断是否存在：

- 使用了和题面不同的参照系。
- 某个描述中的关键步骤可以被直接跳过。
- 生成器和 `main()` 自洽但题意被弱化。
- 输入线索不足以唯一确定输出。

随机验证通过不代表题目通过；若发现语义偏差，即使 100/100 自检通过，也必须修改代码或标记失败。

## 审查清单

每道题必须满足：

- `main()` 的逻辑来自新题 `des.md`，不是旧题照搬。
- `main()` 对同类合法输入有泛化能力，不只识别一个固定模板。
- 输入和输出网格尺寸都符合 ARC 规范：任何一维都 ≤ 30（即不超过 30×30）；放大类规则（平铺、加边框、整数倍缩放）也要保证放大后的输出仍 ≤ 30×30。
- `generate_input()` 每次能随机生成多样样例，不固定死。
- 随机维度覆盖题目关键自由度：对象数量、位置、颜色、方向、尺寸、分支或形状参数。
- 生成器保证不重叠、不越界、标记唯一、匹配关系无歧义。
- 如果题目有多个分支（分支即规则语义中的不同情况），生成器应让各分支出现概率平均，一些较为难发现规律的分支可以占比多一些，一个输入输出网格对中可以包含多个分支示例。
- 禁止用颜色 id 、颜色英文拼写等做人类不可见的排序、算术或隐藏映射。
- 颜色变化必须有结构线索，能从 2-4 组样例中推断。
- 可视化清晰，避免对象粘连、覆盖混乱、过密不可读。
- 参考题只用于实现套路、约束处理和生成器质量校准，不能替代新题规则。

发现问题时，parent 直接修改目标代码并重新运行阶段三验证。

## 输出汇总

最终向用户报告：

```text
| 文件夹 | 题号 | 目标文件 | 相似参考数 | 随机验证 | 状态 |
|--------|------|----------|------------|----------|------|
| opus/1 | 1 | `<out_base>/1.py` | 8 | 100/100, variants=... | 通过 |
```

如果有失败项，说明失败原因、已修复内容，或明确标记未完成。

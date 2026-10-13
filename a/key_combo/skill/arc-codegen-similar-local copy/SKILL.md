---
name: arc-codegen-fable-local
description: key_combo fable 本地代码生成。从 /home/lixinyu/snap/a/key_combo/des/fable 下的新题 Markdown 生成 ARC 代码实现 prompts；有标签/主要机制/来源原语时优先使用，没有时从 docstring、输入结构、规则和关键洞察抽取机制信号；参考题组只作弱提示，禁止照搬旧题规则。
---

# key_combo fable 代码生成

## 适用场景

当用户已经有 key_combo fable 新题描述文件或目录：

```text
/home/lixinyu/snap/a/key_combo/des/fable/1/017.md
/home/lixinyu/snap/a/key_combo/des/fable/1/
```

并希望根据每个 `## 新题 N` 自动生成对应 Python 代码文件：

```text
/home/lixinyu/snap/a/key_combo/code/fable/1/017/1.py
/home/lixinyu/snap/a/key_combo/code/fable/1/017/2.py
...
```


## 核心思路

1. 读取 key_combo fable 新题 `.md` 文件，如 `001.md`、`017.md`。
2. 按 `## 新题 N` 分块。
3. md 文件名作为输出文件夹名，例如 `017.md` 的“新题 1/2”写为 `017/1.py`、`017/2.py`。
4. 从 `/home/lixinyu/snap/new/ARC_code` 收集旧题 `main()` docstring 和代码静态特征。
5. 从 `/home/lixinyu/snap/new/annotations` 读取结构化 annotation。
6. 对新题和旧题抽取机制标签。每个 `## 新题 N` 必须独立处理，禁止混入同一 Markdown 中其他题目的 tags 或显式参考题。对 fable 新题按以下优先级处理：
   - 若当前题块已有 `tags:`，逐条原样保留并直接使用，不再运行关键词标签提取。
   - 只有当前题块没有 `tags:` 时，才从该题自己的 `标签`、`输入结构`、`规则`、`关键洞察`、`来源原语`、`组合逻辑`、`依赖链` 和 `主要机制` 中回退提取对象、操作、关系、输出形态和代码特征。
   - 若当前题正文显式提到具体 `puzzle_id`，该题作为高权重参考；同文件其他题块的 id 不得进入当前题。
   - 文件头的 `参考题组` 只作为各题共享的低权重提示，不视为该新题的规则来源。
   - `object:*`：对象结构，如 closed region、frame、path、template、marker。
   - `operation:*`：核心操作，如 flood fill、ray cast、shape match、rotate、translate、count、recolor。
   - `relation:*`：空间关系，如 inside、adjacent、nearest、aligned、between。
   - `output:*`：输出形态，如 same size、crop、single row。
   - `code:*`：代码特征，如 BFS、bbox、connected components、rotation、ray loop、random retry。
7. 按机制相似度、文本相似度、代码特征相似度综合打分，并用 MMR 减少重复参考题。若文件头写了 `参考题组: /home/lixinyu/snap/a/key_combo/select/.../017.md`，脚本会读取该参考题组文件里的 `puzzle_id`，但只给低权重加成，避免直接照搬。
8. 为每道新题生成一个代码实现 prompt，prompt 中包含：
   - 新题描述。
   - 相似参考题路径、docstring、annotation、相似度细节。
   - 相似参考题在 `/home/lixinyu/snap/new/spce_dis` 中对应的输入生成 YAML spec；用于参考 `generate_input()` 的输入分布、构造步骤、分支覆盖、随机范围和拒绝条件。
   - 必读参考代码路径。
   - 公共参考工具箱 `/home/lixinyu/snap/new/solvers/arc_utils.py`。
   - `main()`/`generate_input()` 实现约束。
   - 随机多样性和反退化验证命令。

参考题信号必须分层：当前题块明确写出的参考题 id 属于高权重显式参考；Markdown 文件头引用的参考题组只属于低权重共享提示；同文件其他题块内的 tags 和参考题 id 必须完全隔离。

已有参考题信息直接从 `/home/lixinyu/snap/a/code` 下的 JSON 读取，不再扫描代码目录重算：

- `tags`：来自 `/home/lixinyu/snap/a/code/arc1_full_plus_arc2_train_index.json`
- `docstring` / `annotation` / `tokens` / `code_features`：来自 `/home/lixinyu/snap/a/code/arc1_reference_index.json` 和 `/home/lixinyu/snap/a/code/arc_reference_index.json`

## 参数约定

- `DES_PATHS`：一个或多个 key_combo fable 新题 Markdown 文件路径，如 `001.md`、`017.md`。
- `ROOT`：可选，包含若干新题文件夹的根目录；例如 `/home/lixinyu/snap/a/key_combo/des/fable`。脚本会自动处理子目录中的 `*.md` 新题文件。
- `FOLDERS`：要处理的子文件夹，例如 `1 2`。
- `SOURCE`：参考题来源，`eval`、`train` 或 `both`，默认 `both`；fable 新题显式参考题可能来自 train 或 eval，通常用 `both`。
- `TAG_INDEX`：已有题 tags 来源，固定使用 `/home/lixinyu/snap/a/code/arc1_full_plus_arc2_train_index.json`。
- `REFERENCE_JSON`：已有题 docstring / annotation / tokens / code_features 来源，固定使用 `/home/lixinyu/snap/a/code/arc1_reference_index.json` 和 `/home/lixinyu/snap/a/code/arc_reference_index.json`。
- `SPEC_DIR`：参考题输入生成 spec 根目录，默认 `/home/lixinyu/snap/new/spce_dis`；脚本在相似题选定后按 `<dataset>/<puzzle_id>_spec.yaml` 自动映射。spec 不参与相似度计算；存在时注入完整 YAML，缺失时在 prompt 中明确标记。
- `TOP_K`：每道新题注入的相似参考题数量，默认 `8`。
- `MAX_AGENTS`：并行代码代理数量，默认 `5`；机器负载较高时降到 `3`，资源充足且任务较轻时可升到 `6`。
- `OUT_DIR`：必须传入，key_combo fable 使用 `/home/lixinyu/snap/a/key_combo/code/fable`。所有生成产物（代码、`codegen_prompts/`、`similarity_reports/`、`similar_ids/`、manifest）写到 `<OUT_DIR>/<轮次>/<md文件名不含后缀>/` 下；例如输入 `/home/lixinyu/snap/a/key_combo/des/fable/1/017.md` 时，实际输出到 `/home/lixinyu/snap/a/key_combo/code/fable/1/017`。
- 目标代码路径为 `<out_base>/<index>.py`。
- 如果历史流程必须无扩展名，可在脚本中传 `--code_suffix ''`，但默认使用 `.py`。

## 阶段一：生成相似参考 prompt

示例：由调用者显式传入 key_combo fable 新题描述路径，处理 `fable/1`。

```bash
DES_PATHS="/home/lixinyu/snap/a/key_combo/des/fable/1/017.md"
SOURCE=both
TOP_K=8
OUT_DIR=/home/lixinyu/snap/a/key_combo/code/fable   # 实际输出到 /home/lixinyu/snap/a/key_combo/code/fable/1/017

python "/home/lixinyu/snap/a/key_combo/skill/arc-codegen-similar-local copy/scripts/build_arc_codegen_prompts_with_similarity.py" \
  $(printf -- ' --des_path %s' $DES_PATHS) \
  --tag_index /home/lixinyu/snap/a/code/arc1_full_plus_arc2_train_index.json \
  --reference_json /home/lixinyu/snap/a/code/arc1_reference_index.json \
  --reference_json /home/lixinyu/snap/a/code/arc_reference_index.json \
  --spec_dir /home/lixinyu/snap/new/spce_dis \
  --source "$SOURCE" \
  --top_k "$TOP_K" \
  --candidate_pool 40 \
  --mmr_lambda 0.18 \
  --out_dir "$OUT_DIR"
```

如果用户明确要求按根目录批量，也可以使用：

```bash
python "/home/lixinyu/snap/a/key_combo/skill/arc-codegen-similar-local copy/scripts/build_arc_codegen_prompts_with_similarity.py" \
  --root /home/lixinyu/snap/a/key_combo/des/fable \
  --folders 1 2 \
  --tag_index /home/lixinyu/snap/a/code/arc1_full_plus_arc2_train_index.json \
  --reference_json /home/lixinyu/snap/a/code/arc1_reference_index.json \
  --reference_json /home/lixinyu/snap/a/code/arc_reference_index.json \
  --spec_dir /home/lixinyu/snap/new/spce_dis \
  --out_dir /home/lixinyu/snap/a/key_combo/code/fable
```

每个新题文件会在 `out_base` 下生成，`out_base` = `<OUT_DIR>/<父目录名>/<md文件名不含后缀>`：

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

读取所有位于 `out_base` 下的 prompt：

```text
<out_base>/codegen_prompts/*.md
```

对每个 prompt，启动代码代理执行。执行指令必须是：

```text
请严格执行下面的 ARC 新题代码生成提示词。必须把代码写入 prompt 指定的目标路径，并运行 prompt 里的自检命令。生成代码前必须把 `/home/lixinyu/snap/new/solvers/arc_utils.py` 作为公共参考工具箱快速阅读，并阅读 prompt 中各参考题的 `<generation_spec>`；spec 只用于借鉴输入生成策略、随机维度、分支覆盖和拒绝条件，不能覆盖新题规则。若直接 import 工具箱，必须保证目标文件可被 `SourceFileLoader` 直接加载。

[粘贴 codegen_prompts/<index>.md 的完整内容]
```

执行每个代码生成 prompt 时，还必须遵守以下语义一致性硬约束：

```text
1. 输出必须唯一可判定
   输入中的线索必须足以唯一确定目标对象、目标操作和结果。
   不允许依赖含糊解释，或让同一可见线索在不同语境下产生不同答案。
   注意不可多个不同样例的输入网格处理后的输出网格都相同

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

parent 必须检查每个 `main()` 的核心逻辑，判断是否存在：

- 使用了和题面不同的参照系。
- 某个描述中的关键步骤可以被直接跳过。
- 生成器和 `main()` 自洽但题意被弱化。
- 输入线索不足以唯一确定输出。
- 不同样例输入网格处理后输出相同

代理自检通过不代表题目通过；若发现语义偏差，即使 100/100 自检通过，也必须修改代码或标记失败。

## 审查清单

每道题必须满足：

- 不可为了方便实现而偷懒，从而简化新题的代码实现
- `main()` 的逻辑来自新题 Markdown，不是旧题照搬。
- `main()` 对同类合法输入有泛化能力，不只识别一个固定模板。
- 输入和输出网格尺寸都符合 ARC 规范：任何一维都 ≤ 30（即不超过 30×30）；放大类规则（平铺、加边框、整数倍缩放）也要保证放大后的输出仍 ≤ 30×30。
- `generate_input()` 每次能随机生成多样样例，不固定死。
- 若题目描述中提及若干，generate_input()中需保证输入网格中的object为多数而不是单独
- 若题目描述中提及最短路径，则输入网格中需保证存在多条路径对选择造成干扰，确保最短路径的存在。
- 随机维度覆盖题目关键自由度：对象数量、位置、颜色、方向、尺寸、分支或形状参数。
- 生成器保证不重叠、不越界、标记唯一、匹配关系无歧义。
- 如果题目有多个分支（分支即规则语义中的不同情况），生成器应让各分支出现概率平均，一些较为难发现规律的分支可以占比多一些，一个输入输出网格对中可以包含多个分支示例。
- 禁止用颜色 id 、颜色英文拼写等做人类不可见的排序、算术或隐藏映射。
- 颜色变化必须有结构线索，能从 2-4 组样例中推断。
- 可视化清晰，避免对象粘连、覆盖混乱、过密不可读。
- 参考题只用于实现套路、约束处理和生成器质量校准，不能替代新题规则。
- 如果某个 fable 文件没有标签、主要机制或参考题，仍然必须只根据新题题面本身生成代码；不要臆造来源原语，也不要把相似参考题当作新题规则。
- 生成代码时把 `/home/lixinyu/snap/new/solvers/arc_utils.py` 作为公共参考工具箱；可直接 import 或摘取必要函数，但目标代码必须保持可独立加载、自检可运行。

发现问题时，parent 直接修改目标代码，重新执行对应 prompt 中的自检，并再次进行语义审查。

## 输出汇总

最终向用户报告：

```text
| 文件夹 | 题号 | 目标文件 | 相似参考数 | 随机验证 | 状态 |
|--------|------|----------|------------|----------|------|
| fable/1/017 | 1 | `<out_base>/1.py` | 8 | 100/100, variants=... | 通过 |
```

如果有失败项，说明失败原因、已修复内容，或明确标记未完成。

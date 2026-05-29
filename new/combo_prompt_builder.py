"""
Build a prompt for Codex to implement a fused combo variant of an ARC-AGI-2 task.

Usage:
    python combo_prompt_builder.py <task_id> <index> [output_path]

Example:
    python combo_prompt_builder.py 2c181942 1
    # → writes prompt to ceshi/tem3/combo_prompt_2c181942_1.txt

Reads the combo YAML at ceshi/des_combo/{task_id}.yaml, locates the combo by
index, extracts each component dim/alt description from the dimension analysis
YAML, attaches the original solver, and — for any component that already has a
single-variant code.py — injects it as reference code.
"""
from __future__ import annotations
import ast
import re
import sys, textwrap, yaml
from pathlib import Path

BASE = Path(__file__).resolve().parent


def _sanitize_combo_yaml_text(raw: str) -> str:
    sanitized_lines = []
    pattern = re.compile(r'^(\s*[A-Za-z0-9_]+:\s*)"(.+)"\s*$')
    for line in raw.splitlines():
        m = pattern.match(line)
        if not m:
            sanitized_lines.append(line)
            continue
        prefix, body = m.groups()
        sanitized_body = body.replace("'", "''")
        sanitized_lines.append(f"{prefix}'{sanitized_body}'")
    return "\n".join(sanitized_lines) + ("\n" if raw.endswith("\n") else "")


def _load_yaml(path: Path):
    raw = path.read_text()
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return yaml.safe_load(_sanitize_combo_yaml_text(raw))


def _has_four_part_main_comment(solver_code: str) -> bool:
    try:
        module = ast.parse(solver_code)
    except SyntaxError:
        return False

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            doc = ast.get_docstring(node)
            if not doc:
                return False
            input_idx = doc.find("Input structure:")
            if input_idx == -1:
                return False
            intro = doc[:input_idx].strip()
            return bool(intro) and ("Rule:" in doc) and ("Key insight:" in doc)
    return False


def build_prompt(task_id: str, index: int) -> str:
    # ── Load combo YAML ───────────────────────────────────────────────
    combo_yaml_path = BASE / "ceshi" / "des_combo" / f"{task_id}.yaml"
    if not combo_yaml_path.exists():
        raise FileNotFoundError(
            f"Combo YAML not found: {combo_yaml_path}\n"
            f"Run /gen-combo-propose {task_id} first."
        )
    combo_data = _load_yaml(combo_yaml_path)

    combo = None
    for c in combo_data["combos"]:
        if c["index"] == index:
            combo = c
            break
    if combo is None:
        raise ValueError(f"Index {index} not found in {combo_yaml_path}")

    combo_spec = combo["combo_spec"]
    combo_name = combo["name"]
    combo_description = combo["description"]

    components = [part.split("/") for part in combo_spec.split("+")]

    # ── Load dimension analysis YAML ──────────────────────────────────
    dim_yaml_path = BASE / "ceshi" / "yaml" / f"dim_analysis_v9_{task_id}.yaml"
    if not dim_yaml_path.exists():
        candidates = sorted(BASE.glob(f"ceshi/yaml/dim_analysis_*_{task_id}.yaml"))
        dim_yaml_path = candidates[-1] if candidates else dim_yaml_path
    dim_data = _load_yaml(dim_yaml_path)

    dims_by_id = {d["id"]: d for d in dim_data["dimensions"]}

    # ── Build per-component descriptions ──────────────────────────────
    component_blocks = []
    for dim_id, alt_idx_str in components:
        alt_idx = int(alt_idx_str)
        dim = dims_by_id.get(dim_id)
        if dim is None:
            component_blocks.append(
                f"- **{dim_id}/{alt_idx_str}**: (维度未在 YAML 中找到)"
            )
            continue
        alt = dim["alternatives"][alt_idx]
        component_blocks.append(textwrap.dedent(f"""\
        - **{dim_id}/{alt_idx_str}** — {dim['name']} (`{dim['tag']}`)
          - 当前值: {dim['current']}
          - 改为: {alt['name']} — {alt['description']}
          - 来源: {alt.get('source', '?')}（参考任务: {alt.get('source_task', 'N/A')}）"""))

    component_section = "\n".join(component_blocks)

    # ── Load original solver code ─────────────────────────────────────
    solver_path = BASE / "arc2_solver_select" / f"{task_id}.py"
    solver_code = solver_path.read_text()
    has_four_part_main_comment = _has_four_part_main_comment(solver_code)

    # ── Collect existing single-variant reference code ────────────────
    ref_code_blocks = []
    for dim_id, alt_idx_str in components:
        code_path = BASE / "ceshi" / "code" / task_id / dim_id / alt_idx_str / "code.py"
        if code_path.exists():
            code_content = code_path.read_text()
            ref_code_blocks.append(textwrap.dedent(f"""\
            ### 参考代码: {dim_id}/{alt_idx_str}

            文件: `{code_path}`

            ```python
            {code_content}
            ```"""))

    ref_section = ""
    if ref_code_blocks:
        ref_section = (
            "\n\n## 已有单变体参考代码\n\n"
            "以下组成项已有单变体实现，可作为融合时的参考：\n\n"
            + "\n\n".join(ref_code_blocks)
        )

    # ── Load spec file if available ──────────────────────────────────
    spec_path = BASE / "spce_dis" / f"{task_id}_spec.yaml"
    spec_content = spec_path.read_text() if spec_path.exists() else ""

    spec_section = ""
    if spec_content:
        spec_section = textwrap.dedent(f"""\

    ## 原始任务规格（Spec）

    以下是从原始 solver 代码和示例中提炼的完整任务规格，包含精确的数值约束和生成策略。
    Spec 本身就是原题代码生成时的参考文件，因此：
    - 在**设计分析**阶段，参考 spec 理解原始任务的完整设计细节（对象结构、空间关系、生成策略等），以此为基础思考融合变体的改动如何落实
    - 在**实现**阶段，参考 spec 中的数值约束（网格尺寸、对象数量/尺寸、颜色数等），确保融合变体在物理上可实现

    ```yaml
    {spec_content.strip()}
    ```
    """)

    # ── Variant output path ───────────────────────────────────────────
    variant_dir = BASE / "ceshi" / "code_combo" / task_id / str(index)
    variant_path = variant_dir / "code.py"

    docstring_instruction = ""
    if has_four_part_main_comment:
        docstring_instruction = textwrap.dedent("""\

    额外要求：原始代码的 `main()` 已包含四段式说明注释/文档字符串（开头概述 + `Input structure:` + `Rule:` + `Key insight:`）。
    生成的新代码里也必须在 `main()` 上方保留这一结构，并根据**融合后的新规则**与**原注释提供的信息**重新编写；可以借鉴原注释的组织方式，但**不要照抄原文**，要让内容准确反映融合后的输入结构、规则和关键洞察。
    """)

    # ── Assemble prompt ───────────────────────────────────────────────
    prompt = textwrap.dedent(f"""\
    你要修改一个 ARC-AGI-2 任务的 solver 和 generator，实现一个**多维度融合变体**。

    ## 原始任务: {task_id}
    **摘要**: {dim_data['summary']}

    ## 要实现的融合组合
    - **组合名称**: {combo_name}
    - **combo_spec**: {combo_spec}

    ### 融合描述（实现规格）

    {combo_description}

    ### 各组成项的原始变体详情

    以下是 combo_spec 中每个组成项在维度分析 YAML 中的原始描述，帮助你理解每项改动的来源和细节：

    {component_section}

    ## 原始 solver + generator 代码

    文件: `{solver_path}`

    ```python
    {solver_code}
    ```
    {spec_section}{docstring_instruction}{ref_section}

    ## 你的任务

    1. **理解原代码**: 弄清 `main()` 和 `generate_input()` 的逻辑。

    2. **阅读参考代码**（如有）: 上面列出的已有单变体代码是各组成项单独实现的版本，帮助你理解每项变更的实现方式。融合版本应综合参考但不能简单拼接。

    3. **设计分析（写代码前必须完成）**:

       综合「融合描述」和「各组成项详情」，将融合规则落实到代码层面。融合描述给出了整体的 input→output 规则，各组成项详情给出了每项改动的具体含义——两者结合才是完整的实现规格。如果上面提供了原始任务规格（Spec），请将其作为设计分析的重要参考——spec 详细描述了原始任务中各对象的结构、空间关系、生成策略等，理解这些细节有助于你准确判断融合改动的影响范围和实现方式。

       a) **具体化与行为追踪**: 融合后的变体在 ARC 的像素网格中具体长什么样？用一个具体小例子追踪：原始逻辑在每个被改动的环节是怎么执行的 → 融合后应该怎么执行的 → 二者的差异是什么。

       b) **复杂度校准**: 检查你的设计是否过于简化。过于简化会导致两个问题：一是容易产生冲突或重复（例如不同对象变得难以区分，匹配出现歧义），二是变体的视觉效果不明显，肉眼看不出和原任务的区别。同时不要让关键自由度过多；如果规则依赖少数档位、类别或对应关系，应控制在少量取值，并让它们在每个样例中有重复支撑。

       c) **隐式假设审查**: 各组成项的改动可能破坏原代码中的哪些隐式假设？多项改动叠加后是否引入新的冲突？逐行检查 main() 中与变体相关的代码段：每行代码依赖了什么前提？这个前提在融合后是否仍然成立？

       d) **放置合法性**: 生成器在放置对象时，必须保证各个体之间不重叠、不越界，且放置结果符合题意。

       e) **视觉可观察性**: ARC 要求人类从 2-4 个 input→output 样例中肉眼推断规则。禁止使用任何人类不可视因素参与排序等操作（例如黑色=0，禁用）。做一次思想实验：假设你是解题者，看到几组变体生成的 input→output，能否看出融合后的规则？注意不要使用颜色值等人类不可视因素。如果变体的变化规律在2-3组像素网格上无法推出（例如对大量纯色对称对象做不同变换、无法从少量例子归纳颜色对应关系、不同情况的变换涉及正逆向旋转镜像等多种不同变化等，即全部的变化逻辑或者说是变化规律无法在3个输入输出示例对中体现），就需要调整设计使差异可观察。

       f) **可视化整洁度**: 即使逻辑合理，如果可视化后的输出网格看上去乱七八糟也不行。检查：同一网格区域是否存在多种元素覆盖叠加导致无法分辨哪个是哪个？不同对象、区域、标记之间在视觉上是否能清晰区分？如果融合后的网格在视觉上杂乱不堪，需要调整设计（如减少同一区域的叠加层次、增大对象间距、简化标记方式等）。

       g) **变化的可推断性**: 如果融合涉及颜色映射或颜色变化，务必确保颜色种类不宜过多，且颜色变化必须有足够的结构性线索（如位置、形状、分区等伴随特征）让人类能从少量样例中推断出颜色变化规则。避免出现大量颜色变化却无任何提示的情况——如果解题者无法从样例归纳出"什么条件导致什么颜色"，这个融合设计就

       将以上分析写在代码注释中或输出中，然后再开始写代码。

    4. **修改 `main()`**: 按照融合描述和各组成项详情实现新规则。确保各项改动在推理链中有真正的协同，不是各改各的。
    5. **修改 `generate_input()`**: 确保生成的输入能正确触发融合后的新规则。
    6. **保证 generator 的正确性**: 沿用 output-first 模式或适当调整，但必须保证每个输入经过新 `main()` 后不崩溃、不触发 assert、输出正确。
    7. **测试**: 运行以下测试，确保成功率 ≥ 80%:

    ```bash
    {sys.executable} -c "
    import numpy as np, importlib.util, time, sys
    spec = importlib.util.spec_from_file_location('variant', '{variant_path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok, fail = 0, 0
    t0 = time.time()
    for i in range(50):
        try:
            inp = mod.generate_input()
            out = mod.main(inp)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 3: print(f'Trial {{i}}: {{e}}')
    elapsed = time.time() - t0
    print(f'Success: {{ok}}/50, Time: {{elapsed:.1f}}s ({{elapsed/50*1000:.0f}}ms/call)')
    if ok < 40: sys.exit(1)
    "
    ```

    如果测试失败，分析错误原因，修改代码，重新测试。反复直到通过。

    8. **验证融合行为确实发生**: `main(input)==output` 只能验证求解器不报错，无法验证输入是否真正体现了融合描述中的特征行为——退化为原始行为的输入同样能通过。测试通过后，定义一个量化指标衡量融合核心行为，统计生成样本中该指标的分布。只有当关键控制变量不起作用、样本按融合逻辑执行与按原逻辑执行在相同输入上输出相同结果时，才算退化并需要修正生成器。
    9. **审查 `main()` 的新隐式假设**: 修改 `main()` 后，它可能对输入产生原始版本中不存在的新假设（如对象间需特定空间关系、匹配特征需可区分、处理顺序依赖前一步结果等）。逐一检查 `main()` 中的每个循环、匹配和条件分支：这段逻辑对输入做了什么假设？`generate_input()` 是否通过构造保证了这些假设？

    ## 参考资源
    - `{BASE / 'solvers' / 'arc_utils.py'}` — 可用的工具函数
    - `{BASE / 'ARC-data' / 'ARC-AGI-2' / 'data' / 'evaluation' / (task_id + '.json')}` — 原任务的 I/O 样例

    ## 输出

    将完整的融合变体代码（包含修改后的 `main()` 和 `generate_input()`）保存到:
    `{variant_path}`

    代码文件开头加一段注释说明这是什么融合变体:
    ```python
    # Combo variant of {task_id} (index {index})
    # combo_spec: {combo_spec}
    # Name: {combo_name}
    ```

    如果原始代码的 `main()` 已有四段式说明注释/文档字符串，则新代码也必须在 `main()` 上方写出同结构的四部分说明：
    1. 开头概述
    2. `Input structure:`
    3. `Rule:`
    4. `Key insight:`

    确保目录 `{variant_dir}` 存在（如不存在则创建）。
    """)

    # Inject source_task solver references
    source_refs = []
    for dim_id, alt_idx_str in components:
        dim = dims_by_id.get(dim_id)
        if dim is None:
            continue
        alt = dim["alternatives"][int(alt_idx_str)]
        src_task = alt.get("source_task")
        if src_task and src_task != "null":
            src_solver = BASE / "arc2_solver_select" / f"{src_task}.py"
            if src_solver.exists():
                source_refs.append(
                    f"    - `{src_solver}` — 参考任务 {src_task} 的 solver"
                    f"（{dim_id}/{alt_idx_str} 的灵感来源）"
                )
    if source_refs:
        prompt += "\n" + "\n".join(source_refs) + "\n"

    return prompt


if __name__ == "__main__":
    task_id = sys.argv[1]
    index = int(sys.argv[2])
    out_path = (
        sys.argv[3] if len(sys.argv) > 3
        else str(BASE / "ceshi" / "tem3" / f"combo_prompt_{task_id}_{index}.txt")
    )
    prompt = build_prompt(task_id, index)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(prompt)
    print(f"Combo prompt for {task_id} index {index}: "
          f"{len(prompt)} chars → {out_path}")

"""
Build a prompt for a subagent to implement one variant of an ARC-AGI-2 task.

Usage:
    python variant_prompt_builder.py <task_id> <dim_id> <alt_index> [output_path]

Example:
    python variant_prompt_builder.py 67e490f4 D1 0
    # → writes prompt to tmp/variant_prompt_67e490f4_D1_alt0.txt

The subagent modifies the original solver+generator to implement the variant,
tests it, and saves the working code.
"""
from __future__ import annotations
import ast
import json, sys, textwrap, yaml
from pathlib import Path

BASE = Path(__file__).resolve().parent


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


def build_prompt(task_id: str, dim_id: str, alt_index: int) -> str:
    # ── Load dimension analysis ───────────────────────────────────────
    yaml_path = BASE / "ceshi" / "yaml" / f"dim_analysis_v9_{task_id}.yaml"
    if not yaml_path.exists():
        yaml_candidates = sorted(BASE.glob(f"ceshi/yaml/dim_analysis_*_{task_id}.yaml"))
        yaml_path = yaml_candidates[-1] if yaml_candidates else yaml_path
    data = yaml.safe_load(yaml_path.read_text())

    dim = None
    for d in data["dimensions"]:
        if d["id"] == dim_id:
            dim = d
            break
    if dim is None:
        raise ValueError(f"Dimension {dim_id} not found in {yaml_path}")

    alt = dim["alternatives"][alt_index]

    # ── Load original solver code ─────────────────────────────────────
    solver_path = BASE / "arc2_solver_select" / f"{task_id}.py"
    solver_code = solver_path.read_text()
    has_four_part_main_comment = _has_four_part_main_comment(solver_code)

    # ── Load spec file if available ──────────────────────────────────
    spec_path = BASE / "spce_dis" / f"{task_id}_spec.yaml"
    spec_content = spec_path.read_text() if spec_path.exists() else ""

    # ── Variant output path ───────────────────────────────────────────
    variant_dir = BASE / "ceshi" / "code" / task_id / dim_id / str(alt_index)
    variant_file = "code.py"
    variant_path = variant_dir / variant_file

    spec_section = ""
    if spec_content:
        spec_section = textwrap.dedent(f"""\

    ## 原始任务规格（Spec）

    以下是从原始 solver 代码和示例中提炼的完整任务规格，包含精确的数值约束和生成策略。
    Spec 本身就是原题代码生成时的参考文件，因此：
    - 在**设计分析**阶段，参考 spec 理解原始任务的完整设计细节（对象结构、空间关系、生成策略等），以此为基础思考变体的改动如何落实
    - 在**实现**阶段，参考 spec 中的数值约束（网格尺寸、对象数量/尺寸、颜色数等），确保变体在物理上可实现

    ```yaml
    {spec_content.strip()}
    ```
    """)

    docstring_instruction = ""
    if has_four_part_main_comment:
        docstring_instruction = textwrap.dedent("""\

    额外要求：原始代码的 `main()` 已包含四段式说明注释/文档字符串（开头概述 + `Input structure:` + `Rule:` + `Key insight:`）。
    生成的新代码里也必须保留这一结构，并根据**新的变体规则**与**原注释提供的信息**重新编写；可以借鉴原注释的组织方式，但**不要照抄原文**，要让内容准确反映变体后的输入结构、规则和关键洞察。
    """)
    else:
        docstring_instruction = textwrap.dedent("""\ """)

    prompt = textwrap.dedent(f"""\
    你要修改一个 ARC-AGI-2 任务的 solver 和 generator，实现一个变体。

    ## 原始任务: {task_id}
    **摘要**: {data['summary']}

    ## 要实现的变体
    - **维度**: {dim_id} — {dim['name']} (`{dim['tag']}`)
    - **当前值**: {dim['current']}
    - **改为**: {alt['name']} — {alt['description']}
    - **来源**: {alt.get('source', '?')}（参考任务: {alt.get('source_task', 'N/A')}）

    ## 修改原则 
    以 {dim_id} 为核心变更，不要随意改动与该创新无关的部分。但如果其他维度需要做适配性调整才能让 {dim_id} 的变化在视觉上可观察、在逻辑上可行（例如：变换规则改为旋转/翻转，但原始对象是纯色对称的导致变换不可见，则应让对象带不对称内部结构），这种配合修改是允许且必要的。

    ## 原始 solver + generator 代码

    文件: `{solver_path}`

    ```python
    {solver_code}
    ```
    {spec_section}{docstring_instruction}
    ## 你的任务

    1. **理解原代码**: 弄清 `main()` 和 `generate_input()` 的逻辑。

    2. **阅读参考任务**（如有 source_task）: 变体描述中的 source_task 是该变体的灵感来源。**必须先阅读其 solver 代码**，理解参考任务中相关机制的实现方式和复杂度，作为你设计变体的基线。不存在就跳过。

    3. **设计分析（写代码前必须完成，可仿照spec书写，尽可能的详细以供下面代码生成使用）**:

       YAML 中的变体描述给出了变更的大致方向，但细节不够具体。你需要自行思考并确定完整的实现细节。如果上面提供了原始任务规格（Spec），请将其作为设计分析的重要参考——spec 详细描述了原始任务中各对象的结构、空间关系、生成策略等，理解这些细节有助于你准确判断变体改动的影响范围和实现方式。在写代码之前，完成以下分析：

       a) **具体化与行为追踪**: 变体在 ARC 的像素网格中具体长什么样？变更的核心行为如何体现在输入/输出的结构上？不要停留在抽象概念上，要想清楚具体的形状、颜色、空间关系等细节。然后，确定变体的变化会影响整个流程（输入构造、solver 处理、输出生成等）中的哪个环节，在这个环节上用一个具体的小例子追踪：原始逻辑在这里是怎么执行的 → 变体要求的逻辑应该是怎么执行的 → 二者的差异是什么。

       b) **复杂度校准**: 检查你的设计是否过于简化。过于简化会导致两个问题：一是容易产生冲突或重复（例如不同对象变得难以区分，匹配出现歧义），二是变体的视觉效果不明显，肉眼看不出和原任务的区别。同时不要让关键自由度过多；如果规则依赖少数档位、类别或对应关系，应控制在少量取值，并让它们在每个样例中有重复支撑。

       c) **隐式假设审查**: 变体改变了原任务的某个属性。原始代码中哪些写法依赖了旧属性的不变性？逐行检查 main() 中与变体相关的代码段：每行代码依赖了什么前提？这个前提在变体中是否仍然成立？特别注意那些在原任务中"恰好无害"但在变体中会出错的模式（例如：原任务中某个值始终相同，所以无条件赋值不会产生覆盖问题——但变体让该值可变后，无条件赋值就会破坏已有的正确结果）。

       d) **放置合法性**: 生成器在放置对象时，必须保证各个体之间不重叠、不越界，且放置结果符合题意（例如对象之间的空间关系、对齐方式、间距等应与任务规则一致）。同时要考虑放置后的网格是否能被 solver 正确处理并产生预期的输出。

       e) **视觉可观察性**: ARC 要求人类从 2-4 个 input→output 样例中肉眼推断规则。**禁止使用颜色id等不可见因素参与任何排序或者颜色变换（例如使用黑色=0参与排序等，禁用）**。做一次思想实验：假设你是解题者，看到几组变体生成的 input→output，能否看出变体的规则？如果变体的变化规律在2-3组像素网格上无法推出（例如对大量纯色对称对象做不同变换、无法从少量例子归纳颜色对应关系、不同情况的变换涉及正逆向旋转镜像等多种不同变化等，即全部的变化逻辑或者说是变化规律无法在3个输入输出示例对中体现），就需要调整设计使差异可观察。若规则本身允许多个合法分支或子情况，不要在生成器中长期固定成其中一种，除非题意本来就要求固定。

       f) **可视化整洁度**: 即使逻辑合理，如果可视化后的输出网格看上去乱七八糟也不行。检查：同一网格区域是否存在多种元素覆盖叠加导致无法分辨哪个是哪个？不同对象、区域、标记之间在视觉上是否能清晰区分？如果变体后的网格在视觉上杂乱不堪，需要调整设计（如减少同一区域的叠加层次、增大对象间距、简化标记方式等）。

       g) **颜色变化可推断性**: 如果变体涉及颜色映射或颜色变化，务必确保颜色种类不宜过多，且颜色变化必须有足够的结构性线索（如位置、形状、分区等伴随特征）让人类能从少量样例中推断出颜色变化规则。避免出现大量颜色变化却无任何提示的情况——如果解题者无法从样例归纳出"什么条件导致什么颜色"，这个变体设计就不合格。

       将以上分析写在代码注释中或输出中，然后再开始写代码。

    4. **修改 `main()`**: 基于第 3 步的设计分析实现变体的新规则。只改需要改的部分，其他逻辑保持不变。
    5. **修改 `generate_input()`**: 确保生成的输入能正确触发新规则。生成的输入必须能被新 `main()` 正确求解。
    6. **保证 generator 的正确性**: 原 generator 采用 output-first 模式（先构造预期输出 → 反推输入 → 用 `main(input)` 验证一致）。修改变体时，你可以沿用或调整这个模式，但必须保证：生成的每个输入经过新 `main()` 后不崩溃、不触发 assert、输出正确。
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

    8. **验证变体行为确实发生**: `main(input)==output` 只能验证求解器不报错，无法验证输入是否真正体现了变体的特征行为——退化为原始行为的输入同样能通过。测试通过后，定义一个量化指标衡量变体核心行为（如交互深度、参与对象数等），统计生成样本中该指标和关键分支的分布。少量样本与原始行为相似不为退化；只有当关键控制变量不起作用，样本按照新的变体逻辑执行与原逻辑执行在相同输入网格上输出相同结果时，才算退化并需要修正生成器。
    9. **审查 `main()` 的新隐式假设**: 修改 `main()` 后，它可能对输入产生原始版本中不存在的新假设（如对象间需特定空间关系、匹配特征需可区分、处理顺序依赖前一步结果等）。逐一检查 `main()` 中的每个循环、匹配和条件分支：这段逻辑对输入做了什么假设？`generate_input()` 是否通过构造保证了这些假设？

    ## 参考资源
    - `{BASE / 'solvers' / 'arc_utils.py'}` — 可用的工具函数
    - `{BASE / 'ARC-data' / 'ARC-AGI-2' / 'data' / 'evaluation' / (task_id + '.json')}` — 原任务的 I/O 样例

    只参考上述文件和原始 solver 代码。不要查看其他变体的实现代码（如 variants/ 目录下的已有文件），避免引入其他维度的改动。""")

    if alt.get("source_task") and alt["source_task"] != "null":
        src_task = alt["source_task"]
        src_solver = BASE / "arc2_solver_select" / f"{src_task}.py"
        if src_solver.exists():
            prompt += f"\n    - `{src_solver}` — **必读**：参考任务 {src_task} 的 solver（变体灵感来源，用于校准实现复杂度）"

    prompt += textwrap.dedent(f"""

    ## 输出

    将完整的变体代码（包含修改后的 `main()` 和 `generate_input()`）保存到:
    `{variant_path}`

    代码文件开头加一段注释说明这是什么变体:
    ```python
    # Variant of {task_id}
    # Dimension: {dim_id} — {dim['name']} ({dim['tag']})
    # Change: {dim['current'][:60]}... → {alt['name']}
    ```

    确保目录 `{variant_dir}` 存在（如不存在则创建）。
    """)

    return prompt


if __name__ == "__main__":
    task_id = sys.argv[1]
    dim_id = sys.argv[2]
    alt_index = int(sys.argv[3])
    out_path = (sys.argv[4] if len(sys.argv) > 4
                else str(BASE / "ceshi" / "tem2" / f"variant_prompt_{task_id}_{dim_id}_alt{alt_index}.txt"))
    prompt = build_prompt(task_id, dim_id, alt_index)
    Path(out_path).write_text(prompt)
    print(f"Variant prompt for {task_id} {dim_id} alt{alt_index}: "
          f"{len(prompt)} chars → {out_path}")

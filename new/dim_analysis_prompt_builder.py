"""
Build a structured dimension-analysis prompt for one ARC-AGI-2 task.

Usage:
    python dim_analysis_prompt_builder.py <task_id> [output_path]

The prompt produces a Chinese markdown dimension analysis for human review,
with per-tag guidance and cross-task inspiration.
"""
from __future__ import annotations
import ast, json, sys, textwrap
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent

TYPE_TAGS = {
    "move_to_match":      ["match_cond", "src", "dst", "transform", "trajectory", "grid_shape", "multi_step", "symbol_instructions"],
    "move_along_path":    ["path", "obstacle", "trajectory", "src", "dst", "transform", "grid_shape", "multi_step", "symbol_instructions"],
    "move_in_order":      ["match_cond", "src", "dst", "transform", "grid_shape", "multi_step", "symbol_instructions"],
    "transform_in_place": ["transform", "src", "grid_shape", "multi_step", "symbol_instructions"],
}

TAG_DESCRIPTIONS = {
    "move_to_match": {
        "src":                  "What moves — source objects to be placed/matched/docked.",
        "dst":                  "Where it goes — destination slots/anchors/holes that receive src.",
        "match_cond":           "How to pair — the KEY INSIGHT: what makes src fit dst (color, shape, size, connector, etc.).",
        "transform":            "What changes — recoloring, erasing, filling, swapping, scaling after placement.",
        "trajectory":           "What's left behind — 'keep src' or default disappear; painting during movement.",
        "grid_shape":           "Output framing — 'tight crop', 'expand if needed', etc.",
        "multi_step":           "Multi-step sequencing — iterative processing, chained operations, ordered passes.",
        "symbol_instructions":  "Symbolic encoding — control pixels, legends, or other encoded parameters.",
    },
    "move_along_path": {
        "src":                  "What emits/moves/starts — the source object that initiates the path.",
        "path":                 "Route rules — direction, navigation, stopping conditions, collision behavior.",
        "obstacle":             "Path blockers/interactors — intermediate objects that deflect, block, split, or modify the moving entity along its path (e.g. shelves, walls, mirrors). Analyze their count, shape, spatial arrangement, spacing, and role differentiation (functional vs distractor).",
        "trajectory":           "What gets painted — color, pattern, continuity (continuous/intermittent/cycle/none).",
        "dst":                  "Endpoint/stopping condition — grid edge, obstacle, target.",
        "transform":            "What changes — recolor, erase, grow, surround on hit objects.",
        "grid_shape":           "Output framing — crop, same size, etc.",
        "multi_step":           "Multi-step sequencing — iterative processing, chained operations, ordered passes.",
        "symbol_instructions":  "Symbolic encoding — control pixels, legends, or other encoded parameters.",
    },
    "move_in_order": {
        "src":                  "What moves — objects to be ordered and placed.",
        "dst":                  "Where they go — slots or accumulation target.",
        "match_cond":           "Ordering/pairing rule — what determines sequence and placement.",
        "transform":            "What changes — recolor, rotate, merge during placement.",
        "grid_shape":           "Output framing.",
        "multi_step":           "Multi-step sequencing — iterative processing, chained operations, ordered passes.",
        "symbol_instructions":  "Symbolic encoding — control pixels, legends, or other encoded parameters.",
    },
    "transform_in_place": {
        "src":                  "What gets transformed — objects identified for in-place modification.",
        "transform":            "What changes — the core transformation rule.",
        "grid_shape":           "Output framing.",
        "multi_step":           "Multi-step sequencing — iterative processing, chained operations, ordered passes.",
        "symbol_instructions":  "Symbolic encoding — control pixels, legends, or other encoded parameters.",
    },
}

# A = same-type only, B = same-type priority + cross-type, C = all types, D = meta
TAG_SCOPE = {
    "match_cond": "A", "path": "A", "obstacle": "B", "trajectory": "A",
    "src": "B", "dst": "B",
    "transform": "C", "grid_shape": "C",
    "multi_step": "D", "symbol_instructions": "D",
}

RAW_FIELD_TO_TAG = {
    "match_cond": "match_cond", "src": "src", "dst": "dst",
    "transform": "transform", "transform2": "transform",
    "trajectory": "trajectory", "path": "path", "obstacle": "obstacle",
    "grid_shape": "grid_shape", "symbol_instructions": "symbol_instructions",
    "multi_step": "multi_step", "notes": "notes", "concepts": "concepts",
}


def extract_docstring(solver_code: str) -> str | None:
    try:
        tree = ast.parse(solver_code)
        return ast.get_docstring(tree)
    except SyntaxError:
        return None


def build_prompt(task_id: str) -> str:
    ann_all = json.loads((BASE / "arc-agi-2-eval-annotations-merge.json").read_text())
    tags_all = json.loads((BASE / "arc2_mechanism_tags_raw.json").read_text())
    solver_path = BASE / "arc2_solver_select" / f"{task_id}.py"

    task_ann  = ann_all[task_id]
    task_type = task_ann["type"]
    task_tags = tags_all.get(task_id, [])

    solver_code = solver_path.read_text() if solver_path.exists() else ""
    docstring = extract_docstring(solver_code) if solver_code else None

    used_tag_names = {t["tag"] for t in task_tags}
    relevant_tags = TYPE_TAGS.get(task_type, TYPE_TAGS["move_to_match"])
    tag_descs = TAG_DESCRIPTIONS.get(task_type, TAG_DESCRIPTIONS["move_to_match"])

    # ── Build field→mechanisms from raw tags ──────────────────────────
    tag_to_mechs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for pid, ptags in tags_all.items():
        for t in ptags:
            mapped = RAW_FIELD_TO_TAG.get(t["field"])
            if mapped:
                tag_to_mechs[mapped][t["tag"]].append(pid)

    # ── Collect tasks by type ─────────────────────────────────────────
    same_type_ids = set()
    same_type_anns: dict[str, dict] = {}
    all_anns: dict[str, dict] = {}
    for pid, a in sorted(ann_all.items()):
        if pid == task_id:
            continue
        all_anns[pid] = a
        if a.get("type") == task_type:
            same_type_ids.add(pid)
            same_type_anns[pid] = a

    # ── Section: Docstring ────────────────────────────────────────────
    doc_section = ""
    if docstring:
        doc_section = f"### Task Description\n{docstring}\n"

    # ── Section: Tag system table ─────────────────────────────────────
    tag_table = "| Tag | Role |\n|-----|------|\n"
    for tag in relevant_tags:
        desc = tag_descs.get(tag, "")
        tag_table += f"| **{tag}** | {desc} |\n"

    # ── Section: Annotation JSON ──────────────────────────────────────
    ann_json = json.dumps(task_ann, indent=2, ensure_ascii=False)
    tags_json = json.dumps(task_tags, indent=2, ensure_ascii=False)

    # ── Section: Catalog per tag (scope-aware) ────────────────────────
    catalog_section = ""

    for tag in relevant_tags:
        scope = TAG_SCOPE.get(tag, "B")
        if scope == "D":
            continue

        mechs = tag_to_mechs.get(tag, {})
        if not mechs:
            continue

        catalog_section += f"\n#### `{tag}`\n"

        for mech_name, task_ids in sorted(mechs.items(),
                                          key=lambda x: -len(x[1])):
            is_used = mech_name in used_tag_names
            same_tasks = [t for t in task_ids if t in same_type_ids]
            cross_tasks = [t for t in task_ids
                           if t not in same_type_ids and t != task_id]

            if is_used:
                catalog_section += f"- `{mech_name}` [✓ USED]\n"
                continue

            if scope == "A":
                if not same_tasks:
                    continue
                snippet, spid = _find_snippet(same_tasks, same_type_anns, tag)
                if spid:
                    catalog_section += f"- `{mech_name}` [✗] — **{spid}**: {snippet}\n"
                else:
                    catalog_section += f"- `{mech_name}` [✗] — e.g. {', '.join(same_tasks[:2])}\n"

            elif scope == "B":
                snippet, spid = _find_snippet(same_tasks, same_type_anns, tag)
                if spid:
                    catalog_section += f"- `{mech_name}` [✗] [same-type] — **{spid}**: {snippet}\n"
                elif same_tasks:
                    catalog_section += f"- `{mech_name}` [✗] [same-type] — e.g. {', '.join(same_tasks[:2])}\n"
                else:
                    snippet, spid = _find_snippet(cross_tasks, all_anns, tag)
                    if spid:
                        catalog_section += f"- `{mech_name}` [✗] [cross-type] — **{spid}**: {snippet}\n"
                    elif cross_tasks:
                        catalog_section += f"- `{mech_name}` [✗] [cross-type] — e.g. {', '.join(cross_tasks[:2])}\n"

            elif scope == "C":
                all_tasks_for_mech = same_tasks + cross_tasks
                snippet, spid = _find_snippet(all_tasks_for_mech, all_anns, tag)
                if spid:
                    marker = " [same-type]" if spid in same_type_ids else ""
                    catalog_section += f"- `{mech_name}` [✗]{marker} — **{spid}**: {snippet}\n"
                elif all_tasks_for_mech:
                    catalog_section += f"- `{mech_name}` [✗] — e.g. {', '.join(all_tasks_for_mech[:2])}\n"

    # ── Meta-dimensions (D scope — all types) ─────────────────────────
    meta_fields = ("multi_step", "symbol_instructions", "notes")
    meta_examples = []
    for pid, a in sorted(all_anns.items()):
        for mf in meta_fields:
            val = a.get(mf, "")
            if val and isinstance(val, str) and len(val) > 10:
                meta_examples.append((pid, mf, val[:120],
                                      pid in same_type_ids))

    if meta_examples:
        catalog_section += "\n#### Meta-dimensions (`multi_step`, `symbol_instructions`, `notes`)\n"
        catalog_section += "Cross-cutting structural modifiers — sequencing, iteration, cardinality, symbolic encoding.\n"
        same_meta = [(p, f, v) for p, f, v, s in meta_examples if s]
        cross_meta = [(p, f, v) for p, f, v, s in meta_examples if not s]
        for pid, mf, val in same_meta:
            catalog_section += f"- **{pid}** [{mf}] [same-type]: {val}\n"
        for pid, mf, val in cross_meta:
            catalog_section += f"- **{pid}** [{mf}]: {val}\n"

    # ── Load spec file if available ───────────────────────────────────
    spec_path = BASE / "spce_dis" / f"{task_id}_spec.yaml"
    spec_content = spec_path.read_text() if spec_path.exists() else ""

    # ── Assemble ──────────────────────────────────────────────────────
    solver_ref = str(solver_path) if solver_path.exists() else "(not available)"

    spec_section = ""
    if spec_content:
        spec_section = textwrap.dedent(f"""\
    ### 任务规格（Spec）

    以下是从 solver 代码和示例中提炼的完整任务规格，包含精确的数值约束和生成策略。
    Spec 本身就是原题代码生成时的参考文件，详细描述了任务中各对象的结构、空间关系、生成策略等设计细节。在提出替代方案时：
    - 参考 spec 理解原始任务的完整设计，以此为基础思考哪些维度可以变化、如何变化
    - 留意 spec 中的数值约束（网格尺寸、对象数量/尺寸、颜色数等），明显不可行的方案可以跳过

    ```yaml
    {spec_content.strip()}
    ```
    """)

    prompt = textwrap.dedent(f"""\
    你正在对一个 ARC-AGI-2 谜题任务进行"维度分析"，作为任务变体生成流水线的第一步。

    ## 任务: {task_id}

    {doc_section}
    ### Annotation Tag System (`{task_type}`)

    {tag_table}

    ### 当前任务的结构化标注
    ```json
    {ann_json}
    ```

    ### 当前任务已有的机制标签
    ```json
    {tags_json}
    ```

    ### Solver 代码
    完整实现: `{solver_ref}` — 请阅读以了解详细逻辑。

    {spec_section}
    ### 术语表（中文描述请统一使用以下译法）
    - sprite → 精灵（小型可移动对象）
    - frame → 框架（大型模板结构）
    - hole → 洞（框架内的空缺区域）
    - blob → 色块（彩色像素连通区域）
    - ray → 射线（发射的线段）
    - port → 端口（射线发射点）
    - trail/trajectory → 轨迹（路径上的涂色痕迹）
    - background/bg → 背景
    - connected component → 连通分量
    - bounding box/bbox → 外接矩形
    - tiebreak → 消歧规则
    - distractor → 干扰项
    
    **关键规则**:
    1. 以上面的 tag 体系为起点组织分析，确保每个相关 tag 至少有一个维度。但 tag 体系只是引导框架，不是限制——如果你在阅读 solver 代码和 spec 后，发现任务中存在重要的可变方面没有被现有 tag 覆盖（例如路径中的障碍物属性、对象间的空间关系、特定的数值约束等），**必须**主动补充为独立维度（tag 填 `meta` 或自拟合适的 tag 名）。好的维度分析的标准是"全面覆盖任务的所有可变方面"，而不是"恰好匹配 tag 列表"。
    2. 对每个维度，考虑**取反/否定**当前值（如 max→min, most→least, include→exclude, allow→forbid, thick→thin）。但只提出能产生合理、有趣的 ARC 谜题的反转。
    3. 参考下方目录中的 "✗" 条目作为灵感。标有 "[same-type]" 的条目来自其他 `{task_type}` 任务，尤其相关。
    4. 考虑**结构性元维度**：src-dst 基数关系（1:1, many:1, 1:many）、多步序列、迭代/链式处理、符号编码参数。
    5. 平衡替代方案的来源：目录中的机制、跨任务灵感、取反、创新。
    6. 目录中的机制仅供参考和启发，**不是限制**。鼓励你提出目录中没有的全新替代方案——最好的变体往往来自创造性组合或全新想法。
    7. 创新优先级：优先提出能改变解题者推理过程的替代方案（如改变排序规则、匹配条件、变换逻辑等）。注意：输出构造方式的改变（如裁剪范围、保留/删除哪些元素、着色规则）也属于有效创新。仅改变视觉外观的方案价值较低，但如果确实有趣仍可列出。
    8. ARC 可解性与可视性：变体规则必须能被人类从样例中推断出来。禁止将颜色 ID 作为数值参与排序、算术或其他隐藏运算。禁止变换规则包含随机逻辑（如"随机选色""随机选位置"作为规则的一部分），但 generator 用于生成多样化样本的参数随机化不受此限制。人类看到的是图像而非数字矩阵，所有变化必须有可见的结构性线索。
    9. 可行性与设计意识：变体在 ARC 框架下可实现（网格不超过30×30、颜色不超过10种等）。如果上面提供了任务规格（Spec），将其作为理解原始任务设计的重要参考，但 spec 中的具体参数（如图案类型数、对象数量、网格精确尺寸等）是原题的参数，变体可以合理调整。只有当变体在 ARC 物理限制下根本无法实现时才排除。
    10. 题目适配性：变体可以改变推理链中的某个关键步骤（如匹配条件从颜色改为形状），只要整体任务结构（源对象→变换→输出）保持连贯即可。不要求变体与原题在每个方面都相似。
    11. 尽量为每个维度列出 2-4 个创新方案，宁缺毋滥，确保每个方案都符合上述 ARC 可解性标准。

    ## 机制目录 & 跨任务灵感

    按 annotation tag 分组。范围策略:
    - 类型特定 tag (`match_cond`, `path`, `trajectory`): 仅展示同类型任务的机制。
    - 对象 tag (`src`, `dst`): 同类型优先，跨类型补充。
    - 动作 tag (`transform`, `grid_shape`): 所有类型平等。
    - Meta tag: 所有类型。
    {catalog_section}

    ## 输出格式 — YAML

    请输出一个合法的 YAML 文档。描述性文本用中文，技术术语（tag 名、机制名、task ID）保留英文。
    不要用 markdown 代码围栏包裹，直接输出 YAML 内容。

    严格遵循以下结构：

    ```yaml
    task_id: {task_id}
    task_type: {task_type}
    summary: "一句话描述任务的核心逻辑"

    dimensions:
      - id: D1
        name: "维度的中文名称"
        tag: match_cond
        current: "当前任务使用的具体机制/参数"
        alternatives:
          - name: "替代方案的中文名称"
            description: "具体描述做什么改变，以及为什么作为 ARC 谜题会有趣"
            source: catalog_tag_name / inversion / creative
            source_task: task_id_or_null
            delta_lines: 15
          - name: "另一个替代方案"
            description: "..."
            source: "..."
            source_task: null
            delta_lines: 10

      - id: D2
        name: "..."
        tag: src
        current: "..."
        alternatives:
          - ...
    ```

    规则:
    - 每个维度必须有 tag 字段。可使用上方 tag 体系中的标准 tag，也可使用 `meta` 或自拟 tag 名来表示标准体系未覆盖的重要维度（如对象属性、空间关系、数值约束等）。
    - delta_lines = 修改 solver 代码的预估行数（整数）。
    - 确保覆盖所有相关的标准 tag，同时主动补充你认为重要的额外维度。维度数量不设上限——尽量全面。
    - source: 目录机制名 / 源任务ID / inversion（取反）/ creative（创新）。
    - source_task: 来源任务的 ID（目录/跨任务来源必填，inversion/creative 填 null）。
    - 只提出能产生有趣、可解的 ARC 谜题的变体。
    - description 中简要说明为什么它作为 ARC 谜题有趣（视觉差异、推理难度变化等）。
    - 字符串值中如果包含冒号或特殊字符，请用双引号包裹。
    - 直接输出 YAML，不要加 markdown 围栏。
    """)

    return prompt


def _find_snippet(task_ids: list[str], anns: dict[str, dict],
                  tag: str) -> tuple[str, str | None]:
    """Find the first annotation snippet for `tag` from the given task list."""
    for tid in task_ids[:5]:
        a = anns.get(tid, {})
        val = a.get(tag, "") or a.get(f"{tag}2", "")
        if val and isinstance(val, str) and len(val) > 5:
            return val[:100], tid
    return "", None


if __name__ == "__main__":
    task_id = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else str(BASE / "ceshi" / "tem1" / f"dim_analysis_prompt_{task_id}.txt")
    prompt = build_prompt(task_id)
    Path(out_path).write_text(prompt)
    print(f"Prompt for {task_id}: {len(prompt)} chars, "
          f"{prompt.count(chr(10))} lines → {out_path}")

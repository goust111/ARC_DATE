from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_ARC_CODE_DIR = Path("/home/lixinyu/snap/new/ARC_code")
DEFAULT_ANNOTATION_DIR = Path("/home/lixinyu/snap/new/annotations")
DEFAULT_TMP_DIR = Path("/home/lixinyu/snap/BARC/tmp")
DEFAULT_SPEC_DIR = Path("/home/lixinyu/snap/new/spce_dis")
DEFAULT_TAG_INDEX_PATH = Path("/home/lixinyu/snap/a/code/arc1_full_plus_arc2_train_index.json")
DEFAULT_REFERENCE_JSON_PATHS = (
    Path("/home/lixinyu/snap/a/code/arc1_reference_index.json"),
    Path("/home/lixinyu/snap/a/code/arc_reference_index.json"),
)
SKIP_FILENAMES = {"arc_utils.py", "__init__.py"}


@dataclass(frozen=True)
class ReferencePuzzle:
    puzzle_id: str
    source: str
    code_path: str
    docstring: str
    annotation_path: str | None
    annotation: dict[str, Any] | None
    annotation_type: str | None
    tags: frozenset[str]
    tokens: frozenset[str]
    code_features: frozenset[str]


@dataclass(frozen=True)
class NewPuzzle:
    index: str
    description: str
    tags: frozenset[str]
    tokens: frozenset[str]
    expected_code_features: frozenset[str]
    explicit_reference_ids: frozenset[str]
    group_reference_ids: frozenset[str]


@dataclass(frozen=True)
class PuzzleInput:
    puzzle_dir: Path
    des_path: Path


@dataclass(frozen=True)
class GenerationSpec:
    path: str
    content: str


@dataclass(frozen=True)
class ScoredReference:
    ref: ReferencePuzzle
    score: float
    mechanism_score: float
    lexical_score: float
    code_feature_score: float
    overlap_tags: tuple[str, ...]


TAG_KEYWORDS: dict[str, list[str]] = {
    "object:closed_region": ["closed region", "enclosure", "enclosed", "封闭", "围成", "容器", "区域"],
    "object:frame": ["frame", "border", "hollow square", "边框", "框", "围栏"],
    "object:hole": ["hole", "cavity", "socket", "洞", "孔", "缺口", "空腔"],
    "object:sprite": ["sprite", "shape", "component", "object", "形状", "图形", "对象", "连通"],
    "object:line_segment": ["line segment", "bar", "segment", "直线", "线段", "横线", "竖线"],
    "object:path": ["path", "track", "corridor", "channel", "路径", "轨道", "通道", "迷宫"],
    "object:rectangle": ["rectangle", "rectangular", "矩形"],
    "object:square": ["square", "正方形", "方块"],
    "object:template": ["template", "legend", "模板", "图例"],
    "object:marker": ["marker", "标记", "seed", "种子", "arrow", "箭头", "clock", "指针"],
    "object:axis": ["axis", "separator", "fold line", "中心线", "分隔", "折叠线", "十字线"],
    "object:quadrant": ["quadrant", "象限"],
    "object:grid_edge": ["edge", "boundary", "边缘", "边界"],
    "operation:flood_fill": ["flood", "reachable", "fill region", "扩散", "填充", "水位"],
    "operation:bfs_distance": ["distance", "manhattan", "shortest", "距离", "曼哈顿", "最短"],
    "operation:parity": ["parity", "odd", "even", "奇数", "偶数", "奇偶"],
    "operation:ray_cast": ["ray", "shoot", "hit", "射线", "发射", "命中"],
    "operation:shape_match": ["shape match", "same shape", "rotation/flip", "匹配", "形状匹配", "旋转关系"],
    "operation:rotate": ["rotate", "rotation", "clockwise", "旋转", "顺时针", "逆时针"],
    "operation:reflect": ["reflect", "mirror", "symmetry", "flip", "镜像", "翻折", "对称"],
    "operation:translate": ["translate", "slide", "move", "relocate", "平移", "滑动", "移动", "传送"],
    "operation:count": ["count", "number", "area", "height", "width", "计数", "数量", "面积", "宽度", "长度"],
    "operation:sort_order": ["sort", "order", "sequence", "排序", "顺序", "序列"],
    "operation:draw_line": ["draw line", "connect", "bridge", "ray", "连线", "桥", "射线"],
    "operation:recolor": ["recolor", "color transfer", "染色", "变色", "颜色轮换", "重新着色"],
    "operation:crop_resize": ["crop", "output height", "output width", "单行", "裁剪", "输出为"],
    "operation:repair_anomaly": ["anomaly", "majority", "repair", "多数", "修复", "污损", "异常"],
    "relation:inside": ["inside", "interior", "within", "内部", "以内"],
    "relation:adjacent": ["adjacent", "touch", "neighbor", "相邻", "紧邻", "接触"],
    "relation:nearest": ["nearest", "closest", "最近"],
    "relation:aligned": ["aligned", "same row", "same column", "对齐", "同行", "同列"],
    "relation:between": ["between", "gap", "间隔", "之间"],
    "relation:overlap": ["overlap", "projection", "覆盖", "重叠", "投影"],
    "relation:direction": ["direction", "up", "down", "left", "right", "方向", "上", "下", "左", "右"],
    "relation:row_col": ["row", "column", "horizontal", "vertical", "行", "列", "水平", "垂直"],
    "output:same_size": ["same size", "unchanged", "保持不变", "原尺寸"],
    "output:crop": ["crop", "tight crop", "裁剪"],
    "output:single_row": ["single row", "1 row", "1行", "单行"],
    "output:new_canvas": ["new grid", "output grid", "新网格", "输出为"],
}


CODE_FEATURE_PATTERNS: dict[str, list[str]] = {
    "code:connected_components": ["components_mask", "find_components", "label_mask", "scipy_label", "ndimage.label"],
    "code:bbox": ["bbox", "crop_bbox", "bbox_coords", "r0", "c0", "r1", "c1"],
    "code:bfs": ["deque", "popleft", "neighbors4", "neighbors8"],
    "code:rotation": ["rot90", "rotate", "np.fliplr", "np.flipud"],
    "code:ray_loop": ["while 0 <=", "while 0<=", "ray", "hit"],
    "code:counter": ["Counter", "most_common"],
    "code:output_resize": ["np.full((", "np.zeros((", "return np.array", "crop"],
    "code:random_retry": ["for _attempt", "GEN_RETRIES", "failed after", "random_free"],
}


TAG_WEIGHTS = {
    "operation": 3.0,
    "object": 2.2,
    "relation": 1.4,
    "output": 1.5,
    "type": 1.8,
    "code": 0.7,
    "catalog": 1.0,
}


def _prefix(tag: str) -> str:
    return tag.split(":", 1)[0] if ":" in tag else "other"


def _tag_weight(tag: str) -> float:
    return TAG_WEIGHTS.get(_prefix(tag), 1.0)


def flatten_annotation(annotation: dict[str, Any] | None) -> str:
    if not annotation:
        return ""
    parts: list[str] = []

    def walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            for key, val in value.items():
                parts.append(str(key))
                walk(val)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            parts.append(str(value))

    walk(annotation)
    return "\n".join(parts)


def tokenize(text: str) -> frozenset[str]:
    lower = text.lower()
    return frozenset(re.findall(r"[a-z0-9_]{2,}", lower))


def extract_tags(text: str, *, annotation_type: str | None = None, catalog_terms: set[str] | None = None) -> frozenset[str]:
    lower = text.lower()
    tags: set[str] = set()
    for tag, keywords in TAG_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lower or keyword in text:
                tags.add(tag)
                break

    if annotation_type:
        tags.add(f"type:{annotation_type}")

    if catalog_terms:
        for term in catalog_terms:
            normalized = term.replace("_", " ")
            if normalized in lower:
                tags.add(f"catalog:{term}")

    return frozenset(tags)


def extract_code_features(code: str) -> frozenset[str]:
    features: set[str] = set()
    for feature, patterns in CODE_FEATURE_PATTERNS.items():
        if any(pattern in code for pattern in patterns):
            features.add(feature)
    return frozenset(features)


def expected_code_features_from_tags(tags: frozenset[str]) -> frozenset[str]:
    lowered = {tag.lower() for tag in tags}

    def has(prefix: str, *terms: str) -> bool:
        return any(
            tag.startswith(f"{prefix}:") and any(term in tag for term in terms)
            for tag in lowered
        )

    out: set[str] = set()
    if (
        has("object", "closed_region", "enclosed_region", "封闭区域")
        or has(
            "operation",
            "flood_fill",
            "bfs_distance",
            "shortest_path",
            "pathfinding",
            "最短路径",
            "距离场",
            "寻路",
        )
        or has("relation", "shortest_path", "最短通路")
        or has("condition", "unique_shortest_path", "最短路径唯一")
    ):
        out.update({"code:bfs", "code:connected_components"})
    if any(tag.startswith("object:") for tag in lowered):
        out.update({"code:connected_components", "code:bbox"})
    if has("operation", "rotate", "reflect", "shape_match", "旋转", "镜像", "形状匹配"):
        out.add("code:rotation")
    if has("operation", "ray_cast", "ray_extend", "射线"):
        out.add("code:ray_loop")
    if has("operation", "count", "计数"):
        out.add("code:counter")
    if has("operation", "crop_resize", "裁剪") or has("output", "single_row", "new_canvas", "单行", "新网格"):
        out.add("code:output_resize")
    return frozenset(out)


def weighted_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = set(a) | set(b)
    inter = set(a) & set(b)
    denom = sum(_tag_weight(t) for t in union)
    return 0.0 if denom == 0 else sum(_tag_weight(t) for t in inter) / denom


def token_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return inter / union if union else 0.0


def extract_main_docstring(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            doc = ast.get_docstring(node, clean=True)
            return doc.strip() if doc else None
    return None


def load_annotation(annotation_dir: Path, puzzle_id: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    path = annotation_dir / f"{puzzle_id}.json"
    if not path.is_file():
        return None, None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None, None, None
    ann_type = data.get("type")
    return data, str(path), str(ann_type) if ann_type else None


def load_catalog_terms(tmp_dir: Path) -> set[str]:
    catalog_path = tmp_dir / "mechanism_catalog_unified.json"
    if not catalog_path.is_file():
        return set()
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return set()
    return {str(k).lower() for k in data.keys()}


def collect_references(
    arc_code_dir: Path,
    annotation_dir: Path,
    *,
    source: str,
    catalog_terms: set[str],
) -> list[ReferencePuzzle]:
    source_dirs = ["eval", "train"] if source == "both" else [source]
    refs: list[ReferencePuzzle] = []
    for source_dir in source_dirs:
        folder = arc_code_dir / source_dir
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.py")):
            if path.name in SKIP_FILENAMES:
                continue
            doc = extract_main_docstring(path)
            if not doc:
                continue
            code = path.read_text(encoding="utf-8", errors="ignore")
            annotation, ann_path, ann_type = load_annotation(annotation_dir, path.stem)
            ann_text = flatten_annotation(annotation)
            code_features = extract_code_features(code)
            tags = set(extract_tags(f"{doc}\n{ann_text}", annotation_type=ann_type, catalog_terms=catalog_terms))
            tags.update(code_features)
            refs.append(
                ReferencePuzzle(
                    puzzle_id=path.stem,
                    source=source_dir,
                    code_path=str(path),
                    docstring=doc,
                    annotation_path=ann_path,
                    annotation=annotation,
                    annotation_type=ann_type,
                    tags=frozenset(tags),
                    tokens=tokenize(f"{doc}\n{ann_text}"),
                    code_features=code_features,
                )
            )
    return refs


def reference_to_json(ref: ReferencePuzzle) -> dict[str, Any]:
    return {
        "puzzle_id": ref.puzzle_id,
        "source": ref.source,
        "code_path": ref.code_path,
        "docstring": ref.docstring,
        "annotation_path": ref.annotation_path,
        "annotation": ref.annotation,
        "annotation_type": ref.annotation_type,
        "tags": sorted(ref.tags),
        "tokens": sorted(ref.tokens),
        "code_features": sorted(ref.code_features),
    }


def reference_from_json(data: dict[str, Any]) -> ReferencePuzzle:
    return ReferencePuzzle(
        puzzle_id=str(data["puzzle_id"]),
        source=str(data["source"]),
        code_path=str(data["code_path"]),
        docstring=str(data.get("docstring", "")),
        annotation_path=data.get("annotation_path"),
        annotation=data.get("annotation"),
        annotation_type=data.get("annotation_type"),
        tags=frozenset(map(str, data.get("tags", []))),
        tokens=frozenset(map(str, data.get("tokens", []))),
        code_features=frozenset(map(str, data.get("code_features", []))),
    )


def write_reference_index(path: Path, refs: list[ReferencePuzzle], *, arc_code_dir: Path, annotation_dir: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "arc_code_dir": str(arc_code_dir),
        "annotation_dir": str(annotation_dir),
        "source": source,
        "count": len(refs),
        "references": [reference_to_json(ref) for ref in refs],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_reference_index(path: Path) -> list[ReferencePuzzle]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs_data = data.get("references")
    if not isinstance(refs_data, list):
        raise RuntimeError(f"Invalid reference index, missing references list: {path}")
    return [reference_from_json(item) for item in refs_data]


def load_tag_index(path: Path) -> dict[str, frozenset[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    refs_data = data.get("references")
    if not isinstance(refs_data, list):
        raise RuntimeError(f"Invalid tag index, missing references list: {path}")
    out: dict[str, frozenset[str]] = {}
    for item in refs_data:
        if not isinstance(item, dict):
            continue
        puzzle_id = str(item.get("puzzle_id", ""))
        if not puzzle_id:
            continue
        out[puzzle_id] = frozenset(map(str, item.get("tags", [])))
    return out


def load_references_from_jsons(
    reference_paths: list[Path],
    *,
    tag_index_path: Path,
    source: str,
) -> list[ReferencePuzzle]:
    tag_map = load_tag_index(tag_index_path)
    refs_by_id: dict[str, ReferencePuzzle] = {}
    for path in reference_paths:
        for ref in load_reference_index(path):
            if source != "both" and ref.source != source:
                continue
            tags = tag_map.get(ref.puzzle_id, frozenset())
            refs_by_id[ref.puzzle_id] = ReferencePuzzle(
                puzzle_id=ref.puzzle_id,
                source=ref.source,
                code_path=ref.code_path,
                docstring=ref.docstring,
                annotation_path=ref.annotation_path,
                annotation=ref.annotation,
                annotation_type=ref.annotation_type,
                tags=tags,
                tokens=ref.tokens,
                code_features=ref.code_features,
            )
    return [refs_by_id[puzzle_id] for puzzle_id in sorted(refs_by_id)]


def extract_explicit_reference_ids(text: str) -> frozenset[str]:
    # ARC puzzle ids are hexadecimal-like ids. Treat ids named in 来源原语/参考题
    # as first-class signals instead of rediscovering them by similarity only.
    return frozenset(re.findall(r"\b[0-9a-f]{8,10}\b", text, flags=re.IGNORECASE))


def extract_reference_markdown_paths(text: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    pattern = re.compile(r"/home/lixinyu/snap/a/key_combo/select/[^\s`，,]+\.md")
    for match in pattern.findall(text):
        paths.append(Path(match))
    return tuple(dict.fromkeys(paths))


def load_reference_ids_from_markdown(path: Path) -> frozenset[str]:
    if not path.is_file():
        return frozenset()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return frozenset()
    ids = re.findall(r"^- puzzle_id:\s*`?([0-9a-f]{8,10})`?", text, flags=re.MULTILINE | re.IGNORECASE)
    return frozenset(ids)


def extract_referenced_group_ids(text: str) -> frozenset[str]:
    ids: set[str] = set()
    for path in extract_reference_markdown_paths(text):
        ids.update(load_reference_ids_from_markdown(path))
    return frozenset(ids)


STRUCTURED_HINT_KEYWORDS = (
    "提取的原语",
    "原语",
    "操作类",
    "对象类",
    "空间关系类",
    "条件/判断类",
    "组合逻辑",
    "来源原语",
    "依赖链",
    "主要机制",
    "参考题",
    "参考题组",
    "tags",
    "标签",
)


def extract_declared_primitive_terms(text: str) -> frozenset[str]:
    terms: set[str] = set()
    for inner in re.findall(r"[（(]([A-Za-z][A-Za-z0-9_/\- ]{1,80})[）)]", text):
        for part in re.split(r"[/,，、\s]+", inner):
            normalized = part.strip().lower().replace("-", "_")
            if len(normalized) >= 2 and re.fullmatch(r"[a-z][a-z0-9_]*", normalized):
                terms.add(normalized)
    for item in re.findall(r"\b(operation|object|relation|condition|output|code):([A-Za-z0-9_\-]+)\b", text):
        terms.add(item[1].lower().replace("-", "_"))
    return frozenset(terms)


DECLARED_TAG_PREFIXES = {
    "operation",
    "object",
    "relation",
    "condition",
    "output",
    "code",
    "type",
    "primitive",
}


def extract_declared_tags(body: str) -> frozenset[str]:
    """Return the current puzzle's Markdown tags verbatim, without keyword remapping."""
    lines = body.splitlines()
    tags: list[str] = []
    in_tags = False
    tags_indent = -1
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if re.fullmatch(r"-?\s*tags\s*:\s*", stripped, flags=re.IGNORECASE):
            in_tags = True
            tags_indent = indent
            continue
        if not in_tags:
            continue
        if not stripped:
            continue
        if indent <= tags_indent or not stripped.startswith("-"):
            break
        value = stripped[1:].strip().strip("`")
        match = re.match(r"^([A-Za-z_]+)\s*:\s*(.+)$", value)
        if match is None or match.group(1).lower() not in DECLARED_TAG_PREFIXES:
            continue
        tag_value = re.sub(r"\s+", " ", match.group(2).strip())
        tags.append(f"{match.group(1).lower()}:{tag_value}")
    return frozenset(tags)


def extract_structured_hint_text(body: str) -> str:
    hint_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("> - ") or any(key in stripped for key in STRUCTURED_HINT_KEYWORDS):
            hint_lines.append(stripped.lstrip("> "))
    return "\n".join(hint_lines)


def extract_file_header(full_text: str) -> str:
    match = re.search(r"^##\s+", full_text, flags=re.MULTILINE)
    return full_text[: match.start()] if match else full_text


def make_new_puzzle(index: str, desc: str, *, full_text: str) -> NewPuzzle:
    declared_tags = extract_declared_tags(desc)
    if declared_tags:
        tags = declared_tags
        primitive_terms = frozenset()
    else:
        hint_text = extract_structured_hint_text(desc)
        signal_text = f"{hint_text}\n{desc}"
        tags = extract_tags(signal_text, catalog_terms=None)
        primitive_terms = extract_declared_primitive_terms(signal_text)
        tags = frozenset(set(tags) | {f"primitive:{term}" for term in primitive_terms})
    expected = expected_code_features_from_tags(tags)
    explicit_refs = extract_explicit_reference_ids(desc)
    group_refs = extract_referenced_group_ids(extract_file_header(full_text))
    token_text = f"{desc}\n{' '.join(sorted(tags))}\n{' '.join(sorted(primitive_terms))}"
    return NewPuzzle(
        index=index,
        description=desc,
        tags=tags,
        tokens=tokenize(token_text),
        expected_code_features=expected,
        explicit_reference_ids=explicit_refs,
        group_reference_ids=group_refs,
    )


def parse_heading_puzzles(text: str) -> list[tuple[str, str]]:
    heading = re.compile(r"^##\s+新题\s*([0-9A-Za-z_-]+)[^\n]*\n", re.MULTILINE)
    matches = list(heading.finditer(text))
    if not matches:
        heading = re.compile(r"^##\s+([0-9A-Za-z_-]+)(?:\.\s+[^\n]*)?\n", re.MULTILINE)
        matches = list(heading.finditer(text))
    blocks: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[start:end].strip()))
    return blocks


def parse_new_puzzles(des_path: Path) -> list[NewPuzzle]:
    text = des_path.read_text(encoding="utf-8")
    puzzles: list[NewPuzzle] = []
    heading_blocks = parse_heading_puzzles(text)
    if heading_blocks:
        for fallback_idx, (idx, body) in enumerate(heading_blocks, start=1):
            index = str(idx).strip() if str(idx).strip() else str(fallback_idx)
            puzzles.append(make_new_puzzle(index, body.strip(), full_text=text))
        return puzzles

    raise RuntimeError(f"No key_combo '## 新题 N' headings found in {des_path}")
    return puzzles


def load_generation_specs(spec_dir: Path) -> dict[str, GenerationSpec]:
    if not spec_dir.is_dir():
        return {}
    specs: dict[str, GenerationSpec] = {}
    for path in sorted(spec_dir.glob("*/*_spec.yaml")):
        puzzle_id = path.name.removesuffix("_spec.yaml")
        if puzzle_id in specs:
            raise RuntimeError(
                f"Duplicate generation specs for {puzzle_id}: "
                f"{specs[puzzle_id].path}, {path}"
            )
        specs[puzzle_id] = GenerationSpec(
            path=str(path),
            content=path.read_text(encoding="utf-8", errors="replace").rstrip(),
        )
    return specs


def score_reference(puzzle: NewPuzzle, ref: ReferencePuzzle) -> ScoredReference:
    mechanism = weighted_jaccard(puzzle.tags, ref.tags)
    lexical = token_jaccard(puzzle.tokens, ref.tokens)
    code_score = weighted_jaccard(puzzle.expected_code_features, ref.code_features)
    final = 0.68 * mechanism + 0.20 * lexical + 0.12 * code_score
    if ref.puzzle_id in puzzle.explicit_reference_ids:
        final += 1.0
        mechanism = max(mechanism, 1.0)
    elif ref.puzzle_id in puzzle.group_reference_ids:
        final += 0.18
    overlap = tuple(sorted(set(puzzle.tags) & set(ref.tags)))
    return ScoredReference(
        ref=ref,
        score=final,
        mechanism_score=mechanism,
        lexical_score=lexical,
        code_feature_score=code_score,
        overlap_tags=overlap,
    )


def select_references(
    puzzle: NewPuzzle,
    refs: list[ReferencePuzzle],
    *,
    top_k: int,
    candidate_pool: int,
    mmr_lambda: float,
) -> list[ScoredReference]:
    scored = sorted((score_reference(puzzle, ref) for ref in refs), key=lambda x: x.score, reverse=True)
    pool = scored[: max(candidate_pool, top_k)]
    selected: list[ScoredReference] = []
    while pool and len(selected) < top_k:
        best_i = 0
        best_val = -math.inf
        for i, item in enumerate(pool):
            if not selected:
                val = item.score
            else:
                diversity_penalty = max(weighted_jaccard(item.ref.tags, chosen.ref.tags) for chosen in selected)
                val = item.score - mmr_lambda * diversity_penalty
            if val > best_val:
                best_i = i
                best_val = val
        selected.append(pool.pop(best_i))
    return selected


def format_annotation(annotation: dict[str, Any] | None) -> str:
    if not annotation:
        return "No annotation JSON was found."
    return json.dumps(annotation, ensure_ascii=False, indent=2)


def build_reference_block(
    item: ScoredReference,
    index: int,
    *,
    include_code_excerpt: bool,
    max_code_chars: int,
    generation_specs: dict[str, GenerationSpec],
) -> str:
    ref = item.ref
    generation_spec = generation_specs.get(ref.puzzle_id)
    spec_path = generation_spec.path if generation_spec is not None else "N/A"
    spec_content = (
        f"```yaml\n{generation_spec.content}\n```"
        if generation_spec is not None
        else "No input-generation spec was found for this reference puzzle."
    )
    code_excerpt = ""
    if include_code_excerpt:
        code = Path(ref.code_path).read_text(encoding="utf-8", errors="ignore")
        if len(code) > max_code_chars:
            code = code[:max_code_chars] + "\n# ... truncated ..."
        code_excerpt = f"""
<code_excerpt>
```python
{code}
```
</code_excerpt>"""

    return f"""<reference_puzzle rank="{index}" source="{ref.source}" puzzle_id="{ref.puzzle_id}">
<similarity>
score={item.score:.4f}
mechanism={item.mechanism_score:.4f}
lexical={item.lexical_score:.4f}
code_feature={item.code_feature_score:.4f}
overlap_tags={", ".join(item.overlap_tags) if item.overlap_tags else "none"}
</similarity>
<paths>
code={ref.code_path}
annotation={ref.annotation_path or "N/A"}
generation_spec={spec_path}
</paths>
<description>
{ref.docstring}
</description>
<annotation>
{format_annotation(ref.annotation)}
</annotation>
<generation_spec>
{spec_content}
</generation_spec>{code_excerpt}
</reference_puzzle>"""


def build_codegen_prompt(
    puzzle: NewPuzzle,
    puzzle_dir: Path,
    out_base: Path,
    selected: list[ScoredReference],
    *,
    include_code_excerpt: bool,
    max_code_chars: int,
    code_suffix: str,
    generation_specs: dict[str, GenerationSpec],
) -> str:
    target_path = out_base / f"{puzzle.index}{code_suffix}"
    refs_text = "\n\n".join(
        build_reference_block(
            item,
            i,
            include_code_excerpt=include_code_excerpt,
            max_code_chars=max_code_chars,
            generation_specs=generation_specs,
        )
        for i, item in enumerate(selected, start=1)
    )
    selected_paths = "\n".join(f"- `{item.ref.code_path}`" for item in selected)
    selected_spec_paths = "\n".join(
        f"- `{generation_specs[item.ref.puzzle_id].path}`"
        for item in selected
        if item.ref.puzzle_id in generation_specs
    ) or "- none"
    selected_ids = ", ".join(f"{item.ref.source}/{item.ref.puzzle_id}" for item in selected)
    return f"""你要为一个新的 ARC-AGI 风格题目实现代码。代码文件必须包含 `main(grid: np.ndarray) -> np.ndarray` 和 `generate_input() -> np.ndarray`。

## 新题位置

- 描述目录：`{puzzle_dir}`
- 题号：`{puzzle.index}`
- 目标代码路径：`{target_path}`

## 新题描述

<new_puzzle index="{puzzle.index}">
{puzzle.description}
</new_puzzle>

## 机制相似参考题

下面为参考题。每道参考题还附带其输入生成 spec。参考题只能用于学习对象识别方式、随机生成策略、约束、拒绝条件和边界处理，禁止照搬旧题规则或固定参数。

已选参考：{selected_ids}

{refs_text}

## 必读参考代码路径

执行前请至少快速阅读这些文件的 `main()` 和 `generate_input()`，重点看它们如何避免退化输入、如何随机放置对象、如何验证隐式假设：

{selected_paths}

## 参考输入生成 spec 路径

下面 YAML spec 也已嵌入各自的 `<reference_puzzle>`。它们只用于帮助设计 `generate_input()` 的输入分布、随机维度、构造步骤、分支覆盖与反退化检查；新题 Markdown 始终是唯一规则来源，不得把参考题的对象、颜色、尺寸或变换规则复制成新题规则。

{selected_spec_paths}

## 公共参考工具箱

执行前还必须快速阅读 `/home/lixinyu/snap/new/solvers/arc_utils.py`。它是 ARC 公共工具箱，包含统一的 `(r, c)` 坐标约定、bbox、连通域、裁剪、几何变换、颜色和网格操作等公共实现。生成代码时优先参考或复用这些函数；若直接 import，请确保目标文件仍可被 `SourceFileLoader` 直接加载，自检命令能独立运行。

## 实现要求

1. `main(grid)` 必须实现新题描述中的确定性转换规则，不能只识别 `generate_input()` 的固定模板。禁止不同的input网格输出相同的output网格。
2. `generate_input()` 必须随机生成多样样例，不能固定死。随机维度应覆盖：网格大小、对象数量、位置、颜色、方向、形状参数或关键分支。注意若描述中存在若干，需实现而不是只有单个，包括object或者对应关系等等。且若涉及到最短路径，则输入网格中必涉及到多条路径，对选择造成干扰。
3. 生成器必须保证输入满足 `main()` 的隐式假设：对象不重叠、不越界、特殊标记唯一或可区分、匹配关系无歧义、关键分支出现概率相同。
4. 禁止把 ARC 颜色 id 、亮度、拼写等当作人类不可见的数值语义参与排序、大小比较或隐藏映射。颜色变化必须有可见结构线索。注意颜色亮度、拼写等也是不可见信息，若描述中出现违规的颜色id或者亮度，可自行使用少量的固定颜色替代，但注意生成时要在三个以内输入输出网格中对所有变换100%覆盖度。
5. 输出样例要视觉清晰，人类必须能从输入输出对中肉眼辨认出规律。具体要求：避免对象粘连、覆盖关系混乱、同一像素多重语义；不同逻辑步骤产生的结果不要大面积层叠在一起导致无法分辨哪个颜色属于哪个步骤；输入中各元素之间保持足够间距，输出中新增的标记/填充/路径等应与背景和已有元素在颜色上有明显区分；若规则涉及多层叠加，生成时控制对象密度和尺寸，确保叠加区域只占少数像素而非铺满整个网格。
6. 若规则有多个分支，生成器不能长期固定成一个分支；应让分支在随机样例中合理出现。分支概率不必平均——难以发现的隐蔽分支（如少数情况下才触发的特殊规则）应适当提高出现概率，确保在 3 个训练示例中至少出现一次，否则求解者无法归纳该规则。对于颜色映射类规则，确保 2 个左右的示例即可覆盖所有映射关系（每种颜色→颜色的映射至少在某个示例中出现过）。
7. 代码风格参考 `ARC_code`：使用 `numpy`，并把 `/home/lixinyu/snap/new/solvers/arc_utils.py` 作为公共参考工具箱，优先复用其中已经稳定的 bbox、连通域、裁剪、变换和颜色工具；目标文件必须能被 `SourceFileLoader` 直接加载。
8. 可以在目标目录放公共 helper 文件，或从 `arc_utils.py` 摘取少量必要函数；若只实现一道题，优先保持目标文件自包含，减少额外依赖。
9. `main(grid)` 必须带一段 docstring，按 `ARC_code` 参考题的格式分四部分书写：首行用一句话概述转换目标；随后 `Input structure:` 描述输入网格中的对象/背景/标记/边界/组件；`Rule:` 给出从输入到输出的确定性变换规则；`Key insight:` 点明解题核心洞见。该 docstring 供下游相似度/索引流程（按 `main` 的 docstring 抽取机制标签）使用，缺失会导致此题无法被收录为参考题。
10. 网格尺寸必须符合 ARC 规范：`generate_input()` 产生的输入网格、以及 `main()` 返回的输出网格，行数和列数都必须 ≤ 30（即任何一维都不超过 30，整体不超过 30×30）。随机化网格大小时把上界封顶在 30；若规则会放大网格（如平铺、加边框、整数倍缩放），必须保证放大后的输出仍 ≤ 30×30，必要时收紧输入尺寸或放大倍数的随机范围，确保 100% 的随机样例都不越过 30×30。

## 写入要求

将完整代码写入：

`{target_path}`

默认使用 `.py` 后缀，便于编辑器、lint 和 Python 工具识别；如果调用方传入空后缀，则按无扩展名文件写入。

## 自检要求

写完后必须运行类似下面的验证，并修复失败：

```bash
python - <<'PY'
from importlib.machinery import SourceFileLoader
from pathlib import Path
import hashlib
import numpy as np

p = Path("{target_path}")
mod = SourceFileLoader("new_arc_task", str(p)).load_module()
sigs = set()
for i in range(100):
    inp = mod.generate_input()
    out = mod.main(inp)
    assert isinstance(inp, np.ndarray)
    assert isinstance(out, np.ndarray)
    assert inp.ndim == 2 and max(inp.shape) <= 30, f"input grid exceeds 30x30: {{inp.shape}}"
    assert out.ndim == 2 and max(out.shape) <= 30, f"output grid exceeds 30x30: {{out.shape}}"
    sigs.add(hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest())
print("variants", len(sigs))
assert len(sigs) >= 20
PY
python -m py_compile "{target_path}"
```

最后清理验证产生的 `__pycache__`。
"""


def write_outputs(
    out_base: Path,
    puzzle: NewPuzzle,
    selected: list[ScoredReference],
    prompt: str,
    *,
    prompt_subdir: str,
    report_subdir: str,
    ids_subdir: str,
    code_suffix: str,
    generation_specs: dict[str, GenerationSpec],
) -> dict[str, Any]:
    prompt_dir = out_base / prompt_subdir
    report_dir = out_base / report_subdir
    ids_dir = out_base / ids_subdir
    prompt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = prompt_dir / f"{puzzle.index}.md"
    report_path = report_dir / f"{puzzle.index}.json"
    ids_path = ids_dir / f"{puzzle.index}.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    ids_path.write_text("\n".join(f"{item.ref.source}/{item.ref.puzzle_id}" for item in selected) + "\n", encoding="utf-8")
    report = {
        "puzzle_index": puzzle.index,
        "target_path": str(out_base / f"{puzzle.index}{code_suffix}"),
        "new_tags": sorted(puzzle.tags),
        "expected_code_features": sorted(puzzle.expected_code_features),
        "explicit_reference_ids": sorted(puzzle.explicit_reference_ids),
        "group_reference_ids": sorted(puzzle.group_reference_ids),
        "selected": [
            {
                "rank": rank,
                "puzzle_id": item.ref.puzzle_id,
                "source": item.ref.source,
                "code_path": item.ref.code_path,
                "annotation_path": item.ref.annotation_path,
                "generation_spec_path": generation_specs[item.ref.puzzle_id].path
                if item.ref.puzzle_id in generation_specs
                else None,
                "score": item.score,
                "mechanism_score": item.mechanism_score,
                "lexical_score": item.lexical_score,
                "code_feature_score": item.code_feature_score,
                "overlap_tags": list(item.overlap_tags),
            }
            for rank, item in enumerate(selected, start=1)
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "puzzle_index": puzzle.index,
        "target_path": str(out_base / f"{puzzle.index}{code_suffix}"),
        "prompt_path": str(prompt_path),
        "report_path": str(report_path),
        "ids_path": str(ids_path),
        "selected_ids": [f"{item.ref.source}/{item.ref.puzzle_id}" for item in selected],
    }


def markdown_inputs_for_dir(path: Path) -> list[PuzzleInput]:
    return [
        PuzzleInput(puzzle_dir=path, des_path=md_path)
        for md_path in sorted(path.glob("*.md"))
        if md_path.is_file()
    ]


def discover_puzzle_inputs(args: argparse.Namespace) -> list[PuzzleInput]:
    inputs: list[PuzzleInput] = []
    if args.des_path:
        for p in args.des_path:
            path = Path(p)
            if path.suffix.lower() != ".md":
                raise RuntimeError(f"--des_path must point to a Markdown file: {path}")
            inputs.append(PuzzleInput(puzzle_dir=path.parent, des_path=path))
    if args.puzzle_dir:
        for p in args.puzzle_dir:
            path = Path(p)
            inputs.extend(markdown_inputs_for_dir(path))
    if args.root:
        root = Path(args.root)
        if args.folders:
            for folder in args.folders:
                path = root / folder
                inputs.extend(markdown_inputs_for_dir(path))
        else:
            inputs.extend(
                PuzzleInput(puzzle_dir=root, des_path=md_path)
                for md_path in sorted(root.glob("*.md"))
                if md_path.is_file()
            )
            for path in sorted(root.iterdir()):
                if path.is_dir():
                    inputs.extend(markdown_inputs_for_dir(path))
    unique: list[PuzzleInput] = []
    seen: set[Path] = set()
    for item in inputs:
        resolved = item.des_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(PuzzleInput(puzzle_dir=item.puzzle_dir.resolve(), des_path=resolved))
    if not unique:
        raise RuntimeError("No puzzle inputs specified. Use --des_path, --puzzle_dir, or --root.")
    for item in unique:
        if not item.des_path.is_file():
            raise RuntimeError(f"Missing Markdown description file: {item.des_path}")
    return unique


def output_base_for_input(puzzle_input: PuzzleInput, out_dir: Path | None) -> Path:
    if out_dir is None:
        raise RuntimeError("--out_dir is required for key_combo fable code generation")
    return out_dir / puzzle_input.puzzle_dir.name / puzzle_input.des_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ARC code-generation prompts with annotation-based similarity references."
    )
    parser.add_argument("--puzzle_dir", action="append", default=[], help="key_combo round directory containing 001.md, 002.md, etc.; may be repeated")
    parser.add_argument("--des_path", action="append", default=[], help="path to a key_combo Markdown puzzle description file; may be repeated")
    parser.add_argument("--root", default=None, help="root containing numbered key_combo round directories")
    parser.add_argument("--folders", nargs="*", default=None, help="folder names under --root, e.g. 1 2")
    parser.add_argument("--arc_code_dir", type=Path, default=DEFAULT_ARC_CODE_DIR)
    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--tmp_dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument(
        "--spec_dir",
        type=Path,
        default=DEFAULT_SPEC_DIR,
        help="root containing <dataset>/*_spec.yaml input-generation references",
    )
    parser.add_argument("--source", choices=["eval", "train", "both"], default="both")
    parser.add_argument(
        "--tag_index",
        type=Path,
        default=DEFAULT_TAG_INDEX_PATH,
        help="JSON whose references provide authoritative puzzle tags",
    )
    parser.add_argument(
        "--reference_json",
        action="append",
        type=Path,
        default=None,
        help="reference index JSON containing docstring/annotation/tokens/code_features; may be repeated",
    )
    parser.add_argument(
        "--write_reference_index",
        type=Path,
        default=None,
        help="deprecated; kept for compatibility but not used by the JSON-based loader",
    )
    parser.add_argument(
        "--refresh_reference_index",
        action="store_true",
        help="deprecated; kept for compatibility but not used by the JSON-based loader",
    )
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--candidate_pool", type=int, default=40)
    parser.add_argument("--mmr_lambda", type=float, default=0.18)
    parser.add_argument("--prompt_subdir", default="codegen_prompts")
    parser.add_argument("--report_subdir", default="similarity_reports")
    parser.add_argument("--ids_subdir", default="similar_ids")
    parser.add_argument("--code_suffix", default=".py", help="suffix for generated code files; use '' for extensionless files")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="required output root; all generated artifacts (code, codegen_prompts, "
        "similarity_reports, similar_ids, manifest) go under <out_dir>/<puzzle_dir_name>/<md_stem>/.",
    )
    parser.add_argument("--include_reference_code", action="store_true", help="embed truncated reference code excerpts")
    parser.add_argument("--max_code_chars", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    puzzle_inputs = discover_puzzle_inputs(args)
    catalog_terms = load_catalog_terms(args.tmp_dir)
    reference_paths = args.reference_json if args.reference_json else list(DEFAULT_REFERENCE_JSON_PATHS)
    refs = load_references_from_jsons(
        reference_paths,
        tag_index_path=args.tag_index,
        source=str(args.source),
    )
    generation_specs = load_generation_specs(args.spec_dir)
    print(
        "[OK] loaded reference JSONs: "
        + ", ".join(str(path) for path in reference_paths)
        + f" ({len(refs)} refs)"
    )
    print(f"[OK] loaded tag index: {args.tag_index}")
    print(f"[OK] loaded generation specs: {args.spec_dir} ({len(generation_specs)} specs)")
    if args.write_reference_index is not None:
        write_reference_index(
            args.write_reference_index,
            refs,
            arc_code_dir=args.arc_code_dir,
            annotation_dir=args.annotation_dir,
            source=args.source,
        )
        print(f"[OK] wrote reference index: {args.write_reference_index} ({len(refs)} refs)")
    if not refs:
        raise RuntimeError("No reference puzzles loaded from reference JSONs")

    for puzzle_input in puzzle_inputs:
        puzzle_dir = puzzle_input.puzzle_dir
        des_path = puzzle_input.des_path
        out_base = output_base_for_input(puzzle_input, args.out_dir)
        dir_manifest: list[dict[str, Any]] = []
        puzzles = parse_new_puzzles(des_path)
        for puzzle in puzzles:
            selected = select_references(
                puzzle,
                refs,
                top_k=int(args.top_k),
                candidate_pool=int(args.candidate_pool),
                mmr_lambda=float(args.mmr_lambda),
            )
            prompt = build_codegen_prompt(
                puzzle,
                puzzle_dir,
                out_base,
                selected,
                include_code_excerpt=bool(args.include_reference_code),
                max_code_chars=int(args.max_code_chars),
                code_suffix=str(args.code_suffix),
                generation_specs=generation_specs,
            )
            record = write_outputs(
                out_base,
                puzzle,
                selected,
                prompt,
                prompt_subdir=str(args.prompt_subdir),
                report_subdir=str(args.report_subdir),
                ids_subdir=str(args.ids_subdir),
                code_suffix=str(args.code_suffix),
                generation_specs=generation_specs,
            )
            dir_manifest.append(record)
            print(f"[OK] {puzzle_dir.name}/{puzzle.index}: prompt={record['prompt_path']}")
            print(f"[OK] selected: {', '.join(record['selected_ids'])}")

        out_base.mkdir(parents=True, exist_ok=True)
        manifest_path = out_base / "codegen_manifest.json"
        manifest_path.write_text(json.dumps(dir_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()

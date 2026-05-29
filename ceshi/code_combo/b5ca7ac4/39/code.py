# Combo variant of b5ca7ac4 (index 39)
# combo_spec: D1/1+D3/0+D4/1
# Name: 边框厚度标识+可变尺寸+面积分组
#
# Design analysis
# a) 具体化与行为追踪
# - 输入里只有两类可移动对象，并且三条线索被绑定在一起：
#   7x7 大框 = 2 像素厚边框 = 大面积组；
#   5x5 小框 = 1 像素厚边框 = 小面积组。
# - 两类对象都保持“单色边框 + 单色 3x3 中心”的原题外观，但所有对象共用同一种边框颜色，
#   因此颜色不再提供任何分组信息。
# - 融合后的空间规则沿用原题：一组贴左边并按原列从小到大处理，另一组贴右边并按原列从大到小处理，
#   行位置保持不变，行重叠则只进第二层。
# - 具体追踪：原题里一个 5x5 青框会去左侧，一个 5x5 红框会去右侧；现在若两者都换成同色边框，
#   但一个是 7x7 厚框、一个是 5x5 薄框，则 7x7 厚框去左侧，5x5 薄框去右侧。差异来自可见几何属性，
#   而不是颜色 id。
#
# b) 复杂度校准
# - 只使用 5x5/7x7 两档，避免“面积分组”变成过宽的自由度，同时让人类能在少量样例中稳定看出
#   “大框一组、小框一组”。
# - 7x7 恰好对应 2 像素边框包 3x3 中心，5x5 恰好对应 1 像素边框包 3x3 中心，厚度与尺寸完全一致，
#   避免出现“厚但不大”或“大但不厚”的冲突样本。
# - 所有框共享同一边框颜色，主动打掉原题的颜色二分法，迫使观察者使用厚度/尺寸/面积三者中的任意
#   一个可见几何线索归纳同一条规则。
#
# c) 隐式假设审查
# - 原 solver 固定扫描 5x5 且只认红/青边框；融合后必须先扫 7x7 厚框，再扫 5x5 薄框，并允许任意
#   非背景边框色。
# - 7x7 厚框的中心 5x5 会伪装成薄框，因此检测顺序必须是厚框优先，并屏蔽被厚框完全包含的 5x5 窗口。
# - 原代码把“组”直接绑定到颜色；现在组绑定到面积，且面积与厚度、尺寸一一对应。
# - 原代码的 inward 位移固定是 5；现在左组宽度为 7、右组宽度为 5，偏移要按各组自身宽度计算。
# - 原代码默认所有对象高度都一样；现在重叠判断必须显式使用各对象自己的尺寸。
#
# d) 放置合法性
# - 左侧最多两层 7x7，右侧最多两层 5x5，因此画布扩到 26x26，给左右堆叠留出明确间隔。
# - 生成器继续采用 output-first：先决定输出里的边层/内层行，再反推输入列，最后用 main(input)==output
#   验证。
# - 输入对象要求彼此至少留 1 格空隙，降低同色边框相邻时出现伪检测窗口的风险。
#
# e) 视觉可观察性
# - 观察者会看到：所有边框颜色都一样，但大厚框总去左边，小薄框总去右边；因此规则无法退回成颜色分组。
# - 输出同时保留原题“谁先靠边、谁被挤到第二列”的动态效果，让融合后的差异集中在“按面积/尺寸/厚度
#   分组”而不是别的隐藏机制上。
#
# f) 可视化整洁度
# - 仍只允许每侧 edge + inward 两层，避免三层以上堆叠造成混乱。
# - 左区横向占 14 列，右区横向占 10 列，中间保留空隙；对象内部都是纯色 3x3，小样例里也容易辨认。
#
# g) 变化的可推断性
# - 本变体不引入颜色映射；颜色只承担“区分不同对象”的身份作用。
# - 由于框架共享边框色，且 5x5/7x7 在样例中重复出现，人类可以从 2-4 组样例中直接归纳
#   “几何大小/边框厚度决定去向”。
#
# h) main() 的新假设与 generator 的对应保证
# - main() 假设合法对象只有两种：5x5 薄框和 7x7 厚框，中心都是 3x3 纯色块；generator 只构造这两类。
# - main() 假设 5x5 真对象不会嵌在 7x7 真对象内部；generator 不会造这种结构，因此“厚框优先 + 包含抑制”
#   是安全的。
# - main() 假设每组最多两层；generator 通过输出行分配保证只会出现边层和一层 inward。

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


GRID_H = 26
GRID_W = 26
THIN_SIZE = 5
THICK_SIZE = 7
INNER_SIZE = 3
THIN_THICKNESS = 1
THICK_THICKNESS = 2
MAX_INWARD = 2
ROW_ASSIGN_RETRIES = 80
INPUT_COL_RETRIES = 160
GEN_RETRIES = 1200
N_LARGE_RANGE = (3, 4)
N_SMALL_RANGE = (3, 4)
INPUT_MIN_GAP = 1


def _span_overlaps(a0: int, a_len: int, b0: int, b_len: int) -> bool:
    return not (int(a0 + a_len - 1) < int(b0) or int(b0 + b_len - 1) < int(a0))


def _rects_overlap(a: tuple[int, int, int], b: tuple[int, int, int], min_gap: int = 0) -> bool:
    ar, ac, asz = map(int, a)
    br, bc, bsz = map(int, b)
    pad = int(min_gap)
    return not (
        ar + asz - 1 + pad < br
        or br + bsz - 1 + pad < ar
        or ac + asz - 1 + pad < bc
        or bc + bsz - 1 + pad < ac
    )


def _any_pairwise_overlap(entries: list[dict], min_gap: int = 0) -> bool:
    rects = [(int(e["r0"]), int(e["c0"]), int(e["size"])) for e in entries]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_overlap(rects[i], rects[j], min_gap=min_gap):
                return True
    return False


def _uniform_non_bg(values: np.ndarray, bg: int) -> int | None:
    if values.size == 0:
        return None
    if int(values.min()) != int(values.max()):
        return None
    color = int(values.flat[0])
    if color == bg:
        return None
    return color


def _make_patch(size: int, thickness: int, frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((size, size), int(frame_color), dtype=int)
    patch[thickness : thickness + INNER_SIZE, thickness : thickness + INNER_SIZE] = int(inner_color)
    return patch


def _thick_border_mask() -> np.ndarray:
    mask = np.zeros((THICK_SIZE, THICK_SIZE), dtype=bool)
    mask[:THICK_THICKNESS, :] = True
    mask[-THICK_THICKNESS:, :] = True
    mask[:, :THICK_THICKNESS] = True
    mask[:, -THICK_THICKNESS:] = True
    return mask


THICK_BORDER_MASK = _thick_border_mask()


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    found: list[dict] = []
    thick_boxes: list[tuple[int, int, int, int]] = []

    for r0 in range(H - THICK_SIZE + 1):
        for c0 in range(W - THICK_SIZE + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + THICK_SIZE - 1), int(c0 + THICK_SIZE - 1)))
            frame_col = _uniform_non_bg(patch[THICK_BORDER_MASK], bg)
            if frame_col is None:
                continue
            inner_col = _uniform_non_bg(patch[THICK_THICKNESS : THICK_THICKNESS + INNER_SIZE,
                                              THICK_THICKNESS : THICK_THICKNESS + INNER_SIZE], bg)
            if inner_col is None or int(inner_col) == int(frame_col):
                continue

            found.append(
                {
                    "group": "large",
                    "size": THICK_SIZE,
                    "thickness": THICK_THICKNESS,
                    "frame": int(frame_col),
                    "inner": int(inner_col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch.copy(),
                }
            )
            thick_boxes.append((int(r0), int(c0), int(r0 + THICK_SIZE - 1), int(c0 + THICK_SIZE - 1)))

    for r0 in range(H - THIN_SIZE + 1):
        for c0 in range(W - THIN_SIZE + 1):
            if any(
                tr0 <= r0
                and tc0 <= c0
                and tr1 >= r0 + THIN_SIZE - 1
                and tc1 >= c0 + THIN_SIZE - 1
                for tr0, tc0, tr1, tc1 in thick_boxes
            ):
                continue

            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + THIN_SIZE - 1), int(c0 + THIN_SIZE - 1)))
            border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
            frame_col = _uniform_non_bg(border, bg)
            if frame_col is None:
                continue
            inner_col = _uniform_non_bg(
                patch[THIN_THICKNESS : THIN_THICKNESS + INNER_SIZE, THIN_THICKNESS : THIN_THICKNESS + INNER_SIZE],
                bg,
            )
            if inner_col is None or int(inner_col) == int(frame_col):
                continue

            found.append(
                {
                    "group": "small",
                    "size": THIN_SIZE,
                    "thickness": THIN_THICKNESS,
                    "frame": int(frame_col),
                    "inner": int(inner_col),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch.copy(),
                }
            )

    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move framed squares to opposite edges according to the large-area vs. small-area group.

    Input structure:
    - The background is the most common color.
    - Every movable object is a square frame with a uniform 3x3 center, but there are two geometries:
      7x7 frames with a 2-pixel border and 5x5 frames with a 1-pixel border.
    - All frames may share the same border color, so color is not the grouping cue.

    Rule:
    - Large 7x7 thick-border frames go to the left edge; small 5x5 thin-border frames go to the right edge.
    - Each frame keeps its original top row.
    - Large frames are processed by original column ascending; small frames are processed by original column descending.
    - If a frame row-overlaps an already placed edge frame from its own group, it shifts inward by its group's width
      instead of taking the edge lane.

    Key insight: Thickness, size, and area all encode the same split, so the solver must read the visible geometry of
    each frame rather than its border color.
    """
    g = np.array(grid, dtype=int)
    _H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    frames = _detect_frames(g, bg)

    large_frames = sorted((f for f in frames if f["group"] == "large"), key=lambda f: int(f["c0"]))
    large_edge_rows: list[int] = []
    for frame in large_frames:
        r0 = int(frame["r0"])
        has_overlap = any(_span_overlaps(r0, THICK_SIZE, er, THICK_SIZE) for er in large_edge_rows)
        dest_c = THICK_SIZE if has_overlap else 0
        paste(out, frame["patch"], r0, dest_c, transparent=bg)
        if dest_c == 0:
            large_edge_rows.append(r0)

    small_frames = sorted((f for f in frames if f["group"] == "small"), key=lambda f: -int(f["c0"]))
    small_edge_rows: list[int] = []
    for frame in small_frames:
        r0 = int(frame["r0"])
        has_overlap = any(_span_overlaps(r0, THIN_SIZE, er, THIN_SIZE) for er in small_edge_rows)
        dest_c = int(W - 2 * THIN_SIZE) if has_overlap else int(W - THIN_SIZE)
        paste(out, frame["patch"], r0, dest_c, transparent=bg)
        if dest_c == int(W - THIN_SIZE):
            small_edge_rows.append(r0)

    return out


def _max_non_overlapping_rows(size: int) -> int:
    return (GRID_H + size - 1) // size


def _assign_output_rows(n: int, size: int) -> tuple[list[int], list[int]] | None:
    max_edge = _max_non_overlapping_rows(size)
    min_inward = max(1, n - max_edge)
    max_inward = min(MAX_INWARD, n - 1)
    if min_inward > max_inward:
        return None

    for _ in range(ROW_ASSIGN_RETRIES):
        n_inward = random.randint(min_inward, max_inward)
        n_edge = n - n_inward

        all_rows = list(range(GRID_H - size + 1))
        random.shuffle(all_rows)
        edge_rows: list[int] = []
        for r in all_rows:
            if all(not _span_overlaps(r, size, er, size) for er in edge_rows):
                edge_rows.append(int(r))
                if len(edge_rows) == n_edge:
                    break
        if len(edge_rows) < n_edge:
            continue

        random.shuffle(all_rows)
        inward_rows: list[int] = []
        for r in all_rows:
            if not any(_span_overlaps(r, size, er, size) for er in edge_rows):
                continue
            if any(_span_overlaps(r, size, ir, size) for ir in inward_rows):
                continue
            inward_rows.append(int(r))
            if len(inward_rows) == n_inward:
                break
        if len(inward_rows) == n_inward:
            return edge_rows, inward_rows
    return None


def _build_output_objects(
    rows_edge: list[int],
    rows_inward: list[int],
    size: int,
    thickness: int,
    frame_color: int,
    inner_colors: list[int],
    edge_col: int,
    inward_col: int,
    group: str,
) -> list[dict]:
    objects: list[dict] = []
    color_idx = 0
    for r0 in rows_edge:
        patch = _make_patch(size, thickness, frame_color, inner_colors[color_idx])
        objects.append(
            {
                "group": group,
                "r0": int(r0),
                "c0": int(edge_col),
                "size": int(size),
                "thickness": int(thickness),
                "is_edge": True,
                "patch": patch,
            }
        )
        color_idx += 1
    for r0 in rows_inward:
        patch = _make_patch(size, thickness, frame_color, inner_colors[color_idx])
        objects.append(
            {
                "group": group,
                "r0": int(r0),
                "c0": int(inward_col),
                "size": int(size),
                "thickness": int(thickness),
                "is_edge": False,
                "patch": patch,
            }
        )
        color_idx += 1
    return objects


def _build_processing_order(objects: list[dict]) -> list[dict]:
    edges = [obj for obj in objects if bool(obj["is_edge"])]
    inwards = [obj for obj in objects if not bool(obj["is_edge"])]
    size = int(objects[0]["size"])

    deps = [
        [idx for idx, edge in enumerate(edges) if _span_overlaps(int(inward["r0"]), size, int(edge["r0"]), size)]
        for inward in inwards
    ]

    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

    inward_indices = list(range(len(inwards)))
    random.shuffle(inward_indices)
    for inward_idx in inward_indices:
        latest_pos = max(
            (
                pos
                for dep in deps[inward_idx]
                for pos, (kind, idx) in enumerate(order)
                if kind == "edge" and idx == dep
            ),
            default=-1,
        )
        insert_pos = random.randint(latest_pos + 1, len(order))
        order.insert(insert_pos, ("inward", inward_idx))

    return [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]


def _conflicts(candidate: dict, placed: list[dict], blocked: list[dict], min_gap: int) -> bool:
    rect_a = (int(candidate["r0"]), int(candidate["c0"]), int(candidate["size"]))
    for other in placed + blocked:
        rect_b = (int(other["r0"]), int(other["c0"]), int(other["size"]))
        if _rects_overlap(rect_a, rect_b, min_gap=min_gap):
            return True
    return False


def _derive_input_positions(objects: list[dict], ascending: bool, blocked: list[dict] | None = None) -> list[dict] | None:
    blocked = [] if blocked is None else blocked

    for _ in range(INPUT_COL_RETRIES):
        ordered = _build_processing_order(objects)
        placed: list[dict] = []
        ok = True

        for idx, obj in enumerate(ordered):
            size = int(obj["size"])
            out_c = int(obj["c0"])
            if ascending:
                lower = 0 if idx == 0 else int(placed[-1]["c0"]) + 1
                upper = GRID_W - size
            else:
                lower = 0
                upper = GRID_W - size if idx == 0 else int(placed[-1]["c0"]) - 1
            if lower > upper:
                ok = False
                break

            candidates = [c for c in range(lower, upper + 1) if int(c) != out_c]
            random.shuffle(candidates)

            chosen = None
            for c0 in candidates:
                candidate = {
                    "group": obj["group"],
                    "r0": int(obj["r0"]),
                    "c0": int(c0),
                    "size": int(size),
                    "thickness": int(obj["thickness"]),
                    "patch": obj["patch"],
                }
                if _conflicts(candidate, placed, blocked, min_gap=INPUT_MIN_GAP):
                    continue
                chosen = candidate
                break

            if chosen is None:
                ok = False
                break
            placed.append(chosen)

        if ok and not _any_pairwise_overlap(placed, min_gap=INPUT_MIN_GAP):
            return placed

    return None


def _group_counts_from_detection(frames: list[dict]) -> tuple[int, int]:
    n_large = sum(1 for frame in frames if frame["group"] == "large")
    n_small = sum(1 for frame in frames if frame["group"] == "small")
    return n_large, n_small


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    bg = int(random.randint(0, 9))
    frame_choices = [c for c in range(10) if c not in {bg, 2, 8}]
    if not frame_choices:
        return None
    frame_color = int(random.choice(frame_choices))

    n_large = int(random.randint(*N_LARGE_RANGE))
    n_small = int(random.randint(*N_SMALL_RANGE))

    large_rows = _assign_output_rows(n_large, THICK_SIZE)
    if large_rows is None:
        return None
    small_rows = _assign_output_rows(n_small, THIN_SIZE)
    if small_rows is None:
        return None
    large_edge_rows, large_inward_rows = large_rows
    small_edge_rows, small_inward_rows = small_rows

    available_inner = [c for c in range(10) if c not in {bg, frame_color}]
    if len(available_inner) < n_large + n_small:
        return None
    random.shuffle(available_inner)
    large_inner = available_inner[:n_large]
    small_inner = available_inner[n_large : n_large + n_small]

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    large_out = _build_output_objects(
        large_edge_rows,
        large_inward_rows,
        THICK_SIZE,
        THICK_THICKNESS,
        frame_color,
        large_inner,
        edge_col=0,
        inward_col=THICK_SIZE,
        group="large",
    )
    small_out = _build_output_objects(
        small_edge_rows,
        small_inward_rows,
        THIN_SIZE,
        THIN_THICKNESS,
        frame_color,
        small_inner,
        edge_col=GRID_W - THIN_SIZE,
        inward_col=GRID_W - 2 * THIN_SIZE,
        group="small",
    )

    output_objects = large_out + small_out
    if _any_pairwise_overlap(output_objects):
        return None
    for obj in output_objects:
        paste(output, obj["patch"], int(obj["r0"]), int(obj["c0"]))

    large_input = _derive_input_positions(large_out, ascending=True)
    if large_input is None:
        return None
    small_input = _derive_input_positions(small_out, ascending=False, blocked=large_input)
    if small_input is None:
        return None

    input_objects = large_input + small_input
    if _any_pairwise_overlap(input_objects, min_gap=INPUT_MIN_GAP):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for obj in input_objects:
        paste(inp, obj["patch"], int(obj["r0"]), int(obj["c0"]))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    detected = _detect_frames(inp, bg)
    det_large, det_small = _group_counts_from_detection(detected)
    if det_large != n_large or det_small != n_small:
        return None

    info = {
        "bg": int(bg),
        "frame_color": int(frame_color),
        "n_large": int(n_large),
        "n_small": int(n_small),
        "large_inward": int(len(large_inward_rows)),
        "small_inward": int(len(small_inward_rows)),
    }
    return inp, output, info


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, _out, _info = result
            return inp
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

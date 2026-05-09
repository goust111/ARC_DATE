# Combo variant of 35ab12c3 (index 1)
# combo_spec: D2/1+D10/1
# Name: 仅对角线路径+仅闭合多边形

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior + trace
#    - 原题允许 H / V / 45° 连接，且 2 点开放线也合法；融合后这两条都收紧：
#      只有完整闭合的 4 顶点对角菱形会扩张，任何正交段或开放线都不动。
#    - 一个最小追踪例子：
#      shape 色点为 {(0,2),(2,4),(4,2),(2,0)}，这是等臂菱形。
#      old: 4 条边都可画，输出补齐整个菱形；new: 仍补齐整个菱形。
#      distractor 色点为 {(0,0),(0,5)}，这是 2 点横线。
#      old: len==2 且水平对齐，输出补齐横线；new: len<4，直接跳过，保持两孤点。
#    - 因此融合后的可见差异不是“形状画法变了”，而是：
#      只有对角闭环被视作形状；任何开放或正交组都失去可扩张资格。
#
# b) Complexity calibration
#    - 若只限制 solver、不限制 generator，样本会大量退化成“几乎不发生变化”。
#    - 若保留三角形/五边形等闭环，45°-only 会让大部分候选边失效，训练样例会很脏。
#    - 所以 generator 只产出一种主形状：4 顶点等臂菱形；再固定加入 1 个 2 点正交
#      distractor，既保持视觉稳定，也保证和原题逻辑存在稳定分歧。
#
# c) Hidden assumptions audit on main()
#    - _segment_cells 依赖“端点沿允许方向对齐”；融合后仍成立，因为每条合法边都先经过
#      严格对角判定。
#    - 原题允许 len==2 的开放线；这个分支会破坏“仅闭合多边形”，因此 combo solver 必须
#      把最小顶点数提高到 4，并要求整个 cycle 全部合法，否则整色跳过。
#    - angle_cycle 不保存拓扑，只保存几何环序；因此 generator 必须只构造凸菱形，使角序
#      就是目标边序。
#    - carried shadow 仍从带异色邻居的起始顶点平移复制；generator 必须保证该 shadow
#      anchor 只贴着一个顶点，且偏移不与任何菱形边共线。
#
# d) Placement legality
#    - 继续采用 output-first：先画完整菱形及其影子，再把输入退化成 4 个顶点和单个
#      shadow-anchor 像素。
#    - patch bbox 覆盖主形状与影子副本，放置时用 random_free_location(..., padding=1,
#      connectivity=8) 保证对象间至少隔 1 格背景。
#    - orth distractor 自身就是 input==output 的惰性补丁，不与其它对象接触。
#
# e) Visual observability
#    - 人类从 2-4 个样例能直接看到：所有会变化的对象都长成菱形边框，横线/竖线两点组
#      不再被连起来。
#    - 训练集中重复出现“菱形扩张 + 正交两点不动”，足以同时暴露 diagonal-only 与
#      closed-only 两条规则。
#
# f) Visual cleanliness
#    - 只保留单一主形状族，避免 pentagon/hexagon 之类混合方向边带来的噪声。
#    - 影子只复制一层，且只能是与菱形边不共线的 8 邻域偏移，视觉上是清晰的平行轮廓。
#
# g) Change inferability
#    - 颜色不携带额外编码；每种颜色只标识一个对象或一个影子层。
#    - 关键结构线索来自“4 顶点等臂对角闭环”和“2 点正交孤组”，而不是颜色值本身。
#
# New solver assumptions introduced by the combo
# ----------------------------------------------
# 1) 真正会扩张的 shape 色组恰好由 4 个顶点组成，并形成一个等臂对角菱形。
# 2) 任何会变化的 shape 都必须整圈可画；不能再接受“部分边可画”的半成品。
# 3) 若存在 shadow，则其 anchor 只邻接一个顶点，且全局留白保证没有别的异色邻居干扰。

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_diag_only(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return dr != 0 and abs(dr) == abs(dc)


def _aligned_legacy(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _diamond_like_cycle(cycle: list[tuple[int, int]]) -> bool:
    if len(cycle) != 4:
        return False

    edge_lengths: list[int] = []
    for i in range(4):
        r0, c0 = cycle[i]
        r1, c1 = cycle[(i + 1) % 4]
        if not _aligned_diag_only(r0, c0, r1, c1):
            return False
        edge_lengths.append(abs(r1 - r0))

    if min(edge_lengths) <= 0 or len(set(edge_lengths)) != 1:
        return False

    rows = sorted({r for r, _ in cycle})
    cols = sorted({c for _, c in cycle})
    if len(rows) != 3 or len(cols) != 3:
        return False

    mid_r = rows[1]
    mid_c = cols[1]
    if sum(r == mid_r for r, _ in cycle) != 2:
        return False
    if sum(c == mid_c for _, c in cycle) != 2:
        return False
    if sum(r < mid_r for r, _ in cycle) != 1 or sum(r > mid_r for r, _ in cycle) != 1:
        return False
    if sum(c < mid_c for _, c in cycle) != 1 or sum(c > mid_c for _, c in cycle) != 1:
        return False
    return True


def _start_index_and_carried(
    cycle: list[tuple[int, int]],
    g: np.ndarray,
    bg: int,
    col: int,
) -> tuple[int, list[tuple[int, int, int]]]:
    h, w = g.shape
    seed_idxs: list[int] = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_idxs.append(i)
                    break

    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

    carried: list[tuple[int, int, int]] = []
    sr0, sc0 = cycle[start]
    for drn, dcn in DIR8:
        rr = sr0 + drn
        cc = sc0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                carried.append((drn, dcn, v))
    return start, carried


def _solve(
    grid: np.ndarray,
    *,
    aligned_fn,
    min_vertices: int,
    require_full_cycle: bool,
    require_diamond: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < min_vertices:
            continue

        cycle = _angle_cycle(pts_rc)
        if require_diamond and not _diamond_like_cycle(cycle):
            continue

        start, carried = _start_index_and_carried(cycle, g, bg, col)
        cycle = cycle[start:] + cycle[:start]

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        valid_cycle = True
        for (r0, c0), (r1, c1) in edges:
            if not aligned_fn(r0, c0, r1, c1):
                valid_cycle = False
                if require_full_cycle:
                    break
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                valid_cycle = False
                if require_full_cycle:
                    break
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        if require_full_cycle and not valid_cycle:
            continue

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        h, w = g.shape
        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not aligned_fn(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# Overview:
#   This combo keeps the original "connect same-color vertices and copy one neighboring
#   shadow color in parallel" mechanic, but only for fully closed 4-vertex diamonds.
# Input structure:
#   Each actionable shape color appears as four isolated vertices of an equal-arm diamond.
#   At least one shadow color appears as one 8-neighbor pixel next to exactly one diamond vertex.
#   One extra distractor color is always a 2-pixel horizontal or vertical pair.
# Rule:
#   Ignore any color group that is not a complete diagonal diamond. For each valid diamond,
#   draw all four diagonal edges, then copy the chosen neighboring color along the same
#   closed boundary translated by the anchor offset when that translated cell is background.
# Key insight:
#   The fused constraint is not merely "diagonal edges" or "closed loops" separately; the
#   only expandable objects are diagonal closed loops, which in this task reduces to diamonds.
def main(grid: np.ndarray) -> np.ndarray:
    return _solve(
        grid,
        aligned_fn=_aligned_diag_only,
        min_vertices=4,
        require_full_cycle=True,
        require_diamond=True,
    )


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(
        grid,
        aligned_fn=_aligned_legacy,
        min_vertices=2,
        require_full_cycle=False,
        require_diamond=False,
    )


# --------------- generator (combo D2/1 + D10/1) ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
MIN_SHADOWS_PER_INSTANCE = 1
DISTRACTOR_PROB = 1.0
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

DIAMOND_ARM_RANGE = (2, 6)
ORTH_DISTRACTOR_SPAN_RANGE = (4, 11)

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 400


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_diamond_vertices() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    raw = _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])
    cycle = _angle_cycle(raw)
    if not _diamond_like_cycle(cycle):
        raise RuntimeError("diamond builder produced invalid cycle")
    return cycle


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _build_diamond_vertices()
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, shape_color, int(shadow_color), offset)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _make_orth_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    span = int(np.random.randint(*ORTH_DISTRACTOR_SPAN_RANGE))
    horizontal = bool(np.random.rand() < 0.5)
    if horizontal:
        patch = np.full((1, span + 1), BG, dtype=int)
        patch[0, 0] = color
        patch[0, span] = color
    else:
        patch = np.full((span + 1, 1), BG, dtype=int)
        patch[0, 0] = color
        patch[span, 0] = color
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    if max_shadows < MIN_SHADOWS_PER_INSTANCE:
        return None
    shadow_count = int(np.random.randint(MIN_SHADOWS_PER_INSTANCE, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_orth_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if has_distractor and np.array_equal(_legacy_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")

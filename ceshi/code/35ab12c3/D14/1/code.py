# Variant of 35ab12c3
# Dimension: D14 — 阴影锚点检测规则 (src)
# Change: 8邻居检测——在起始顶点（最小坐标的有阴影邻居的顶点）的8邻居中寻找不同颜色非背景像素作为阴影锚点，偏移量=锚点位置-顶点位置
#   → 锚点可在任意顶点附近
#
# Design analysis for the variant
# --------------------------------
# a) Concrete behavior
#    原题里，某个形状色 `col` 先被还原成顶点循环，然后只查看“起始顶点”附近的异色像素；
#    这些像素的相对偏移会被复制到整条轮廓上，形成 1 组阴影。D14 改成：每个顶点都要检查，
#    所有顶点附近的异色像素都可以独立触发整形状的平行阴影。
#    例子：若一个矩形的左上角有一个红色锚点 `(-1,0)`，右下角有一个蓝色锚点 `(1,1)`，
#    则输出应同时出现一份红色的上移矩形轮廓和一份蓝色的右下移矩形轮廓。
#    原逻辑只会保留“起始顶点”那一份，另一份会被完全忽略。
# b) Complexity calibration
#    为了让变体可见，生成器强制每个样例至少有 2 个阴影，且至少 1 个形状在不同顶点上带 2 个锚点。
#    这样样例不会退化成原题的“单顶点单阴影”。
# c) Hidden assumptions in the original solver
#    - `seed_idxs` / `start`: 原代码假设阴影信息只需要决定一个起始顶点。变体后这个起点只影响轮廓旋转，
#      不再控制阴影收集范围，因此保留它只作稳定排序用途。
#    - `carried` 只从 `cycle[0]` 读取：这是 D14 直接破坏的旧假设，必须改成扫描全部顶点。
#    - Phase 2 用 `_draw_line_bg_only`：这隐含“不同阴影副本最好不要互相覆盖，否则顺序会吞掉后者”。
#      因此生成器会避免同一形状的多个阴影副本彼此重叠。
#    - `_clear_in_original(... allow=set())`: 阴影线的内部路径不能穿过任何输入非背景像素。
#      因此生成器还要避免一个阴影的输入锚点落到另一个阴影副本的内部路径上。
# d) Placement legality
#    仍采用 output-first patch 构造。每个形状 patch 先画主轮廓，再尝试选 1-2 个不同顶点的锚点。
#    候选锚点要求：8 邻域、不是主轮廓上的格子、只邻接该形状的唯一一个顶点、偏移不与任一主边共线。
#    组合出多个锚点后，再要求它们生成的整形状平移副本两两不重叠，最后用 `main(inp) == out` 做 oracle 验证。
# e) Visual observability
#    人眼在 2-4 个样例中能直接看到“同一形状的多个顶点各自挂着不同颜色小点，输出出现多组不同方向的平行轮廓”。
#    这里没有使用颜色 id 做排序或映射；颜色只作为可见分组标签。
# f) Visual cleanliness
#    通过限制每个形状最多 2 个阴影、要求阴影副本彼此不重叠、并保留对象间 1 格缓冲，避免画面混乱。
# g) Color inference
#    每个阴影颜色只出现为一个输入锚点和一组输出阴影，语义稳定：颜色就是该组阴影的颜色。
#    不存在隐式颜色变换表。
#
# Post-change solver assumptions and generator guarantees
# ------------------------------------------------------
# 1. Solver assumes every shadow anchor pixel is 8-adjacent to exactly one vertex of its host shape.
#    Generator enforces this in `_shadow_candidates`.
# 2. Solver assumes distinct shadow copies of the same shape do not need tie-breaking on overlap.
#    Generator enforces pairwise-disjoint shifted shape-cell sets.
# 3. Solver still treats colors with <2 pixels as non-shapes; generator keeps every shadow color as a single input pixel.

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
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


def _collect_shadow_specs(
    g: np.ndarray, cycle: list[tuple[int, int]], col: int, bg: int
) -> list[tuple[int, int, int]]:
    """Collect unique shadow anchors from all vertices, not only the chosen start vertex."""
    h, w = g.shape
    hits: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for vr, vc in cycle:
        for drn, dcn in DIR8:
            rr = vr + drn
            cc = vc + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    hits.setdefault((rr, cc), []).append((drn, dcn, v))

    carried: list[tuple[int, int, int]] = []
    for anchor_rc in sorted(hits):
        choices = hits[anchor_rc]
        if len(choices) == 1:
            carried.append(choices[0])
    return carried


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs = []
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

        cycle = cycle[start:] + cycle[:start]
        carried = _collect_shadow_specs(g, cycle, col, bg)

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

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
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator (D14 multi-anchor variant) ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
MIN_TOTAL_SHADOWS = 2
PRIMARY_SHAPE_SHADOWS = 2
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHADOW_COMBO_ATTEMPTS = 100
SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _shift_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _shadow_candidates(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> list[dict[str, object]]:
    directions = _edge_dirs(vertices)
    candidates: list[dict[str, object]] = []
    for vidx, anchor in enumerate(vertices):
        for dr, dc in DIR8:
            if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
                continue
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in shape_cells:
                continue
            adjacent_vertices = sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1)
            if adjacent_vertices != 1:
                continue
            candidates.append(
                {
                    "vertex_idx": vidx,
                    "anchor": anchor,
                    "offset": (dr, dc),
                    "anchor_pos": pos,
                    "shifted_cells": _shift_cells(shape_cells, (dr, dc)),
                }
            )
    return candidates


def _choose_shadows(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    shadow_colors: list[int],
) -> list[tuple[tuple[int, int], tuple[int, int], int]] | None:
    if not shadow_colors:
        return []

    candidates = _shadow_candidates(vertices, shape_cells)
    if not candidates:
        return None

    for _ in range(SHADOW_COMBO_ATTEMPTS):
        order = np.random.permutation(len(candidates))
        used_vertices: set[int] = set()
        used_offsets: set[tuple[int, int]] = set()
        used_shifted: list[set[tuple[int, int]]] = []
        selected: list[dict[str, object]] = []

        for cand_idx in order:
            cand = candidates[int(cand_idx)]
            vertex_idx = int(cand["vertex_idx"])
            offset = tuple(cand["offset"])
            shifted_cells = cand["shifted_cells"]
            if vertex_idx in used_vertices or offset in used_offsets:
                continue
            if any(not shifted_cells.isdisjoint(prev) for prev in used_shifted):
                continue
            selected.append(cand)
            used_vertices.add(vertex_idx)
            used_offsets.add(offset)
            used_shifted.append(shifted_cells)
            if len(selected) == len(shadow_colors):
                break

        if len(selected) != len(shadow_colors):
            continue

        out: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        for cand, shadow_color in zip(selected, shadow_colors):
            out.append((tuple(cand["anchor"]), tuple(cand["offset"]), int(shadow_color)))
        return out

    return None


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadows: list[tuple[tuple[int, int], tuple[int, int], int]],
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    for anchor, (dr, dc), _shadow_color in shadows:
        coords.append((anchor[0] + dr, anchor[1] + dc))
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


def _make_shape_patch(shape_color: int, shadow_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)

    shadows = _choose_shadows(vertices, cells, shadow_colors)
    if shadows is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadows)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    seen_anchor_vertices: set[tuple[int, int]] = set()
    for anchor, offset, shadow_color in shadows:
        local_anchor = (anchor[0] - r0, anchor[1] - c0)
        seen_anchor_vertices.add(local_anchor)
        _paint_shadow(out, shape_color, shadow_color, offset)
        sa = (local_anchor[0] + offset[0], local_anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w):
            return None
        if inp[sa] != BG:
            return None
        inp[sa] = shadow_color
        if sum(1 for vr, vc in shifted_vertices if max(abs(sa[0] - vr), abs(sa[1] - vc)) == 1) != 1:
            return None

    if len(shadow_colors) >= 2 and len(seen_anchor_vertices) < 2:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < MIN_TOTAL_SHADOWS:
        return None
    shadow_count = int(np.random.randint(MIN_TOTAL_SHADOWS, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, list[int]]:
    if len(shadow_colors) < PRIMARY_SHAPE_SHADOWS:
        return {}
    primary = int(np.random.randint(shape_count))
    mapping: dict[int, list[int]] = {primary: [int(v) for v in shadow_colors[:PRIMARY_SHAPE_SHADOWS]]}
    remaining = shadow_colors[PRIMARY_SHAPE_SHADOWS:]
    if remaining:
        others = [idx for idx in range(shape_count) if idx != primary]
        extra_target = int(np.random.choice(others)) if others else primary
        mapping.setdefault(extra_target, []).append(int(remaining[0]))
    return mapping


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
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    multi_anchor_shape_count = 0
    for idx, shape_color in enumerate(shape_colors):
        placed = False
        wanted_shadows = shadow_map.get(idx, [])
        if len(wanted_shadows) >= 2:
            multi_anchor_shape_count += 1
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), wanted_shadows)
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if multi_anchor_shape_count < 1:
        return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")

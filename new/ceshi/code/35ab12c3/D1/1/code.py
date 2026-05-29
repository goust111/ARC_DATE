# Variant of 35ab12c3
# Dimension: D1 — 匹配条件——同色配对规则 (match_cond)
# Change: color_match——同色像素配对（s1.color == s2.color），同一颜色的所有像素按极角排序后形成多... → 形状匹配配对
#
# Design analysis
# a) Concrete behavior:
#    原题把“同色点集”当作一个 shape；这里改成“同局部 marker 形状的中心点集”当作一个 shape。
#    输入中的顶点不再总是单像素，而是三种固定局部模式之一：
#    - hbar: 3 格横条，中心在中点
#    - vbar: 3 格竖条，中心在中点
#    - plus: 5 格十字，中心在交叉点
#    同一模式的所有中心点按极角排序后连线；颜色只负责决定这条线/多边形用什么颜色画，
#    不再决定哪些顶点属于同一组。为让差异可见，生成器会让不同模式复用同一种颜色。
#    小例子：若有两组都用红色，一组是 hbar 顶点，一组是 plus 顶点，原逻辑会把全部红色像素
#    混成一个组；变体只会分别连接 hbar 中心和 plus 中心。
# b) Complexity calibration:
#    只用 3 种 marker，避免自由度过大；每个样例放 2-3 个 pattern-group，且强制至少两组复用同色，
#    让“不是按颜色连”的现象在 2-3 个样例中足够稳定可见。
# c) Hidden assumptions review:
#    - 原 `colors = ...` 假定颜色就是组 id；这里失效，必须先做组件识别和 pattern 分类。
#    - 原 `_clear_in_original(... allow={col})` 假定同色像素都属于同一组；这里会误穿过别的同色组。
#      现在改成 `allow_coords`，只允许穿过当前 pattern-group 自己的 marker 像素。
#    - 原 seed 检测把“不同色邻居”当 shadow；这里 marker 本身是同色多像素，所以 seed 只能看
#      “不属于 marker 的额外邻居像素”。生成器把 shadow 锚点限制在对角邻居，避免跟 marker 混淆。
# d) Placement legality:
#    生成器保证 marker 组件彼此不重叠、不 4-连通合并；不同 patch 以 output 结果做 padding 放置，
#    避免主线、shadow、marker 互相覆盖。shadow 只允许对角偏移，且锚点只 8 邻接一个中心。
# e) Visual observability:
#    解题者能直接看到：同样颜色的顶点有时不会彼此相连，只有同样的小形状会连。marker 形状是
#    肉眼可辨的局部几何特征，不依赖颜色 id 排序或隐藏属性。
# f) Visual cleanliness:
#    marker 只用 hbar / vbar / plus 三种整洁形状；对象间保留至少 1 格输出缓冲，避免杂乱重叠。
# g) Color inferability:
#    颜色不做映射，只保留原题“线的颜色 = 该组顶点颜色；shadow 颜色 = 邻接锚点颜色”。
#    变体只改变分组条件，因此颜色规则依旧容易读。
# h) Variant metric used after testing:
#    统计样本中“存在至少一对不同 pattern-group 复用同色”的比例，以及这些样本上
#    `main(input)` 与“原始按颜色分组的 solver”输出不同的比例；这两个指标若接近 1，说明
#    变体行为确实被触发而非退化。
# i) New solver assumptions audited:
#    - 只识别 hbar / vbar / plus 三种 marker；生成器只产这三类。
#    - 同一 pattern-group 的中心颜色一致；生成器保证同一 pattern 不会出现多种主色。
#    - shadow 锚点不是 marker 组件的一部分，且与组外对象保持间距；生成器保证。

from __future__ import annotations

import math
from collections import deque

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    get_bg,
    get_bg as _bg_color,
    neighbors8,
    paste,
    random_free_location,
)

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

PATTERN_MASKS = {
    "hbar": {(0, -1), (0, 0), (0, 1)},
    "vbar": {(-1, 0), (0, 0), (1, 0)},
    "plus": {(-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)},
}
PATTERN_IDS = ["hbar", "vbar", "plus"]


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


def _clear_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow_coords: set[tuple[int, int]],
) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        if (rr, cc) in allow_coords:
            continue
        if int(g[rr, cc]) != bg:
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


def _same_color_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            col = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                cells.add((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": col, "cells": cells})
    return comps


def _recognize_marker(cells: set[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    n = len(cells)
    if n not in (3, 5):
        return None
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if sum(rs) % n != 0 or sum(cs) % n != 0:
        return None
    center = (sum(rs) // n, sum(cs) // n)
    if center not in cells:
        return None
    rel = {(r - center[0], c - center[1]) for r, c in cells}
    for pid in PATTERN_IDS:
        if rel == PATTERN_MASKS[pid]:
            return pid, center
    return None


def _detect_markers(g: np.ndarray, bg: int) -> list[dict]:
    markers: list[dict] = []
    for comp in _same_color_components(g, bg):
        found = _recognize_marker(comp["cells"])
        if found is None:
            continue
        pid, center = found
        markers.append(
            {
                "pattern": pid,
                "center": center,
                "color": int(comp["color"]),
                "cells": set(comp["cells"]),
            }
        )
    return markers


def _extra_neighbors(g: np.ndarray, center: tuple[int, int], marker_cells: set[tuple[int, int]], bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    r0, c0 = center
    carried: list[tuple[int, int, int]] = []
    for dr, dc in DIR8:
        rr = r0 + dr
        cc = c0 + dc
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if (rr, cc) in marker_cells:
            continue
        v = int(g[rr, cc])
        if v != bg:
            carried.append((dr, dc, v))
    return carried


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    markers = _detect_markers(g, bg)
    for pattern_id in PATTERN_IDS:
        group = [m for m in markers if m["pattern"] == pattern_id]
        if len(group) < 2:
            continue

        group_colors = {m["color"] for m in group}
        if len(group_colors) != 1:
            continue
        col = next(iter(group_colors))

        pts_rc = [m["center"] for m in group]
        cycle = _angle_cycle(pts_rc)
        center_to_cells = {m["center"]: set(m["cells"]) for m in group}
        allow_coords = set().union(*(m["cells"] for m in group))

        seed_idxs = []
        for i, center in enumerate(cycle):
            if _extra_neighbors(g, center, center_to_cells[center], bg):
                seed_idxs.append(i)
        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
        cycle = cycle[start:] + cycle[:start]

        carried = _extra_neighbors(g, cycle[0], center_to_cells[cycle[0]], bg)

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow_coords):
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
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow_coords=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)  # 2-3 pattern-groups; at most one per pattern.
MAX_SHADOWS_CAP = 2
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 6)
LINE_TOTAL_LENGTH = 16
LINE_MIN_GAP = 4
TRIANGLE_SIDE_RANGE = (4, 8)
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
DIAMOND_ARM_RANGE = (3, 6)
PENTAGON_ROOF_RANGE = (3, 5)
PENTAGON_WALL_RANGE = (3, 6)
HEXAGON_SLANT_RANGE = (2, 4)
HEXAGON_WALL_RANGE = (3, 5)

SHAPE_WEIGHTS = np.array([0.30, 0.20, 0.18, 0.14, 0.10, 0.08], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 400


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


def _shape_line_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _pattern_cells(center: tuple[int, int], pattern_id: str) -> set[tuple[int, int]]:
    return {(center[0] + dr, center[1] + dc) for dr, dc in PATTERN_MASKS[pattern_id]}


def _footprints_well_formed(vertices: list[tuple[int, int]], pattern_id: str) -> bool:
    footprints = [_pattern_cells(v, pattern_id) for v in vertices]
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            if footprints[i] & footprints[j]:
                return False
            for r0, c0 in footprints[i]:
                for r1, c1 in footprints[j]:
                    if abs(r0 - r1) + abs(c0 - c1) == 1:
                        return False
    return True


def _choose_shadow(vertices: list[tuple[int, int]], marker_cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIAG_DIRS:
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in marker_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _shape_bbox(
    vertices: list[tuple[int, int]],
    pattern_id: str,
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    marker_cells = set().union(*[_pattern_cells(v, pattern_id) for v in vertices])
    line_cells = _shape_line_cells(vertices)
    coords = list(marker_cells | line_cells)
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in line_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _new_cells_from_solver(inp: np.ndarray) -> int:
    out = main(inp)
    return int(np.count_nonzero(out != inp))


def _make_shape_patch(shape_color: int, pattern_id: str, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    if not _footprints_well_formed(vertices, pattern_id):
        return None

    marker_cells = set().union(*[_pattern_cells(v, pattern_id) for v in vertices])
    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, marker_cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _shape_bbox(vertices, pattern_id, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for vr, vc in shifted_vertices:
        for rr, cc in _pattern_cells((vr, vc), pattern_id):
            inp[rr, cc] = int(shape_color)

    if shadow is not None:
        (ar, ac), offset = shadow
        sa = (ar - r0 + offset[0], ac - c0 + offset[1])
        if inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)

    out = main(inp)
    if np.array_equal(out, inp):
        return None
    if _new_cells_from_solver(inp) < 4:
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


def _sample_shape_colors(shape_count: int) -> list[int]:
    pool = list(np.random.permutation(COLORS))
    if shape_count == 2:
        shared = int(pool[0])
        return [shared, shared]
    shared = int(pool[0])
    other = int(pool[1])
    order = [shared, shared, other]
    np.random.shuffle(order)
    return order


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None]:
    shape_colors = _sample_shape_colors(shape_count)
    used = set(shape_colors)
    free = [c for c in COLORS if c not in used]
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(free) - int(has_distractor))
    shadow_count = int(np.random.randint(max_shadows + 1))
    np.random.shuffle(free)
    shadow_colors = [int(c) for c in free[:shadow_count]]
    distractor_color = int(free[shadow_count]) if has_distractor else None
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
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    pattern_ids = list(np.random.choice(PATTERN_IDS, size=shape_count, replace=False))

    shape_colors, shadow_colors, distractor_color = _sample_palette(shape_count, has_distractor)
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    for idx, (shape_color, pattern_id) in enumerate(zip(shape_colors, pattern_ids)):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), pattern_id, shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    unique_shape_colors = len(set(shape_colors))
    total_colors = unique_shape_colors + len(shadow_colors) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None

    # Force the core variant behavior to be visible in almost every sample:
    # at least two groups share a color, and the solver adds enough structure.
    if unique_shape_colors >= shape_count:
        return None
    if int(np.count_nonzero(out != inp)) < 12:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")

# Variant of 35ab12c3
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: continuous_trail——沿路径的每个背景格子都涂上形状颜色，形成连续实线... → 间歇跳跃轨迹
#
# Design analysis
# ===============
# variant_spec:
#   preserved_core:
#     - 输入仍由若干同色顶点组构成；每组代表一条折线或一个多边形的顶点集合。
#     - 起始顶点若 8 邻接一个异色像素，则该像素仍然表示一层平行偏移的阴影颜色。
#     - 边的可绘制性仍由 H/V/45° 对齐与“原输入该边中间不能被其他颜色阻塞”决定。
#   D4_change:
#     - 原题会把每条可绘制边上的所有背景格都涂成该形状颜色。
#     - 本变体改为“虚线轨迹”：
#       - 若该颜色组只有 2 个顶点，则对每条边使用 `paint 1, skip 1`。
#       - 若该颜色组有 3 个及以上顶点，则对每条边使用 `paint 2, skip 1`。
#     - 两端顶点始终保留着色；周期只作用于边的内部格子。
#     - 阴影不是连续复制，而是沿完全相同的节拍复制，所以主轨迹和阴影轨迹同样是虚线。
#   why_this_is_observable:
#     - 输入里直接可见“同色顶点数量”，所以控制信号不是颜色 id 这类不可见信息。
#     - 节拍只保留两档，便于人从 2-4 组样例中归纳：
#       - 2 点组 -> 稀疏虚线
#       - 3+ 点组 -> 稍密虚线
#     - 生成器保证大多数网格里同时出现两类形状，并拒绝那些虚线效果退化成实线的几何。
#   concrete_trace:
#     - 例：同色顶点在 `(2,2)` 与 `(2,7)`，原题会输出 `2,2..7` 全连成 6 格线段。
#     - 本变体中该组顶点数为 2，所以内部格 `(2,3..6)` 按 `涂1跳1` 处理：
#       - 顶点 `(2,2)`、`(2,7)` 保留；
#       - `(2,3)` 涂、`(2,4)` 空、`(2,5)` 涂、`(2,6)` 空。
#     - 若起点邻接一个阴影色偏移 `(1,0)`，则阴影平行线也在 `(3,2..7)` 上按同样节拍出现。
#   implicit_assumption_audit:
#     - 原 `main()` 的 `_draw_line_bg_only()` 隐含“轨迹连续填满”这一旧前提；变体下该函数必须改成按周期跳涂。
#     - `_clear_in_original()` 只检查原输入是否有阻塞，不依赖轨迹是否连续，仍然成立。
#     - `out = g.copy()` 很关键：主图形顶点和单个阴影锚点已在输入里，所以即使虚线阶段跳过某些端点位置，
#       输出里的真实顶点仍被保留；但阴影复制的其余端点不在输入里，因此绘制函数必须显式“总是保留边端点”。
#     - 生成器若直接用虚线输出补丁做碰撞检测，会把虚线空洞当作可放置空位，导致别的对象钻进原本应保留的几何走廊，
#       从而让 solver 的 `_clear_in_original()` 失败。因此放置阶段必须使用“连续保留占位图(reserve)”。
#   placement_legality:
#     - 本地 patch 仍采用 output-first：先画期望输出，再反推出输入只保留顶点和单个阴影锚点。
#     - 但额外维护 reserve patch：主边和阴影边都按连续线保留位置，用于全局随机放置与 1 格缓冲。
#     - 这样即使实际输出是虚线，也不会有别的对象落进其跳过的格子。
#   cleanliness:
#     - 阴影仍只允许非共线偏移，避免主边与阴影边在同一直线上缠绕。
#     - 几何生成会优先包含一个 2 点组和一个 3+ 点组，并拒绝“虚线与实线无区别”的形状，保证视觉上能看到 D4 改动。

import math
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

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


def _segment_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


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


def _trail_pattern(vertex_count: int) -> tuple[int, int]:
    if vertex_count <= 2:
        return 1, 1
    return 2, 1


def _trail_hit(idx: int, steps: int, paint_len: int, skip_len: int) -> bool:
    if idx == 0 or idx == steps:
        return True
    cycle = paint_len + skip_len
    return ((idx - 1) % cycle) < paint_len


def _draw_pattern_bg_only(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    bg: int,
    paint_len: int,
    skip_len: int,
) -> None:
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for idx, (rr, cc) in enumerate(_segment_cells(r0, c0, r1, c1)):
        if out[rr, cc] == bg and _trail_hit(idx, steps, paint_len, skip_len):
            out[rr, cc] = col


def _draw_pattern_solid(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    paint_len: int,
    skip_len: int,
) -> None:
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for idx, (rr, cc) in enumerate(_segment_cells(r0, c0, r1, c1)):
        if _trail_hit(idx, steps, paint_len, skip_len):
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
        paint_len, skip_len = _trail_pattern(len(cycle))

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

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
            _draw_pattern_bg_only(out, r0, c0, r1, c1, col, bg, paint_len, skip_len)

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
                _draw_pattern_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg, paint_len, skip_len)

    return out


# --------------- generator ---------------

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: shape geometry ----
LINE_VERTICES_RANGE = (3, 5)       # 3-4 collinear/polyline vertices for the multi-vertex line family
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 3
PAIR_SEGMENT_LEN_RANGE = (5, 11)   # 5-10 cells between the two endpoints
TRIANGLE_SIDE_RANGE = (4, 8)
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
DIAMOND_ARM_RANGE = (3, 6)
PENTAGON_ROOF_RANGE = (3, 5)
PENTAGON_WALL_RANGE = (4, 7)
HEXAGON_SLANT_RANGE = (3, 5)
HEXAGON_WALL_RANGE = (4, 6)

# ---- CFG: shape mixture ----
SHAPE_WEIGHTS = np.array([0.24, 0.18, 0.18, 0.14, 0.13, 0.13], dtype=float)

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

# ---- CFG: retry budgets ----
SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


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


def _build_segment_pair() -> list[tuple[int, int]]:
    gap = int(np.random.randint(*PAIR_SEGMENT_LEN_RANGE))
    return _transform_coords([(0, 0), (0, gap)])


def _build_line(n_vertices: int | None = None) -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE)) if n_vertices is None else int(n_vertices)
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


_MULTI_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices(kind: str = "any") -> list[tuple[int, int]]:
    if kind == "pair":
        return _build_segment_pair()
    if kind == "multi":
        weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
        idx = int(np.random.choice(len(_MULTI_BUILDERS), p=weights))
        return _MULTI_BUILDERS[idx]()
    if np.random.rand() < 0.35:
        return _build_segment_pair()
    return _make_vertices("multi")


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _dashed_shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    paint_len, skip_len = _trail_pattern(len(vertices))
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        steps = _segment_len(a, b)
        for idx, cell in enumerate(_gen_segment_cells(a, b)):
            if _trail_hit(idx, steps, paint_len, skip_len):
                cells.add(cell)
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
    src = (out == shape_color)
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _paint_shape_output(out: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    paint_len, skip_len = _trail_pattern(len(vertices))
    for a, b in _shape_edges(vertices):
        _draw_pattern_solid(out, a[0], a[1], b[0], b[1], color, paint_len, skip_len)


def _paint_shape_reserve(out: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    for a, b in _shape_edges(vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=color)


def _shape_has_visible_dash(vertices: list[tuple[int, int]]) -> bool:
    solid = _shape_cells(vertices)
    dashed = _dashed_shape_cells(vertices)
    return dashed != solid


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    vertices = _make_vertices(kind)
    cells = _shape_cells(vertices)

    if not _shape_has_visible_dash(vertices):
        return None

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    reserve = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    _paint_shape_output(out, shifted_vertices, shape_color)
    _paint_shape_reserve(reserve, shifted_vertices, shape_color)

    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, shape_color, int(shadow_color), offset)
        _paint_shadow(reserve, shape_color, int(shadow_color), offset)
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

    if np.array_equal(out, reserve):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out, reserve


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


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
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


def _try_place(
    inp: np.ndarray,
    out: np.ndarray,
    occ: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    reserve: np.ndarray,
) -> bool:
    try:
        loc = random_free_location(
            occ,
            reserve,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(occ, reserve, loc[0], loc[1], transparent=BG)
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _shape_kinds(shape_count: int) -> list[str]:
    kinds = ["pair", "multi"]
    while len(kinds) < shape_count:
        kinds.append("any")
    kinds = kinds[:shape_count]
    np.random.shuffle(kinds)
    return kinds


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)
    occ = np.full((H, W), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    kinds = _shape_kinds(shape_count)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx), kind=kinds[idx])
            if patch is None:
                continue
            if _try_place(inp, out, occ, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, occ, *_make_distractor_patch(distractor_color)):
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

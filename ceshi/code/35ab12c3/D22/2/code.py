# Variant of 35ab12c3
# Dimension: D22 — 变换操作——输出中对输入像素的处理 (transform)
# Change: 保持源像素（keep_source）——输入中的顶点像素和阴影锚点像素在输出中保持不变，连线和阴影仅涂在背景格子上... → 顶点重着色为阴影颜色
#
# Design analysis
# ===============
# variant_summary:
#   input:
#     与原题相同。每个形状仍然只在输入里保留同色顶点；若该形状带阴影，则在某个起始顶点
#     的 8 邻域里保留一个阴影锚点像素。干扰色块仍然保持原样。
#   output:
#     与原题相比，主线段和阴影线段的几何完全不变；唯一的 D22 变化是：
#     若某个形状存在唯一的对应阴影色，则该形状所有顶点在输出中都不再保留原形状色，
#     而是整体改成该阴影色。阴影锚点像素本身仍保持阴影色。
#   concrete_trace:
#     例子：输入里有两个蓝色顶点 (2,2),(2,6)，并且 (1,2) 有一个红色阴影锚点。
#     原题输出：蓝色水平线连通两顶点，(1,2)..(1,6) 画红色平行阴影，但两个端点仍是蓝色。
#     变体输出：水平线中间仍是蓝色，红色阴影仍在上方；差异在于 (2,2),(2,6) 两个顶点
#     也变成红色，于是出现“顶点颜色和阴影颜色交换”的局部效果。
# complexity_calibration:
#   - 不引入新的几何类型，也不改变原有连线/阴影推断机制，只改输出端对顶点的着色。
#   - 为了让 D22 肉眼可见，generator 强制每个样本至少有 1 个带阴影的形状；否则新旧输出可能相同。
#   - 继续保持“每个形状至多一个阴影锚点”，避免一个顶点同时对应多个阴影色导致顶点改色歧义。
# implicit_assumptions_review:
#   - 原 `out = g.copy()` 依赖 keep_source：输入像素默认直接保留到输出。
#     变体下这个前提对“形状顶点”不再成立，所以必须在绘制结束后显式覆写这些顶点。
#   - `_draw_line_bg_only(...)` 的“只涂背景”前提仍成立。变体并不要求主线/阴影去覆盖其他实体，
#     只是要求最后顶点换色，因此不应把整条线的覆盖逻辑改成强制写入。
#   - `seed_idxs` / `carried` 的邻接推断仍成立，因为输入构造没有改，阴影锚点仍位于起始顶点的 8 邻域。
#   - 顶点改色必须放在两阶段绘制之后：若提前改色，基于形状色复制阴影的逻辑会丢失源顶点。
# placement_legality:
#   - patch 的 bbox 仍覆盖形状本体、阴影锚点、以及完整平移后的阴影轮廓。
#   - 仍使用 `random_free_location(..., transparent=BG, padding=1)` 按输出 patch 放置，
#     因而不同对象的最终可见输出不会重叠，也保留 >=1 格缓冲。
# visual_observability:
#   - 每个样本至少一个阴影形状，且该形状的所有顶点一起改色；解题者能在 2-4 个样例中
#     看到“只要附近有阴影锚点，输出顶点就和阴影同色”的重复证据。
#   - 干扰物和无阴影形状保持原行为，帮助对比“哪些对象会发生改色”。
# color_inference:
#   - 不使用颜色 id 排序来决定改成哪种颜色；改色完全由局部可见结构决定：
#     起始顶点旁边那个阴影锚点是什么颜色，顶点就改成什么颜色。
#   - generator 只给每个形状一个阴影色，因此这种对应关系在视觉上是单值、可归纳的。
# solver_generator_contract:
#   - solver 对“会改色”的唯一新假设是：若一个形状需要改色，则其起始顶点邻域里收集到的
#     非本色锚点像素属于同一个颜色。generator 通过“每形状最多一个阴影锚点”保证这一点。

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

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        vertex_recolor = None
        carried_colors = {ncol for _drn, _dcn, ncol in carried}
        if len(carried_colors) == 1:
            vertex_recolor = next(iter(carried_colors))

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

        if vertex_recolor is not None:
            for r, c in cycle:
                out[r, c] = vertex_recolor

    return out


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
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
    vertices = _make_vertices()
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
        for r, c in shifted_vertices:
            out[r, c] = int(shadow_color)
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
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows <= 0:
        return None
    shadow_count = int(np.random.randint(1, max_shadows + 1))
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

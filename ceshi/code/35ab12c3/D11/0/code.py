# Variant of 35ab12c3
# Dimension: D11 — 颜色选择范围 (meta)
# Change: 从{1,2,3,4,6,7,8}中选择颜色（排除0=背景和5），每组使用唯一颜色... → 允许颜色复用（多组共享同一颜色）
#
# Design analysis for the D11 variant
# ----------------------------------
# concrete_behavior:
#   - 原题把“同色的所有像素”直接视为同一组，再整体排序并连边。
#   - 变体里，同一种颜色可以出现在多个空间分离的形状组中；人类看到的输入会出现
#     “两个远离的 6 形状顶点群”“两个远离的 3 折线群”等现象。
#   - 输出仍然对每个局部顶点群各自连线，并且只把各自起点旁边的阴影锚点沿平行路径复制。
# behavior_trace:
#   - 原逻辑示例：同色 6 有四个点 A,B,C,D，程序把 A,B,C,D 一起按角度排序，可能错误地
#     把左上局部形状和右下局部形状混成一个大多边形。
#   - 新逻辑示例：仍然先看颜色 6，但先在 6 色像素之间建立“若两点 H/V/45° 对齐且中间
#     只有背景或同色，则可视为同一局部形状的一条候选连接”的图。A,B 连通，C,D 连通，
#     但 A/C、A/D、B/C、B/D 都不满足候选连接，因此拆成两个连通分量，分别独立求解。
# complexity_calibration:
#   - 只允许少量颜色复用：2-4 个形状里至少有一对复用色，但不会把所有对象都压成同一色。
#   - 阴影色和干扰色仍保持与形状主色分离，避免把 D11 变成“所有类别都重复颜色”的无序题。
#   - 复用色形状之间额外施加“任意顶点对都不能 H/V/45° 对齐”的放置约束，避免 solver 误并组。
# implicit_assumption_review:
#   - 原 `pts_rc = argwhere(g == col)` 假设“颜色=组”。D11 下失效，必须拆成局部组。
#   - 原 `seed_idxs` / `carried` 依赖“当前 cycle 就是一个形状”；因此 cycle 的输入必须先按局部组分开。
#   - 原 `_clear_in_original(... allow={col})` 在变体下仍成立，因为局部组内部路径仍允许穿过同色顶点。
#   - 新 solver 的隐藏假设是：不同同色组之间不存在可直连候选边。生成器显式保证这点。
# placement_legality:
#   - 仍然保持对象输出区域之间至少 1 格缓冲，防止主线/阴影互相覆盖。
#   - 对复用颜色的不同形状，除了普通不重叠，还要求它们的输入顶点跨组两两都不形成 H/V/45° 对齐。
# visual_observability:
#   - 在 2-4 个样例里，解题者会反复看到“同一颜色在两个远处局部群里各自独立扩张”的现象，
#     因而能归纳出“不能再按颜色全局合并，必须按空间局部分组”。
# visual_cleanliness:
#   - 不引入重叠层、递归阴影或共享阴影色；仍保留原题的稀疏布局与背景缓冲，保证画面整洁。

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste

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


def _same_color_groups(g: np.ndarray, col: int, bg: int) -> list[list[tuple[int, int]]]:
    """Split one color into local shape candidates by aligned-clear connectivity."""
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    n = len(pts_rc)
    if n == 0:
        return []
    if n == 1:
        return [[pts_rc[0]]]

    adj = [set() for _ in range(n)]
    for i in range(n):
        r0, c0 = pts_rc[i]
        for j in range(i + 1, n):
            r1, c1 = pts_rc[j]
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            adj[i].add(j)
            adj[j].add(i)

    groups: list[list[tuple[int, int]]] = []
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp_idx = []
        while stack:
            cur = stack.pop()
            comp_idx.append(cur)
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        groups.append([pts_rc[k] for k in comp_idx])

    groups.sort(key=lambda grp: min(grp))
    return groups


def _shape_group_stats(grid: np.ndarray) -> dict[str, int]:
    """Helper for variant validation: count local shape groups after same-color splitting."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    per_color_group_counts: dict[int, int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        count = sum(1 for grp in _same_color_groups(g, col, bg) if len(grp) >= 2)
        if count:
            per_color_group_counts[int(col)] = count
    shape_group_count = sum(per_color_group_counts.values())
    reused_color_count = sum(1 for count in per_color_group_counts.values() if count >= 2)
    reused_group_surplus = sum(count - 1 for count in per_color_group_counts.values() if count >= 2)
    return {
        "shape_group_count": int(shape_group_count),
        "reused_color_count": int(reused_color_count),
        "reused_group_surplus": int(reused_group_surplus),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        for pts_rc in _same_color_groups(g, col, bg):
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


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1

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


def _has_reused_shape_color(shape_colors: list[int]) -> bool:
    return len(set(shape_colors)) < len(shape_colors)


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    """Sample shape colors with forced reuse, while keeping shadow/distractor colors distinct."""
    min_unique_shapes = max(1, (shape_count + 1) // 2)
    max_unique_shapes = shape_count - 1
    unique_shape_count = int(np.random.randint(min_unique_shapes, max_unique_shapes + 1))

    shape_palette = list(map(int, np.random.permutation(COLORS)[:unique_shape_count]))
    shape_colors = shape_palette.copy()
    extra_count = shape_count - unique_shape_count
    if extra_count > 0:
        extras = np.random.choice(shape_palette, size=extra_count, replace=True).tolist()
        shape_colors.extend(int(v) for v in extras)
    np.random.shuffle(shape_colors)

    shadow_pool = [col for col in COLORS if col not in shape_palette]
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(shadow_pool) - int(has_distractor))
    if max_shadows < 0:
        return None

    shadow_count = int(np.random.randint(max_shadows + 1))
    shadow_pool = list(map(int, np.random.permutation(shadow_pool)))
    shadow_colors = shadow_pool[:shadow_count]
    distractor_color = shadow_pool[shadow_count] if has_distractor else None
    return list(map(int, shape_colors)), list(map(int, shadow_colors)), None if distractor_color is None else int(distractor_color)


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _dilate_8(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = int(radius)
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r_src = slice(max(0, -dr), h - max(0, dr))
            c_src = slice(max(0, -dc), w - max(0, dc))
            r_dst = slice(max(0, dr), h - max(0, -dr))
            c_dst = slice(max(0, dc), w - max(0, -dc))
            out[r_dst, c_dst] |= mask[r_src, c_src]
    return out


def _candidate_locations(canvas_out: np.ndarray, patch_out: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch_out.shape
    H, W = canvas_out.shape
    if h > H or w > W:
        return []

    occupied = canvas_out != BG
    blocked = _dilate_8(occupied, PLACEMENT_PADDING)
    rr, cc = np.nonzero(patch_out != BG)
    if len(rr) == 0:
        return []

    candidates: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if blocked[r0 + rr, c0 + cc].any():
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _cross_group_alignment_exists(
    new_vertices: list[tuple[int, int]],
    prior_vertices: list[tuple[int, int]],
) -> bool:
    for r0, c0 in new_vertices:
        for r1, c1 in prior_vertices:
            if _aligned_8(r0, c0, r1, c1):
                return True
    return False


def _try_place_shape(
    inp: np.ndarray,
    out: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    shape_color: int,
    placed_shape_groups: list[dict[str, object]],
) -> bool:
    local_vertices = [tuple(map(int, p)) for p in np.argwhere(patch_in == shape_color)]
    candidates = _candidate_locations(out, patch_out)
    valid: list[tuple[int, int]] = []
    for r0, c0 in candidates:
        global_vertices = [(r0 + r, c0 + c) for r, c in local_vertices]
        bad = False
        for info in placed_shape_groups:
            if int(info["shape_color"]) != shape_color:
                continue
            if _cross_group_alignment_exists(global_vertices, info["vertices"]):
                bad = True
                break
        if not bad:
            valid.append((r0, c0))

    if not valid:
        return False

    loc = valid[int(np.random.randint(len(valid)))]
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    placed_shape_groups.append(
        {
            "shape_color": int(shape_color),
            "vertices": [(loc[0] + r, loc[1] + c) for r, c in local_vertices],
        }
    )
    return True


def _try_place_patch(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    candidates = _candidate_locations(out, patch_out)
    if not candidates:
        return False
    loc = candidates[int(np.random.randint(len(candidates)))]
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
    if not _has_reused_shape_color(shape_colors):
        return None

    shadow_map = _assign_shadows(shape_count, shadow_colors)
    placed_shape_groups: list[dict[str, object]] = []

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place_shape(inp, out, patch[0], patch[1], int(shape_color), placed_shape_groups):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place_patch(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    present_colors = set(shape_colors) | set(shadow_colors)
    if distractor_color is not None:
        present_colors.add(int(distractor_color))

    stats = _shape_group_stats(inp)
    if stats["reused_color_count"] < 1 or stats["reused_group_surplus"] < 1:
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if set(colors_present(inp, exclude=[BG])) != present_colors:
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

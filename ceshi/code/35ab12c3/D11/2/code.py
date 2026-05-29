# Variant of 35ab12c3
# Dimension: D11 — 颜色选择范围 (meta)
# Change: 从{1,2,3,4,6,7,8}中选择颜色（排除0=背景和5），每组使用唯一颜色... → 仅使用2种颜色

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg as _bg_color, neighbors8, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the D11 variant
#
# a) 具体化与行为追踪
#    - 变体后，整张图只允许两种非背景色：
#      1) SHAPE_COLOR: 所有形状的顶点都用这一种颜色。
#      2) SHADOW_COLOR: 所有阴影锚点与输出中的所有阴影轨迹都用这一种颜色。
#    - 原题依赖“每种颜色就是一个组”；现在颜色不再区分组，必须先按空间位置把输入切成若干局部岛。
#    - 小例子：
#      原逻辑：蓝色两端点 + 蓝色三角形顶点 + 红色阴影锚点，solver 直接按颜色把蓝色点归成一组。
#      变体：所有顶点都变成同一色、所有阴影锚点都变成另一色；solver 若仍按颜色分组，会把整图所有顶点误并到一起。
#      新逻辑：先把所有非背景像素按“局部紧凑星座”切成空间岛；每个岛内部再按原题逻辑连线、平移复制阴影。
#
# b) 复杂度校准
#    - 不能保留原题的 distractor：一旦 distractor 和真实顶点同色，它在视觉上与真实形状顶点无区别，会把推理退化成猜测。
#    - 也不能让边太长、组间太近，否则不同形状会在“只看空间”的条件下混淆。
#    - 因此本变体保留 2-3 个形状，但把每个形状限制为局部紧凑对象，且每个形状都带一个阴影锚点，让第二种颜色稳定出现。
#
# c) 原 solver 中与变体相关的隐式前提审查
#    - `colors = sorted(set(np.unique(g)) - {bg})`
#      旧前提：每个形状组颜色唯一。
#      新变体：不成立。全图只有顶点色和阴影色，不能再按颜色迭代组。
#    - `pts_rc = np.argwhere(g == col)`
#      旧前提：`col` 对应一个完整形状。
#      新变体：不成立。`shape_color` 会收集到所有形状的全部顶点。
#    - `seed_idxs` / `carried`
#      旧前提：不同色邻居就是当前颜色组的阴影锚点，不会来自别的形状。
#      新变体：只有在先做空间分组后，这个前提才重新成立。
#    - `_clear_in_original(... allow={col})`
#      这里依赖的是“同一形状顶点沿线可以跨过同色顶点”，这个在变体里仍成立；但前提是传入的端点已经属于同一空间岛。
#
# d) 放置合法性
#    - 生成器继续使用 output-first。
#    - 每个 shape patch 的输入只保留顶点和一个阴影锚点；输出补全主体折线/多边形，并复制阴影。
#    - patch 放置时使用比 solver 分组半径更大的留白，保证不同形状的输入像素不会被错误并岛。
#    - 每个 patch 自身必须在“局部像素图”上只形成一个稀疏岛，否则 solver 的空间分组会拆错。
#
# e) 视觉可观察性
#    - 人类在 2-4 个样例里仍能看到：同色顶点在各自局部区域内被连成形状；另一种颜色总是贴着某个起点并在输出里形成平行阴影。
#    - 关键差异清晰可见：不能再靠“不同颜色对应不同组”来读题，而要靠“局部星座 + 邻近阴影锚点”来归组。
#    - solver 与 generator 不使用颜色 id 的数值含义做任何语义映射；角色只由出现频次和局部结构区分。
#
# f) 可视化整洁度
#    - 取消 distractor，减少视觉噪声。
#    - 形状局部紧凑、patch 间保留明显空白，避免全图同色点混成一团。
#
# g) 颜色变化可推断性
#    - 非背景色恒为 2 种，且跨样例保持固定语义：较多的那种是顶点/主体，较少的那种是阴影锚点/阴影轨迹。
#    - 这不是颜色 id 到功能的硬编码，而是生成器稳定制造的结构性线索；人类可直接观察到。
#
# 9) 新 solver 的隐式假设与生成器保证
#    - 假设 1：顶点色在输入中的总像素数严格多于阴影色。
#      生成器保证：每个 shape 至少 3 个顶点，且每个 shape 只有 1 个阴影锚点。
#    - 假设 2：同一 shape 的输入像素在 Chebyshev 半径 `GROUP_LINK_RADIUS` 的稀疏图中连通。
#      生成器保证：每条边长至多 `MAX_EDGE_STEP`，局部 patch 紧凑。
#    - 假设 3：不同 shape 的输入像素在上述稀疏图中不连通。
#      生成器保证：patch 放置 padding >= `GROUP_LINK_RADIUS`。
#    - 假设 4：每个 shadow 锚点只贴着本 shape 的一个顶点。
#      生成器保证：shadow offset 从满足“只邻接一个顶点”的候选中选择。
# -----------------------------------------------------------------------------

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# ---- Solver / generator shared grouping constraint ----
GROUP_LINK_RADIUS = 4
MAX_EDGE_STEP = 4


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


def _color_roles(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    coords = [tuple(map(int, p)) for p in np.argwhere(g != bg)]
    if not coords:
        return None, None

    counts: dict[int, int] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for r, c in coords:
        col = int(g[r, c])
        counts[col] = counts.get(col, 0) + 1
        if col not in first_seen:
            first_seen[col] = (r, c)

    ordered = sorted(counts, key=lambda col: (-counts[col], first_seen[col][0], first_seen[col][1]))
    shape_color = int(ordered[0])
    shadow_color = int(ordered[1]) if len(ordered) > 1 else None
    return shape_color, shadow_color


def _sparse_components(points_rc: list[tuple[int, int]], radius: int) -> list[list[tuple[int, int]]]:
    if not points_rc:
        return []

    comps: list[list[tuple[int, int]]] = []
    used = [False] * len(points_rc)
    for start in range(len(points_rc)):
        if used[start]:
            continue
        used[start] = True
        stack = [start]
        comp: list[tuple[int, int]] = []
        while stack:
            i = stack.pop()
            pr, pc = points_rc[i]
            comp.append(points_rc[i])
            for j, (qr, qc) in enumerate(points_rc):
                if used[j]:
                    continue
                if max(abs(pr - qr), abs(pc - qc)) <= radius:
                    used[j] = True
                    stack.append(j)
        comps.append(sorted(comp))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    shape_color, shadow_color = _color_roles(g, bg)
    if shape_color is None:
        return out

    islands = _sparse_components([tuple(map(int, p)) for p in np.argwhere(g != bg)], GROUP_LINK_RADIUS)
    for island in islands:
        pts_rc = [(r, c) for r, c in island if int(g[r, c]) == shape_color]
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
                if 0 <= rr < h and 0 <= cc < w and shadow_color is not None and int(g[rr, cc]) == shadow_color:
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
                if v != bg and v != shape_color:
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={shape_color}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, shape_color, bg)

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

# ---- CFG: canvas / placement ----
CANVAS_H_RANGE = (17, 23)          # H in [17, 22]
CANVAS_W_JITTER = (-2, 3)          # W = clip(H + randint(lo, hi), 17, 22)
CANVAS_W_CLIP = (17, 22)
SHAPE_COUNT_RANGE = (2, 4)         # 2-3 shapes
PLACEMENT_PADDING = GROUP_LINK_RADIUS
PLACEMENT_PAD_CONNECTIVITY = 8
GENERATE_RETRIES = 400
SHAPE_PLACEMENT_RETRIES = 80
MAX_PATCH_SIDE = 9

# ---- CFG: geometry ----
LINE_VERTICES_RANGE = (3, 6)       # 3-5 vertices
LINE_MIN_GAP = 2
LINE_MAX_GAP = MAX_EDGE_STEP
TRIANGLE_SIDE_RANGE = (2, 5)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 5)
DIAMOND_ARM_RANGE = (2, 4)
PENTAGON_ROOF_RANGE = (2, 4)
PENTAGON_WALL_RANGE = (2, 5)
HEXAGON_SLANT_RANGE = (2, 4)
HEXAGON_WALL_RANGE = (2, 5)
SHAPE_WEIGHTS = np.array([0.30, 0.18, 0.18, 0.14, 0.10, 0.10], dtype=float)


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
    gaps_left = n_vertices - 1
    total = int(np.random.randint(gaps_left * LINE_MIN_GAP, gaps_left * LINE_MAX_GAP + 1))
    remaining = total
    pos = [0]
    for i in range(gaps_left):
        tail = gaps_left - i - 1
        lo = max(LINE_MIN_GAP, remaining - tail * LINE_MAX_GAP)
        hi = min(LINE_MAX_GAP, remaining - tail * LINE_MIN_GAP)
        gap = int(np.random.randint(lo, hi + 1))
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


def _single_sparse_island(inp: np.ndarray) -> bool:
    pts = [tuple(map(int, p)) for p in np.argwhere(inp != BG)]
    return len(_sparse_components(pts, GROUP_LINK_RADIUS)) == 1


def _make_shape_patch(shape_color: int, shadow_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)
    shadow = _choose_shadow(vertices, cells)
    if shadow is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if max(h, w) > MAX_PATCH_SIDE:
        return None

    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)
    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]

    for a, b in _shape_edges(shifted_vertices):
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > MAX_EDGE_STEP:
            return None
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)

    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    (ar, ac), offset = shadow
    anchor = (ar - r0, ac - c0)
    _paint_shadow(out, shape_color, shadow_color, offset)
    sa = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
        return None
    inp[sa] = shadow_color

    nb_hits = sum(1 for rr, cc in neighbors8(anchor[0], anchor[1]) if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color)
    if nb_hits != 1:
        return None
    if not _single_sparse_island(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _sample_palette() -> tuple[int, int]:
    shape_color, shadow_color = np.random.choice(COLORS, size=2, replace=False).tolist()
    return int(shape_color), int(shadow_color)


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
    if loc is None:
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

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    shape_color, shadow_color = _sample_palette()

    for _ in range(shape_count):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(shape_color, shadow_color)
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    present = colors_present(inp, exclude=[BG])
    if set(present) != {shape_color, shadow_color}:
        return None
    if np.count_nonzero(inp == shape_color) <= np.count_nonzero(inp == shadow_color):
        return None
    if len(_sparse_components([tuple(map(int, p)) for p in np.argwhere(inp != BG)], GROUP_LINK_RADIUS)) != shape_count:
        return None
    if _bg_color(inp) != BG or _bg_color(out) != BG:
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

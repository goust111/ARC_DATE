# Variant of 35ab12c3
# Dimension: D5 — 阴影/平行偏移机制 (path)
# Change: parallel_offset——起始顶点的8邻居中的非背景非同色像素作为阴影锚点，其相对偏移被应用到整条形状路径上，生成平行的阴影副本
#         → direction_from_shape——阴影方向不再由邻居锚点给出，而是由形状自身的朝向推出
#
# Design analysis
# - a) 具体化与行为追踪
#   原题里，输入中的阴影色像素必须贴着某个顶点；solver 先找这个邻接像素，再把“顶点->邻接像素”的单位偏移复制到整条路径上。
#   这个变体保留“同色顶点连成主形状、异色形成平行阴影副本”的输出风格，但改成：
#   1. 有阴影的形状仍然带一个单像素异色标记，用来声明阴影颜色；
#   2. 这个标记不再充当方向锚点，而是放在主形状包围盒内部、并且本身属于未来阴影路径上的一个像素；
#   3. 阴影偏移方向由形状本身决定：取唯一最长边，计算它的两条单位垂线方向，再选“背离其余顶点”的那一侧。
#   具体小例子：若三角形三个顶点是 (0,0),(0,8),(4,4)，最长边是上方水平边；其余顶点在这条边下方，
#   因此阴影方向取向上 (-1,0)，整条三角形路径整体上移 1 格得到阴影。输入里的阴影色单像素放在阴影拷贝内部的一格，
#   不是任何顶点的 8 邻居，所以旧规则无法从“邻居偏移”恢复阴影。
# - b) 复杂度校准
#   若继续让所有带阴影形状都使用原始等边/对称轮廓，则“最长边垂线方向”的符号会不稳定，变体不可见。
#   因此只对“带阴影形状”做必要适配：生成独一条最长边、且其余顶点都在这条边同侧的非对称三角形。
#   其他无阴影形状仍沿用原题混合形状，避免把整个任务简化成单一模板。
# - c) 隐式假设审查（原 main 中与 D5 相关的旧前提）
#   1. `seed_idxs` / `start` 依赖“某个顶点 8 邻域里有异色锚点”，此假设在变体中失效，必须移除。
#   2. `carried` 把起始顶点邻居的相对偏移直接带到全部边上，这正是旧 D5，需要改成从形状几何计算一个统一偏移。
#   3. 原 shadow 画线时 `_clear_in_original(... allow=set())` 默认阴影路径内部没有任何原始输入像素。
#      变体里输入会保留一个位于阴影路径内部的颜色标记，因此必须放宽为 `allow={shadow_color}`，否则会错误拒绝合法阴影。
#   4. 原 solver 默认“阴影锚点本身是某条偏移边的端点，所以 out 里保留它不会妨碍画线”。
#      变体中标记像素可能位于偏移边内部，因此仍需依赖 `out` 初始拷贝保留它，并让画线仅填背景。
# - d) 放置合法性
#   生成器继续 output-first：先画主形状与阴影，再反推输入，仅保留顶点和一个内部阴影标记。
#   带阴影三角形的 patch bbox 按“主形状 + 整体平移后的阴影副本”统一包裹，因此不会越界；
#   贴到总画布时仍使用原题的 `random_free_location(... padding=1)`，保证对象之间留白。
# - e) 视觉可观察性
#   多个样例里，阴影总出现在“最长边的外侧、远离剩余顶点的一边”；内部异色像素只负责说明阴影颜色，
#   自身位置不再给方向。由于标记不与任何主顶点 8 邻接，人眼不会再把它看成旧题的偏移锚点。
# - f) 可视化整洁度
#   只允许每个带阴影形状有一个内部标记；标记选在阴影路径上、且避开所有主顶点的 8 邻域，
#   减少局部拥挤与“像是额外顶点”的误读。
# - g) 颜色变化可推断性
#   颜色机制尽量不动：每个阴影仍有自己的独立颜色，并且输入里仍能看到该颜色，只是从“顶点邻接锚点”改为“内部标记”。
#   不使用颜色 id 排序或隐藏映射来决定几何行为。
# - h) 变体验证指标
#   生成后的样本应满足：阴影标记 100% 不与主形状顶点 8 邻接；同一输入喂给原始 solver 时，至少一部分带阴影样本输出会与新 solver 不同。
# - i) main 的新假设与 generator 保证
#   1. 若某形状带阴影，则其严格内部恰有一个单像素异色标记；generator 保证这一点。
#   2. 该形状存在唯一最长边，且其余顶点均位于这条边同侧；generator 只对带阴影形状使用专门三角形来保证。
#   3. 标记像素属于真实阴影路径；generator 直接从已生成的 shadow 副本中选点，保证 solver 恢复后与 output 一致。

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


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _strict_interior_marker_color(
    singletons: dict[int, tuple[int, int]],
    vertices: list[tuple[int, int]],
) -> int | None:
    arr = np.array(vertices, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    candidates = [col for col, (r, c) in singletons.items() if r0 < r < r1 and c0 < c < c1]
    if len(candidates) != 1:
        return None
    return int(candidates[0])


def _shape_shadow_offset(
    vertices: list[tuple[int, int]],
    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[int, int] | None:
    if not drawable_edges:
        return None

    lengths = [max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in drawable_edges]
    max_len = max(lengths)
    if lengths.count(max_len) != 1:
        return None

    edge_idx = lengths.index(max_len)
    (r0, c0), (r1, c1) = drawable_edges[edge_idx]
    others = [p for p in vertices if p != (r0, c0) and p != (r1, c1)]
    if not others:
        return None

    sr, sc = _step_dir(r0, c0, r1, c1)
    perp_a = (-sc, sr)
    perp_b = (sc, -sr)
    mr = (r0 + r1) / 2.0
    mc = (c0 + c1) / 2.0
    orr = sum(r for r, _ in others) / len(others)
    occ = sum(c for _, c in others) / len(others)

    dot_a = (orr - mr) * perp_a[0] + (occ - mc) * perp_a[1]
    dot_b = (orr - mr) * perp_b[0] + (occ - mc) * perp_b[1]
    if dot_a == dot_b:
        return None
    return perp_a if dot_a < dot_b else perp_b


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    all_colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    coords_by_color = {
        col: [tuple(map(int, p)) for p in np.argwhere(g == col)]
        for col in all_colors
    }
    singletons = {col: pts[0] for col, pts in coords_by_color.items() if len(pts) == 1}

    for col in all_colors:
        pts_rc = coords_by_color[col]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
        cycle = cycle[start:] + cycle[:start]

        edges = _shape_edges(cycle)
        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        shadow_col = _strict_interior_marker_color(singletons, cycle)
        if shadow_col is None:
            continue

        offset = _shape_shadow_offset(cycle, drawable_edges)
        if offset is None:
            continue
        drn, dcn = offset

        for (r0, c0), (r1, c1) in drawable_edges:
            rr0 = r0 + drn
            cc0 = c0 + dcn
            rr1 = r1 + drn
            cc1 = c1 + dcn
            if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                continue
            if not _aligned_8(rr0, cc0, rr1, cc1):
                continue
            if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow={shadow_col}):
                continue
            _draw_line_bg_only(out, rr0, cc0, rr1, cc1, shadow_col, bg)

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
SHADOW_TRIANGLE_ARM_RANGE = (3, 7)

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


def _build_shadow_triangle() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*SHADOW_TRIANGLE_ARM_RANGE))
    return _transform_coords([(0, 0), (0, 2 * arm), (arm, arm)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _patch_bbox(shape_cells: set[tuple[int, int]], shadow_offset: tuple[int, int] | None) -> tuple[int, int, int, int]:
    coords = list(shape_cells)
    if shadow_offset is not None:
        dr, dc = shadow_offset
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


def _select_shadow_marker(
    shadow_mask: np.ndarray,
    vertices: list[tuple[int, int]],
) -> tuple[int, int] | None:
    arr = np.array(vertices, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    cr = sum(r for r, _ in vertices) / len(vertices)
    cc = sum(c for _, c in vertices) / len(vertices)

    candidates: list[tuple[float, int, int]] = []
    for r, c in np.argwhere(shadow_mask):
        rr = int(r)
        cc2 = int(c)
        if not (r0 < rr < r1 and c0 < cc2 < c1):
            continue
        if any(max(abs(rr - vr), abs(cc2 - vc)) <= 1 for vr, vc in vertices):
            continue
        dist = (rr - cr) ** 2 + (cc2 - cc) ** 2
        candidates.append((dist, rr, cc2))
    if not candidates:
        return None
    _, rr, cc2 = min(candidates)
    return int(rr), int(cc2)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    if shadow_color is None:
        vertices = _make_vertices()
        shadow_offset = None
    else:
        vertices = _build_shadow_triangle()
        shadow_offset = _shape_shadow_offset(vertices, _shape_edges(vertices))
        if shadow_offset is None:
            return None

    cells = _shape_cells(vertices)
    r0, c0, r1, c1 = _patch_bbox(cells, shadow_offset)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow_color is not None and shadow_offset is not None:
        _paint_shadow(out, shape_color, int(shadow_color), shadow_offset)
        marker = _select_shadow_marker(out == int(shadow_color), shifted_vertices)
        if marker is None:
            return None
        if inp[marker] != BG:
            return None
        inp[marker] = int(shadow_color)

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

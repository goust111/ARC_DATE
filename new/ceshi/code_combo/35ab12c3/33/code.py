# Combo variant of 35ab12c3 (index 33)
# combo_spec: D4/0+D5/1+D10/3
# Name: 嵌套形状+多层阴影+间歇跳跃轨迹
#
# Design analysis
# ===============
# [a] 具体化与行为追踪
# - 原题里，一个颜色通常只是一条折线/一个闭环的顶点，输出把这些顶点用连续实线连接；若起始顶点旁边有异色像素，
#   就再沿一个固定偏移复制一层平行线。
# - 融合后，一个“主形状颜色”不再对应单个环，而是对应两层同色闭合多边形：
#   外环顶点 + 内环顶点，二者严格嵌套，且都只走 H/V/45° 边。
# - 输出不再是连续轮廓，而是把外环和内环都画成虚线；同时起始顶点旁边不止一个异色锚点，而是 2-3 个链式锚点，
#   每一层都把“整套外环+内环”按累计偏移复制成新的虚线阴影轮廓。
# - 例子：
#   1. 输入某色有 8 个点，分别是外矩形四角和内矩形四角；原题会尝试把 8 点排成一个单环，逻辑已不成立。
#   2. 现在 solver 先把这 8 点拆成两个合法矩形：outer 和 inner。
#   3. 假设 outer 左上角旁边依次有颜色 7、8 两个锚点，偏移分别是 (-1,0) 与 (0,-1)。
#   4. 输出先画主色 outer/inner 的虚线双环，再画累计偏移 (-1,0) 的颜色 7 双环，再画累计偏移 (-1,-1) 的颜色 8 双环。
# - 和原题的差异不只是“多一层阴影”或“换成虚线”，而是：
#   单环 -> 双环嵌套；单层平移 -> 2-3 层链式累计平移；连续线 -> 统一虚线。
#
# [b] 复杂度校准
# - 若保留原题 2-4 个 shape 且每个都强制 2-3 层阴影，会超出 7 色预算；因此这里把主形状数限制为 1-2。
# - 两形状时固定为 2 层阴影 + 2 层阴影；单形状时允许 2 或 3 层，并可选一个 distractor。
# - 嵌套形状只采样两类视觉清晰、对齐稳定的模板：矩形环与菱形环。这样“同色双环”在 2-4 个样例中很容易被观察到，
#   同时避免任意多边形导致子集拆分歧义过大。
# - 虚线节拍统一为 `paint 2, skip 1`。这样规则直接可见，且不会把认知负担分散到额外的颜色或形状映射上。
#
# [c] 隐式假设审查
# - 原 `main()` 的 `_angle_cycle(points_of_color)` 默认“同色点集对应单条轨迹”。嵌套后失效，因此这里只把它用于
#   验证某个候选子集是否能闭合成一个环，而不是直接解释整个颜色组。
# - 原 `carried` 默认阴影像素都贴在主顶点旁边，且每个邻居都表示一层直接平移。多层链式阴影不满足这个前提，
#   所以改为从起始顶点出发读取一条 2-3 节的异色锚点链，并使用累计偏移。
# - 原 `_draw_line_bg_only()` 隐含“整段连续涂满”。融合后改为 `_draw_pattern_bg_only()`，只在虚线节拍命中的格子着色，
#   但仍强制保留端点，使输入里的顶点/锚点在输出中继续成立为轮廓顶点。
# - 原 solver 对阴影颜色 `len(points)<2` 自动忽略，这个前提仍成立：阴影颜色在输入里依然只出现一个锚点像素，
#   由其父 shape 消费，不会被单独当成 shape。
# - 新 solver 假设：
#   1. 一个主颜色恰好能拆成两个不相交、严格嵌套、覆盖全部同色点的闭合多边形。
#   2. 第一层阴影锚点只贴着外环的一个起始顶点，后续层形成简单链。
#   generator 分别通过模板化构造和链合法性检查来保证。
#
# [d] 放置合法性 / 可观察性 / 整洁度 / 可推断性
# - generator 对每个局部 patch 维护两份几何：
#   1. 真正输出是虚线双环；
#   2. `reserve` 是所有主环和阴影环的连续实线占位。
#   全局放置用 `reserve` 做碰撞检测，防止其他对象钻进虚线空洞，从而破坏 solver 的“原输入边中间无阻塞”假设。
# - 多层阴影要求累计偏移后的整套双环彼此不重叠，且阴影锚点链除了前驱/后继外不接触别的顶点或锚点。
# - 视觉上线索是结构性的，不依赖颜色值：
#   同色 8 点 -> 两个嵌套闭环；邻接彩色链 -> 多层外扩阴影；输出所有轮廓均为虚线。
# - 量化退化检查应关注三件事是否同时成立：
#   1. 主颜色是否都能拆成嵌套双环；
#   2. 每个 shape 是否真的有 2-3 层链式阴影；
#   3. 输出中主环是否明显少于其连续轮廓占位，证明出现了虚线跳涂而非退化回原题实线。

import itertools
import math

import numpy as np
from scipy import ndimage

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]

CANVAS_H_RANGE = (16, 23)
CANVAS_W_JITTER = (-2, 3)
CANVAS_W_CLIP = (16, 22)

SHAPE_PLANS = [
    (1, [2]),
    (1, [3]),
    (2, [2, 2]),
]
SHAPE_PLAN_WEIGHTS = np.array([0.30, 0.34, 0.36], dtype=float)
SHAPE_TEMPLATE_KINDS = ("rect", "diamond")
DISTRACTOR_PROB = 0.40
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

OUTER_RECT_H_RANGE = (8, 11)
OUTER_RECT_W_RANGE = (8, 11)
RECT_INSET_MIN = 2
RECT_INNER_MIN_SIDE = 4
OUTER_DIAMOND_ARM_RANGE = (4, 7)
INNER_DIAMOND_ARM_MIN = 2

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

CHAIN_RETRIES = 100
SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500
PLAN_REALIZATION_RETRIES = 80

PAINT_LEN = 2
SKIP_LEN = 1


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _trail_hit(idx: int, steps: int) -> bool:
    if idx == 0 or idx == steps:
        return True
    cycle = PAINT_LEN + SKIP_LEN
    return ((idx - 1) % cycle) < PAINT_LEN


def _draw_pattern_bg_only(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], col: int, bg: int) -> None:
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    for idx, (rr, cc) in enumerate(_segment_cells(a[0], a[1], b[0], b[1])):
        if out[rr, cc] == bg and _trail_hit(idx, steps):
            out[rr, cc] = col


def _polygon_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _polygon_masks(vertices: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(np.array(vertices, dtype=int))
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    outline_local = np.zeros((h + 2, w + 2), dtype=bool)
    shifted = [(r - r0 + 1, c - c0 + 1) for r, c in vertices]
    for a, b in _polygon_edges(shifted):
        draw_line(outline_local, a[0], a[1], end_r=b[0], end_c=b[1], color=True)

    filled_local = ndimage.binary_fill_holes(outline_local)

    outline = np.zeros(shape, dtype=bool)
    filled = np.zeros(shape, dtype=bool)
    outline[r0 : r1 + 1, c0 : c1 + 1] = outline_local[1:-1, 1:-1]
    filled[r0 : r1 + 1, c0 : c1 + 1] = filled_local[1:-1, 1:-1]
    return outline, filled


def _analyze_polygon(g: np.ndarray, col: int, pts_rc: list[tuple[int, int]], bg: int) -> dict | None:
    if len(pts_rc) < 4 or len(pts_rc) > 6:
        return None

    cycle = _angle_cycle(pts_rc)
    for (r0, c0), (r1, c1) in _polygon_edges(cycle):
        if not _aligned_8(r0, c0, r1, c1):
            return None
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
            return None

    outline, filled = _polygon_masks(cycle, g.shape)
    if not outline.any():
        return None
    vals = g[filled]
    if np.any((vals != bg) & (vals != col)):
        return None
    if int(filled.sum()) <= len(cycle):
        return None

    return {
        "cycle": cycle,
        "outline": outline,
        "filled": filled,
        "interior": filled & ~outline,
        "area": int(filled.sum()),
    }


def _nested_cycles(g: np.ndarray, col: int, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    n = len(pts_rc)
    if n < 8 or n > 12:
        return None

    candidates: dict[frozenset[int], dict] = {}
    for size in range(4, min(6, n - 2) + 1):
        for idxs in itertools.combinations(range(n), size):
            info = _analyze_polygon(g, col, [pts_rc[i] for i in idxs], bg)
            if info is not None:
                candidates[frozenset(idxs)] = info

    if not candidates:
        return None

    all_idxs = frozenset(range(n))
    best: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None = None
    best_score: tuple[int, int] | None = None
    keys = list(candidates.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if a & b:
                continue
            if a | b != all_idxs:
                continue
            pa = candidates[a]
            pb = candidates[b]
            outer, inner = (pa, pb) if pa["area"] >= pb["area"] else (pb, pa)
            if np.any(inner["filled"] & ~outer["interior"]):
                continue
            if np.any(outer["outline"] & inner["filled"]):
                continue
            if not all(bool(outer["interior"][r, c]) for r, c in inner["cycle"]):
                continue
            score = (outer["area"] - inner["area"], outer["area"])
            if best_score is None or score > best_score:
                best_score = score
                best = (outer["cycle"], inner["cycle"])
    return best


def _find_start_vertex(cycle: list[tuple[int, int]], g: np.ndarray, col: int, bg: int) -> tuple[int, int]:
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
        idx = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        return cycle[idx]
    return min(cycle)


def _extract_shadow_chain(
    g: np.ndarray,
    start_rc: tuple[int, int],
    base_col: int,
    bg: int,
    max_layers: int = 3,
) -> list[tuple[int, int, int]]:
    h, w = g.shape
    chain: list[tuple[int, int, int]] = []
    used_colors = {base_col}
    used_positions = {start_rc}
    prev: tuple[int, int] | None = None
    cur = start_rc

    for _ in range(max_layers):
        candidates: list[tuple[int, int, int, int, int]] = []
        for drn, dcn in DIR8:
            rr = cur[0] + drn
            cc = cur[1] + dcn
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if prev is not None and (rr, cc) == prev:
                continue
            if (rr, cc) in used_positions:
                continue
            v = int(g[rr, cc])
            if v == bg or v in used_colors:
                continue
            candidates.append((rr, cc, drn, dcn, v))
        if not candidates:
            break
        rr, cc, drn, dcn, v = min(candidates, key=lambda x: (x[0], x[1], x[4]))
        chain.append((drn, dcn, v))
        used_colors.add(v)
        used_positions.add((rr, cc))
        prev, cur = cur, (rr, cc)
    return chain


def _draw_cycles_with_offset(
    out: np.ndarray,
    cycles: list[list[tuple[int, int]]],
    col: int,
    bg: int,
    offset: tuple[int, int] = (0, 0),
) -> None:
    dr, dc = offset
    for cycle in cycles:
        shifted = [(r + dr, c + dc) for r, c in cycle]
        for a, b in _polygon_edges(shifted):
            _draw_pattern_bg_only(out, a, b, col, bg)


def _solid_cycle_cells(cycle: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _polygon_edges(cycle):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _translate_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _transform_nested_coords(
    outer: list[tuple[int, int]],
    inner: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    arr = np.array(outer + inner, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    coords = [tuple(map(int, p)) for p in arr]
    return coords[: len(outer)], coords[len(outer) :]


def _build_nested_rectangle() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h = int(np.random.randint(*OUTER_RECT_H_RANGE))
    w = int(np.random.randint(*OUTER_RECT_W_RANGE))
    max_inset = min((h - RECT_INNER_MIN_SIDE) // 2, (w - RECT_INNER_MIN_SIDE) // 2)
    if max_inset < RECT_INSET_MIN:
        raise ValueError("rectangle too small for nested inset")
    inset = int(np.random.randint(RECT_INSET_MIN, max_inset + 1))
    outer = [(0, 0), (0, w), (h, w), (h, 0)]
    inner = [(inset, inset), (inset, w - inset), (h - inset, w - inset), (h - inset, inset)]
    return _transform_nested_coords(outer, inner)


def _build_nested_diamond() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    arm = int(np.random.randint(*OUTER_DIAMOND_ARM_RANGE))
    max_inner = arm - 2
    if max_inner < INNER_DIAMOND_ARM_MIN:
        raise ValueError("diamond too small for nested inset")
    inner_arm = int(np.random.randint(INNER_DIAMOND_ARM_MIN, max_inner + 1))
    outer = [(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]
    inner = [(arm - inner_arm, arm), (arm, arm + inner_arm), (arm + inner_arm, arm), (arm, arm - inner_arm)]
    return _transform_nested_coords(outer, inner)


def _make_nested_vertices(kind: str | None = None) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if kind is None:
        kind = SHAPE_TEMPLATE_KINDS[int(np.random.randint(len(SHAPE_TEMPLATE_KINDS)))]
    if kind == "rect":
        return _build_nested_rectangle()
    if kind == "diamond":
        return _build_nested_diamond()
    raise ValueError(f"unknown nested template kind: {kind}")

def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _all_patch_coords(
    outer: list[tuple[int, int]],
    inner: list[tuple[int, int]],
    anchor_positions: list[tuple[int, int]],
    cumulative_offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    coords = list(outer) + list(inner) + list(anchor_positions)
    base_cells = _solid_cycle_cells(outer) | _solid_cycle_cells(inner)
    coords.extend(base_cells)
    for off in cumulative_offsets:
        coords.extend(_translate_cells(base_cells, off))
    return coords


def _choose_shadow_steps(
    outer: list[tuple[int, int]],
    inner: list[tuple[int, int]],
    layer_count: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    start = min(outer)
    shape_cells = _solid_cycle_cells(outer) | _solid_cycle_cells(inner)

    for _ in range(CHAIN_RETRIES):
        used_steps: set[tuple[int, int]] = set()
        existing_points = set(outer) | set(inner)
        anchor_positions: list[tuple[int, int]] = []
        cumulative_offsets: list[tuple[int, int]] = []
        steps: list[tuple[int, int]] = []
        prev_pos = start
        cum = (0, 0)
        ok = True

        for _layer in range(layer_count):
            candidates: list[tuple[int, int]] = []
            for off in DIR8:
                if off in used_steps:
                    continue
                new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
                if new_pos in shape_cells or new_pos in existing_points:
                    continue
                forbidden_adj = [pt for pt in existing_points if pt != prev_pos]
                if any(_is_adjacent(new_pos, pt) for pt in forbidden_adj):
                    continue
                new_cum = (cum[0] + off[0], cum[1] + off[1])
                candidates.append(off)

            if not candidates:
                ok = False
                break

            off = candidates[int(np.random.randint(len(candidates)))]
            prev_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
            cum = (cum[0] + off[0], cum[1] + off[1])
            used_steps.add(off)
            steps.append(off)
            anchor_positions.append(prev_pos)
            cumulative_offsets.append(cum)
            existing_points.add(prev_pos)

        if ok:
            return steps, anchor_positions, cumulative_offsets
    return None


def _draw_solid_reserve(patch: np.ndarray, cycles: list[list[tuple[int, int]]], color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    for cycle in cycles:
        shifted = [(r + dr, c + dc) for r, c in cycle]
        for a, b in _polygon_edges(shifted):
            draw_line(patch, a[0], a[1], end_r=b[0], end_c=b[1], color=color)


def _make_shape_patch(
    shape_color: int,
    shadow_colors: list[int],
    preferred_kind: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    kind_order: list[str] = []
    if preferred_kind is not None:
        kind_order.append(preferred_kind)
    for kind in SHAPE_TEMPLATE_KINDS:
        if kind not in kind_order:
            kind_order.append(kind)

    outer: list[tuple[int, int]] | None = None
    inner: list[tuple[int, int]] | None = None
    for kind in kind_order:
        try:
            outer, inner = _make_nested_vertices(kind)
            break
        except ValueError:
            continue
    if outer is None or inner is None:
        return None

    chain = _choose_shadow_steps(outer, inner, len(shadow_colors))
    if chain is None:
        return None
    _steps, anchor_positions, cumulative_offsets = chain

    coords = _all_patch_coords(outer, inner, anchor_positions, cumulative_offsets)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    outer = [(r - r0, c - c0) for r, c in outer]
    inner = [(r - r0, c - c0) for r, c in inner]
    anchor_positions = [(r - r0, c - c0) for r, c in anchor_positions]
    cycles = [outer, inner]

    inp = np.full((h, w), BG, dtype=int)
    reserve = np.full((h, w), BG, dtype=int)
    base_solid = np.full((h, w), BG, dtype=int)

    for r, c in outer + inner:
        inp[r, c] = int(shape_color)
    for (r, c), col in zip(anchor_positions, shadow_colors):
        if inp[r, c] != BG:
            return None
        inp[r, c] = int(col)

    _draw_solid_reserve(reserve, cycles, int(shape_color), (0, 0))
    _draw_solid_reserve(base_solid, cycles, int(shape_color), (0, 0))
    for off, col in zip(cumulative_offsets, shadow_colors):
        _draw_solid_reserve(reserve, cycles, int(col), off)

    out = main(inp)

    nested = _nested_cycles(inp, int(shape_color), BG)
    if nested is None:
        return None
    start = _find_start_vertex(nested[0], inp, int(shape_color), BG)
    shadow_chain = _extract_shadow_chain(inp, start, int(shape_color), BG, max_layers=3)
    if len(shadow_chain) != len(shadow_colors):
        return None
    if not np.array_equal(out.shape, reserve.shape):
        return None
    if np.array_equal(out, reserve):
        return None
    if int(np.count_nonzero(out == int(shape_color))) >= int(np.count_nonzero(base_solid == int(shape_color))):
        return None
    for col in shadow_colors:
        if int(np.count_nonzero(out == int(col))) <= 1:
            return None
    return inp, out, reserve


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _sample_shape_plan() -> tuple[list[int], bool]:
    weights = SHAPE_PLAN_WEIGHTS / SHAPE_PLAN_WEIGHTS.sum()
    idx = int(np.random.choice(len(SHAPE_PLANS), p=weights))
    shape_count, layer_counts = SHAPE_PLANS[idx]
    has_distractor = shape_count == 1 and bool(np.random.rand() < DISTRACTOR_PROB)
    return list(layer_counts), has_distractor


def _sample_template_kinds(shape_count: int) -> list[str]:
    if shape_count <= 0:
        return []
    if shape_count == 1:
        return [SHAPE_TEMPLATE_KINDS[int(np.random.randint(len(SHAPE_TEMPLATE_KINDS)))]]
    if shape_count == 2:
        return [str(v) for v in np.random.permutation(SHAPE_TEMPLATE_KINDS)]
    return [SHAPE_TEMPLATE_KINDS[int(np.random.randint(len(SHAPE_TEMPLATE_KINDS)))] for _ in range(shape_count)]


def _sample_palette(layer_counts: list[int], has_distractor: bool) -> tuple[list[tuple[int, list[int]]], int | None] | None:
    total = len(layer_counts) + sum(layer_counts) + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    ptr = 0
    plans: list[tuple[int, list[int]]] = []
    for count in layer_counts:
        base = int(palette[ptr])
        ptr += 1
        shadows = [int(v) for v in palette[ptr : ptr + count]]
        ptr += count
        plans.append((base, shadows))
    distractor = int(palette[ptr]) if has_distractor else None
    return plans, distractor


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


# Overview:
# This combo interprets each main color as two nested same-color polygons and draws
# both rings as dashed outlines. A 2-3 pixel shadow-anchor chain near the outer ring
# recursively generates multi-layer dashed copies of the entire nested shape.
#
# Input structure:
# - Background is 0.
# - A main shape color appears as exactly two closed polygons of the same color:
#   an outer polygon and a smaller inner polygon fully inside it.
# - Each main shape also has 2-3 distinct shadow colors, each appearing as one pixel.
#   These shadow pixels form a simple 8-neighbor chain starting from one outer vertex.
# - Optional distractor colors appear as 1-3 isolated pixels that do not form polygons.
#
# Rule:
# - For every main color, split its same-color points into two disjoint nested polygons.
# - Draw both polygons as dashed H/V/45° outlines using a fixed `paint 2, skip 1` rhythm.
# - Read the shadow-anchor chain from the chosen outer start vertex; after each step,
#   copy the entire dashed outer+inner ring by the cumulative offset in that shadow color.
# - Distractors stay unchanged.
#
# Key insight:
# The visible output is not a filled annulus. It is a nested pair of dashed contours,
# and the short multicolor chain near one outer vertex encodes successive whole-shape
# offsets, so every shadow layer is another dashed copy of the same two-ring geometry.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        cycles = _nested_cycles(g, col, bg)
        if cycles is None:
            continue
        outer, inner = cycles
        _draw_cycles_with_offset(out, [outer, inner], col, bg, (0, 0))

        start = _find_start_vertex(outer, g, col, bg)
        shadow_chain = _extract_shadow_chain(g, start, col, bg, max_layers=3)
        cum = (0, 0)
        for drn, dcn, shadow_col in shadow_chain:
            cum = (cum[0] + drn, cum[1] + dcn)
            _draw_cycles_with_offset(out, [outer, inner], shadow_col, bg, cum)

    return out


def _gen_attempt(layer_counts: list[int] | None = None, has_distractor: bool | None = None) -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occ = np.full((h, w), BG, dtype=int)

    if layer_counts is None or has_distractor is None:
        layer_counts, has_distractor = _sample_shape_plan()
    palette = _sample_palette(layer_counts, has_distractor)
    if palette is None:
        return None
    shape_plans, distractor_color = palette
    template_kinds = _sample_template_kinds(len(shape_plans))
    total_colors = len(layer_counts) + sum(layer_counts) + int(has_distractor)

    for (shape_color, shadow_colors), template_kind in zip(shape_plans, template_kinds):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_colors, preferred_kind=template_kind)
            if patch is None:
                continue
            if _try_place(inp, out, occ, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, occ, *_make_distractor_patch(int(distractor_color))):
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
        layer_counts, has_distractor = _sample_shape_plan()
        for _ in range(PLAN_REALIZATION_RETRIES):
            result = _gen_attempt(layer_counts, has_distractor)
            if result is not None:
                return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")

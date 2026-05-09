# Combo variant of 13e47133 (index 9)
# combo_spec: D6/0+D7/0
# Name: 裁剪到单区域+嵌套矩形框架墙壁
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题把整张图中每个非墙连通域都独立填成“按边界距离循环”的同心色带，最后输出仍与输入同尺寸。
# - D7 把墙几何改成若干层嵌套闭合矩形框，于是非墙区域不再都是实心矩形；大多数区域变成“外边界+内边界”
#   同时存在的矩形环带。若某环带的 marker 距该环带边界深度为 0,1，则填色会从内外两侧同时向中间收缩。
# - D6 再把输出改成只保留一个目标区域的紧凑裁剪。为了让融合真的可见，generator 强制“marker 数最多”的目标
#   是一个非最外层环带，而不是中心实心区。于是输出是该环带 bbox 的局部图：环带被重绘，中心洞保留下来。
# - 具体小例子：若 4 层框产生外环/中环/内环/中心四个区域，marker 数分别是 1,2,0,1，则原题会把四区都重绘且
#   保留整张图；融合后只选中环，输出裁成它的 bbox，并在 bbox 中留下一个墙色洞。
#
# b) 复杂度校准
# - 选择规则只保留一个自由度：比较各区域 marker 数。generator 保证唯一最大值，避免 solver 依赖不可见 tie-break。
# - 框数控制在 3-6；目标环带要求不是最外层且厚度足以容纳至少两个连续深度，这样输出既明显不同于原题，也不会
#   因过薄而退化成单色环。
# - 非目标区域只放 0-1 个 marker，偶尔给一个 runner-up 2 个 marker（仅当目标有 3 个 marker 时），既能支撑
#   “先选区域”这一步，又不把输入搞得过于拥挤。
#
# c) 隐式假设审查
# - 原 `out = g.copy()` 依赖输出尺寸等于输入；融合后不成立，必须改成只构造目标区域 bbox 的局部输出。
# - 原 `for pts in comps:` 会处理全部区域；融合后要先统计每区 marker 数，再只处理目标区。
# - 原代码默认区域 bbox 内全是该区域像素；环带不满足这一点，所以局部输出不能直接整块填满，必须保留 mask，
#   并把 mask 外的洞位显式留成墙色。
# - 原 `free = g != wall`、`_cc4(free)`、`_boundary_distance(mask)` 并不要求墙是直线，闭合框架仍满足。
# - 原 `dist_to_color[d]` 默认同一区域每个深度至多需要一个 marker；generator 继续保证这一点。
#
# d) 放置合法性
# - 生成器先按递增 inset 画 1 像素厚矩形框，所有框天然不越界、不重叠。
# - target gap 至少 4 个像素 inset，确保对应环带自由厚度至少 3，从而存在深度 0 和 1 两层可放 marker。
# - marker 不再按 bbox 对角线硬放，而是从真实连通域中按指定 boundary distance 选点，避免落到墙上或洞里。
#
# e) 视觉可观察性
# - 输入里最显眼的结构是同色嵌套矩形框；输出却只剩其中一层内部的局部图，而且局部图中心带洞，足以在 2-3 个
#   样例里肉眼看出“先选环带，再裁剪”。
# - 目标环带总是非最外层，所以输出尺寸显著小于输入，避免退化成“整图照常填，只是外圈更醒目”。
# - 颜色顺序完全由可见的几何深度决定，不用颜色编号做任何排序。
#
# f) 可视化整洁度
# - 每个区域只有底色和少量 marker；墙始终单色；输出只保留一个局部 bbox，因此信息层次清楚。
# - 洞位统一留成墙色，解题者能直接把它看成“该环带围住的空腔”，不会和填色带混在一起。
#
# g) 变化的可推断性
# - 颜色变化仍沿用原题：marker 深度决定调色板顺序，区域内按 `distance mod len(palette)` 循环。
# - 新信息只有两层：先找 marker 最多的区域，再只输出该区域的局部 bbox；目标被构造成带洞环带，因此裁剪后的
#   非矩形感是稳定、重复出现的，而不是个别样例偶然现象。

import random

import numpy as np

from arc_utils import components_mask


DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _bbox_from_points(pts):
    rs = [int(r) for r, _c in pts]
    cs = [int(c) for _r, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neigh (Chebyshev) distance to boundary for True-cells in component_mask.
    Boundary cells are those that touch False or the grid edge in their 8-neighborhood.
    Returns dist array with -1 outside component.
    """
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    is_boundary = True
                    break
                if not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _corner_key(corner: int, r: int, c: int, r0: int, c0: int, r1: int, c1: int):
    rr = int(r - r0)
    cc = int(c - c0)
    hh = int(r1 - r0)
    ww = int(c1 - c0)
    if corner == 0:
        return rr + cc, rr, cc
    if corner == 1:
        return rr + (ww - cc), rr, -cc
    if corner == 2:
        return (hh - rr) + cc, -rr, cc
    return (hh - rr) + (ww - cc), -rr, -cc


def _component_infos(grid: np.ndarray, wall: int):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    infos = []
    for pts_arr in _cc4(g != int(wall)):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        dist = _boundary_distance(comp_mask)
        vals = g[comp_mask]
        base = _mode(vals)
        r0, c0, r1, c1 = _bbox_from_points(pts)
        area = len(pts)
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        marker_count = sum(int(g[r, c]) != int(base) for r, c in pts)
        comp_depths = dist[comp_mask]
        max_depth = int(comp_depths.max()) if comp_depths.size else -1
        infos.append(
            {
                "pts": pts,
                "mask": comp_mask,
                "dist": dist,
                "base": int(base),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(area),
                "bbox_area": int(bbox_area),
                "marker_count": int(marker_count),
                "max_depth": int(max_depth),
                "has_hole": bool(area < bbox_area),
                "touches_border": bool(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1),
            }
        )
    return infos


def _select_target_region(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    infos = _component_infos(g, int(wall))
    assert infos

    best = None
    best_key = None
    for info in infos:
        r0, c0, _r1, _c1 = info["bbox"]
        key = (
            int(info["marker_count"]),
            int(info["has_hole"]),
            int(info["area"]),
            -int(r0),
            -int(c0),
        )
        if best is None or key > best_key:
            best = info
            best_key = key
    return int(bg), int(wall), best


def _palette_for_region(grid: np.ndarray, info, r0: int, c0: int, local_dist: np.ndarray):
    g = np.asarray(grid, dtype=int)
    base = int(info["base"])
    dist_to_color: dict[int, int] = {}
    for r, c in sorted(info["pts"]):
        rr = int(r - r0)
        cc = int(c - c0)
        val = int(g[r, c])
        if val == int(base):
            continue
        d = int(local_dist[rr, cc])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = int(val)

    if not dist_to_color:
        return [int(base)]

    ds = sorted(dist_to_color)
    palette = []
    if ds[0] > 0:
        palette.append(int(base))
    for d in ds:
        palette.append(int(dist_to_color[d]))
    return palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the richest nested-frame region and repaint only that local chamber.

    Input structure:
    - One wall color forms several 1px closed rectangular frames nested inside each other, with non-wall regions appearing as rectangular rings plus a center chamber.
    - Every region has a dominant base color and may contain a few isolated marker pixels whose Chebyshev depths within that specific region are consecutive.

    Rule:
    - Count marker pixels in each 4-connected non-wall region and choose the unique region with the largest count.
    - Build that region's palette by sorting its marker colors by boundary distance; if the first marker starts deeper than distance 0, prepend the base color.
    - Output the chosen region's tight bbox only: paint the region cells with `palette[distance mod len(palette)]`, and keep any hole cells inside the bbox as wall-colored void.

    Key insight: The crop target is still chosen by marker richness, but under nested rectangular frames that target is intentionally a ring-shaped region, so the usual distance-based fill now survives inside a cropped bbox as a band wrapped around a visible hole.
    """
    g = np.asarray(grid, dtype=int)
    _bg, wall, target = _select_target_region(g)

    r0, c0, r1, c1 = target["bbox"]
    out_h = int(r1 - r0 + 1)
    out_w = int(c1 - c0 + 1)

    local_mask = np.zeros((out_h, out_w), dtype=bool)
    for r, c in target["pts"]:
        local_mask[int(r - r0), int(c - c0)] = True

    local_dist = _boundary_distance(local_mask)
    palette = _palette_for_region(g, target, int(r0), int(c0), local_dist)

    out = np.full((out_h, out_w), int(wall), dtype=int)
    plen = len(palette)
    for r, c in target["pts"]:
        rr = int(r - r0)
        cc = int(c - c0)
        out[rr, cc] = int(palette[int(local_dist[rr, cc]) % plen])
    return out


H_MIN, H_MAX = 18, 30
W_MIN, W_MAX = 18, 30

N_FRAMES_MIN, N_FRAMES_MAX = 3, 6
MIN_FRAME_STEP = 2
MAX_FRAME_STEP = 6
TARGET_GAP_MIN_STEP = 4
MIN_INNER_FRAME_DIM = 5

MARKERS_PER_REGION_MAX = 3
TARGET_MIN_MARKERS = 2
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 8

GEN_MAX_ATTEMPTS = 500


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return int(bg), int(wall), colors


def _draw_frame(mask: np.ndarray, rect):
    r0, c0, r1, c1 = rect
    mask[r0, c0:c1 + 1] = 1
    mask[r1, c0:c1 + 1] = 1
    mask[r0:r1 + 1, c0] = 1
    mask[r0:r1 + 1, c1] = 1


def _frame_rects(h: int, w: int, rng: random.Random):
    max_inset = min((h - MIN_INNER_FRAME_DIM) // 2, (w - MIN_INNER_FRAME_DIM) // 2)
    if max_inset < TARGET_GAP_MIN_STEP + MIN_FRAME_STEP:
        return None

    max_frames = min(N_FRAMES_MAX, max_inset // MIN_FRAME_STEP + 1)
    if max_frames < N_FRAMES_MIN:
        return None

    n_frames = rng.randint(N_FRAMES_MIN, max_frames)
    n_gaps = n_frames - 1
    target_gap = rng.randint(1, n_gaps - 1)

    min_steps = [MIN_FRAME_STEP] * n_gaps
    min_steps[target_gap] = TARGET_GAP_MIN_STEP
    if sum(min_steps) > max_inset:
        return None

    steps = list(min_steps)
    budget = int(max_inset - sum(steps))
    # Prefer giving the designated target gap enough width to expose depth 2.
    target_extra = min(2, MAX_FRAME_STEP - steps[target_gap], budget)
    steps[target_gap] += int(target_extra)
    budget -= int(target_extra)

    gap_order = list(range(n_gaps))
    rng.shuffle(gap_order)
    for gi in gap_order:
        extra_cap = MAX_FRAME_STEP - steps[gi]
        if extra_cap <= 0 or budget <= 0:
            continue
        extra = rng.randint(0, min(extra_cap, budget))
        steps[gi] += int(extra)
        budget -= int(extra)

    inset = 0
    rects = [(0, 0, h - 1, w - 1)]
    for step in steps:
        inset += int(step)
        rects.append((inset, inset, h - 1 - inset, w - 1 - inset))
    return rects


def _choose_target_info(infos, shape, rng: random.Random):
    h, w = shape
    candidates = []
    for idx, info in enumerate(infos):
        r0, c0, r1, c1 = info["bbox"]
        is_outer = r0 == 1 and c0 == 1 and r1 == h - 2 and c1 == w - 2
        if not info["has_hole"]:
            continue
        if info["touches_border"] or is_outer:
            continue
        if info["max_depth"] < TARGET_MIN_MARKERS - 1:
            continue
        candidates.append((idx, info["area"]))
    if not candidates:
        return None
    max_area = max(area for _idx, area in candidates)
    viable = [idx for idx, area in candidates if area >= max_area // 2]
    return int(rng.choice(viable))


def _plan_marker_counts(infos, target_idx: int, rng: random.Random):
    counts = [0] * len(infos)
    target_depth = int(infos[target_idx]["max_depth"])
    target_count = 3 if target_depth >= 2 else 2
    counts[target_idx] = int(target_count)

    others = [i for i in range(len(infos)) if i != target_idx]
    rng.shuffle(others)

    runner = None
    for i in others:
        need = target_count - 1
        if int(infos[i]["max_depth"]) >= need - 1:
            runner = i
            counts[i] = int(need)
            break
    if runner is None:
        fallback = [i for i in others if int(infos[i]["max_depth"]) >= 0]
        if not fallback:
            return None
        runner = rng.choice(fallback)
        counts[runner] = 1

    for i in others:
        if i == runner:
            continue
        depth = int(infos[i]["max_depth"])
        if depth < 0:
            continue
        if rng.random() < 0.45:
            counts[i] = 1

    if sum(1 for i, n in enumerate(counts) if i != target_idx and n > 0) == 0:
        return None
    if any(counts[i] >= counts[target_idx] for i in range(len(counts)) if i != target_idx):
        return None
    return counts


def _pick_region_markers(info, n_markers: int, rng: random.Random, force_start_zero: bool):
    if n_markers == 0:
        return []

    max_depth = int(info["max_depth"])
    if n_markers > min(MARKERS_PER_REGION_MAX, max_depth + 1):
        return None

    start_hi = int(min(1, max_depth - (n_markers - 1)))
    if start_hi < 0:
        return None
    start_d = 0 if force_start_zero else rng.randint(0, start_hi)
    corner = rng.randrange(4)
    r0, c0, r1, c1 = info["bbox"]
    used = set()
    cells = []
    for offset in range(n_markers):
        target_d = int(start_d + offset)
        candidates = [
            (r, c)
            for r, c in info["pts"]
            if int(info["dist"][r, c]) == target_d and (r, c) not in used
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda rc: _corner_key(corner, rc[0], rc[1], r0, c0, r1, c1))
        rr, cc = candidates[0]
        used.add((rr, cc))
        cells.append((target_d, int(rr), int(cc)))
    return cells


def _stamp_markers(grid: np.ndarray, infos, counts, marker_pool: list[int], target_idx: int, rng: random.Random):
    total = sum(counts)
    if not (DISTINCT_MARKER_COLORS_MIN <= total <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if total > len(marker_pool):
        return None

    colors = rng.sample(marker_pool, total)
    color_i = 0
    for idx, (info, n_markers) in enumerate(zip(infos, counts)):
        force_start_zero = idx == target_idx
        cells = _pick_region_markers(info, int(n_markers), rng, force_start_zero)
        if cells is None:
            return None
        for _d, rr, cc in cells:
            if int(grid[rr, cc]) != int(info["base"]):
                return None
            grid[rr, cc] = int(colors[color_i])
            color_i += 1
    return True


def _validate_color_order(grid: np.ndarray, bg: int, wall: int):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != int(bg):
        return False
    return len(order) >= 2 and int(vals[order[1]]) == int(wall)


def _gen_attempt(rng: random.Random):
    h = rng.randint(H_MIN, H_MAX)
    w = rng.randint(W_MIN, W_MAX)

    rects = _frame_rects(h, w, rng)
    if rects is None:
        return None

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask = np.zeros((h, w), dtype=int)
    for rect in rects:
        _draw_frame(wall_mask, rect)

    grid = np.full((h, w), bg, dtype=int)
    grid[wall_mask == 1] = wall

    infos = _component_infos(grid, int(wall))
    if not (N_FRAMES_MIN <= len(infos) <= N_FRAMES_MAX):
        return None

    target_idx = _choose_target_info(infos, grid.shape, rng)
    if target_idx is None:
        return None

    counts = _plan_marker_counts(infos, target_idx, rng)
    if counts is None:
        return None

    if _stamp_markers(grid, infos, counts, marker_pool, target_idx, rng) is None:
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    _bg2, wall2, selected = _select_target_region(grid)
    if int(wall2) != int(wall):
        return None
    if int(selected["marker_count"]) != int(counts[target_idx]):
        return None
    if not selected["has_hole"]:
        return None
    if selected["touches_border"]:
        return None

    r0, c0, r1, c1 = selected["bbox"]
    if r0 <= 1 or c0 <= 1 or r1 >= h - 2 or c1 >= w - 2:
        return None

    out = main(grid)
    if out.shape == grid.shape:
        return None
    if int(np.count_nonzero(out == wall)) == 0:
        return None
    if np.array_equal(out, grid[r0:r1 + 1, c0:c1 + 1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

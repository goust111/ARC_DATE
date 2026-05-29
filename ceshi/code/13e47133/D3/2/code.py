# Variant of 13e47133
# Dimension: D3 — 源对象：标记像素的定义与放置 (src)
# Change: 每个区域内 0-3 个孤立彩色像素，位于区域角落对角线上，按切比雪夫距离递增放置... → 标记像素按行优先顺序确定调色板
#
# Design analysis for this variant
# a) Concrete behavior:
#    - 原题中，一个 region 里的 marker 基本沿角落对角线排布，所以“按边界距离排序”和“按读图顺序理解”
#      很容易重合。这里把 marker 改成可散布在矩形 region 内任意位置，输出仍然是按边界距离分层的同心框，
#      但调色板顺序改成“把 region 里所有 marker 按 (row, col) 行优先读出来”。
#    - 小例子：若某个 5x6 region 有两个 marker，红色在上方内部格 (d=1)，蓝色在左下角边界格 (d=0)。
#      原题 palette = [蓝, 红]；本变体 palette = [红, 蓝]。于是边界一圈在原题会涂蓝，在本题会涂红，
#      同一个输入会得到明显不同的框带颜色顺序。
# b) Complexity calibration:
#    - 仍然限制每区 0-3 个 marker，避免自由度过高。
#    - 为了不把“同一距离多个 marker 如何处理”引入成新的核心谜面，生成器保持同一区域 marker 的边界距离互异。
#    - 至少强制一个 region 满足“行优先顺序 != 按距离顺序”，防止样本退化回原题。
# c) Implicit assumption audit against the original main():
#    - 原 `dist_to_color[d] = v` 假设 palette 位置由距离决定，并且每个距离至多保留一个颜色。
#      该假设在新题里失效，因为 marker 顺序与距离脱钩，所以必须改成显式的 row-major marker 列表。
#    - 原代码里对 `pts` 的遍历顺序没有显式排序；在旧题里这对“同距离 tie-break”影响不大，
#      但新题直接依赖扫描顺序，所以必须先对 `(r, c)` 排序。
#    - 原“若最小距离 > 0 则 base 领头”与新排序规则并不冲突；这是边界色缺省规则，不是 marker 排序规则，
#      因此保留。
# d) Placement legality:
#    - 生成器只在 wall 分出的矩形 region 内放 marker，不越界、不压墙、不互相重叠。
#    - marker 之间保持 Chebyshev 距离 > 1，确保它们仍然是视觉上分离的单像素提示。
# e) Visual observability:
#    - 特殊 region 强制包含“上方内部 marker + 下方边界 marker”，这样输入中能直观看到
#      “上面的 marker 先读到”，而输出边界色又会跟它对上，足以支持人类归纳“行优先而非按深度”。
#    - 同时保留一些只含内部 marker 的 region，让解题者继续看到 base 作为缺省边界色的情况。
# f) Visual cleanliness:
#    - 继续使用原题的墙分区和矩形区域，不引入额外形状层叠。
#    - marker 数量很少，且彼此分离，不会和同心框输出叠成难以辨认的杂讯。
# g) Color inference:
#    - 颜色本身不参与排序；排序只由可见的空间位置 `(row, col)` 决定。
#    - 全局 marker 颜色种类仍控制在 3-8 种，足够支持模式观察，但不会多到让颜色映射失控。

import random

import numpy as np

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


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


def _build_component_mask(h: int, w: int, pts) -> np.ndarray:
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        comp_mask[r, c] = True
    return comp_mask


def _row_major_markers(g: np.ndarray, pts, base: int, dist: np.ndarray):
    markers = []
    for r, c in sorted((int(r), int(c)) for r, c in pts):
        v = int(g[r, c])
        if v != base:
            markers.append((r, c, v, int(dist[r, c])))
    return markers


def _palette_row_major(markers, base: int) -> list[int]:
    if not markers:
        return [int(base)]
    palette: list[int] = []
    if min(d for _, _, _, d in markers) > 0:
        palette.append(int(base))
    palette.extend(int(v) for _, _, v, _ in markers)
    return palette


def _palette_distance_sorted(markers, base: int) -> list[int]:
    if not markers:
        return [int(base)]
    dist_to_color: dict[int, int] = {}
    for _r, _c, v, d in markers:
        if d not in dist_to_color:
            dist_to_color[d] = int(v)
    ds = sorted(dist_to_color)
    palette: list[int] = []
    if ds[0] > 0:
        palette.append(int(base))
    for d in ds:
        palette.append(int(dist_to_color[d]))
    return palette


def _solve_with_palette_rule(grid: np.ndarray, use_row_major: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = _build_component_mask(h, w, pts)
        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        markers = _row_major_markers(g, pts, base, dist)
        palette = _palette_row_major(markers, base) if use_row_major else _palette_distance_sorted(markers, base)

        m = len(palette)
        for r, c in pts:
            out[r, c] = palette[int(dist[r, c]) % m]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill partitioned regions with concentric colored bands whose palette is read from scattered marker pixels.

    Input structure:
    - The grid is split into regions by wall-colored orthogonal lines.
    - Each region has one dominant base color plus 0-3 isolated marker pixels of other colors, and those markers may appear anywhere inside the region.

    Rule:
    - For each region, scan its non-base marker pixels in row-major order to form the palette.
    - If the region has no boundary marker, prepend the base color before the scanned marker colors.
    - Paint the region by Chebyshev distance from its boundary using palette[distance mod palette_length], yielding cyclic concentric rectangular bands.

    Key insight: The geometry of the output bands is unchanged from the original task; the only thing that moved is how marker colors are ordered, so the decisive clue is that top-to-bottom / left-to-right marker order now controls the color cycle.
    """
    return _solve_with_palette_rule(grid, use_row_major=True)


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 2]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 600


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _place_walls(H: int, W: int, rng: random.Random):
    """
    Build a wall mask by recursive rectangular subdivision.

    Returns (wall_mask, region_bboxes). wall_mask[r, c] == 1 on wall cells.
    Each bbox = (r0, c0, r1, c1) inclusive for a 4-connected free region.
    """
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= SPLITTABLE_LEN
            can_v = (c1 - c0 + 1) >= SPLITTABLE_LEN
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axes)

        if axis == "h":
            rr = rng.randint(r0 + MIN_REGION_DIM, r1 - MIN_REGION_DIM)
            wall_mask[rr, c0:c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0:r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _rect_cells(rect):
    r0, c0, r1, c1 = rect
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            d = min(r - r0, r1 - r, c - c0, c1 - c)
            cells.append((r, c, d))
    return cells


def _well_separated(chosen, rr: int, cc: int) -> bool:
    for r, c, _d in chosen:
        if max(abs(rr - r), abs(cc - c)) <= 1:
            return False
    return True


def _sample_layout(rect, n: int, allow_boundary: bool, rng: random.Random):
    if n <= 0:
        return []
    cells = _rect_cells(rect)
    if not allow_boundary:
        cells = [cell for cell in cells if cell[2] > 0]
    if not cells:
        return None

    for _ in range(80):
        pool = cells[:]
        rng.shuffle(pool)
        chosen = []
        used_d = set()
        for rr, cc, d in pool:
            if d in used_d:
                continue
            if not _well_separated(chosen, rr, cc):
                continue
            chosen.append((rr, cc, d))
            used_d.add(d)
            if len(chosen) == n:
                return chosen
    return None


def _discordant_layout(rect, rng: random.Random):
    r0, c0, r1, c1 = rect
    layouts = [
        [(r0 + 1, c0 + 1, 1), (r1, c1, 0)],
        [(r0 + 1, c1 - 1, 1), (r1, c0, 0)],
    ]
    chosen = list(rng.choice(layouts))

    extras = []
    for rr, cc, d in _rect_cells(rect):
        if d <= 1:
            continue
        if not _well_separated(chosen, rr, cc):
            continue
        extras.append((rr, cc, d))
    if extras and rng.random() < 0.5:
        chosen.append(rng.choice(extras))
    return chosen


def _base_lead_layout(rect, rng: random.Random):
    h = rect[2] - rect[0] + 1
    w = rect[3] - rect[1] + 1
    max_d = _max_cheb_depth(h, w)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if upper <= 0:
        return []

    desired = 1 if upper == 1 else rng.choice([1, 2])
    for n in range(desired, 0, -1):
        layout = _sample_layout(rect, n, allow_boundary=False, rng=rng)
        if layout is not None:
            return layout
    return []


def _random_layout(rect, rng: random.Random):
    h = rect[2] - rect[0] + 1
    w = rect[3] - rect[1] + 1
    max_d = _max_cheb_depth(h, w)
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return []

    desired = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    desired = min(desired, upper)
    if desired == 0:
        return []

    allow_boundary_first = rng.random() < 0.6
    for n in range(desired, 0, -1):
        for allow_boundary in (allow_boundary_first, not allow_boundary_first):
            layout = _sample_layout(rect, n, allow_boundary=allow_boundary, rng=rng)
            if layout is not None:
                return layout
    return []


def _stamp_markers(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    if len(rects) < 2:
        return None

    special_idx = rng.randrange(len(rects))
    other_indices = [i for i in range(len(rects)) if i != special_idx]
    base_idx = rng.choice(other_indices)

    used = set()
    layouts = []
    for i, rect in enumerate(rects):
        if i == special_idx:
            layout = _discordant_layout(rect, rng)
        elif i == base_idx:
            layout = _base_lead_layout(rect, rng)
        else:
            layout = _random_layout(rect, rng)
        layouts.append(layout)

    for layout in layouts:
        if len(layout) > len(marker_pool):
            return None

    for layout in layouts:
        colors = rng.sample(marker_pool, len(layout))
        for color, (rr, cc, _d) in zip(colors, layout):
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
            used.add(color)
    return used


def _validate_color_order(grid, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    comps = _cc4(g != wall)

    marker_regions = 0
    discordant_regions = 0
    base_lead_regions = 0
    boundary_marker_regions = 0

    for pts in comps:
        comp_mask = _build_component_mask(h, w, pts)
        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        markers = _row_major_markers(g, pts, base, dist)
        if not markers:
            continue
        marker_regions += 1
        if min(d for _, _, _, d in markers) > 0:
            base_lead_regions += 1
        else:
            boundary_marker_regions += 1
        if _palette_row_major(markers, base) != _palette_distance_sorted(markers, base):
            discordant_regions += 1

    return {
        "marker_regions": marker_regions,
        "discordant_regions": discordant_regions,
        "base_lead_regions": base_lead_regions,
        "boundary_marker_regions": boundary_marker_regions,
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)
    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out_new = main(grid)
        out_old = _solve_with_palette_rule(grid, use_row_major=False)
    except Exception:
        return None

    if np.array_equal(grid, out_new):
        return None
    if np.array_equal(out_new, out_old):
        return None

    metrics = _variant_metrics(grid)
    if metrics["discordant_regions"] < 1:
        return None
    if metrics["marker_regions"] < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

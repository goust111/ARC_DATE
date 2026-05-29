# Combo variant of 13e47133 (index 14)
# combo_spec: D3/2+D8/0
# Name: 任意位置标记+颜色值排序调色板
#
# Design analysis for this combo variant
# a) Concrete behavior:
#    - 原题里，region 内 0-3 个 marker 往往沿某个角落的对角线摆放，palette 由 marker 到边界的
#      Chebyshev 距离决定，所以“哪个颜色先涂到边界”本质上由位置深浅控制。
#    - 本融合版保留“按边界距离分层涂出同心矩形框”的输出几何，但把 palette 顺序改成：
#      先收集 region 内全部非 base 的 marker 颜色，再按 ARC 调色板索引从小到大排序。
#      marker 的空间位置只影响“是否存在边界 marker，从而是否需要 base 打头”，不再决定 marker 之间的顺序。
#    - 例子：某个 6x6 region 有三个 marker，颜色分别为 8、9、3，位置的边界深度依次是 d=1、d=2、d=0。
#      原题 palette 会按距离得到 [3, 8, 9] 或 [base, ...]；本题无论这三个 marker 放在哪里，
#      marker 部分都固定排序成 [3, 8, 9]。若它们在输入里按行优先读出来是 [8, 9, 3]，则本题会同时区别于
#      原题的“按距离排”和 D3/2 单变体的“按行优先排”。
# b) Complexity calibration:
#    - 仍限制每区最多 3 个 marker，避免把题面自由度放得过大。
#    - 生成器继续保持同一区域 marker 的边界距离互异，这样视觉变化聚焦在“排序准则变了”，而不是
#      “同距离多 marker 如何解读”。
#    - 至少强制一个 region 同时满足：marker 不在原题那种角落对角线上、颜色排序 != 距离排序、
#      颜色排序 != 行优先排序。这样样本不会退化成原题或某个单变体。
# c) Implicit assumption audit against the original main():
#    - 原代码的 `dist_to_color[d] = v` 假设 palette 位置由距离决定；融合后该前提失效，必须改成先显式收集
#      marker 列表，再按颜色值排序。
#    - 原题里 marker 位置和 palette 顺序高度耦合；融合后位置只保留“最小距离是否为 0”这个作用，
#      所以 solver 仍需计算边界距离，但只用它判断是否 prepend base，并用于最终分层上色。
#    - 原代码没有显式排序 `pts`；在本题的核心规则里这不再影响最终 palette，但对对照分析函数
#      （row-major / distance baselines）仍需稳定排序，否则无法可靠检测样本是否体现融合差异。
# d) Placement legality:
#    - 墙仍通过递归矩形切分生成，region 是不重叠的矩形自由区。
#    - marker 只放在所属矩形内部，绝不压墙、不越界、不重叠，并保持两两 Chebyshev 距离 > 1，
#      确保它们视觉上是孤立单像素提示。
# e) Visual observability:
#    - 关键 region 强制使用 3 个分散 marker，并让“颜色升序”同时不同于“按深度顺序”和“按读图顺序”，
#      这样从输入到输出能直接看到：边界带的颜色并不跟最浅 marker 绑定，也不跟最先读到的 marker 绑定。
#    - 另保留至少一个只含内部 marker 的 region，让解题者仍能观察到“若没有边界 marker，则 base 先占边界层”。
# f) Visual cleanliness:
#    - 继续沿用原题的墙分区与同心框输出，不叠加额外对象类型。
#    - marker 数量少且彼此分离；region 仍是规则矩形，因此输入和输出都保持干净、可分区阅读。
# g) Change inferability:
#    - 这里的排序依据是 ARC 颜色索引这一离散全序。它不像位置那样直接可见，因此生成器会重复制造
#      “同一组颜色在不同位置下仍按同一顺序生效”的证据，并强制至少一个 region 同时打破距离序和读图序，
#      让解题者能够从少量样例归纳出“位置被忽略，颜色类别本身决定顺序”。
#    - 新 solver 的额外假设只有两点：marker 是非 base 的孤立彩色像素；base/wall 仍可由全局众数和次众数恢复。
#      generator 显式构造并验证这两点。

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
        comp_mask[int(r), int(c)] = True
    return comp_mask


def _collect_markers(g: np.ndarray, pts, base: int, dist: np.ndarray):
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
    for r, c, v, d in sorted(markers, key=lambda item: (item[3], item[0], item[1])):
        if d not in dist_to_color:
            dist_to_color[d] = int(v)
    ds = sorted(dist_to_color)
    palette: list[int] = []
    if ds[0] > 0:
        palette.append(int(base))
    palette.extend(int(dist_to_color[d]) for d in ds)
    return palette


def _palette_color_sorted(markers, base: int) -> list[int]:
    if not markers:
        return [int(base)]
    palette: list[int] = []
    if min(d for _, _, _, d in markers) > 0:
        palette.append(int(base))
    palette.extend(sorted(int(v) for _, _, v, _ in markers))
    return palette


def _solve_with_palette_rule(grid: np.ndarray, palette_mode: str) -> np.ndarray:
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
        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        markers = _collect_markers(g, pts, base, dist)

        if palette_mode == "color":
            palette = _palette_color_sorted(markers, base)
        elif palette_mode == "distance":
            palette = _palette_distance_sorted(markers, base)
        elif palette_mode == "row_major":
            palette = _palette_row_major(markers, base)
        else:
            raise ValueError(f"unknown palette_mode: {palette_mode}")

        m = len(palette)
        for r, c in pts:
            out[int(r), int(c)] = palette[int(dist[int(r), int(c)]) % m]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with cyclic concentric bands whose palette is determined only by marker colors.

    Input structure:
    - Orthogonal wall lines split the grid into several non-wall regions.
    - Each region has one dominant base color plus 0-3 isolated marker pixels of other colors, and those markers may appear at arbitrary cells instead of corner diagonals.

    Rule:
    - For each region, collect all non-base marker colors and sort them by ARC palette index to form the marker portion of the palette.
    - If the region has no marker on boundary depth 0, prepend the base color before those sorted marker colors.
    - Paint each region cell by Chebyshev distance from the region boundary using palette[distance mod palette_length], producing repeating concentric rectangular frames.

    Key insight: Marker placement no longer tells you the order of the color cycle; position only says whether the base color should occupy the outermost band, while the actual palette order comes from the marker colors themselves.
    """
    return _solve_with_palette_rule(grid, palette_mode="color")


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
GEN_MAX_ATTEMPTS = 900

# Corner encoding: (row_from, col_from, dr, dc)
_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


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


def _is_old_diagonal_layout(rect, layout) -> bool:
    if len(layout) <= 1:
        return False

    ds = sorted(int(d) for _, _, d in layout)
    if ds != list(range(ds[0], ds[0] + len(ds))):
        return False

    r0, c0, r1, c1 = rect
    normalized = {(int(r), int(c), int(d)) for r, c, d in layout}
    for row_from, col_from, dr, dc in _CORNERS:
        br = r0 if row_from == 0 else r1
        bc = c0 if col_from == 0 else c1
        target = {(br + dr * d, bc + dc * d, d) for d in ds}
        if normalized == target:
            return True
    return False


def _sample_layout(rect, n: int, allow_boundary: bool, rng: random.Random, reject_old_style: bool):
    if n <= 0:
        return []

    cells = _rect_cells(rect)
    if not allow_boundary:
        cells = [cell for cell in cells if cell[2] > 0]
    if not cells:
        return None

    for _ in range(120):
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
                if reject_old_style and _is_old_diagonal_layout(rect, chosen):
                    break
                return chosen
    return None


def _combo_conflict_layout(rect, rng: random.Random):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if min(h, w) < 5:
        return None

    options = [
        [(r0 + 1, c0 + 2, 1), (r1 - 2, c0 + 2, 2), (r1, c1 - 1, 0)],
        [(r0 + 1, c1 - 2, 1), (r1 - 2, c1 - 2, 2), (r1, c0, 0)],
    ]
    rng.shuffle(options)
    for layout in options:
        if all(_well_separated(layout[:i], rr, cc) for i, (rr, cc, _d) in enumerate(layout)):
            if not _is_old_diagonal_layout(rect, layout):
                return layout
    return None


def _base_lead_layout(rect, rng: random.Random):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if min(h, w) < 5:
        return None

    options = [
        [(r0 + 1, c1 - 2, 1), (r1 - 2, c0 + 2, 2)],
        [(r0 + 2, c0 + 1, 1), (r1 - 2, c1 - 2, 2)],
    ]
    rng.shuffle(options)
    for layout in options:
        if all(_well_separated(layout[:i], rr, cc) for i, (rr, cc, _d) in enumerate(layout)):
            if not _is_old_diagonal_layout(rect, layout):
                return layout
    return None


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

    prefer_boundary = rng.random() < 0.6
    for n in range(desired, 0, -1):
        for allow_boundary in (prefer_boundary, not prefer_boundary):
            layout = _sample_layout(
                rect,
                n,
                allow_boundary=allow_boundary,
                rng=rng,
                reject_old_style=(n >= 2),
            )
            if layout is not None:
                return layout
    return []


def _assign_colors(layout, marker_pool: list[int], rng: random.Random, strategy: str):
    if not layout:
        return []

    if strategy == "conflict":
        colors = sorted(rng.sample(marker_pool, len(layout)))
        low, mid, high = colors
        by_depth = {0: mid, 1: high, 2: low}
        return [(rr, cc, by_depth[d]) for rr, cc, d in layout]

    colors = rng.sample(marker_pool, len(layout))
    rng.shuffle(colors)
    return [(rr, cc, color) for color, (rr, cc, _d) in zip(colors, layout)]


def _stamp_markers(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    if len(rects) < 2:
        return None

    rich_rects = [i for i, rect in enumerate(rects) if _max_cheb_depth(rect[2] - rect[0] + 1, rect[3] - rect[1] + 1) >= 2]
    if len(rich_rects) < 2:
        return None

    conflict_idx = rng.choice(rich_rects)
    base_choices = [i for i in rich_rects if i != conflict_idx]
    base_idx = rng.choice(base_choices)
    empty_choices = [i for i in range(len(rects)) if i not in {conflict_idx, base_idx}]
    empty_idx = rng.choice(empty_choices) if empty_choices else None

    layouts = []
    for i, rect in enumerate(rects):
        if i == conflict_idx:
            layout = _combo_conflict_layout(rect, rng)
            strategy = "conflict"
        elif i == base_idx:
            layout = _base_lead_layout(rect, rng)
            strategy = "base"
        elif i == empty_idx:
            layout = []
            strategy = "empty"
        else:
            layout = _random_layout(rect, rng)
            strategy = "random"
        if layout is None:
            return None
        layouts.append((strategy, layout))

    used = set()
    for strategy, layout in layouts:
        if len(layout) > len(marker_pool):
            return None
        for rr, cc, color in _assign_colors(layout, marker_pool, rng, strategy):
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
            used.add(int(color))
    return used


def _validate_color_order(grid, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _rect_from_points(pts):
    rows = [int(r) for r, _c in pts]
    cols = [int(c) for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    comps = _cc4(g != wall)

    marker_regions = 0
    arbitrary_regions = 0
    color_vs_distance_regions = 0
    color_vs_row_regions = 0
    base_lead_regions = 0

    for pts in comps:
        comp_mask = _build_component_mask(h, w, pts)
        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        markers = _collect_markers(g, pts, base, dist)
        if not markers:
            continue

        marker_regions += 1
        if min(d for _, _, _, d in markers) > 0:
            base_lead_regions += 1
        if len(markers) >= 2 and not _is_old_diagonal_layout(_rect_from_points(pts), [(r, c, d) for r, c, _v, d in markers]):
            arbitrary_regions += 1

        color_palette = _palette_color_sorted(markers, base)
        if color_palette != _palette_distance_sorted(markers, base):
            color_vs_distance_regions += 1
        if color_palette != _palette_row_major(markers, base):
            color_vs_row_regions += 1

    return {
        "marker_regions": marker_regions,
        "arbitrary_regions": arbitrary_regions,
        "color_vs_distance_regions": color_vs_distance_regions,
        "color_vs_row_regions": color_vs_row_regions,
        "base_lead_regions": base_lead_regions,
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
        out_distance = _solve_with_palette_rule(grid, palette_mode="distance")
        out_row = _solve_with_palette_rule(grid, palette_mode="row_major")
    except Exception:
        return None

    if np.array_equal(grid, out_new):
        return None
    if np.array_equal(out_new, out_distance):
        return None
    if np.array_equal(out_new, out_row):
        return None

    metrics = _variant_metrics(grid)
    if metrics["marker_regions"] < 2:
        return None
    if metrics["arbitrary_regions"] < 2:
        return None
    if metrics["color_vs_distance_regions"] < 1:
        return None
    if metrics["color_vs_row_regions"] < 1:
        return None
    if metrics["base_lead_regions"] < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

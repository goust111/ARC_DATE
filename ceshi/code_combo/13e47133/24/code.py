# Combo variant of 13e47133 (index 24)
# combo_spec: D6/0+D4/0+D11/0
# Name: 裁剪单区域+深度限制+统一标记数

"""
Design analysis
===============

a) 具体化与行为追踪
- 原题输入是被墙分开的多个区域；每个区域里有若干异色标记，按它们到区域边界
  的切比雪夫距离建调色板，然后整块区域一直循环填满。
- 本融合版把三处改动绑在一起：
  1. 所有区域的标记数统一为同一个 N，因此不能再靠“标记最多”选目标区。
  2. 目标区改为“最大区域”；输出只保留它的紧凑裁剪。
  3. 该区域不再填满到底，只给距离边界 d <= N 的层着色，更深处恢复主体色。
- 具体例子：若三块矩形区域面积分别是 24/35/20，且每块都正好有 2 个标记。
  原题会把三块都做成完整同心框；旧 D6 会因标记数相同失去区分力；本题则先选
  面积 35 的最大区域，再按其标记深度建调色板，只填到 d=2，中心保留为空白内核。

b) 复杂度校准
- 统一标记数把“选区依据”从局部计数切到全局几何大小，视觉上与 D6 单独变体显著不同。
- N 仍限制在 1..3，且同一张图里会重复出现在所有区域，解题者可直接肉眼验证。
- 生成器要求目标区域比截断前的完整填充至少少掉一个可见内核，避免 D4 退化成原题。

c) 隐式假设审查
- 原 `out = g.copy()` 依赖“输出与输入同尺寸且墙保留”，融合后失效，必须改为只构造目标区域裁剪。
- 原“遍历全部区域并全部回写”也失效，因为现在只输出一个区域。
- 原 D6 里“按标记数最多选目标区”的假设被 D11 直接打破；solver 必须改成按区域面积选。
- 原无条件 `palette[d % m]` 依赖“填到底”，融合后必须在 d > N 时改回 base。
- 调色板构造本身仍沿用原题：标记按深度排序，若最浅标记不在边界，则先放 base。

d) 放置合法性
- 生成器继续使用整行/整列墙递归切分，所以区域天然互不重叠、都在画布内且 4 连通。
- 每个区域的 N 个标记沿某个角的对角线放置，深度连续，保证都落在本区域内且互不覆盖。
- 目标区域额外要求足够深，使裁剪输出里一定留下未填充内核，从而让深度上限可见。

e) 视觉可观察性
- 输入里所有区域都有相同数量的标记，解题者能先发现“标记数不再是选区线索”。
- 输出只保留最大那块区域，且该区域中心留空；因此两条新规律都能从少量样例直接看出。
- 规则只依赖可见几何量：区域大小、标记个数、标记到边界的相对深度；不依赖颜色编号。

f) 可视化整洁度
- 区域保持矩形，裁剪后没有墙残留，也没有多对象重叠。
- 标记仍是孤立像素，输出只含同心框和一个空心内核，图面结构比原题更简洁。

g) 变化的可推断性
- 颜色顺序仍只由可见的深度顺序决定，不引入新的颜色映射表。
- 真正新增的是“先按最大区域裁剪”和“只填到统一的 N 层深度”；二者都有明确几何证据。
"""

import random

import numpy as np

from arc_utils import components_mask


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


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


def _region_infos(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    infos = []
    for pts_arr in _cc4(g != wall):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        vals = g[comp_mask]
        base = _mode(vals)
        markers = [(r, c) for r, c in pts if int(g[r, c]) != int(base)]
        r0, c0, r1, c1 = _bbox_from_points(pts)
        infos.append(
            {
                "pts": pts,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "base": int(base),
                "marker_count": len(markers),
                "area": len(pts),
            }
        )
    return int(bg), int(wall), infos


def _select_target_region(grid: np.ndarray):
    _bg, _wall, infos = _region_infos(grid)
    assert infos
    best = None
    best_key = None
    for info in infos:
        r0, c0, _r1, _c1 = info["bbox"]
        key = (int(info["area"]), -int(r0), -int(c0))
        if best is None or key > best_key:
            best = info
            best_key = key
    return best


def _build_local_region_view(grid: np.ndarray, region_info: dict):
    g = np.asarray(grid, dtype=int)
    pts = region_info["pts"]
    base = int(region_info["base"])
    r0, c0, r1, c1 = region_info["bbox"]
    out_h = int(r1 - r0 + 1)
    out_w = int(c1 - c0 + 1)

    comp_mask = np.zeros((out_h, out_w), dtype=bool)
    for r, c in pts:
        comp_mask[int(r - r0), int(c - c0)] = True

    dist = _boundary_distance(comp_mask)
    dist_to_color: dict[int, int] = {}
    marker_count = 0
    for r, c in pts:
        rr = int(r - r0)
        cc = int(c - c0)
        v = int(g[r, c])
        if v == base:
            continue
        marker_count += 1
        d = int(dist[rr, cc])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = v

    if dist_to_color:
        palette: list[int] = []
        ds = sorted(dist_to_color)
        if ds[0] > 0:
            palette.append(base)
        for d in ds:
            palette.append(int(dist_to_color[d]))
    else:
        palette = [base]

    return comp_mask, dist, palette, marker_count, base


def _render_selected_region(grid: np.ndarray, *, capped: bool) -> np.ndarray:
    region = _select_target_region(grid)
    comp_mask, dist, palette, marker_count, base = _build_local_region_view(grid, region)
    out = np.full(comp_mask.shape, base, dtype=int)
    m = len(palette)

    for r in range(comp_mask.shape[0]):
        for c in range(comp_mask.shape[1]):
            if not comp_mask[r, c]:
                continue
            d = int(dist[r, c])
            if (not capped) or d <= marker_count:
                out[r, c] = int(palette[d % m])
            else:
                out[r, c] = base
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the largest partition and draw only its first few concentric layers.

    Input structure:
    - One wall color splits the canvas into several 4-connected regions.
    - Every region has the same number N of isolated marker pixels, and each region otherwise keeps one dominant base color.
    - Marker pixels in a region lie at consecutive Chebyshev depths from one corner toward the interior.

    Rule:
    - Ignore walls and compare region sizes; choose the single largest region and crop the output to that region's tight bounding box.
    - Inside the chosen region, order marker colors by boundary distance to form the palette, prepending the base color if the shallowest marker is not on the boundary.
    - Let N be the marker count of that region. Paint only cells whose boundary distance d satisfies d <= N using palette[d % palette_length]; leave deeper cells as the base color.

    Key insight: Because all regions carry the same number of markers, marker count no longer identifies the target region. The useful signals are: largest region decides the crop, while the shared marker count sets how many concentric layers survive before the center is left blank.
    """
    return _render_selected_region(np.asarray(grid, dtype=int), capped=True)


def _full_fill_selected_region(grid: np.ndarray) -> np.ndarray:
    return _render_selected_region(np.asarray(grid, dtype=int), capped=False)


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 4
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Marker planning ----
MARKERS_PER_REGION_MAX = 3
START_DIST_MAX = 1
# Require at least one deeper ring beyond the N-layer cap; the final oracle below
# still rejects cases where the truncated center is too small to read cleanly.
MIN_VISIBLE_EXTRA_DEPTH = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 8

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 500


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _place_walls(h: int, w: int, rng: random.Random):
    wall_mask = np.zeros((h, w), dtype=int)
    rects = [(0, 0, h - 1, w - 1)]
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
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
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


def _largest_rect_index(rects):
    areas = [(rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1) for rect in rects]
    best_area = max(areas)
    if areas.count(best_area) != 1:
        return None
    return areas.index(best_area)


def _start_options(max_d: int, n_markers: int, *, require_visible_cap: bool) -> list[int]:
    options = []
    for start_d in range(START_DIST_MAX + 1):
        if start_d + n_markers - 1 > max_d:
            continue
        if require_visible_cap:
            needed = n_markers + MIN_VISIBLE_EXTRA_DEPTH + (1 if start_d > 0 else 0)
            if max_d < needed:
                continue
        options.append(start_d)
    return options


def _choose_marker_count(rects, target_idx: int, rng: random.Random):
    depths = []
    for r0, c0, r1, c1 in rects:
        depths.append(_max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1))

    feasible = []
    for n_markers in range(1, MARKERS_PER_REGION_MAX + 1):
        if any(depth < n_markers - 1 for depth in depths):
            continue
        if _start_options(depths[target_idx], n_markers, require_visible_cap=True):
            feasible.append(n_markers)
    if not feasible:
        return None

    # Bias slightly toward deeper branches so 1/2/3-marker cases all appear
    # in generated batches instead of collapsing mostly to the shallowest cap.
    weight_map = {1: 1, 2: 2, 3: 5}
    weights = [weight_map[n] for n in feasible]
    return rng.choices(feasible, weights=weights)[0]


def _plan_region_starts(rects, target_idx: int, n_markers: int, rng: random.Random):
    starts = []
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect
        max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
        options = _start_options(
            max_d,
            n_markers,
            require_visible_cap=(idx == target_idx),
        )
        if not options:
            return None
        starts.append(rng.choice(options))
    return starts


def _marker_cells_for_rect(rect, n_markers: int, start_d: int, rng: random.Random):
    r0, c0, r1, c1 = rect
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    return [
        (start_d + k, br + dr * (start_d + k), bc + dc * (start_d + k))
        for k in range(n_markers)
    ]


def _assign_region_colors(n_markers: int, marker_pool: list[int], used_colors: set[int], rng: random.Random):
    colors = []
    unused = [c for c in marker_pool if c not in used_colors]
    take = min(n_markers, len(unused))
    if take:
        picked = rng.sample(unused, take)
        colors.extend(picked)
        used_colors.update(picked)
    while len(colors) < n_markers:
        choices = [c for c in marker_pool if c not in colors]
        colors.append(rng.choice(choices))
    return colors


def _stamp_markers(grid, rects, starts, n_markers: int, bg: int, marker_pool: list[int], rng: random.Random):
    used = set()
    for rect, start_d in zip(rects, starts):
        cells = _marker_cells_for_rect(rect, n_markers, start_d, rng)
        colors = _assign_region_colors(n_markers, marker_pool, used, rng)
        for (_d, rr, cc), col in zip(cells, colors):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = int(col)
    return used


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != int(bg):
        return False
    return len(order) >= 2 and int(vals[order[1]]) == int(wall)


def _gen_attempt(rng: random.Random):
    h = rng.randint(H_MIN, H_MAX)
    w = rng.randint(W_MIN, W_MAX)
    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(h, w, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    target_idx = _largest_rect_index(rects)
    if target_idx is None:
        return None

    n_markers = _choose_marker_count(rects, target_idx, rng)
    if n_markers is None:
        return None

    starts = _plan_region_starts(rects, target_idx, n_markers, rng)
    if starts is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_markers(grid, rects, starts, n_markers, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    _bg, _wall, infos = _region_infos(grid)
    counts = [int(info["marker_count"]) for info in infos]
    if any(count != n_markers for count in counts):
        return None

    areas = [int(info["area"]) for info in infos]
    best_area = max(areas)
    if areas.count(best_area) != 1:
        return None

    selected = _select_target_region(grid)
    if int(selected["marker_count"]) != n_markers:
        return None
    if tuple(selected["bbox"]) != tuple(rects[target_idx]):
        return None

    out = main(grid)
    full = _full_fill_selected_region(grid)
    r0, c0, r1, c1 = selected["bbox"]
    raw_crop = grid[r0 : r1 + 1, c0 : c1 + 1]

    if out.shape == grid.shape:
        return None
    if np.array_equal(out, raw_crop):
        return None
    if np.array_equal(out, full):
        return None
    if int(np.count_nonzero(out != full)) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

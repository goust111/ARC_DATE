# Variant of 13e47133
# Dimension: D6 — 网格形状：输出的裁剪方式 (grid_shape)
# Change: 输出与输入网格尺寸完全相同... → 裁剪到单个区域
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 输入仍是被单像素墙分隔的多个矩形区域；每个区域主体颜色相同，区域内有 0..3 个异色标记，
#   标记沿某个角落向内的对角线，且它们到区域边界的切比雪夫距离连续。
# - 原题：对每个区域都建立调色板并填满整张图，墙保留。
# - 变体：先找“标记数最多”的那个区域，只对它执行原题的调色板填充，然后把输出裁成该区域的紧凑 bbox。
# - 小例子追踪：若三块区域的标记数分别是 1, 3, 0，原题会输出三块区域都出现同心框；
#   现在只会保留那块有 3 个标记的区域，并输出它单独的填充结果，外部墙和其他区域全部消失。
#
# b) 复杂度校准
# - 选择规则只引入一个新的自由度：先比较“每个区域里有多少个异色标记”。
# - 生成器强制唯一最大值，避免 solver 依赖看不见的 tie-break。
# - 仍保留原题的边界距离 / 调色板循环机制，因此变化集中在 D6 的输出裁剪层。
#
# c) 隐式假设审查
# - 原 `out = g.copy()` 与“遍历所有区域并原地回写”依赖输出尺寸与输入相同；变体后不成立，
#   必须改为只构造目标区域的局部输出。
# - 原 `for pts in comps:` 对每个区域都填色；变体后必须先统计各区域标记数，再只处理被选中的区域。
# - 原来墙颜色保留在输出中；变体后输出里根本不应出现墙，因此不能再复制整张输入作为底板。
# - 原调色板构造本身仍成立，因为目标区域内部的“base + markers by distance”语义未变。
#
# d) 放置合法性
# - 生成器继续用整行/整列墙递归切分，所以区域互不重叠、都在画布内、且天然是 4 连通。
# - 标记只放在所属矩形区域内的单条对角线上，距离连续，不会越界，也不会彼此重叠。
# - 目标区域要求至少容纳 2 个标记，从而裁剪后的输出中能看到明确的填充效果。
#
# e) 视觉可观察性
# - 解题者肉眼可以先数各分区的异色点，再发现输出只对应“点最多”的那一块。
# - 生成器保证至少还有一个非目标区域带少量标记，避免“唯一有标记的区域”把规则退化成“挑有点的区域”。
#
# f) 可视化整洁度
# - 区域保持矩形，裁剪后输出就是单个整洁矩形，不引入孔洞或墙残留。
# - 标记总数受控，不会让输入过于杂乱，也不会压过同心框模式本身。
#
# g) 颜色变化可推断性
# - 不使用颜色 id 排序或映射；颜色顺序仅由各标记到边界的可见几何距离决定。
# - 各区域内部的颜色变化规律与原题一致，因此样例中的新信息只在“选哪块并裁掉其余内容”上。

import random

import numpy as np

from arc_utils import components_mask


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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
    q = []

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
    for pts in _cc4(g != wall):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        vals = g[comp_mask]
        base = _mode(vals)
        markers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) != int(base)]
        r0, c0, r1, c1 = _bbox_from_points(pts)
        infos.append(
            {
                "pts": [(int(r), int(c)) for r, c in pts],
                "base": int(base),
                "marker_count": len(markers),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
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
        key = (int(info["marker_count"]), int(info["area"]), -int(r0), -int(c0))
        if best is None or key > best_key:
            best = info
            best_key = key
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Isolate the most-marked region and fill it with concentric bands derived from its markers.

    Input structure:
    - The grid is partitioned by wall-colored lines into several 4-connected regions.
    - Each region has one dominant base color plus a few isolated marker pixels at different Chebyshev depths from that region's boundary.

    Rule:
    - Count how many marker pixels each region contains and choose the unique region with the largest count.
    - Inside that region, build the palette exactly as in the original task: order marker colors by boundary distance, prepend the base color when the first marker is not on the boundary, then fill by palette[distance mod palette_length].
    - Output only the chosen region's tight crop; walls and all other regions are discarded.

    Key insight: The new step is not a new color transformation but a region-selection stage: first identify the richest marker set, then apply the old concentric-frame rule locally and crop to that single compartment.
    """
    g = np.asarray(grid, dtype=int)
    target = _select_target_region(g)

    pts = target["pts"]
    base = int(target["base"])
    r0, c0, r1, c1 = target["bbox"]
    out_h = int(r1 - r0 + 1)
    out_w = int(c1 - c0 + 1)

    comp_mask = np.zeros((out_h, out_w), dtype=bool)
    for r, c in pts:
        comp_mask[int(r - r0), int(c - c0)] = True

    dist = _boundary_distance(comp_mask)

    dist_to_color = {}
    for r, c in pts:
        rr = int(r - r0)
        cc = int(c - c0)
        v = int(g[r, c])
        if v == int(base):
            continue
        d = int(dist[rr, cc])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = v

    if dist_to_color:
        palette = []
        ds = sorted(dist_to_color)
        if ds[0] > 0:
            palette.append(int(base))
        for d in ds:
            palette.append(int(dist_to_color[d]))
    else:
        palette = [int(base)]

    out = np.full((out_h, out_w), int(base), dtype=int)
    m = len(palette)
    for r, c in pts:
        rr = int(r - r0)
        cc = int(c - c0)
        out[rr, cc] = int(palette[int(dist[rr, cc]) % m])
    return out


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
TARGET_MIN_MARKERS = 2
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 8

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


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


def _choose_target_rect(rects, rng: random.Random):
    candidates = []
    for i, rect in enumerate(rects):
        r0, c0, r1, c1 = rect
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        depth = _max_cheb_depth(h, w)
        if depth >= TARGET_MIN_MARKERS:
            candidates.append((i, depth, h * w))
    if not candidates:
        return None
    max_area = max(area for _i, _d, area in candidates)
    large = [item for item in candidates if item[2] >= max_area // 2]
    return rng.choice(large)[0]


def _marker_cells_for_rect(rect, n_markers: int, rng: random.Random):
    if n_markers <= 0:
        return []
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    max_d = _max_cheb_depth(h, w)
    if n_markers > min(MARKERS_PER_REGION_MAX, max_d):
        return None

    max_start = max_d - n_markers
    start_d = rng.randint(0, min(1, max_start))
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    return [
        (start_d + k, br + dr * (start_d + k), bc + dc * (start_d + k))
        for k in range(n_markers)
    ]


def _plan_marker_counts(rects, target_idx: int, rng: random.Random):
    depths = []
    for rect in rects:
        r0, c0, r1, c1 = rect
        depths.append(_max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1))

    target_depth = depths[target_idx]
    target_count = 3 if target_depth >= 3 and rng.random() < 0.8 else 2
    counts = [0] * len(rects)
    counts[target_idx] = target_count

    runner_candidates = [
        i for i, d in enumerate(depths) if i != target_idx and d >= target_count - 1
    ]
    if runner_candidates:
        runner = rng.choice(runner_candidates)
        counts[runner] = target_count - 1

    for i, depth in enumerate(depths):
        if i == target_idx or counts[i] > 0 or depth <= 0:
            continue
        counts[i] = rng.choice([0, 1]) if depth >= 1 else 0

    if sum(1 for i, n in enumerate(counts) if i != target_idx and n > 0) == 0:
        fallback = [i for i, d in enumerate(depths) if i != target_idx and d >= 1]
        if not fallback:
            return None
        counts[rng.choice(fallback)] = 1

    if any(n >= target_count for i, n in enumerate(counts) if i != target_idx):
        return None
    return counts


def _assign_region_colors(n_markers: int, marker_pool: list[int], used_colors: set[int], rng: random.Random):
    if n_markers == 0:
        return []
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


def _stamp_markers(grid, rects, counts, bg, marker_pool, rng: random.Random):
    used = set()
    placements = []
    for rect, n_markers in zip(rects, counts):
        cells = _marker_cells_for_rect(rect, n_markers, rng)
        if cells is None:
            return None, None
        colors = _assign_region_colors(n_markers, marker_pool, used, rng)
        for (_d, rr, cc), col in zip(cells, colors):
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = int(col)
            placements.append((rect, rr, cc, int(col)))
    return used, placements


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

    target_idx = _choose_target_rect(rects, rng)
    if target_idx is None:
        return None

    counts = _plan_marker_counts(rects, target_idx, rng)
    if counts is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, _placements = _stamp_markers(grid, rects, counts, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    selected = _select_target_region(grid)
    if selected["marker_count"] != counts[target_idx]:
        return None

    max_other = max(
        (counts[i] for i in range(len(counts)) if i != target_idx),
        default=-1,
    )
    if counts[target_idx] <= max_other:
        return None
    if selected["marker_count"] <= max_other:
        return None

    target_bbox = rects[target_idx]
    if tuple(selected["bbox"]) != tuple(target_bbox):
        return None

    out = main(grid)
    if out.shape == grid.shape:
        return None
    r0, c0, r1, c1 = selected["bbox"]
    if np.array_equal(out, grid[r0 : r1 + 1, c0 : c1 + 1]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

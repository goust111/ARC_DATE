# Variant of 13e47133
# Dimension: D2 — 轨迹：颜色交替模式 (trajectory)
# Change: 调色板按距离循环：palette[dist % len]，标记颜色与背景色交替... → 渐变式颜色循环
#
# design_analysis:
#   concrete_behavior:
#     - 输入仍是被单色墙分开的多个矩形区域；每个区域大多是底色，少数标记像素沿某个角的对角线向内摆放。
#     - 输出仍按“到区域边界的切比雪夫距离”在区域内画同心矩形带，但颜色轨迹改为单向推进：
#       第 0 层用调色板第 0 色，第 1 层用第 1 色，……，到最后一种颜色后不再回绕，而是把更深的所有层都涂成最后一种颜色。
#     - 例子：若某区域调色板是 [7, 8, 4]，距离序列原题是 7,8,4,7,8,4,...；本变体变为 7,8,4,4,4,...。
#   flow_impact:
#     - 原逻辑唯一的核心旧假设是“更深层会周期性回到外层颜色”，实现上就是 palette[d % m]。
#     - 变体把该假设替换为“颜色只前进不回头”，实现上应改为 palette[min(d, m - 1)]。
#     - 生成器必须同步适配：若所有区域深度都 <= 最后一个标记层，则新旧规则输出相同，视觉上无法观察到 D2 变化。
#   complexity_calibration:
#     - 仍保留 0..3 个标记、3..6 个区域、同一套墙体分割机制，避免引入额外自由度。
#     - 为了让新规则可见，每个样本强制至少存在一个“焦点区域”：最大深度 >= 3，且放置 3 个连续深度标记。
#       这样该区域必然出现 d=3 的格子，原题会回到第 0 色，而变体会继续保持第 2 色。
#   implicit_assumption_review:
#     - bg = 全局众数、wall = 非 bg 中众数：仍成立，所以生成器继续验证 wall 为第二众数。
#     - base = 区域众色：仍依赖“标记只是少数像素”，因此每个区域最多 3 个标记且区域面积远大于 3。
#     - dist_to_color 每个距离只取首个颜色：仍要求每个区域同一深度至多一个标记；生成器继续沿单条对角线放置连续深度标记来保证。
#     - 对所有区域直接写 out[r, c]：仍安全，因为各 free-region 互不重叠，墙不参与重绘。
#   placement_legality:
#     - 墙由递归整行/整列切分产生，非墙连通块保持矩形，因此角点对角线放置的标记一定处在对应区域内。
#     - 焦点区域只在 max_depth >= 3 时选取，并固定使用 d=0,1,2 三个标记，故不会越界且必有更深层触发“封顶不回绕”。
#   observability_and_cleanliness:
#     - 规则只依赖可见的几何深度，不依赖颜色 id 排序。
#     - 颜色种类上限仍是 8 个标记色，且每个区域的线索是局部、整洁、可复用的同心框图案，不会造成视觉混乱。
#   generator_guarantees_for_new_solver:
#     - 每个样本都要求 main(input) 与旧循环版求解器输出不同，防止退化回原题。
#     - 至少一个区域有“最后一种颜色吞掉内核”的现象，保证人类能从少量样例中观察到新轨迹。

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

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


def _solve(grid: np.ndarray, *, cyclic: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d < 0:
                continue
            if d not in dist_to_color:
                dist_to_color[d] = v

        palette: list[int] = []
        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            idx = d % m if cyclic else min(d, m - 1)
            out[r, c] = palette[idx]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint wall-separated regions by reading a depth-ordered palette from marker pixels.

    Input structure:
    - One wall color divides the board into several non-wall regions.
    - Inside each region, the dominant color is the region base and a few isolated marker pixels sit at increasing Chebyshev depths from the boundary.

    Rule:
    - For each region, sort marker colors by boundary depth; if the shallowest marker is not on the boundary, insert the base color at the front.
    - Fill the region by boundary distance, but use the palette as a one-way gradient: layer 0 uses the first color, layer 1 the second, and once the last palette color is reached it continues through every deeper layer.

    Key insight: Marker depth still defines the palette order, but the trajectory is no longer periodic; the innermost marker color absorbs the entire remaining core instead of restarting the palette.
    """
    return _solve(grid, cyclic=False)


def _main_cyclic(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, cyclic=True)


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
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Variant visibility ----
FOCUS_MIN_DEPTH = 3

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _place_walls(H: int, W: int, rng: random.Random):
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
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _marker_coords(rect, start_d: int, n: int, rng: random.Random):
    r0, c0, r1, c1 = rect
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    return [(start_d + k, br + dr * (start_d + k), bc + dc * (start_d + k)) for k in range(n)]


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    start_d = rng.randint(0, min(START_DIST_MAX, max_d - n))
    coords = _marker_coords(rect, start_d, n, rng)
    colors = rng.sample(marker_pool, n)
    return [(d, col, rr, cc) for (d, rr, cc), col in zip(coords, colors)]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _place_focus_markers(rect, marker_pool: list[int], rng: random.Random):
    coords = _marker_coords(rect, start_d=0, n=3, rng=rng)
    colors = rng.sample(marker_pool, 3)
    return [(d, col, rr, cc) for (d, rr, cc), col in zip(coords, colors)]


def _stamp_markers(grid, rects, bg, marker_pool, focus_index: int, rng: random.Random):
    used = set()
    focus_rect = rects[focus_index]

    for d, col, rr, cc in _place_focus_markers(focus_rect, marker_pool, rng):
        if not (focus_rect[0] <= rr <= focus_rect[2] and focus_rect[1] <= cc <= focus_rect[3]):
            return None
        if grid[rr, cc] != bg:
            return None
        grid[rr, cc] = col
        used.add(col)

    for i, rect in enumerate(rects):
        if i == focus_index:
            continue
        for _d, col, rr, cc in _place_markers(rect, marker_pool, rng):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)
    return used


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _choose_focus_region(rects, rng: random.Random):
    candidates = []
    for i, rect in enumerate(rects):
        depth = _max_cheb_depth(rect[2] - rect[0] + 1, rect[3] - rect[1] + 1)
        if depth >= FOCUS_MIN_DEPTH:
            candidates.append(i)
    if not candidates:
        return None
    return rng.choice(candidates)


def _variant_visible(inp: np.ndarray) -> bool:
    return not np.array_equal(main(inp), _main_cyclic(inp))


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    focus_index = _choose_focus_region(rects, rng)
    if focus_index is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_markers(grid, rects, bg, marker_pool, focus_index, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    if not _variant_visible(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

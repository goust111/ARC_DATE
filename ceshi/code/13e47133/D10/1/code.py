# Variant of 13e47133
# Dimension: D10 — 距离度量方式 (meta)
# Change: 使用切比雪夫距离（8-邻域）计算到区域边界的距离... → 使用欧几里得距离（取整）

# Design analysis for the D10 variant:
# spec:
#   concretization_and_trace: |
#     输入仍然是“墙颜色”把底色区域切开的网格，区域里只有少量 marker 像素携带调色板信息。
#     变化点只在“区域深度”的定义：原题把每个格子的层数当作到区域边界的切比雪夫层数，
#     所以矩形或 L 形区域都会长出轴对齐的方框带。这里改成“到区域外部的欧几里得距离取整，
#     并整体减 1，使边界层仍是 0”。小例子：一个 7x7 的 L 形区域里，原逻辑会把很多内层格子
#     归到同一个方框层；新逻辑里靠近拐角的格子会因为斜向离外部更近而提前落到较浅层，
#     因此同一层的轮廓会从方角变成圆角，marker 排序和最终填色都一起改变。
#   complexity_calibration: |
#     如果 generator 继续只产出纯矩形区域，整数欧氏层在很多样例上会与原层数完全重合，
#     变体会退化成看不见的改动。因此 generator 必须稳定产出至少一个较大的非矩形区域，
#     但仍然只保留 3 个分区和 1 个墙连通块，避免规则过度复杂。marker 数量仍限制在每区 1-3，
#     整体颜色数控制在原题范围内。
#   implicit_assumptions_review: |
#     原 main() 里与 D10 相关的依赖有三处：
#     1. `_boundary_distance` 原先假设“每次 8 邻域 BFS +1 就是层号”。变体下这不成立，必须改成
#        欧氏距离变换，不能只换注释。
#     2. marker palette 的顺序直接读取 `dist[r, c]`。如果 solver 的距离函数变了，但 generator
#        仍按旧层数放 marker，就会把颜色放到错误的 palette 位置，因此放置逻辑也必须改成读取
#        新欧氏层。
#     3. `out[r, c] = palette[d % m]` 假设边界层从 0 开始。新的欧氏距离实现要显式减 1，保持边界
#        仍是第 0 层，否则所有 palette 会整体错位。
#   placement_legality: |
#     墙采用单个连通的折线/T 形模板，所有墙像素都有正交墙邻居；自由区域由 4 连通组件定义。
#     marker 只从目标组件内部、指定欧氏层的候选格里选，天然不会越界，也不会跨区重叠。
#     generator 在落点后再运行新 solver 验证，并额外检查新旧 solver 输出确实不同。
#   observability_and_cleanliness: |
#     为了让人眼能看出新规则，非矩形区域必须足够大且至少有 2 层以上深度；否则圆角层根本看不见。
#     视觉上仍只保留三种元素：底色、墙、孤立 marker。marker 按同一角落附近的层递进摆放，
#     不引入依赖颜色 id 的排序，也不叠加额外噪声。
#   color_inference: |
#     颜色规则完全沿用原题：颜色只通过“离边界的层号顺序”进入 palette，绝不按颜色编号变换。
#     每区 marker 数很少、且输出中的同一颜色会重复出现在多个环带中，因此人类仍可从 2-4 个样例
#     归纳“marker 决定循环调色板，距离层决定填色位置”。

import random

import numpy as np
from scipy.ndimage import distance_transform_edt

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute integer Euclidean depth for True-cells in component_mask.

    The raw Euclidean transform measures distance to the padded exterior in
    cell-center units; subtract 1 so that boundary cells remain on layer 0,
    then take the integer part.
    """
    padded = np.pad(component_mask.astype(np.uint8), 1, constant_values=0)
    raw = distance_transform_edt(padded)[1:-1, 1:-1]
    dist = np.full(component_mask.shape, -1, dtype=int)
    layers = np.floor(np.maximum(raw - 1.0, 0.0) + 1e-9).astype(int)
    dist[component_mask] = layers[component_mask]
    return dist


def _boundary_distance_chebyshev(component_mask: np.ndarray) -> np.ndarray:
    """Original 8-neighborhood boundary depth, retained for generator checks."""
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


def _solve_with_distance(grid: np.ndarray, distance_fn) -> np.ndarray:
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
        dist = distance_fn(comp_mask)

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

        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            palette = [int(base)] if ds[0] > 0 else []
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            out[r, c] = palette[int(dist[r, c]) % m]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint wall-delimited regions with cyclic bands indexed by Euclidean depth.

    Input structure:
    - One wall color splits the grid into several non-wall regions.
    - Each region is mostly a single base color, plus a few isolated marker pixels whose depths from the region exterior differ.

    Rule:
    - For each region, sort marker colors by the integer part of their Euclidean distance from the exterior; prepend the base color if the first marker is not on depth 0.
    - Fill the whole region with palette[depth mod palette_length], where depth is the same integer Euclidean layer number for every cell.

    Key insight:
    The markers still define a repeating palette, but the layer index now comes from Euclidean distance, so large irregular regions turn into rounded concentric bands instead of purely rectangular shells.
    """
    return _solve_with_distance(grid, _boundary_distance)


# --------------- generator ---------------

H_MIN, H_MAX = 13, 28
W_MIN, W_MAX = 13, 28

MIN_REGION_DIM = 4
N_REGIONS_MIN, N_REGIONS_MAX = 3, 3
MIN_REGION_AREA, MAX_REGION_AREA = 18, 350

MARKERS_PER_REGION_MAX = 3
START_DIST_MAX = 1
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

MIN_VARIANT_DIFF = 12
GEN_MAX_ATTEMPTS = 800

_ANCHORS = ["tl", "tr", "bl", "br"]


def _main_chebyshev(grid: np.ndarray) -> np.ndarray:
    return _solve_with_distance(grid, _boundary_distance_chebyshev)


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rect_component(pts: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(pts)
    return int(len(pts)) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _canonical_wall_v(H: int, W: int, rng: random.Random) -> np.ndarray:
    wall = np.zeros((H, W), dtype=int)
    r1 = rng.randint(MIN_REGION_DIM, H - MIN_REGION_DIM - 1)
    c1 = rng.randint(MIN_REGION_DIM, W - 2 * MIN_REGION_DIM - 1)
    c2 = rng.randint(c1 + MIN_REGION_DIM, W - MIN_REGION_DIM - 1)
    wall[: r1 + 1, c1] = 1
    wall[r1, c1:] = 1
    wall[r1:, c2] = 1
    return wall


def _canonical_wall_h(H: int, W: int, rng: random.Random) -> np.ndarray:
    wall = np.zeros((H, W), dtype=int)
    r1 = rng.randint(MIN_REGION_DIM, H - 2 * MIN_REGION_DIM - 1)
    r2 = rng.randint(r1 + MIN_REGION_DIM, H - MIN_REGION_DIM - 1)
    c1 = rng.randint(MIN_REGION_DIM, W - MIN_REGION_DIM - 1)
    wall[r1, : c1 + 1] = 1
    wall[r1:, c1] = 1
    wall[r2, c1:] = 1
    return wall


def _random_wall_mask(H: int, W: int, rng: random.Random) -> np.ndarray:
    builders = []
    if H - 2 * MIN_REGION_DIM - 1 >= MIN_REGION_DIM:
        builders.append(_canonical_wall_h)
    if W - 2 * MIN_REGION_DIM - 1 >= MIN_REGION_DIM:
        builders.append(_canonical_wall_v)
    if not builders:
        return np.zeros((H, W), dtype=int)

    wall = rng.choice(builders)(H, W, rng)
    if rng.random() < 0.5:
        wall = np.flipud(wall)
    if rng.random() < 0.5:
        wall = np.fliplr(wall)
    return np.ascontiguousarray(wall)


def _anchor_point(pts: np.ndarray, anchor: str) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(pts)
    if anchor == "tl":
        return r0, c0
    if anchor == "tr":
        return r0, c1
    if anchor == "bl":
        return r1, c0
    return r1, c1


def _layer_cells(pts: np.ndarray, dist: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        d = int(dist[r, c])
        if d >= 0:
            out.setdefault(d, []).append((int(r), int(c)))
    return out


def _choose_marker_layers(
    pts: np.ndarray,
    dist: np.ndarray,
    rng: random.Random,
    *,
    min_markers: int,
) -> list[tuple[int, int]]:
    layers = _layer_cells(pts, dist)
    if not layers:
        return []

    max_d = max(layers)
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper < min_markers:
        return []

    lo = min_markers
    hi = upper
    if lo == hi:
        n = lo
    else:
        n = rng.randint(lo, hi)

    starts = []
    max_start = min(START_DIST_MAX, max_d - n + 1)
    for start_d in range(max_start + 1):
        if all((start_d + k) in layers for k in range(n)):
            starts.append(start_d)
    if not starts:
        return []

    start_d = rng.choice(starts)
    anchor = _anchor_point(pts, rng.choice(_ANCHORS))
    chosen = []
    for d in range(start_d, start_d + n):
        cells = sorted(
            layers[d],
            key=lambda rc: ((rc[0] - anchor[0]) ** 2 + (rc[1] - anchor[1]) ** 2, rc[0], rc[1]),
        )
        chosen.append(cells[0])
    return chosen


def _stamp_markers(
    grid: np.ndarray,
    comps: list[np.ndarray],
    marker_pool: list[int],
    rng: random.Random,
) -> set[int] | None:
    used: set[int] = set()
    remaining = marker_pool[:]
    rng.shuffle(remaining)

    irregular_indices = [i for i, pts in enumerate(comps) if not _is_rect_component(pts)]
    if not irregular_indices:
        return None
    irregular_idx = max(irregular_indices, key=lambda i: len(comps[i]))

    for i, pts in enumerate(comps):
        comp_mask = np.zeros_like(grid, dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        dist = _boundary_distance(comp_mask)

        min_markers = 2 if i == irregular_idx else 1
        coords = _choose_marker_layers(pts, dist, rng, min_markers=min_markers)
        if not coords:
            return None
        if len(remaining) < len(coords):
            return None

        colors = [remaining.pop() for _ in coords]
        for (r, c), col in zip(coords, colors):
            if grid[r, c] != _mode(grid[comp_mask]):
                return None
            grid[r, c] = col
            used.add(col)

    return used


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    new_out = main(grid)
    old_out = _main_chebyshev(grid)
    diff = int(np.sum(new_out != old_out))

    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = grid != wall

    comp_gap = 0
    irregular_gap = 0
    for pts in _cc4(free):
        comp_mask = np.zeros_like(grid, dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        eu = _boundary_distance(comp_mask)
        ch = _boundary_distance_chebyshev(comp_mask)
        gap = int(np.sum((eu != ch) & comp_mask))
        comp_gap = max(comp_gap, gap)
        if not _is_rect_component(pts):
            irregular_gap = max(irregular_gap, gap)

    return {
        "output_diff": diff,
        "max_component_gap": comp_gap,
        "max_irregular_gap": irregular_gap,
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)
    wall_mask = _random_wall_mask(H, W, rng)

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    free = grid != wall
    comps = _cc4(free)
    if not (N_REGIONS_MIN <= len(comps) <= N_REGIONS_MAX):
        return None
    if not all(MIN_REGION_AREA <= len(pts) <= MAX_REGION_AREA for pts in comps):
        return None
    if not any(not _is_rect_component(pts) for pts in comps):
        return None

    used = _stamp_markers(grid, comps, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        old_out = _main_chebyshev(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, old_out):
        return None

    metrics = _variant_metrics(grid)
    if metrics["output_diff"] < MIN_VARIANT_DIFF:
        return None
    if metrics["max_irregular_gap"] == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

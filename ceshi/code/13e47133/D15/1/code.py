# Variant of 13e47133
# Dimension: D15 — 符号指令：调色板构建的特殊规则 (symbol_instructions)
# Change: 无显式符号编码；标记像素的距离隐式决定调色板顺序，距离 > 0 时背景色自动前置... → 标记像素的形状编码额外参数

import random

import numpy as np

from arc_utils import components_mask, neighbors8

# Design analysis for the D15 variant
# a) Concrete behavior:
#    原题在每个区域里放若干单像素标记，直接用这些像素到“最近区域边界”的切比雪夫距离给调色板排序。
#    这个变体把单像素替换成 3x3 的同色 L 形记号；颜色仍然代表调色板颜色，但排序距离不再默认从最近边界测，
#    而是由 L 的朝向指定测量起点：
#      - 朝左: 从该区域的左边界往右数
#      - 朝右: 从该区域的右边界往左数
#      - 朝上: 从该区域的上边界往下数
#      - 朝下: 从该区域的下边界往上数
#    具体例子：一个右朝向 L 若离右边界 1 格、但同时贴着上边界，那么原题会把它当作距离 0 的标记，
#    变体则把它读成距离 1。于是同一批颜色在调色板中的顺序会变化，最终同心框的循环颜色顺序也变化。
# b) Complexity calibration:
#    只使用 4 个朝向、1 种符号形状，避免组合爆炸；每张图中会重复出现多个 L 标记，且不同区域可复用朝向，
#    让“朝向决定从哪条边计距”在 2-4 个样例里可见。符号足够大(3x3)以便肉眼辨认，不会退化成难以看清的像素噪声。
# c) Audit of original solver assumptions:
#    - 原代码把“所有非 base 单元格”都视为单像素 markers，并直接读 dist[r,c]。这在多像素符号下失效。
#    - 原代码默认一个距离层只会遇到一个标记像素；现在必须先识别整块同色 L 组件，再给整块组件一个距离。
#    - 原代码的 output 填充依赖的是区域边界距，这一部分与 D15 无关，保留即可。
# d) Placement legality:
#    生成器只在由墙分割出的矩形区域里放置 L 标记。每个 L 都完整落在同一区域内，颜色单一，不与其他 L 重叠，
#    且留出一圈缓冲避免不同颜色的标记在 8 邻域里粘连。区域尺寸下限调到 5，使 3x3 符号能够稳定放入。
# e) Visual observability:
#    为了让新规则确实可观察，生成器强制至少出现一个“朝向指定边界”和“最近边界”不同的标记。
#    因此解题者不能仅沿用原题的“看离最近边多远”规则，必须解码 L 的朝向。
# f) Visual cleanliness:
#    每个标记都是整齐的 3x3 L，区域仍由直墙切成规整矩形；输入里只有墙、底色和少量 L 标记，输出仍是整洁的同心框。
# g) Color inference:
#    颜色本身不参与排序，排序只由符号朝向 + 到对应边界的可见间距决定，避免使用颜色 id 之类不可见信息。
#    调色板颜色数仍控制在原任务级别内，足够少，便于从结构线索中推断规则。

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

L_SHAPES = {
    "right": {(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)},
    "left": {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)},
    "up": {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)},
    "down": {(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)},
}
DIR_FROM_NAME = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
DIR_NAMES = tuple(L_SHAPES.keys())


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


def _normalize_component(pts: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, c0 = min(rs), min(cs)
    return {(r - r0, c - c0) for r, c in pts}, r0, c0


def _decode_l_direction(pts: list[tuple[int, int]]) -> str | None:
    norm, _r0, _c0 = _normalize_component(pts)
    for name, shape in L_SHAPES.items():
        if norm == shape:
            return name
    return None


def _same_color_components(grid: np.ndarray, comp_mask: np.ndarray, base: int) -> list[tuple[int, list[tuple[int, int]]]]:
    colors = sorted(int(v) for v in np.unique(grid[comp_mask]) if int(v) != base)
    out: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors:
        mask = comp_mask & (grid == col)
        for pts in _cc4(mask):
            out.append((col, [(int(r), int(c)) for r, c in pts]))
    return out


def _ray_gap(region_set: set[tuple[int, int]], cell: tuple[int, int], dr: int, dc: int) -> int:
    r, c = cell
    gap = 0
    while (r + dr, c + dc) in region_set:
        r += dr
        c += dc
        gap += 1
    return gap


def _directed_marker_distance(
    region_set: set[tuple[int, int]], marker_pts: list[tuple[int, int]], direction: str
) -> int:
    dr, dc = DIR_FROM_NAME[direction]
    return min(_ray_gap(region_set, cell, dr, dc) for cell in marker_pts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill walled regions with cyclic concentric bands, but read the palette order from oriented L symbols.

    Input structure:
    - The wall color partitions the grid into rectangular free regions on a shared base color.
    - Each region may contain a few same-colored 3x3 L markers instead of isolated pixels.
    - The marker color is still the palette color, while the L orientation is an extra symbolic instruction.

    Rule:
    - Decode every non-base monochrome component as an L marker and read its orientation.
    - The orientation chooses which side of the region to measure from: left/right/up/down.
    - Sort marker colors by that directed side-distance; if the smallest directed distance is above 0, prepend the base color, then fill the region by palette[boundary_distance mod palette_length].

    Key insight: The output pattern is still driven by ordinary Chebyshev layers, but the palette itself is no longer recovered from nearest-boundary depth; the L symbol tells you which boundary to count from.
    """
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
        region_set = set()
        for r, c in pts:
            rr, cc = int(r), int(c)
            comp_mask[rr, cc] = True
            region_set.add((rr, cc))

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        directed_to_color: dict[int, tuple[tuple[int, int], int]] = {}
        for col, marker_pts in _same_color_components(g, comp_mask, base):
            direction = _decode_l_direction(marker_pts)
            if direction is None:
                continue
            d = _directed_marker_distance(region_set, marker_pts, direction)
            _norm, r0, c0 = _normalize_component(marker_pts)
            if d not in directed_to_color or (r0, c0) < directed_to_color[d][0]:
                directed_to_color[d] = ((r0, c0), int(col))

        palette: list[int] = []
        if directed_to_color:
            ds = sorted(directed_to_color.keys())
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(directed_to_color[d][1]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            rr, cc = int(r), int(c)
            out[rr, cc] = palette[int(dist[rr, cc]) % m]

    return out


# --------------- generator ---------------

# ---- Canvas bounds ----
H_MIN, H_MAX = 12, 30
W_MIN, W_MAX = 15, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 5
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 3, 3, 1]
START_DIST_MAX = 1
MARKER_RETRIES = 60

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400


def _max_axis_gap(h: int, w: int) -> int:
    return max(h - 3, w - 3)


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


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _shape_cells(top: int, left: int, direction: str) -> list[tuple[int, int]]:
    return sorted((top + dr, left + dc) for dr, dc in L_SHAPES[direction])


def _candidate_positions(rect, direction: str, d: int) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    out = []
    if direction == "left":
        left = c0 + d
        if left + 2 > c1:
            return out
        for top in range(r0, r1 - 1):
            out.append(_shape_cells(top, left, direction))
    elif direction == "right":
        left = c1 - d - 2
        if left < c0:
            return out
        for top in range(r0, r1 - 1):
            out.append(_shape_cells(top, left, direction))
    elif direction == "up":
        top = r0 + d
        if top + 2 > r1:
            return out
        for left in range(c0, c1 - 1):
            out.append(_shape_cells(top, left, direction))
    elif direction == "down":
        top = r1 - d - 2
        if top < r0:
            return out
        for left in range(c0, c1 - 1):
            out.append(_shape_cells(top, left, direction))
    return out


def _nearest_boundary_gap_rect(rect, cells: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = rect
    return min(min(r - r0, r1 - r, c - c0, c1 - c) for r, c in cells)


def _reserve_with_buffer(cells: list[tuple[int, int]], blocked: set[tuple[int, int]]):
    for cell in cells:
        blocked.add(cell)
        for nr, nc in neighbors8(*cell):
            blocked.add((int(nr), int(nc)))


def _max_markers_for_rect(rect) -> int:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    cap = min(MARKERS_PER_REGION_MAX, _max_axis_gap(h, w) + 1)
    if min(h, w) <= 5:
        cap = min(cap, 1)
    elif min(h, w) <= 7:
        cap = min(cap, 2)
    return max(0, cap)


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    n_cap = _max_markers_for_rect(rect)
    if n_cap <= 0:
        return [], False

    n = rng.choices(range(n_cap + 1), weights=MARKER_COUNT_WEIGHTS[: n_cap + 1])[0]
    if n == 0:
        return [], False

    max_d = _max_axis_gap(h, w)
    start_hi = min(START_DIST_MAX, max_d - (n - 1))
    if start_hi < 0:
        return None, False
    start_d = rng.randint(0, start_hi)
    distances = [start_d + k for k in range(n)]
    colors = rng.sample(marker_pool, n)

    for _ in range(MARKER_RETRIES):
        blocked: set[tuple[int, int]] = set()
        markers = []
        has_mismatch = False
        ok = True
        for d, col in zip(distances, colors):
            choices = []
            better = []
            dirs = list(DIR_NAMES)
            rng.shuffle(dirs)
            for direction in dirs:
                positions = _candidate_positions(rect, direction, d)
                rng.shuffle(positions)
                for cells in positions:
                    if any(cell in blocked for cell in cells):
                        continue
                    choices.append((direction, cells))
                    if _nearest_boundary_gap_rect(rect, cells) != d:
                        better.append((direction, cells))
            if not choices:
                ok = False
                break
            chosen_pool = better if better else choices
            direction, cells = rng.choice(chosen_pool)
            has_mismatch = has_mismatch or bool(better)
            markers.append((d, int(col), direction, cells))
            _reserve_with_buffer(cells, blocked)
        if ok:
            return markers, has_mismatch
    return None, False


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    used = set()
    mismatch_count = 0
    for rect in rects:
        placed, has_mismatch = _place_markers(rect, marker_pool, rng)
        if placed is None:
            return None, 0
        mismatch_count += int(has_mismatch)
        for _d, col, _direction, cells in placed:
            for rr, cc in cells:
                if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                    return None, 0
                if grid[rr, cc] != bg:
                    return None, 0
            for rr, cc in cells:
                grid[rr, cc] = col
            used.add(col)
    return used, mismatch_count


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _analyze_markers(grid: np.ndarray) -> tuple[int, int]:
    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = grid != wall
    total = 0
    mismatches = 0
    for pts in _cc4(free):
        comp_mask = np.zeros_like(grid, dtype=bool)
        region_set = set()
        for r, c in pts:
            rr, cc = int(r), int(c)
            comp_mask[rr, cc] = True
            region_set.add((rr, cc))
        base = _mode(grid[comp_mask])
        for _col, marker_pts in _same_color_components(grid, comp_mask, base):
            direction = _decode_l_direction(marker_pts)
            if direction is None:
                continue
            total += 1
            directed = _directed_marker_distance(region_set, marker_pts, direction)
            rs = [r for r, _ in marker_pts]
            cs = [c for _, c in marker_pts]
            rect = (min(r for r, _ in pts), min(c for _, c in pts), max(r for r, _ in pts), max(c for _, c in pts))
            nearest = _nearest_boundary_gap_rect(rect, [(r, c) for r, c in marker_pts])
            if directed != nearest:
                mismatches += 1
    return total, mismatches


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)
    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, mismatch_regions = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if mismatch_regions == 0:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    total_markers, mismatches = _analyze_markers(grid)
    if total_markers == 0 or mismatches == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

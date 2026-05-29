# Variant of 13e47133
# Dimension: D1 — 路径机制：同心框架填充方式 (path)
# Change: 从区域边界向内计算切比雪夫距离，以 palette[dist % len] 产生同心矩形框架... → BFS 激活式扩散填充

import random
from collections import deque

import numpy as np

from arc_utils import components_mask


# Design analysis for the D1/path variant
# ---------------------------------------
# spec_like_summary:
#   preserved:
#   - 仍然是 10-30 x 13-30 的网格，1 个全局底色，1 个墙色。
#   - 墙把网格切成 3-6 个 4 连通区域；墙像素在输出中保持不变。
#   - 每个区域输入中依旧主要是底色，只夹杂少量孤立的彩色标记像素。
#   changed_core:
#   - 原题把标记像素按“到区域边界的切比雪夫距离”排成调色板，再按边界深度循环填充，
#     所以输出是同心矩形框。
#   - 变体不再读取“深度顺序”。每个非底色标记都直接当作一个 BFS 种子，以 4 邻域、
#     等速、同时向外扩散；先到达的种子占领该格，形成曼哈顿 Voronoi 式分区。
#   concrete_trace:
#   - 例：一个 7x7 房间，左上角有红种子，右下角有绿种子。
#   - 原题：红/绿只是调色板中的两个槽位，整块区域仍按边界深度涂成矩形环带。
#   - 变体：每个格子比较它到红种子、绿种子的曼哈顿距离，离谁近就染谁；
#     中间出现一条楼梯状分界线，而不是围绕房间边界的矩形环。
#   complexity_calibration:
#   - 若大量区域只有 0-1 个种子，输出会退化成纯色块，BFS 竞争不明显。
#   - 因此生成器强制至少 1 个区域放置 3 个种子，并偏向 2-3 种子区域。
#   - 种子位置从少量可见模板中选取（角点、边中点、中心），避免难以肉眼归纳的随意散点。
#   implicit_assumption_audit:
#   - 原 main() 中 `_boundary_distance(comp_mask)` 假设“路径机制”由区域边界定义；
#     该前提在变体中失效，因此整段距离计算只保留给生成器做反退化对照，不再参与求解。
#   - 原 `dist_to_color[d]` 假设“每个有意义的颜色只需要记录在某个深度槽位里一次”；
#     变体中每个标记都是独立种子，即使深度相同也不能合并。
#   - 原 `palette[d % m]` 会无条件覆盖整块区域，但这是因为原题所有格子的目标颜色
#     只取决于边界深度。变体里目标颜色改为“最先到达的种子颜色”，必须逐步扩散赋值。
#   placement_legality:
#   - 区域仍由递归矩形切分得到，因此每个 bbox 就是真实房间，可直接在 bbox 的模板位置放种子。
#   - 同一区域内模板点互不重合；不同区域被墙隔开，不会发生越界或重叠。
#   visibility_and_tidiness:
#   - 只使用少量模板位置，避免杂乱。
#   - tie-break 采用可见的“种子阅读顺序（从上到下、从左到右）”；但生成器主要用不对称模板，
#     让视觉上更多依赖明显的距离竞争，而不是大量等距平局。
#   color_inference:
#   - 颜色本身不参与排序；颜色只作为扩散源身份出现。
#   - 求解者只需看出“哪个彩色点离某片区域更近”，不需要记忆颜色 id 或隐式编码。


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Legacy helper kept only for generator-side anti-degeneration checks.
    Compute 8-neigh (Chebyshev) distance to boundary for True-cells in component_mask.
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
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood each wall-bounded room outward from its colored seed pixels.

    Input structure:
    - The grid is partitioned into rooms by one wall color.
    - Inside each room, most cells share one base color, with a few isolated marker pixels of other colors.

    Rule:
    - Treat every non-base pixel in a room as a seed and launch a simultaneous 4-neighbor BFS from all seeds.
    - Each wave expands one step at a time; the first seed to reach a cell claims it, with reading-order seed priority on exact same-time ties.
    - Rooms with no seeds stay in their base color, and wall cells are unchanged.

    Key insight: The marker pixels are no longer entries in a boundary-depth palette; they are competing sources whose Manhattan-distance wavefronts carve the room into Voronoi-like territories.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts_arr in comps:
        pts = sorted((int(r), int(c)) for r, c in pts_arr)
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        seeds = [(r, c, int(g[r, c])) for r, c in pts if int(g[r, c]) != base]

        if not seeds:
            for r, c in pts:
                out[r, c] = int(base)
            continue

        owner = np.full((h, w), -1, dtype=int)
        seed_colors = [col for _r, _c, col in seeds]
        q: deque[tuple[int, int]] = deque()

        for idx, (r, c, col) in enumerate(seeds):
            owner[r, c] = idx
            out[r, c] = int(col)
            q.append((r, c))

        while q:
            r, c = q.popleft()
            idx = int(owner[r, c])
            col = int(seed_colors[idx])
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc] and owner[rr, cc] == -1:
                    owner[rr, cc] = idx
                    out[rr, cc] = col
                    q.append((rr, cc))

    return out


# --------------- generator ---------------

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
MARKER_COUNT_WEIGHTS = [1, 2, 4, 5]   # 0/1/2/3 seeds; still allows empty rooms, but favors competition

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 300

ANCHOR_PATTERNS = {
    1: [
        ("tl",),
        ("tr",),
        ("bl",),
        ("br",),
        ("c",),
    ],
    2: [
        ("tl", "br"),
        ("tr", "bl"),
        ("ml", "mr"),
        ("tm", "bm"),
        ("tl", "mr"),
        ("tr", "ml"),
    ],
    3: [
        ("tl", "tr", "br"),
        ("tl", "bl", "br"),
        ("tl", "tr", "bl"),
        ("tr", "bl", "br"),
        ("tl", "c", "br"),
        ("tr", "c", "bl"),
    ],
}


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
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _anchor_pos(rect, name: str) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    if name == "tl":
        return r0, c0
    if name == "tr":
        return r0, c1
    if name == "bl":
        return r1, c0
    if name == "br":
        return r1, c1
    if name == "tm":
        return r0, cm
    if name == "bm":
        return r1, cm
    if name == "ml":
        return rm, c0
    if name == "mr":
        return rm, c1
    if name == "c":
        return rm, cm
    raise ValueError(name)


def _valid_pattern(rect, pattern: tuple[str, ...]) -> bool:
    pts = [_anchor_pos(rect, name) for name in pattern]
    if len(set(pts)) != len(pts):
        return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) < 2:
                return False
    return True


def _choose_markers(rect, marker_pool: list[int], rng: random.Random, forced_n: int | None = None):
    n = forced_n
    if n is None:
        n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(int(n), MARKERS_PER_REGION_MAX)
    if n == 0:
        return []

    candidates = [p for p in ANCHOR_PATTERNS[n] if _valid_pattern(rect, p)]
    if not candidates:
        return []

    pattern = rng.choice(candidates)
    colors = rng.sample(marker_pool, n)
    markers = []
    for color, name in zip(colors, pattern):
        rr, cc = _anchor_pos(rect, name)
        markers.append((int(color), int(rr), int(cc)))
    return markers


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    """
    Place markers into grid (in place).
    Force one region to have 3 competing seeds so the BFS behavior is visible.
    Returns (used_colors, marker_counts) or (None, None) on failure.
    """
    if not rects:
        return None, None

    used = set()
    counts = []
    focus_idx = rng.randrange(len(rects))

    for idx, rect in enumerate(rects):
        forced_n = 3 if idx == focus_idx else None
        markers = _choose_markers(rect, marker_pool, rng, forced_n=forced_n)
        if idx == focus_idx and len(markers) < 3:
            return None, None
        counts.append(len(markers))
        for col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(int(col))
    return used, counts


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original concentric-frame behavior, used only to reject degenerate samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for pts_arr in _cc4(free):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
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

        m = len(palette)
        for r, c in pts:
            out[r, c] = int(palette[int(dist[r, c]) % m])

    return out


def _competitive_region_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    count = 0
    for pts_arr in _cc4(g != wall):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        vals = [int(g[r, c]) for r, c in pts]
        base = _mode(vals)
        seeds = sum(int(g[r, c]) != base for r, c in pts)
        if seeds >= 2:
            count += 1
    return count


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, marker_counts = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None
    if max(marker_counts, default=0) < 2:
        return None
    if _competitive_region_count(grid) == 0:
        return None

    try:
        out = main(grid)
        legacy_out = _legacy_main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, legacy_out):
        return None

    diff_from_legacy = int(np.count_nonzero(out != legacy_out))
    if diff_from_legacy < max(6, min(H, W) // 2):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

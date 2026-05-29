# Combo variant of 13e47133 (index 12)
# combo_spec: D1/2+D3/3+D10/2
# Name: BFS扩散+边缘控制像素+到标记距离

import itertools
import random
from collections import deque

import numpy as np

from arc_utils import components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#   - 原题: 每个墙分隔区域内部有 0-3 个彩色标记。先按“到区域边界的切比雪夫深度”
#     把这些标记排成 palette，再把整块区域按 boundary-distance 循环上色，所以输出
#     是围绕房间边界向内收缩的矩形环带。
#   - 融合后: 彩色标记不再放在区域内部，也不再编码 palette 槽位；它们被放到该区域所接触
#     的整张画布外边缘上，并直接充当 BFS 种子。距离从标记本身开始算，多个种子同时以
#     4 邻域向区域内部扩散，谁先到谁占领该格，平局按种子阅读顺序打破。
#   - 小例子: 顶边房间的顶边上有三个控制像素 A/B/C。原题会把它们误读成边界深度 0 的
#     线索，最多只留下一个颜色槽；融合后则把 A/B/C 当作三个独立源，房间内部出现从
#     顶边往下扩散的三块竞争区域，分界线由到各种子的曼哈顿距离决定。
#   - 差异的关键不是“边缘上也能放点颜色”，而是“这些颜色点从编码 palette 角色变成了
#     真正的扩散源”，因此 D3/3 与 D1/2、D10/2 必须联动实现。
#
# b) complexity_calibration:
#   - 若大多数区域只有 0-1 个种子，输出会退化成纯色覆盖，看不出多源 BFS 的竞争。
#   - 生成器因此强制至少一个边缘区域拥有 3 个种子，并尽量再让另一个边缘区域拥有 2 个
#     种子；其余区域仍允许 0-3 个种子，保证“有竞争”和“有无变化对照”同时出现。
#   - 种子位置只从少量边缘锚点模板中选择（边角与边中点），避免随意散点导致人眼难以
#     归纳，又让“控制像素来自外边缘”在样例中反复出现。
#
# c) implicit_assumption_audit_against_original_main:
#   - `dist = _boundary_distance(comp_mask)` 依赖“每格颜色只由到边界的深度决定”；融合后
#     失效，因为距离场的源头已经从边界改成标记像素。
#   - `dist_to_color[d]` 依赖“同一深度只需保留一个颜色槽”；融合后失效，因为每个控制像素
#     都是独立种子，即使它们同在边缘、同在深度 0，也必须同时保留并竞争空间。
#   - `palette[d % m]` 依赖“整块区域共享一个一维周期场”；融合后失效，因为目标颜色先取决于
#     最近种子是谁，再取决于 BFS 到达顺序，必须显式做多源扩散。
#   - D3 单变体里“边缘控制像素的偏移量编码 palette 顺序”的前提在本融合中也不再需要。
#     一旦 D1 改成 BFS、D10 改成从标记出发，边缘控制像素携带的信息就是它自己的位置，
#     而不是某个隐式 offset 编码。
#
# d) placement_legality:
#   - 仍用递归矩形切墙，因此每个 free component 都是矩形房间，边缘锚点坐标可直接从 bbox
#     推出，不会跨墙落到别的区域。
#   - 只允许在真正位于整张画布外边界上的房间锚点放控制像素；内部房间没有外边缘，因此
#     自动进入“无种子、保持底色”的分支。
#   - 同一区域内锚点去重并要求最小曼哈顿间隔，避免重叠或几乎重合；不同区域被墙隔开，
#     因而不会写到同一格。
#
# e) visual_observability:
#   - 输入里所有非墙、非底色线索都贴在画布外缘，人眼可以先观察“这些彩点属于哪块房间”，
#     再看输出中同色波前如何向房间内部推进。
#   - 输出不再是边界同心框，而是从边缘种子长出来的 Voronoi 式色块；与原题视觉差异显著。
#   - 任何排序或 tie-break 都只依赖肉眼可见的位置阅读顺序，不依赖颜色编号等不可视因素。
#
# f) cleanliness:
#   - 每个区域最多 3 个种子，且只落在边角/边中等稀疏位置，输入仍然整洁。
#   - 墙、底色、控制像素三类对象角色分明；输出中的竞争分区也按房间独立展开，不会互相覆盖。
#
# g) inferability_and_new_solver_assumptions:
#   - 新 solver 假设“组件里的有效线索是该组件落在整张画布边界上的非底色格”；生成器严格只
#     这样放置控制像素，并且不注入任何内部彩色噪声。
#   - `base = _mode(vals)` 假设底色在单个区域内仍占多数；生成器将每区种子数限制在 0-3。
#   - `bg = _mode(g)` 与 `wall = second_mode` 假设全局底色、墙色仍保持前两大频数；生成器
#     在总着色量上做限制，并在每次尝试后显式验证这两个模式关系。


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

ANCHOR_PATTERNS = {
    1: [
        ("tm",),
        ("bm",),
        ("ml",),
        ("mr",),
        ("tl",),
        ("tr",),
        ("bl",),
        ("br",),
    ],
    2: [
        ("tl", "tr"),
        ("bl", "br"),
        ("tl", "bl"),
        ("tr", "br"),
        ("tm", "bm"),
        ("ml", "mr"),
        ("tl", "mr"),
        ("tr", "ml"),
        ("tm", "ml"),
        ("tm", "mr"),
        ("bm", "ml"),
        ("bm", "mr"),
    ],
    3: [
        ("tl", "tm", "tr"),
        ("bl", "bm", "br"),
        ("tl", "ml", "bl"),
        ("tr", "mr", "br"),
        ("tl", "tm", "mr"),
        ("tr", "tm", "ml"),
        ("bl", "bm", "mr"),
        ("br", "bm", "ml"),
        ("tl", "mr", "br"),
        ("tr", "ml", "bl"),
    ],
}


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Legacy helper kept only for generator-side anti-degeneration checks."""
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


def _available_border_anchors(rect, H: int, W: int) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = rect
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    raw = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
        "tm": (r0, cm),
        "bm": (r1, cm),
        "ml": (rm, c0),
        "mr": (rm, c1),
    }
    anchors = {}
    for name, (r, c) in raw.items():
        if r in (0, H - 1) or c in (0, W - 1):
            anchors[name] = (int(r), int(c))
    return anchors


def _pattern_is_valid(anchors: dict[str, tuple[int, int]], pattern: tuple[str, ...]) -> bool:
    if any(name not in anchors for name in pattern):
        return False
    pts = [anchors[name] for name in pattern]
    if len(set(pts)) != len(pts):
        return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) < 2:
                return False
    return True


def _fallback_patterns(anchors: dict[str, tuple[int, int]], n: int) -> list[tuple[str, ...]]:
    names = sorted(anchors)
    patterns = []
    for combo in itertools.combinations(names, n):
        if _pattern_is_valid(anchors, combo):
            patterns.append(combo)
    return patterns


def _candidate_patterns(rect, H: int, W: int, n: int) -> list[tuple[str, ...]]:
    anchors = _available_border_anchors(rect, H, W)
    preferred = [p for p in ANCHOR_PATTERNS[n] if _pattern_is_valid(anchors, p)]
    if preferred:
        return preferred
    return _fallback_patterns(anchors, n)


def _border_seeds_for_component(g: np.ndarray, pts: list[tuple[int, int]], base: int):
    h, w = g.shape
    seeds = []
    for r, c in pts:
        if r not in (0, h - 1) and c not in (0, w - 1):
            continue
        color = int(g[r, c])
        if color != int(base):
            seeds.append((int(r), int(c), color))
    seeds.sort(key=lambda item: (item[0], item[1]))
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood each wall-bounded region inward from colored control pixels placed on the outer canvas edge.

    Input structure:
    - One wall color partitions the grid into 4-connected free regions.
    - Most cells in each region share a dominant base color.
    - Any non-base clues for a region appear on the global grid border cells that belong to that region, not inside the room.

    Rule:
    - For each region, collect its border control pixels and treat them as simultaneous 4-neighbor BFS seeds.
    - Expansion distance is measured from the seed pixels themselves; when multiple waves reach a cell at the same time, the earlier seed in reading order wins.
    - Regions with no border controls remain in the base color, and wall cells stay unchanged.

    Key insight: The edge clues no longer encode boundary-depth palette slots. They are literal spatial origins, so solving requires attaching each border clue to its room and then following multi-source BFS competition from those marker positions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    for pts_arr in _cc4(g != wall):
        pts = sorted((int(r), int(c)) for r, c in pts_arr)
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        seeds = _border_seeds_for_component(g, pts, base)

        if not seeds:
            for r, c in pts:
                out[r, c] = int(base)
            continue

        owner = np.full((h, w), -1, dtype=int)
        seed_colors = [color for _r, _c, color in seeds]
        q: deque[tuple[int, int]] = deque()

        for idx, (r, c, color) in enumerate(seeds):
            owner[r, c] = idx
            out[r, c] = int(color)
            q.append((r, c))

        while q:
            r, c = q.popleft()
            idx = int(owner[r, c])
            color = int(seed_colors[idx])
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc] and owner[rr, cc] == -1:
                    owner[rr, cc] = idx
                    out[rr, cc] = color
                    q.append((rr, cc))

    return out


# ---- Canvas bounds ----
H_MIN, H_MAX = 12, 30
W_MIN, W_MAX = 15, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 4
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Controls per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [2, 2, 4, 5]  # 0/1/2/3

# ---- Distinct control colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 500


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


def _choose_border_markers(
    rect, H: int, W: int, marker_pool: list[int], rng: random.Random, forced_n: int | None = None
):
    n = forced_n
    if n is None:
        n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(int(n), MARKERS_PER_REGION_MAX)
    if n == 0:
        return []

    candidates = _candidate_patterns(rect, H, W, n)
    if not candidates:
        return []

    anchors = _available_border_anchors(rect, H, W)
    pattern = rng.choice(candidates)
    colors = rng.sample(marker_pool, n)
    markers = []
    for color, name in zip(colors, pattern):
        rr, cc = anchors[name]
        markers.append((int(color), int(rr), int(cc)))
    return markers


def _stamp_controls(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    H, W = grid.shape
    edge_rects = [i for i, rect in enumerate(rects) if _candidate_patterns(rect, H, W, 1)]
    focus3 = [i for i in edge_rects if _candidate_patterns(rects[i], H, W, 3)]
    focus2 = [i for i in edge_rects if _candidate_patterns(rects[i], H, W, 2)]

    if not focus3:
        return None

    focus_idx = rng.choice(focus3)
    support_candidates = [i for i in focus2 if i != focus_idx]
    support_idx = rng.choice(support_candidates) if support_candidates else None

    used = set()
    region_seed_counts = []
    seeded_regions = 0
    competitive_regions = 0

    for idx, rect in enumerate(rects):
        forced_n = None
        if idx == focus_idx:
            forced_n = 3
        elif idx == support_idx:
            forced_n = 2

        markers = _choose_border_markers(rect, H, W, marker_pool, rng, forced_n=forced_n)
        if forced_n is not None and len(markers) < forced_n:
            return None

        region_seed_counts.append(len(markers))
        if markers:
            seeded_regions += 1
        if len(markers) >= 2:
            competitive_regions += 1

        for color, rr, cc in markers:
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = int(color)
            used.add(int(color))

    return {
        "used_colors": used,
        "region_seed_counts": region_seed_counts,
        "seeded_regions": seeded_regions,
        "competitive_regions": competitive_regions,
    }


def _validate_color_order(grid, bg: int, wall: int):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _legacy_boundary_main(grid: np.ndarray) -> np.ndarray:
    """Original solver logic kept only to reject generator degeneracy."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    for pts_arr in _cc4(g != wall):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            color = int(g[r, c])
            if color == base:
                continue
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = color

        if dist_to_color:
            ds = sorted(dist_to_color)
            palette = []
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


def _edge_seed_stats(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    h, w = g.shape

    total_seeds = 0
    border_seeds = 0
    competitive_regions = 0
    seeded_regions = 0

    for pts_arr in _cc4(g != wall):
        pts = sorted((int(r), int(c)) for r, c in pts_arr)
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        base = _mode(g[comp_mask])

        region_seed_count = 0
        for r, c in pts:
            color = int(g[r, c])
            if color == int(base):
                continue
            total_seeds += 1
            if r in (0, h - 1) or c in (0, w - 1):
                border_seeds += 1
            region_seed_count += 1

        if region_seed_count > 0:
            seeded_regions += 1
        if region_seed_count >= 2:
            competitive_regions += 1

    return {
        "total_seeds": total_seeds,
        "border_seeds": border_seeds,
        "seeded_regions": seeded_regions,
        "competitive_regions": competitive_regions,
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

    meta = _stamp_controls(grid, rects, bg, marker_pool, rng)
    if meta is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(meta["used_colors"]) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if meta["seeded_regions"] < 2:
        return None
    if meta["competitive_regions"] == 0:
        return None
    if max(meta["region_seed_counts"], default=0) < 3:
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    seed_stats = _edge_seed_stats(grid)
    if seed_stats["total_seeds"] == 0:
        return None
    if seed_stats["border_seeds"] != seed_stats["total_seeds"]:
        return None
    if seed_stats["competitive_regions"] == 0:
        return None

    try:
        out = main(grid)
        legacy_out = _legacy_boundary_main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, legacy_out):
        return None

    diff = int(np.count_nonzero(out != legacy_out))
    if diff < max(10, (H * W) // 18):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

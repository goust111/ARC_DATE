# Combo variant of 13e47133 (index 20)
# combo_spec: D13/2+D5/0
# Name: 对称配对标记+标记保留
#
# Design analysis
# a) 具体化与行为追踪:
#    - 原题里每个区域在某个角的对角线上放 0..3 个单像素 marker；solver 只按“距边界深度”读取这些颜色，
#      然后把整个区域都刷成按深度循环的同心矩形框，所以 marker 自己通常会被覆盖掉。
#    - 本融合版把“每个深度一个 marker”改成“每个深度一对 marker”：若区域 bbox 为 (r0,c0)-(r1,c1)，
#      选定一条角对角线后，在深度 d 放置 p=(corner+d,d) 以及关于区域中心的反点 p'=(r0+r1-p.r, c0+c1-p.c)。
#      这两个点颜色相同、深度相同，因此它们共同代表同一个调色板槽位。
#    - 同时启用 D5/0：输出中不覆盖 marker。例子：某 7x7 区域在深度 [0,2] 放颜色 [A,B] 的两对对称点。
#      调色板仍是 [A,B]；旧题会把深度 2 的 B-pair 刷成 A，因为 2 % 2 = 0；新题则保留这两个 B 点，
#      所以 A 色框上会出现一对中心对称的断口。差异不是“多了两个点”而是“成对点既定义 palette 又保留下来”。
# b) 复杂度校准:
#    - 对称关系只用一种：180 度中心对称；每区仍只有 0..3 个深度槽位，避免自由度过多。
#    - 为了让 D5/0 真的可见，深度允许跳档而非总是连续，例如 [0,2]、[1,3]；这样旧版覆盖输出和新版保留输出
#      会在至少一对 marker 上分叉，但整体仍保持“少量锚点 + 同心框”的简洁视觉。
#    - 对称 pair 重复同一颜色，解题者不会误把两个对称点当成两个独立 palette 槽；跨样例会反复看到“同深度双点”模式。
# c) 对原 main() 的隐式假设审查:
#    - 原代码默认“每个深度至多一个 marker”。融合后这个前提失效，因为同一深度现在有两个点；因此实现改成
#      收集 `depth -> [colors...]`，再把同深度的重复颜色折叠成一个 palette 项。
#    - 原代码最后无条件覆盖整个区域，这与 D5/0 冲突；现在显式构造 `marker_mask`，只给非 marker 位置上色。
#    - `base` 仍依赖区域众数，故生成器继续保证 marker 稀疏、区域至少 3x3，且墙色仍为全局次众数。
#    - 由于 solver 不再依赖“读取顺序打破同深度冲突”，生成器必须保证同一深度的两个 marker 颜色一致。
# d) 放置合法性:
#    - 墙仍由整行/整列递归切分得到，因此自由区域是互不重叠的矩形，bbox 就是区域本体。
#    - 每个 pair 都先算单侧点，再取中心反点；生成器检查两点都在该矩形内、两点不同、且不覆盖墙或其它 marker。
#    - 奇数边长正方形的最深中心格会与自身重合，因此该深度不会被选为 pair 深度。
# e) 视觉可观察性:
#    - 输入里同色双点总以中心对称方式成对出现，输出里对应位置也总是成对保留；这是肉眼可见的结构线索。
#    - 生成器拒绝“新旧 solver 输出相同”的样本，至少保留一对真正打断框纹的 marker。
#    - 排序只依赖可见的空间深度，不依赖颜色编号或隐藏元数据。
# f) 可视化整洁度:
#    - 区域仍是大矩形、marker 仍是稀疏单像素、每区最多 3 对，因此即便加入对称点也不会把局部挤满。
#    - 双点总落在同一条 inward 对角线及其中心反点上，视觉上规则统一，不会出现杂乱散点。
# g) 变化的可推断性:
#    - 颜色变化仍由“深度排序后的 palette”决定；新增信息只是“两个对称点其实是同一个深度样本”以及
#      “这些样本在输出中保留”。两条线索都伴随明确的位置结构，因此 2-4 个样例足以归纳。
# 9) 新增假设审查:
#    - solver 新增假设：同深度多点只是重复采样，不应扩充 palette 长度；生成器用“每深度恰好一对且同色”保证它。
#    - solver 还假设要按坐标保留 marker，而不是按颜色保留；生成器保持 marker 稀疏，避免同色框纹被误当作 marker。

from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

THIS_FILE = Path(__file__).resolve()
ROOT = None
for parent in THIS_FILE.parents:
    if (parent / "solvers" / "arc_utils.py").exists():
        ROOT = parent
        break
if ROOT is None:
    raise RuntimeError("Could not locate project root containing solvers/arc_utils.py")
sys.path.insert(0, str(ROOT / "solvers"))

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


def _palette_from_markers(base: int, dist_to_color: dict[int, int]) -> list[int]:
    if not dist_to_color:
        return [int(base)]
    ds = sorted(dist_to_color)
    palette: list[int] = []
    if ds[0] > 0:
        palette.append(int(base))
    for d in ds:
        palette.append(int(dist_to_color[d]))
    return palette


def _component_mask(shape: tuple[int, int], pts) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[int(r), int(c)] = True
    return mask


def _collect_region_data(g: np.ndarray, pts):
    comp_mask = _component_mask(g.shape, pts)
    base = _mode(g[comp_mask])
    dist = _boundary_distance(comp_mask)
    marker_mask = np.zeros(g.shape, dtype=bool)
    dist_to_colors: dict[int, list[int]] = {}

    for r, c in pts:
        r = int(r)
        c = int(c)
        v = int(g[r, c])
        if v == base:
            continue
        d = int(dist[r, c])
        if d < 0:
            continue
        marker_mask[r, c] = True
        dist_to_colors.setdefault(d, []).append(v)

    dist_to_color = {int(d): _mode(cols) for d, cols in dist_to_colors.items()}
    palette = _palette_from_markers(base, dist_to_color)
    return comp_mask, base, dist, marker_mask, dist_to_color, palette


def _transform_grid(grid: np.ndarray, *, preserve_markers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for pts in _cc4(free):
        _comp_mask, _base, dist, marker_mask, _dist_to_color, palette = _collect_region_data(g, pts)
        m = len(palette)
        for r, c in pts:
            r = int(r)
            c = int(c)
            if preserve_markers and marker_mask[r, c]:
                continue
            d = int(dist[r, c])
            out[r, c] = int(palette[d % m])

    return out


def _reflected_point(rect, r: int, c: int) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return int(r0 + r1 - r), int(c0 + c1 - c)


def _component_rect(pts) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _c in pts]
    cols = [int(c) for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _visible_pair_break_count(depths: tuple[int, ...] | list[int]) -> int:
    ds = sorted(int(d) for d in depths)
    if not ds:
        return 0
    offset = 1 if ds[0] > 0 else 0
    m = len(ds) + offset
    breaks = 0
    for idx, d in enumerate(ds):
        palette_index = idx + offset
        if d % m != palette_index:
            breaks += 1
    return breaks


def _pair_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    old_out = _transform_grid(g, preserve_markers=False)
    new_out = _transform_grid(g, preserve_markers=True)

    total_pairs = 0
    visible_pairs = 0
    diff_cells = int(np.count_nonzero(old_out != new_out))

    for pts in _cc4(g != wall):
        rect = _component_rect(pts)
        _comp_mask, base, dist, _marker_mask, _dist_to_color, _palette = _collect_region_data(g, pts)
        markers_by_depth: dict[int, list[tuple[int, int, int]]] = {}
        for r, c in pts:
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d >= 0:
                markers_by_depth.setdefault(d, []).append((r, c, v))

        for d, items in markers_by_depth.items():
            if len(items) != 2:
                continue
            colors = {int(v) for _r, _c, v in items}
            if len(colors) != 1:
                continue
            pos = {(int(r), int(c)) for r, c, _v in items}
            reflected = {_reflected_point(rect, r, c) for r, c, _v in items}
            if pos != reflected:
                continue
            total_pairs += 1
            r, c, _v = items[0]
            if int(old_out[r, c]) != int(new_out[r, c]):
                visible_pairs += 1

    return {
        "total_pairs": int(total_pairs),
        "visible_pairs": int(visible_pairs),
        "diff_cells": int(diff_cells),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with cyclic Chebyshev bands derived from symmetric marker pairs.

    Input structure:
    - A wall color splits the canvas into 4-connected non-wall regions.
    - Each region has one dominant base color plus 0..3 marker colors; every marker color appears twice at equal boundary depth, in positions that are 180-degree symmetric about that region's center.

    Rule:
    - Within each region, collapse each same-depth symmetric pair into one palette entry and sort those entries by Chebyshev distance from the region boundary; if the shallowest pair is not on the boundary, prepend the base color.
    - Paint only the non-marker cells with palette[depth mod palette_length], producing concentric rectangular bands.
    - Keep both original marker pixels unchanged, so every preserved pair creates matching breaks in the band where those anchors sit.

    Key insight: The doubled markers are not two separate palette samples; they are one depth-coded color shown twice by symmetry. After deduplicating by depth, the region fills exactly like the original frame task except that the paired anchors remain visible.
    """
    return _transform_grid(grid, preserve_markers=True)


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Marker pairs per region ----
MARKER_PAIRS_PER_REGION_MAX = 3
MARKER_PAIR_COUNT_WEIGHTS = [1, 2, 4, 4]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 8

# ---- Variant visibility ----
VISIBLE_PAIR_BREAKS_MIN = 1

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 300

# Corner -> (row_from, col_from, dr, dc)
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


def _single_corner_point(rect, corner, d: int) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    row_from, col_from, dr, dc = corner
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    return int(br + dr * d), int(bc + dc * d)


def _pair_positions(rect, corner, d: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = rect
    r, c = _single_corner_point(rect, corner, int(d))
    rr, cc = _reflected_point(rect, r, c)
    if not (r0 <= r <= r1 and c0 <= c <= c1 and r0 <= rr <= r1 and c0 <= cc <= c1):
        return None
    if (r, c) == (rr, cc):
        return None
    return (int(r), int(c)), (int(rr), int(cc))


def _valid_pair_depths(rect, corner) -> list[int]:
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    valid = []
    for d in range(max_d + 1):
        if _pair_positions(rect, corner, d) is not None:
            valid.append(int(d))
    return valid


def _choose_pair_depths(valid_depths: list[int], rng: random.Random) -> list[int]:
    upper = min(MARKER_PAIRS_PER_REGION_MAX, len(valid_depths))
    if upper <= 0:
        return []

    choices = list(range(upper + 1))
    weights = MARKER_PAIR_COUNT_WEIGHTS[: upper + 1]
    n = rng.choices(choices, weights=weights)[0]
    if n == 0:
        return []

    candidates = list(combinations(valid_depths, n))
    breakful = [cand for cand in candidates if _visible_pair_break_count(cand) > 0]
    if breakful and rng.random() < 0.85:
        picked = rng.choice(breakful)
    else:
        picked = rng.choice(candidates)
    return [int(d) for d in picked]


def _place_marker_pairs(rect, marker_pool: list[int], rng: random.Random):
    corner = rng.choice(_CORNERS)
    valid_depths = _valid_pair_depths(rect, corner)
    depths = _choose_pair_depths(valid_depths, rng)
    if not depths:
        return []

    colors = rng.sample(marker_pool, len(depths))
    pairs = []
    for d, col in zip(depths, colors):
        pair = _pair_positions(rect, corner, int(d))
        if pair is None:
            return []
        (r1, c1), (r2, c2) = pair
        pairs.append((int(d), int(col), int(r1), int(c1), int(r2), int(c2)))
    return pairs


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_marker_pairs(grid, rects, bg, marker_pool, rng):
    used = set()
    for rect in rects:
        for _d, col, r1, c1, r2, c2 in _place_marker_pairs(rect, marker_pool, rng):
            if not (rect[0] <= r1 <= rect[2] and rect[1] <= c1 <= rect[3]):
                return None
            if not (rect[0] <= r2 <= rect[2] and rect[1] <= c2 <= rect[3]):
                return None
            if (r1, c1) == (r2, c2):
                return None
            if grid[r1, c1] != bg or grid[r2, c2] != bg:
                return None
            grid[r1, c1] = int(col)
            grid[r2, c2] = int(col)
            used.add(int(col))
    return used


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_marker_pairs(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    metrics = _pair_metrics(grid)
    if metrics["total_pairs"] < DISTINCT_MARKER_COLORS_MIN:
        return None
    if metrics["visible_pairs"] < VISIBLE_PAIR_BREAKS_MIN:
        return None
    if metrics["diff_cells"] < 2 * VISIBLE_PAIR_BREAKS_MIN:
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

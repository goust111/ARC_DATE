# Variant of 13e47133
# Dimension: D5 — 变换：标记像素在输出中的处理 (transform)
# Change: 标记像素被同心框架的调色板颜色覆盖，不在输出中单独保留... → 保留标记像素不被覆盖
#
# Design analysis
# a) 具体化与行为追踪:
#    - 仍然是被墙分开的多个矩形区域；每个区域大部分是底色，少量单像素 marker 沿某个角的对角线放置。
#    - 新规则只改最后一步：区域内部仍按“到边界的切比雪夫距离”循环着色，但 marker 自己不被刷掉，
#      因此同一条同心框在 marker 位置会断开，marker 像一个被保留下来的锚点。
#    - 小例子：某区域 marker 深度为 [0, 2]，颜色为 [A, B]。原逻辑 palette=[A,B]，深度 2 的位置会被刷成
#      palette[2 % 2] = A，所以 B 会消失；新逻辑在该位置跳过覆盖，保留 B，于是 A 色框上出现一个 B 断点。
# b) 复杂度校准:
#    - 墙、区域分割、调色板构造都保持原题结构，只把“最终覆盖 marker”改成“保护 marker 坐标”。
#    - 为了让 D5 肉眼可见，生成器允许 marker 深度出现跳跃，而不是总是连续深度；这样原题覆盖版与本变体版
#      在同一输入上会出现真实差异，但自由度仍限制在 0..3 个 marker、单角对角线、每个深度至多一个 marker。
# c) 对原 main() 的隐式假设审查:
#    - bg/wall 通过全局众数/次众数识别，前提是底色最多、墙其次；生成器继续保证这一点。
#    - base 通过区域众数识别，前提是 marker 稀疏且不会压过底色；每区最多 3 个 marker，区域至少 3x3。
#    - dist_to_color 依赖“每个深度最多一个 marker”；生成器显式保证。
#    - 原代码最后对每个区域格子无条件赋值，这正是 D5 下会出错的点；新代码改为只给非 marker 格子上色。
# d) 放置合法性:
#    - 区域仍由整行/整列墙递归切分得到，都是不重叠矩形。
#    - marker 只放在所属矩形内的单个底色格子上，深度唯一，不越界，不覆盖墙或其他 marker。
# e) 视觉可观察性:
#    - 生成器拒绝“所有 marker 在旧 solver 下恰好也不变”的样本，至少要有一个 marker 真正形成断点。
#    - 差异完全由空间深度与保留坐标决定，不使用颜色 id 排序或隐藏映射。
# f) 可视化整洁度:
#    - 继续使用稀疏单像素 marker、轴对齐墙和矩形区域，避免多层叠加。
# g) 颜色变化可推断性:
#    - 颜色只按“边界深度排序后的 palette”参与框纹生成；marker 自身的保留由它的位置可见地提示。
# 9) 新增假设审查:
#    - 新 solver 额外假设 marker 需要按“坐标”保护而不是按“颜色”保护，否则同色框纹会被误保留。
#    - generate_input() 用稀疏单像素 marker、每深度唯一、至少一个 visible break 来保证这些假设成立。

from __future__ import annotations

import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
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


def _marker_break_count_from_depths(depths: tuple[int, ...] | list[int]) -> int:
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


def _count_visible_breaks(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = g != wall

    total = 0
    for pts in _cc4(free):
        comp_mask = np.zeros(g.shape, dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        dist_to_color: dict[int, int] = {}
        marker_depths: list[int] = []
        for r, c in pts:
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d < 0:
                continue
            marker_depths.append(d)
            if d not in dist_to_color:
                dist_to_color[d] = v

        if not dist_to_color:
            continue

        palette = _palette_from_markers(base, dist_to_color)
        m = len(palette)
        for d in marker_depths:
            expected_old = int(palette[d % m])
            marker_color = int(dist_to_color[d])
            if expected_old != marker_color:
                total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with cyclic Chebyshev bands while preserving the marker pixels.

    Input structure:
    - A wall color splits the canvas into 4-connected non-wall regions.
    - Each region has one dominant base color plus a few isolated marker pixels of other colors placed at different depths from that region's boundary.

    Rule:
    - Within each region, sort marker colors by boundary depth to form the palette, prepending the base color when the shallowest marker is not on the boundary.
    - Paint the non-marker cells by palette[depth mod palette_length] to create concentric rectangular bands.
    - Leave every original marker pixel unchanged, so the band pattern is interrupted exactly at those anchor cells.

    Key insight: The markers still define the palette by depth, but unlike the original task they are protected source pixels rather than cells to be repainted.
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
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        marker_mask = np.zeros((h, w), dtype=bool)

        dist_to_color: dict[int, int] = {}
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
            if d not in dist_to_color:
                dist_to_color[d] = v

        palette = _palette_from_markers(base, dist_to_color)
        m = len(palette)
        for r, c in pts:
            r = int(r)
            c = int(c)
            if marker_mask[r, c]:
                continue
            d = int(dist[r, c])
            out[r, c] = palette[d % m]

    return out


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
MARKER_COUNT_WEIGHTS = [1, 2, 4, 4]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 8

# ---- Variant visibility ----
VISIBLE_BREAKS_MIN = 1

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 300

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


def _choose_marker_depths(max_d: int, rng: random.Random) -> list[int]:
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return []

    choices = list(range(upper + 1))
    weights = MARKER_COUNT_WEIGHTS[: upper + 1]
    n = rng.choices(choices, weights=weights)[0]
    if n == 0:
        return []

    all_depths = list(range(max_d + 1))
    candidates = list(combinations(all_depths, n))
    breakful = [cand for cand in candidates if _marker_break_count_from_depths(cand) > 0]

    if breakful and rng.random() < 0.8:
        picked = rng.choice(breakful)
    else:
        picked = rng.choice(candidates)
    return list(map(int, picked))


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    """
    Produce 0..MARKERS_PER_REGION_MAX markers along a corner diagonal.
    Depths are strictly increasing but may skip levels so the preserved-marker
    variant creates visible breaks relative to the original overwrite behavior.
    """
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    depths = _choose_marker_depths(max_d, rng)
    if not depths:
        return []

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    colors = rng.sample(marker_pool, len(depths))

    markers = []
    for d, col in zip(depths, colors):
        rr = br + dr * int(d)
        cc = bc + dc * int(d)
        markers.append((int(d), int(col), int(rr), int(cc)))
    return markers


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    used = set()
    for rect in rects:
        for _d, col, rr, cc in _place_markers(rect, marker_pool, rng):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
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

    used = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None
    if _count_visible_breaks(grid) < VISIBLE_BREAKS_MIN:
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

# Variant of 13e47133
# Dimension: D9 — 多步处理与顺序 (multi_step)
# Change: 单步处理：独立地对每个区域执行填充，区域间无交互... → 两步处理：先填充再传播

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVERS_DIR = ROOT / "solvers"
if str(SOLVERS_DIR) not in sys.path:
    sys.path.append(str(SOLVERS_DIR))

from arc_utils import components_mask

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# Design analysis for the D9 variant
# ----------------------------------
# a) 具体化与行为追踪
#    - 输入仍然是“墙 + 多个自由区域 + 区域内少量 marker”的结构，墙厚始终为 1。
#    - 第一步完全保留原题：每个区域独立读出 marker 深度顺序，生成循环 palette，
#      再按 boundary-distance % len(palette) 画出同心矩形框。
#    - 第二步才发生跨区域交互：区域 A 的“最内层颜色”定义为第一步中最大深度层的颜色。
#      这层颜色穿过一格墙，覆盖区域 B 中紧贴这段墙的最外层边界。传播只改一层。
#    - 小例子：左右两个矩形区域被一条竖墙隔开。第一步后左侧最中心颜色若为 4，右侧最外层
#      靠墙的一列原本是 8，那么第二步把这列 8 改成 4；反向同时发生，即右侧中心色也会改写
#      左侧靠墙的一列。原题只会停在第一步，不会出现“中心色影响邻区边框”的现象。
# b) 复杂度校准
#    - 传播不是全局洪泛，只覆盖共享墙对面的外框段，因此规则足够明显但不会把整块区域涂乱。
#    - 仍然只用少量 marker 和单格墙，避免引入新的对象类型或颜色排序规则。
#    - 生成器要求传播至少改动多个区域和若干像素，避免样本退化成“看起来和原题差不多”。
# c) 隐式假设审查
#    - 原 solver 的 `free = g != wall` 仍成立，因为两步都不会改墙。
#    - 原 solver 在单次区域循环里直接写 `out[r, c] = palette[...]`；变体下这不够，因为若在同一轮
#      循环里边填边传播，会让后处理区域错误地读到被传播后的颜色，所以必须先完成整张图的第一步，
#      再做第二步。
#    - 原 solver 用输入中的非 base 像素当 marker；变体仍保持 marker 只存在于输入，因此 palette
#      仍必须从输入图 `g` 读取，而不是从第一步结果读取。
#    - “最内层颜色”依赖最大距离层，因此要求先算完整的 distance map，再决定 donor color。
# d) 放置合法性
#    - 生成器继续用整行/整列墙做矩形递归切分，保证区域 4-连通、互不重叠、墙厚为 1。
#    - marker 只放在各自矩形内部的一条角对角线上，深度连续，不越界，不覆盖墙，不彼此重叠。
#    - 第二步传播只检查“单元格 -> 墙 -> 对面区域”这类正交三连结构，因此与生成器的单格墙假设一致。
# e) 视觉可观察性
#    - 颜色的作用只来自它们在图中的空间位置：marker 的深度决定第一步 palette，区域中心色决定
#      第二步外溢。没有使用颜色 id 大小、黑色是否为 0 等不可见属性。
#    - 人类可从样例中看到：同一块区域的中心颜色，会出现在邻居靠墙的一圈边框上，且必须先有
#      第一步的同心框结构才能解释第二步的来源。
# f) 可视化整洁度
#    - 传播仅覆盖边界一层，墙本身不变，marker 也不会残留成独立噪声，因此输出仍是清晰的
#      “区域框纹 + 邻区边界染色”。
#    - 若某个边界角点同时面对两个不同邻区，传播来源会有歧义。这里显式跳过这类角点，让共享边段
#      保持整洁，也避免人为引入隐藏 tie-break。
# g) 颜色变化可推断性
#    - 第二步不是任意换色，而是“邻区中心色 -> 本区靠墙外框”，有稳定的结构线索。
#    - 生成器筛掉传播过弱的样本，保证 2-3 组样例中就能观察到至少几个明显的溢出边段。
# h) 新 solver 的额外输入假设
#    - 墙必须是一格厚，否则“隔墙看对面 donor”会失效；生成器始终满足这一点。
#    - 共享墙两侧必须真的是不同区域；矩形递归切分天然满足。
#    - 传播歧义只在墙交汇角点出现；solver 选择跳过，生成器则保证长直墙段足够多，角点不是主要信号。


def _cc4(mask: np.ndarray) -> list[np.ndarray]:
    return components_mask(mask, connectivity=4)


def _mode(arr: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _infer_bg_wall(grid: np.ndarray) -> tuple[int, int]:
    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    return bg, wall


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


def _build_palette(grid: np.ndarray, pts: np.ndarray, comp_mask: np.ndarray, dist: np.ndarray) -> list[int]:
    vals = grid[comp_mask]
    base = _mode(vals)

    dist_to_color: dict[int, int] = {}
    for r, c in pts:
        rr, cc = int(r), int(c)
        v = int(grid[rr, cc])
        if v == base:
            continue
        d = int(dist[rr, cc])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = v

    if not dist_to_color:
        return [int(base)]

    ds = sorted(dist_to_color)
    palette: list[int] = []
    if ds[0] > 0:
        palette.append(int(base))
    for d in ds:
        palette.append(int(dist_to_color[d]))
    return palette


def _first_pass(grid: np.ndarray, wall: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    h, w = grid.shape
    out = grid.copy()
    region_id = np.full((h, w), -1, dtype=int)
    inner_colors: list[int] = []

    for rid, pts in enumerate(_cc4(grid != wall)):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            rr, cc = int(r), int(c)
            comp_mask[rr, cc] = True
            region_id[rr, cc] = rid

        dist = _boundary_distance(comp_mask)
        palette = _build_palette(grid, pts, comp_mask, dist)
        max_d = int(dist[comp_mask].max())
        inner_colors.append(int(palette[max_d % len(palette)]))

        m = len(palette)
        for r, c in pts:
            rr, cc = int(r), int(c)
            out[rr, cc] = palette[int(dist[rr, cc]) % m]

    return out, region_id, inner_colors


def _second_pass(step1: np.ndarray, region_id: np.ndarray, inner_colors: list[int]) -> np.ndarray:
    h, w = step1.shape
    out = step1.copy()

    for r in range(h):
        for c in range(w):
            rid = int(region_id[r, c])
            if rid < 0:
                continue

            donor_ids: set[int] = set()
            for dr, dc in DIR4:
                wr, wc = r + dr, c + dc
                br, bc = r + 2 * dr, c + 2 * dc
                if not (0 <= wr < h and 0 <= wc < w and 0 <= br < h and 0 <= bc < w):
                    continue
                if int(region_id[wr, wc]) != -1:
                    continue
                donor = int(region_id[br, bc])
                if donor >= 0 and donor != rid:
                    donor_ids.add(donor)

            if len(donor_ids) == 1:
                donor = next(iter(donor_ids))
                out[r, c] = int(inner_colors[donor])

    return out


def _variant_outputs(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    _bg, wall = _infer_bg_wall(grid)
    step1, region_id, inner_colors = _first_pass(grid, wall)
    step2 = _second_pass(step1, region_id, inner_colors)
    return step1, step2, region_id, inner_colors


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    step1, step2, region_id, _inner_colors = _variant_outputs(grid)
    fill_changed = int(np.count_nonzero(step1 != grid))
    spill_mask = step2 != step1
    spill_cells = int(np.count_nonzero(spill_mask))
    affected_regions = {
        int(region_id[r, c])
        for r, c in np.argwhere(spill_mask)
        if int(region_id[r, c]) >= 0
    }
    spill_colors = {int(step2[r, c]) for r, c in np.argwhere(spill_mask)}
    return {
        "fill_changed": fill_changed,
        "spill_cells": spill_cells,
        "affected_regions": len(affected_regions),
        "spill_color_count": len(spill_colors),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each walled-off region with a cyclic frame pattern, then let the center color spill across walls.

    Input structure:
    - A one-cell-thick wall color partitions the grid into several non-wall regions.
    - Inside each region, the dominant base color is interrupted by a few marker pixels placed at specific Chebyshev depths.

    Rule:
    - First, solve each region exactly like the original task: sort marker colors by boundary depth, prepend the base color if depth 0 is missing, and fill the region with palette[depth mod palette_length].
    - Second, inspect the finished first-pass picture: each region's deepest layer color crosses any shared wall and overwrites the neighboring region's wall-adjacent outer layer by one cell.

    Key insight: The output cannot be produced by local per-region filling alone; you must finish the concentric pattern first, read each region's innermost color from that intermediate state, and only then propagate it to adjacent regions.
    """
    g = np.asarray(grid, dtype=int)
    _step1, out, _region_id, _inner_colors = _variant_outputs(g)
    return out


# --------------- generator ---------------

# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 4
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 3, 4, 4]
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Variant visibility filters ----
MIN_FILL_CHANGED = 8
MIN_SPILL_CELLS = 8
MIN_AFFECTED_REGIONS = 2

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 600

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _place_walls(H: int, W: int, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
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


def _place_markers(rect: tuple[int, int, int, int], marker_pool: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    start_hi = min(START_DIST_MAX, max_d - n + 1)
    start_d = rng.randint(0, max(0, start_hi))

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    colors = rng.sample(marker_pool, n)

    return [
        (start_d + k, col, br + dr * (start_d + k), bc + dc * (start_d + k))
        for k, col in enumerate(colors)
    ]


def _pick_palette(rng: random.Random) -> tuple[int, int, list[int]]:
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    bg: int,
    marker_pool: list[int],
    rng: random.Random,
):
    used: set[int] = set()
    for rect in rects:
        for _d, col, rr, cc in _place_markers(rect, marker_pool, rng):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if int(grid[rr, cc]) != bg:
                return None
            grid[rr, cc] = int(col)
            used.add(int(col))
    return used


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
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

    try:
        step1, out, _region_id, _inner_colors = _variant_outputs(grid)
    except Exception:
        return None

    metrics = _variant_metrics(grid)
    if np.array_equal(grid, out):
        return None
    if metrics["fill_changed"] < MIN_FILL_CHANGED:
        return None
    if metrics["spill_cells"] < MIN_SPILL_CELLS:
        return None
    if metrics["affected_regions"] < MIN_AFFECTED_REGIONS:
        return None
    if np.array_equal(step1, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

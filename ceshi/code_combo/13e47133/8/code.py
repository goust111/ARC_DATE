# Combo variant of 13e47133 (index 8)
# combo_spec: D5/0+D12/0
# Name: 保留标记锚点+墙壁消失

# Design analysis
# a) 具体化与行为追踪:
#    - 仍然是若干矩形区域，被单一墙色分隔；每个区域主体是 base 色，少量单像素 marker 沿某个角的对角线放置。
#    - 原题流程: 先按区域内 marker 到区域边界的切比雪夫深度排序，构出 palette，再把整个区域(含 marker 位置)全部刷成
#      palette[depth mod len(palette)]，墙保持原色不动。
#    - 融合后流程分两处改动:
#      1) 区域内部刷同心框时，marker 坐标不覆盖，保留输入中的原色，形成“框纹断点”。
#      2) 区域刷完后，再把墙像素按“离哪个区域最近”继承该区域最外层框颜色，墙整体消失。
#    - 具体小例子: 两个竖向区域之间隔着 2 列墙。左区 marker 深度为 [0,2]、颜色 [A,B]，所以 palette=[A,B]。
#      原题中深度 2 的 marker 会被重刷成 A；融合后它保留 B，于是左区 A 色外框里出现一个 B 锚点。
#      若右区外层色是 C，则中间两列墙会分别变成 [A,C]，不再保留墙色。
# b) 复杂度校准:
#    - 只保留两项核心变化，不引入额外对象类型。墙布局限制为平行的 2 像素厚条带，避免“最近区域”出现难以观察的四向竞争。
#    - marker 仍限制为每区 0..3 个、单角对角线放置，但允许深度跳跃而非总是连续，这样“保留锚点”在输出中能稳定可见。
#    - 生成器强制每个样本同时出现至少一个 marker 断点和至少一处墙带双色分裂，避免退化成任一单变体或原题。
# c) 隐式假设审查:
#    - `bg` 为全局众数、`wall` 为次众数: 生成器继续保证 base 面积最大、厚墙面积次大。
#    - `base` 为单区域众数: 每区 marker 稀疏，且区域尺寸 >= 3，因此不会压过 base。
#    - `dist_to_color` 默认每个深度最多取一个颜色: 生成器仍保证每区每个深度至多一个 marker。
#    - 原题末尾对所有区域格子无条件赋值；融合后这会破坏 D5，因此改成“非 marker 才刷色”。
#    - 原题默认墙无需后处理；融合后这不成立，因此必须在区域着色后追加一轮基于最近区域的墙填充。
#    - 最近区域判定若存在大面积平局会引入隐藏 tie-break；因此生成器只产出平行 2 厚墙带，使每个墙像素都有唯一最近侧。
# d) 放置合法性:
#    - 区域由平行条带墙分割，矩形互不重叠，墙也不与区域重叠。
#    - 每个 marker 都在所属矩形内部，覆盖前必检验该位置仍是 base 色，因此不会撞墙、越界或覆盖其他 marker。
#    - 深度从合法范围 `0..max_d` 中选取，保证 marker 坐标始终落在对应区域内。
# e) 视觉可观察性:
#    - 解题者能直接看到两类证据: 区域里某些 marker 没被框纹吞掉；原本连续的墙带在输出里消失并分裂成相邻区域的外层色。
#    - 这两类变化都完全由几何关系决定，不依赖颜色编号排序或其他人类不可见属性。
# f) 可视化整洁度:
#    - 使用平行厚墙和稀疏单像素 marker，避免 T/cross 交汇导致的墙填色混战。
#    - 区域内部只有“同心框 + 少量锚点”两层信息，墙消失后仍能清楚分辨原先分区边界的大致位置。
# g) 变化的可推断性:
#    - 颜色变化只来自两条结构化规则: marker 深度决定 palette 顺序，区域最外层色决定墙如何消失。
#    - 生成器拒绝“marker 保留不可见”或“墙两侧外层色一致”的样本，确保 2-3 组示例就能观察到融合规律。
# 9) 新增假设审查:
#    - solver 新增假设: 要按 marker 的坐标保护，而不是按颜色保护；生成器使用孤立单像素 marker 保证这一点可判定。
#    - solver 新增假设: 墙像素可由某个最近区域唯一归属；生成器用无交叉的 2 厚平行墙保证这一点。
#    - generator 通过 `visible_breaks >= 1`、`visible_wall_split == True`、`fused != D5_only`、`fused != D12_only`
#      这几道校验，确保新假设不仅成立，而且在样本中真的起作用。

from __future__ import annotations

import random
import sys
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        sys.path.insert(0, str(_parent / "solvers"))
        break

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


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


def _paint_regions(g: np.ndarray, wall: int, keep_markers: bool):
    """
    Paint all non-wall regions and return `(out, outer_color_map)`.
    `outer_color_map[r, c]` stores the frame-0 color of the region owning `(r, c)`.
    """
    h, w = g.shape
    out = g.copy()
    outer = np.full((h, w), -1, dtype=int)
    free = g != wall

    for pts in _cc4(free):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        marker_mask = np.zeros((h, w), dtype=bool)
        dist_to_color: dict[int, int] = {}

        for r, c in pts:
            r = int(r)
            c = int(c)
            value = int(g[r, c])
            if value == base:
                continue
            d = int(dist[r, c])
            if d < 0:
                continue
            marker_mask[r, c] = True
            if d not in dist_to_color:
                dist_to_color[d] = value

        palette = _palette_from_markers(base, dist_to_color)
        outer_color = int(palette[0])
        m = len(palette)

        for r, c in pts:
            r = int(r)
            c = int(c)
            outer[r, c] = outer_color
            if keep_markers and marker_mask[r, c]:
                continue
            out[r, c] = int(palette[int(dist[r, c]) % m])

    return out, outer


def _paint_walls_from_outer(out: np.ndarray, wall_mask: np.ndarray, outer: np.ndarray) -> None:
    """Recolor each wall cell with the frame-0 color of its nearest region."""
    h, w = out.shape
    if not wall_mask.any():
        return

    dist = np.full((h, w), -1, dtype=int)
    color = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if outer[r, c] == -1:
                continue
            dist[r, c] = 0
            color[r, c] = int(outer[r, c])
            q.append((r, c))

    while q:
        r, c = q.popleft()
        next_dist = dist[r, c] + 1
        col = color[r, c]
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                dist[rr, cc] = next_dist
                color[rr, cc] = col
                q.append((rr, cc))

    out[wall_mask] = color[wall_mask]


def _infer_wall(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    return bg if other.size == 0 else _mode(other)


def _original_like(grid: np.ndarray) -> np.ndarray:
    wall = _infer_wall(grid)
    out, _outer = _paint_regions(np.asarray(grid, dtype=int), wall, keep_markers=False)
    return out


def _d5_only_like(grid: np.ndarray) -> np.ndarray:
    wall = _infer_wall(grid)
    out, _outer = _paint_regions(np.asarray(grid, dtype=int), wall, keep_markers=True)
    return out


def _d12_only_like(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    wall = _infer_wall(g)
    out, outer = _paint_regions(g, wall, keep_markers=False)
    _paint_walls_from_outer(out, g == wall, outer)
    return out


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
    wall = _infer_wall(g)
    total = 0

    for pts in _cc4(g != wall):
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
            value = int(g[r, c])
            if value == base:
                continue
            d = int(dist[r, c])
            if d < 0:
                continue
            marker_depths.append(d)
            if d not in dist_to_color:
                dist_to_color[d] = value

        if not dist_to_color:
            continue

        palette = _palette_from_markers(base, dist_to_color)
        m = len(palette)
        for d in marker_depths:
            if int(palette[d % m]) != int(dist_to_color[d]):
                total += 1
    return total


def _marker_mask_by_region(grid: np.ndarray, wall: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    marker_mask = np.zeros(g.shape, dtype=bool)

    for pts in _cc4(g != wall):
        comp_mask = np.zeros(g.shape, dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True
        base = _mode(g[comp_mask])
        marker_mask |= comp_mask & (g != base)

    return marker_mask


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    wall = _infer_wall(g)
    fused = main(g)
    d5_only = _d5_only_like(g)
    d12_only = _d12_only_like(g)
    marker_mask = _marker_mask_by_region(g, wall)
    wall_mask = g == wall
    return {
        "visible_breaks": _count_visible_breaks(g),
        "walls_recolored": int(np.count_nonzero(fused[wall_mask] != wall)),
        "markers_preserved": int(np.count_nonzero(fused[marker_mask] == g[marker_mask])),
        "vs_d5_cells": int(np.count_nonzero(fused != d5_only)),
        "vs_d12_cells": int(np.count_nonzero(fused != d12_only)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-separated region with cyclic Chebyshev frames, keep the marker anchors visible, and then erase the walls with outer-frame colors.

    Input structure:
    - One wall color partitions the canvas into several 4-connected non-wall regions.
    - Inside each region, one dominant base color is punctured by a few isolated marker pixels placed at specific Chebyshev depths from that region's boundary.

    Rule:
    - For every region, sort marker colors by boundary depth to build its palette, prepending the region base color when the shallowest marker is not on depth 0.
    - Paint concentric rectangular bands by `palette[depth mod len(palette)]`, but skip the marker coordinates so those source pixels remain unchanged as visible anchors.
    - After all regions are painted, recolor each wall pixel with the outermost frame color of the nearest region, making the wall color disappear.

    Key insight: Marker depths still define the cyclic palette exactly as in the original task, but now the markers act as preserved anchor points and the vanished walls simply inherit frame-0 ownership from nearby regions.
    """
    g = np.asarray(grid, dtype=int)
    wall = _infer_wall(g)
    out, outer = _paint_regions(g, wall, keep_markers=True)
    _paint_walls_from_outer(out, g == wall, outer)
    return out


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall layout ----
WALL_THICKNESS = 2
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MIN_REGION_DIM = 3

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 4, 4]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 8

# ---- Visibility filters ----
# Require at least two visibly preserved anchors so D5 is inferable from a few examples.
VISIBLE_BREAKS_MIN = 2

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 500

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _feasible_region_counts(total_len: int) -> list[int]:
    out = []
    for n in range(N_REGIONS_MIN, N_REGIONS_MAX + 1):
        need = n * MIN_REGION_DIM + (n - 1) * WALL_THICKNESS
        if need <= total_len:
            out.append(n)
    return out


def _split_total(total: int, parts: int, min_part: int, rng: random.Random) -> list[int]:
    extra = total - parts * min_part
    cuts = sorted(rng.sample(range(extra + parts - 1), parts - 1)) if parts > 1 else []
    vals = []
    prev = -1
    for cut in cuts + [extra + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return [min_part + v for v in vals]


def _place_walls(H: int, W: int, rng: random.Random):
    """
    Partition the grid into parallel strips separated by 2-cell wall bands.

    Returns `(wall_mask, region_rects, orientation)`.
    """
    options = []
    if _feasible_region_counts(W):
        options.append("v")
    if _feasible_region_counts(H):
        options.append("h")
    if not options:
        return None

    orientation = rng.choice(options)
    total_len = W if orientation == "v" else H
    n_regions = rng.choice(_feasible_region_counts(total_len))
    lengths = _split_total(total_len - (n_regions - 1) * WALL_THICKNESS, n_regions, MIN_REGION_DIM, rng)

    wall_mask = np.zeros((H, W), dtype=int)
    rects = []
    pos = 0
    for i, length in enumerate(lengths):
        if orientation == "v":
            rects.append((0, pos, H - 1, pos + length - 1))
            pos += length
            if i < n_regions - 1:
                wall_mask[:, pos : pos + WALL_THICKNESS] = 1
                pos += WALL_THICKNESS
        else:
            rects.append((pos, 0, pos + length - 1, W - 1))
            pos += length
            if i < n_regions - 1:
                wall_mask[pos : pos + WALL_THICKNESS, :] = 1
                pos += WALL_THICKNESS

    return wall_mask, rects, orientation


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
    picked = rng.choice(breakful) if breakful and rng.random() < 0.8 else rng.choice(candidates)
    return [int(d) for d in picked]


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    """
    Produce 0..3 markers on a corner diagonal.
    Depths stay unique but may skip levels so preserved anchors become visible.
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


def _stamp_markers(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    used = set()
    region_markers = []
    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = int(col)
            used.add(int(col))
        region_markers.append(markers)
    return used, region_markers


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _outer_color_from_markers(markers, bg: int) -> int:
    if not markers:
        return int(bg)
    shallowest = min(markers, key=lambda item: (int(item[0]), int(item[2]), int(item[3])))
    return int(bg) if int(shallowest[0]) > 0 else int(shallowest[1])


def _has_visible_wall_split(region_markers, bg: int) -> bool:
    outer_colors = [_outer_color_from_markers(markers, bg) for markers in region_markers]
    return any(outer_colors[i] != outer_colors[i + 1] for i in range(len(outer_colors) - 1))


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    layout = _place_walls(H, W, rng)
    if layout is None:
        return None
    wall_mask, rects, _orientation = layout

    bg, wall, marker_pool = _pick_palette(rng)
    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, region_markers = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None
    if _count_visible_breaks(grid) < VISIBLE_BREAKS_MIN:
        return None
    if not _has_visible_wall_split(region_markers, bg):
        return None

    try:
        fused = main(grid)
        d5_only = _d5_only_like(grid)
        d12_only = _d12_only_like(grid)
        original = _original_like(grid)
    except Exception:
        return None

    if np.array_equal(grid, fused):
        return None
    if np.array_equal(fused, d5_only):
        return None
    if np.array_equal(fused, d12_only):
        return None
    if np.array_equal(fused, original):
        return None

    if np.count_nonzero(fused[wall_mask == 1] != wall) != int(wall_mask.sum()):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")

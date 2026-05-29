# Combo variant of 80a900e0 (index 24)
# combo_spec: D8/0+D15/2+D18/1
# Name: 颜色反转提取
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题输入只有棋盘格 + 菱形 + 外线段，输出补出射线；射线颜色取外线段颜色。
#    - 本融合变体把原题“完整输出图形”直接作为输入，但同时做颜色职责互换：
#      输入里菱形和外线段同色 `border_color`，真正的射线改成另一色 `ray_color`，
#      而这个 `ray_color` 对应“原本属于菱形”的颜色来源。
#    - 具体例子：若某个对象对的颜色是 `(ray_color=3, border_color=8)`，
#      则输入里会看到 8 色菱形、8 色贴边短线段，以及从短线段端点向外延伸的 3 色射线。
#      输出不再保留菱形或短线段，只留下那些 3 色射线。
#    - 因此 solver 的链路变成：
#      1. 先定位被换成 `border_color` 的菱形模板；
#      2. 再用同色相邻短线段判定每条外射方向；
#      3. 最后读取输入里已经存在的、位于该方向上的连续 `ray_color` 像素并提取出来。
#
# b) 复杂度校准
#    - 若仍允许同一菱形周围出现多种外线段颜色，则“菱形换成哪种颜色”会失去唯一性，
#      且提取任务会混入额外配对歧义；因此生成器收紧为“一个菱形只有一个 `border_color`”。
#    - 若 `ray_color == border_color`，融合会退化成人眼看不出的原题近似形式；因此生成器强制颜色对互异。
#    - 为了让 D18 的“提取而非补全”清晰可见，样本要求有足够多的可见射线像素。
#
# c) 隐式假设审查
#    - 原题的 `diamonds = components_by_color(... if _is_diamond8)` 依赖“菱形颜色独立成组件”；
#      D15 之后菱形与外线段同色且 8 连通，这个前提失效，所以这里改成整图 5x5 模板扫描。
#    - 原题 / D18 单变体可把相邻非背景像素都视作外线段；本融合里如果这样做，会把其他对象的射线误收进来。
#      新 solver 只收集“与菱形同色的相邻像素”，而生成器保证所有 `border_color` 与所有 `ray_color` 两两分离。
#    - D18 的旧提取逻辑默认“射线颜色 = 外线段颜色”；D8/0 使这一点失效。
#      这里改为：先看端点外第一格的实际颜色，把它作为该射线的连续颜色，并要求它不同于 `border_color`。
#    - 模板扫描还新增一个前提：菱形 5x5 框内除四个角外，其余非菱形位置都保持棋盘背景。
#      生成器保留原几何，因此这个前提成立。
#
# d) 放置合法性
#    - 生成器沿用几何回溯放置，要求每个菱形、短线段、完整射线的占用集合互不重叠且不越界。
#    - 候选边若因裁切只剩过短线段，或产生的可见射线太短，会直接拒绝，避免不可辨样本。
#    - 颜色分配使用互不重复的 `(ray_color, border_color)` 对，避免同色射线误贴到别的菱形上。
#
# e) 视觉可观察性
#    - 输入里同一局部同时出现两层角色：中心轮廓/短线段共享 `border_color`，外伸细线共享 `ray_color`。
#      输出则把中心全部删除，只留下 `ray_color` 射线。
#    - 解题者能从 2-4 个样例里观察到两个稳定信号：
#      1. 输出是在做“删除中心、保留外伸线”的提取；
#      2. 被保留的颜色不是短线段颜色，而是另一种与中心换位前对应的颜色。
#
# f) 可视化整洁度
#    - 背景仍是稀疏棋盘格，所有对象保持不重叠，输出只剩细对角线，结构清爽。
#    - 每个菱形只绑定一种边框色，避免中心区域出现多色碎片。
#
# g) 变化的可推断性
#    - 每个对象只涉及两种非背景色：`border_color` 和 `ray_color`，且二者的角色由空间位置固定提示。
#    - 多个对象时，不同颜色对可重复这个“中心色 / 射线色”关系，给人类足够的重复证据去归纳颜色反转提取规则。

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


_DIAMOND_COORDS = (
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
)
_DIAMOND_COORD_SET = frozenset(_DIAMOND_COORDS)
_OPTIONAL_CORNER_OFFSETS = frozenset({(0, 0), (0, 4), (4, 0), (4, 4)})
_DIAMOND_BG_ONLY_OFFSETS = frozenset(
    (r, c)
    for r in range(5)
    for c in range(5)
    if (r, c) not in _DIAMOND_COORD_SET and (r, c) not in _OPTIONAL_CORNER_OFFSETS
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _diamond_cells(r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]


def _find_diamonds(g: np.ndarray, bg_colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = g.shape
    diamonds: list[tuple[int, list[tuple[int, int]]]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            color = int(g[r0 + _DIAMOND_COORDS[0][0], c0 + _DIAMOND_COORDS[0][1]])
            if color in bg_colors:
                continue
            if any(int(g[r0 + dr, c0 + dc]) != color for dr, dc in _DIAMOND_COORDS):
                continue
            if any(int(g[r0 + dr, c0 + dc]) not in bg_colors for dr, dc in _DIAMOND_BG_ONLY_OFFSETS):
                continue
            diamonds.append((color, _diamond_cells(r0, c0)))
    return diamonds


def _outer_line_info(pts: list[tuple[int, int]]):
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _segment_outward(oc: list[tuple[int, int]], dset: set[tuple[int, int]]):
    info = _outer_line_info(oc)
    if info is None:
        return None
    slope_key, endpoints = info
    if slope_key == "rc_diff":
        normals = [(-1, 1), (1, -1)]
    else:
        normals = [(-1, -1), (1, 1)]

    best_n = None
    best_score = -1
    for n in normals:
        score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
        if score > best_score:
            best_score = score
            best_n = n
    if best_n is None or best_score <= 0:
        return None
    return endpoints, (-best_n[0], -best_n[1])


def _extract_rays_only(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    out = _checkerboard(h, w)

    for border_color, dpts in _find_diamonds(g, bg_colors):
        dset = set(dpts)
        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    if int(g[rr, cc]) == border_color:
                        adj_mask[rr, cc] = True

        for oc in _cc8(adj_mask):
            parsed = _segment_outward(oc, dset)
            if parsed is None:
                continue
            endpoints, outward = parsed
            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                ray_color = int(g[rr, cc])
                if ray_color in bg_colors or ray_color == border_color:
                    continue
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == ray_color:
                    out[rr, cc] = ray_color
                    rr += outward[0]
                    cc += outward[1]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the pre-drawn rays from swapped-color diamond figures on a checkerboard.

    Input structure:
    - A 0/1 checkerboard background with one to three hollow 5x5 diamond outlines.
    - Each diamond outline has the same color as its adjacent short outer diagonal segments.
    - The outward rays are already present, but they use the paired inner-diamond color instead of the segment color.

    Rule:
    - Detect each diamond by its fixed 5x5 outline template, even though it shares color with the outer segments.
    - Use the same-color adjacent segments to recover the outward diagonal direction on each occupied edge.
    - Starting just beyond each segment endpoint, keep only the contiguous non-background run that already exists there and discard the diamond plus outer segments.

    Key insight:
    - Because the color swap merges the diamond and its short border segments into one same-color scaffold, color-connected components no longer isolate diamonds; the reliable anchor is the geometric diamond template, and the rays are the differently colored lines extending away from that scaffold.
    """
    g = np.asarray(grid, dtype=int)
    return _extract_rays_only(g)


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
MIN_RAY_CELLS_PER_EDGE = 2
MIN_TOTAL_RAY_PIXELS = 8
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = _diamond_cells(r0, c0)
    all_cells = set(diamond)
    edge_parts = []
    total_rays = 0
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        if len(rays) < MIN_RAY_CELLS_PER_EDGE:
            return None
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells):
            return None
        if any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})
        total_rays += len(rays)
    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "ray_count": total_rays,
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        placements = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]
        placements.sort(key=lambda p: (p["ray_count"], random.random()), reverse=True)
        candidate_cache[edges] = placements

    occupied: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int, ray_total: int) -> bool:
        if i == len(order):
            return ray_total >= MIN_TOTAL_RAY_PIXELS
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            if occupied.isdisjoint(cells):
                occupied.update(cells)
                placed[spec_i] = placement
                if backtrack(i + 1, ray_total + int(placement["ray_count"])):
                    return True
                occupied.difference_update(cells)
                placed[spec_i] = None
        return False

    if not backtrack(0, 0):
        return None
    return [p for p in placed if p is not None]


def _assign_color_pairs(count: int) -> tuple[list[int], list[int]] | None:
    needed = 2 * count
    if needed > len(COLOR_RANGE):
        return None
    chosen = random.sample(list(COLOR_RANGE), needed)
    ray_colors = chosen[:count]
    border_colors = chosen[count:]
    return ray_colors, border_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND))))
        for _ in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    color_pairs = _assign_color_pairs(len(placements))
    if color_pairs is None:
        return None
    ray_colors, border_colors = color_pairs

    inp = _checkerboard(h, w)
    out = _checkerboard(h, w)
    total_ray_pixels = 0

    for pi, placement in enumerate(placements):
        border_color = border_colors[pi]
        ray_color = ray_colors[pi]
        _paint_cells(inp, placement["diamond"], border_color)
        for edge_part in placement["edges"]:
            _paint_cells(inp, edge_part["segment"], border_color)
            _paint_cells(inp, edge_part["rays"], ray_color)
            _paint_cells(out, edge_part["rays"], ray_color)
            total_ray_pixels += len(edge_part["rays"])

    if total_ray_pixels < MIN_TOTAL_RAY_PIXELS:
        return None
    if not _bg_counts_safe(inp):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")

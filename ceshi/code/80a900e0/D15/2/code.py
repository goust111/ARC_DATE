# Variant of 80a900e0
# Dimension: D15 — 菱形颜色与外边框颜色的关系 (meta)
# Change: 菱形和外边框使用不同的非背景色，同一菱形的多条外边框可以共享颜色... → 菱形颜色与外边框颜色成对交换
#
# Design analysis for the D15 variant
# ==================================
# a) 具体化与行为追踪
#    - 输入仍然是 0/1 棋盘格背景，上面放 1~3 个 8 像素空心菱形，以及贴着菱形外侧的 2~4 条短对角线段。
#    - 原题中：菱形保持自身颜色，外线段从两端向外发射射线，射线颜色=外线段颜色。
#    - 本变体中：每个菱形与其外线段颜色组成一个颜色对 `(diamond_color, border_color)`。
#      输出时发生“职责交换”：
#      1. 菱形轮廓整体改涂成 `border_color`
#      2. 所有从该菱形外线段发出的射线改用 `diamond_color`
#      3. 外线段本身保持 `border_color`
#    - 小例子：若输入里某菱形是 3，外线段是 8，则原逻辑输出会是“菱形仍为 3，射线为 8”；
#      变体输出则变为“菱形改为 8，射线为 3”。真正变化发生在输出着色阶段，几何检测与射线方向判断保持原题不变。
#
# b) 复杂度校准
#    - 若同一菱形允许多种外线段颜色，则“菱形该换成哪种颜色”会变得不明确，也不利于人类从少量样例归纳。
#    - 因此生成器收紧一个自由度：同一菱形的所有外线段统一使用一种 `border_color`。
#      这不是引入新机制，只是把原题中“可以共享颜色”的松约束收紧为“总是共享颜色”，从而让 D15 的交换关系稳定可见。
#    - 不同菱形仍可有不同的颜色对，保持样例多样性。
#
# c) 对原 solver 隐式前提的审查
#    - `diamonds = ... if _is_diamond8(...)`：仍然成立，几何不变。
#    - `ocol = int(g[oc[0][0], oc[0][1]])`：原题把 `ocol` 同时当作“线段色”和“射线色”。
#      变体中这个前提失效，射线色必须改为菱形原色 `dcol`。
#    - `out[rr, cc] = ocol`：原题无害，因为输出只新增外线段色；变体中若不改，会直接退化回原题。
#    - `out = g.copy()`：原题不改菱形；变体需要额外覆写菱形轮廓颜色，因此要在每个菱形处理时把其像素改成配对的外线段色。
#    - 由于本实现把“菱形换色”放在按菱形处理的局部循环里，生成器必须保证：每个菱形相邻的外线段都代表同一个 `border_color`。
#
# d) 放置合法性
#    - 仍沿用 output-first 放置：先选菱形位置和占用边，再计算每条边对应的完整射线路径。
#    - 任何候选放置都要求：菱形、外线段、未来射线之间互不重叠，且不同菱形之间占用集合互斥。
#    - 为了让颜色交换在视觉上可观察，额外要求每条外线段至少产生少量可见射线像素，避免“只换了菱形颜色但没有可见射线”的弱样例。
#
# e) 视觉可观察性
#    - 人眼在 2~4 个样例中能直接看到：输入里菱形颜色 A、外线段颜色 B；输出里菱形变成 B，而新出现的射线却是 A。
#    - 这个规律不依赖颜色编号大小，也不依赖排序，只依赖局部结构上的“菱形 vs 外线段/射线”角色差异。
#
# f) 可视化整洁度
#    - 仍保留原题的稀疏棋盘格和不重叠放置。
#    - 每个菱形只对应一种外线段颜色，输出中不会出现一个菱形被多种颜色分块覆盖的杂乱情况。
#
# g) 颜色变化可推断性
#    - 每个样例中每个菱形只涉及两种非背景色，且交换关系由空间角色明确提示：
#      “贴边短线段的颜色进到菱形里，菱形原色跑到外部射线里”。
#    - 生成器让不同菱形使用不同颜色对，避免靠单一样例的巧合颜色误导。

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


_DIAMOND8_OFFSETS = {
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}


def _is_diamond8(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 8:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _DIAMOND8_OFFSETS


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Detect each diamond and its adjacent outer diagonal segments exactly as in 80a900e0.
    - Determine the outward ray direction from each outer segment.
    - Recolor the diamond outline to the outer-segment color paired with that diamond.
    - Emit rays from each segment endpoint using the original diamond color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for dcol, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)

        adj: list[tuple[int, int]] = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
                        adj.append((rr, cc))

        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)
        if not outer_comps:
            continue

        # In generated samples all adjacent segments of one diamond share one color;
        # use majority color as a robust fallback for unexpected mixed-color inputs.
        outer_cells = {(r, c) for comp in outer_comps for r, c in comp}
        swap_color = Counter(int(g[r, c]) for r, c in outer_cells).most_common(1)[0][0]
        for r, c in dpts:
            out[r, c] = swap_color

        for oc in outer_comps:
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, endpoints = info

            if slope_key == "rc_diff":
                normals = [(-1, 1), (1, -1)]
            else:
                normals = [(-1, -1), (1, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                sc = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if sc > best_score:
                    best_score = sc
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = int(dcol)
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
MIN_RAY_CELLS_PER_EDGE = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300

_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
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
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    total_ray_cells = 0
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
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        total_ray_cells += len(rays)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})
    if total_ray_cells <= 0:
        return None
    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]

    occupied: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            if occupied.isdisjoint(cells):
                occupied.update(cells)
                placed[spec_i] = placement
                if backtrack(i + 1):
                    return True
                occupied.difference_update(cells)
                placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_color_pairs(count: int) -> tuple[list[int], list[int]] | None:
    needed = 2 * count
    if needed > len(COLOR_RANGE):
        return None
    chosen = random.sample(list(COLOR_RANGE), needed)
    diamond_colors = chosen[:count]
    border_colors = chosen[count:]
    return diamond_colors, border_colors


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

    pair_colors = _assign_color_pairs(len(placements))
    if pair_colors is None:
        return None
    diamond_colors, border_colors = pair_colors

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        diamond_color = diamond_colors[pi]
        border_color = border_colors[pi]
        # Output is already swapped: diamond takes border color, rays take diamond color.
        _paint_cells(out, placement["diamond"], border_color)
        for edge_part in placement["edges"]:
            _paint_cells(out, edge_part["segment"], border_color)
            _paint_cells(out, edge_part["rays"], diamond_color)
            ray_coords.extend(edge_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    for pi, placement in enumerate(placements):
        _paint_cells(inp, placement["diamond"], diamond_colors[pi])

    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs, cs = np.array(rs), np.array(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")

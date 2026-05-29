# Combo variant of 80a900e0 (index 12)
# combo_spec: D8/0+D15/2
# Name: 完整颜色互换
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍是 0/1 棋盘格，上面放 1~3 个 8 像素空心菱形，以及贴着菱形外侧的 2~4 条短对角线段。
#    - 例子：输入里某个菱形是颜色 A，围着它的所有外线段是颜色 B。
#    - 原题输出：菱形仍是 A，射线长成 B。
#    - 本融合输出：菱形改成 B，外线段仍是 B，射线改成 A。也就是“菱形色”和“射线色”整体对换。
#    - 几何检测、端点选择、法向判定都沿用原题；变化只发生在输出着色职责的分配上。
#
# b) 复杂度校准
#    - 若同一菱形允许多种外线段颜色，则“菱形到底换成哪种颜色”会变得模糊。
#    - 因此生成器收紧自由度：同一菱形的全部外线段统一使用一个 border_color。
#    - 不同菱形仍使用不同的 `(diamond_color, border_color)` 颜色对，避免样例过于单调。
#
# c) 隐式假设审查
#    - `_top_k_colors` 仍可用来识别 0/1 棋盘背景，因为输入中只有背景像素数量占优。
#    - `_is_diamond8` 的几何前提不变；输入里没有射线，因此菱形识别不受融合影响。
#    - `adj` / `_cc8(adj_mask)` 依赖的是“紧邻菱形的非背景像素”，而不是颜色一致性；这个前提仍成立。
#    - 原题把外线段颜色 `ocol` 直接用于射线着色；融合后这条前提失效，射线必须改用菱形原色 `dcol`。
#    - 融合后新增了“菱形需要换色”的步骤，因此 solver 需要先从相邻外线段里提取一个统一的 swap_color。
#    - 为了在异常输入上更稳健，solver 对相邻外线段颜色取众数；生成器则保证它们实际全相同。
#
# d) 放置合法性
#    - 生成器继续采用 output-first：先放置菱形、外线段与完整射线，再把射线抹回棋盘得到输入。
#    - 候选放置要求菱形、线段、未来射线全部互不重叠，不越界，并且不同对象之间的占用集合互斥。
#    - 每条已选边都必须产生至少少量可见射线像素，否则该样本会被拒绝。
#
# e) 视觉可观察性
#    - 人眼应能在少量样例里看到稳定规律：输入里“内层菱形=A，贴边短线=B”；输出里“菱形=B，新射线=A”。
#    - 该规律完全由空间角色提示，不依赖颜色编号大小或任何隐藏排序。
#    - 生成器强制 `diamond_color != border_color`，并保证射线可见，避免退化成与原题难区分的弱样例。
#
# f) 可视化整洁度
#    - 每个菱形只使用一个 border_color，因此输出中不会出现一个菱形被多种外线段颜色争夺的杂乱情况。
#    - 稀疏棋盘格和不重叠放置继续保留，使“菱形 / 外线段 / 射线”三类结构边界清晰。
#
# g) 变化的可推断性
#    - 每个菱形只牵涉两种非背景色，且二者通过“内层轮廓”和“外层发射体”两个角色反复出现。
#    - 多个样例会重复展示相同的结构关系：外线段颜色流入菱形，菱形原色流向射线。

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
    Swap the diamond color and the ray color around each diamond while keeping the same ray geometry.

    Input structure:
    - The background is a 0/1 checkerboard, with those two colors dominating the grid.
    - Each object contains an 8-cell hollow diamond in one non-background color.
    - Each occupied diamond edge has a short adjacent diagonal outer segment in a second non-background color.

    Rule:
    - Recover each diamond and each adjacent outer segment exactly as in the base task.
    - For one diamond, treat the shared outer-segment color as the swap target for the diamond outline.
    - Recolor the whole diamond to that outer color, then emit two outward diagonal rays from every outer-segment endpoint using the original diamond color.

    Key insight:
    - Geometry still comes entirely from the outer segments, but color ownership changes: the segments donate their color to the diamond, and the diamond donates its original color to all emitted rays.
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
                    if int(g[rr, cc]) not in bg_colors:
                        adj.append((rr, cc))

        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)
        if not outer_comps:
            continue

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
                score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if score > best_score:
                    best_score = score
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
MIN_TOTAL_RAY_CELLS = 4
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

_DIAMOND_COORDS = (
    (0, 2),
    (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3),
    (4, 2),
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
    if total_ray_cells < MIN_TOTAL_RAY_CELLS:
        return None
    guard = set()
    for r, c in diamond:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in all_cells:
                guard.add((rr, cc))
    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "guard": frozenset(guard),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            placement
            for r0, c0 in positions
            if (placement := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def rebuild_blocked() -> None:
        blocked.clear()
        blocked.update(occupied)
        for existing in placed:
            if existing is not None:
                blocked.update(existing["guard"])

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            if blocked.isdisjoint(cells) and occupied.isdisjoint(placement["guard"]):
                occupied.update(cells)
                blocked.update(cells)
                blocked.update(placement["guard"])
                placed[spec_i] = placement
                if backtrack(i + 1):
                    return True
                occupied.difference_update(cells)
                placed[spec_i] = None
                rebuild_blocked()
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

    color_pairs = _assign_color_pairs(len(placements))
    if color_pairs is None:
        return None
    diamond_colors, border_colors = color_pairs

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        diamond_color = diamond_colors[pi]
        border_color = border_colors[pi]
        _paint_cells(out, placement["diamond"], border_color)
        for edge_part in placement["edges"]:
            _paint_cells(out, edge_part["segment"], border_color)
            _paint_cells(out, edge_part["rays"], diamond_color)
            ray_coords.extend(edge_part["rays"])

    if len(ray_coords) < MIN_TOTAL_RAY_CELLS:
        return None
    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    for pi, placement in enumerate(placements):
        _paint_cells(inp, placement["diamond"], diamond_colors[pi])

    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs = np.array(rs)
        cs = np.array(cs)
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

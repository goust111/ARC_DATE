# Variant of 80a900e0
# Dimension: D8 — 射线颜色来源 (trajectory)
# Change: 射线颜色与对应的外边框线段颜色相同... → 射线颜色为菱形颜色
#
# Design analysis
# spec_variant:
#   unchanged_geometry:
#     - 背景仍是 0/1 棋盘格，左上角为 1。
#     - 内层对象仍是 5x5 bbox 内的 8 像素空心菱形。
#     - 外层对象仍是贴着菱形四条边外侧的短对角线段，方向判定完全沿用原题。
#     - 每条外线段仍从两个端点向外发射两条对角射线，直到边界或遇到非背景。
#   changed_semantics:
#     - 原题: 射线用外线段颜色。
#     - 本变体: 射线改用对应菱形颜色；外线段只负责提供位置与方向线索。
#   concrete_trace:
#     - 例如某个菱形是红色，右上侧外线段是绿色。
#     - 输入中只看得到红色菱形 + 绿色短线段。
#     - 原题输出会长出绿色射线；本变体输出会长出红色射线。
#     - 所以 solver 不能再把“外线段组件的颜色”直接拿去涂射线，而要先确定该组件属于哪个菱形。
#   complexity_calibration:
#     - 几何关系不变，避免引入额外匹配自由度。
#     - 为了让 D8 差异可观察，生成器强制保留可见射线，且外线段颜色来自非菱形色调板。
#     - 同一菱形通常带 2-4 条边，且尽量给这些边分配不同颜色；这样人眼能看到“多条不同色外线段，
#       但长出来的射线却统一回到内层菱形色”。
#   implicit_assumption_review_main:
#     - `_top_k_colors` / `bg_colors`: 仍只依赖棋盘背景，不受 D8 影响。
#     - `diamonds`: 仍按 8 连通 + 固定 8 像素形状识别，颜色是否会延伸到射线不影响输入识别，因为输入里没有射线。
#     - `adj` / `outer_comps`: 只按“菱形周围的非背景像素”收集外线段，依赖的是邻接几何，不依赖颜色相等，变体后仍成立。
#     - `_outer_line_info` 和 inward/outward 判定: 完全是斜率与相对位置判断，和颜色来源无关。
#     - 原题真正依赖旧属性的行是 `out[rr, cc] = ocol`；这里把颜色写成外线段色。
#       在变体里这会错误复现旧题，所以必须改成对应菱形色 `dcol`。
#     - `if int(out[rr, cc]) in bg_colors`: 仍安全，因为生成器保证射线路径不穿过其他非背景对象；不会要求“穿过同色射线继续画”。
#   placement_legality:
#     - 放置仍采用 output-first：先放菱形、线段和完整射线，再把射线抹回棋盘得到输入。
#     - 候选放置要求菱形、线段、射线全都不重叠；这样 solver 在输入上能稳定恢复。
#     - 生成器还要求总射线像素数达到下限，避免出现“看不到变体”的退化样本。
#   visual_observability:
#     - 如果没有射线像素，或者外线段色恰好等于菱形色，人类无法确认 D8 已变化。
#     - 因此生成器要求 `ray_coords` 足够多，且所有外线段颜色都从“排除所有菱形色”的调色板中选。
#   cleanliness:
#     - 不新增额外标记层，输出仍只有菱形、线段、射线三种结构，视觉上比添加新标志更干净。
#   color_inference:
#     - 颜色映射不是任意置换，而是固定结构关系：每条射线回到其所属菱形颜色。
#     - 线段颜色数量被限制在原调色板内，并尽量在同一菱形周围做局部多样化，利于少量样例归纳。

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
    # Returns (slope_key, endpoints_sorted_by_r) or None if not diagonal collinear.
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        endpoints = sorted(pts, key=lambda p: (p[0], p[1]))
        return ("rc_diff", endpoints)
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        endpoints = sorted(pts, key=lambda p: (p[0], -p[1]))
        return ("rc_sum", endpoints)
    return None


def main(grid: np.ndarray) -> np.ndarray:
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
                        out[rr, cc] = dcol
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
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
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})
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


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for pi, placement in enumerate(placements):
        used_local: set[int] = set()
        edge_order = sorted(
            range(len(placement["edges"])),
            key=lambda ei: (placement["edges"][ei]["weight"], random.random()),
            reverse=True,
        )
        for ei in edge_order:
            options = [c for c in palette if c not in used_local] or palette
            chosen = min(options, key=lambda c: (counts[c], random.random()))
            segment_colors[pi][ei] = chosen
            counts[chosen] += placement["edges"][ei]["weight"]
            used_local.add(chosen)
    return segment_colors


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

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        dcolor = diamond_colors[pi]
        _paint_cells(out, placement["diamond"], dcolor)
        for ei, edge_part in enumerate(placement["edges"]):
            _paint_cells(out, edge_part["segment"], segment_colors[pi][ei])
            _paint_cells(out, edge_part["rays"], dcolor)
            ray_coords.extend(edge_part["rays"])

    if len(ray_coords) < MIN_TOTAL_RAY_CELLS:
        return None
    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
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

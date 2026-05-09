# Variant of 80a900e0
# Dimension: D6 — 外边框线段的位置与形状 (obstacle)
# Change: 3像素对角线段平行于菱形边缘且向外偏移1步，每条线段对应菱形的一条边... → 外边框为单个像素（点标记）

# Design analysis for this D6 variant
# a) Concrete behavior:
#    原题每条外边框是 3 像素对角线段；这里把它压缩为该线段的中心像素。
#    因为原 5x5 菱形四条边对应的 3 像素外线段中心正好落在菱形 bbox 的四个角点，
#    所以输入中的“点标记”固定出现在 (r0,c0)/(r0,c1)/(r1,c0)/(r1,c1) 四个候选位置之一。
#    输出仍然是“保留点标记，并向外补上两条对角射线”；只是射线不再由可见线段端点直接给出，
#    而是要先从点标记相对菱形的位置恢复 edge，再恢复该 edge 的两个虚拟端点。
#    例子：TR 边原本线段为 [(-1,3),(0,4),(1,5)]，现在只保留中心点 (0,4)。
#    原 solver 通过 3 点共线得到 slope=(1,1)，再从端点发射两条 outward=(-1,1) 的射线；
#    新 solver 则通过“点在 bbox 右上角”判断它是 TR 边，恢复虚拟端点为点±(1,1)，
#    再沿同样 outward 方向发射射线。差异只在障碍的可见形状和朝向推理来源。
# b) Complexity calibration:
#    只允许 4 个离散点位（四个角），避免自由度过高；每个样例仍有 2-4 个点标记，
#    多个菱形时会重复出现这些 edge 类型，足够支撑人类归纳“角点决定射线朝向”。
#    同时点标记明显比 3 像素线段更短，视觉上能直接看出这是 D6 的变化。
# c) Implicit assumptions audit of original main():
#    - `_outer_line_info(oc)` 依赖 outer component 至少 2 像素且对角共线；点标记不成立，必须删掉。
#    - `normals` 的选择依赖已知线段斜率；点标记没有斜率，只能改为由 corner→edge 的查表。
#    - `p0/p1` 直接取可见线段端点；点标记下必须恢复“虚拟端点 = marker ± tangent”。
#    - `adj_mask -> _cc8` 原来用于把相邻像素聚成线段；现在输入就是单像素，可简化为直接读四个角点。
# d) Placement legality:
#    generator 继续 output-first。对每个 diamond+edge，先放 diamond，再放 corner marker，再计算两条完整 rays。
#    必须保证 marker/ray 与其他对象不重叠；还要保证至少有可见 ray，否则该 edge 虽合法但退化为“无变化”。
# e) Visual observability:
#    解题者能看到：输入里菱形外侧不再是短斜线，而是角点式单像素标记；输出中每个点都会触发一对同色射线。
#    因为点的位置只有四类，2-3 组样例就足以把“左上/右上/左下/右下 marker 对应哪种 outward 方向”看出来。
# f) Visual cleanliness:
#    仍保持原题的 checkerboard、diamond、同色射线布局，只把 3 像素障碍缩成 1 像素；
#    这会减少遮挡。generator 继续禁止输出态对象/射线互撞，避免同一区域堆叠。
# g) Color inference:
#    颜色规则不变：marker 与其射线同色，diamond 用不同颜色。没有新增颜色映射，因此不会引入不可见的排序规则。
# h) Source-task audit:
#    本变体没有 source_task，可直接基于原题几何做最小修改。
# i) New solver assumptions and generator guarantees:
#    - solver 假设 marker 只会出现在 diamond bbox 四角之一；generator 只在这四个角放点。
#    - solver 假设每个 marker 对应的 rays 与其他非背景对象不冲突；generator 在 output-first 阶段显式检测。
#    - solver 假设每个被选 edge 至少产生 1 个可见 ray cell；generator 用 `MIN_RAY_CELLS` 保证这一点，
#      从而新规则在输出中一定可观察，不会退化成 input==output。

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
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


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _marker_rays(
    shape: tuple[int, int], marker: tuple[int, int], tangent: tuple[int, int], outward: tuple[int, int]
) -> list[tuple[int, int]]:
    a = (marker[0] - tangent[0], marker[1] - tangent[1])
    b = (marker[0] + tangent[0], marker[1] + tangent[1])
    return _ray_cells(shape, a, outward) + _ray_cells(shape, b, outward)


# ── Tunable parameters ──────────────────────────────────────
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_RAY_CELLS = 1
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300

# ── Geometry constants ──────────────────────────────────────
_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_EDGE_INFO = {
    "TL": {"marker": (0, 0), "tangent": (1, -1), "outward": (-1, -1)},
    "TR": {"marker": (0, 4), "tangent": (1, 1), "outward": (-1, 1)},
    "BL": {"marker": (4, 0), "tangent": (1, 1), "outward": (1, -1)},
    "BR": {"marker": (4, 4), "tangent": (1, -1), "outward": (1, 1)},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from single-pixel corner markers adjacent to 5x5 hollow diamonds.

    Input:
    - checkerboard background using the two dominant colors;
    - 8-pixel hollow diamond outlines;
    - optional single-pixel outer markers placed at one or more corners of each diamond bbox.

    Rule:
    - Each marker identifies one diamond edge by its corner position (TL/TR/BL/BR).
    - That corner marker is treated as the midpoint of the original 3-pixel outer segment.
    - Reconstruct the segment's two virtual endpoints as marker ± tangent(edge).
    - From each virtual endpoint, cast a diagonal ray in the edge's outward direction until a
      non-background pixel or the boundary is reached.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    def bbox_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        r0, c0, r1, c1 = _bbox(dpts)
        corner_to_edge = {
            (r0, c0): "TL",
            (r0, c1): "TR",
            (r1, c0): "BL",
            (r1, c1): "BR",
        }

        for (mr, mc), edge in corner_to_edge.items():
            if not (0 <= mr < h and 0 <= mc < w):
                continue
            mcol = int(g[mr, mc])
            if mcol in bg_colors:
                continue

            info = _EDGE_INFO[edge]
            tr, tc = info["tangent"]
            dr, dc = info["outward"]
            for sign in (-1, 1):
                pr = mr + sign * tr
                pc = mc + sign * tc
                rr, cc = pr + dr, pc + dc
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = mcol
                    else:
                        break
                    rr += dr
                    cc += dc

    return out


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[edge]
        marker = (r0 + info["marker"][0], c0 + info["marker"][1])
        rays = _marker_rays((h, w), marker, info["tangent"], info["outward"])
        if len(rays) < MIN_RAY_CELLS:
            return None
        part_cells = [marker] + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append({"marker": marker, "rays": rays, "weight": len(part_cells)})
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


def _assign_marker_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        ((ep["weight"], random.random(), pi, ei)
         for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
        reverse=True,
    )
    marker_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        marker_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return marker_colors


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
    marker_colors = _assign_marker_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = marker_colors[pi][ei]
            _paint_cells(out, [edge_part["marker"]], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

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

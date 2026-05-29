# Variant of 80a900e0
# Dimension: D14 — 外边框激活条件 (meta)
# Change: 所有与菱形相邻的非背景色块组件都作为外边框参与射线发射（无筛选条件）... → 外边框朝向决定是否激活

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8

# Design analysis for the D14 variant
# -----------------------------------
# source_task:
#   none
# concrete_rule:
#   - 输入仍然是棋盘格背景 + 8 像素空心菱形 + 与菱形相邻的短对角外边框段。
#   - 变体只改“哪些外边框会发射射线”：
#       * 若整张网格是“高于宽”（h > w）的竖向画布，只激活菱形上方两边的外边框: TL / TR。
#       * 否则激活菱形下方两边的外边框: BL / BR。
#   - 未激活的外边框仍然保留在输出里，但不会额外长出射线。
# behavior_trace:
#   - 原题中，一个菱形如果四边都有外边框，则四条短线全部向外发射。
#   - 现在同样的输入，在 tall 网格里只有上侧两条可能延长；在 wide/square 网格里只有下侧两条可能延长。
#   - 例如 h=23,w=17 时，同一菱形同时带有 TR 和 BR 两条外边框：
#       * 原逻辑: TR 往右上发射，BR 往右下发射。
#       * 新逻辑: 只保留 TR 的射线；BR 只是短线本体，不再延长。
# complexity_calibration:
#   - 规则只有两个全局档位（top-active / bottom-active），复杂度可控。
#   - 生成器强制每个菱形至少同时拥有 1 条激活边和 1 条未激活边，避免样本退化成原题或看不出差异。
# hidden_assumption_audit:
#   - 原 solver 默认“所有 outer component 都发射”。这在 D14 下不成立，必须在求出 outward 方向后再筛选。
#   - 原 solver 用 outer component 的法向与菱形重叠分数推断 inward/outward；这一点仍成立，因为几何结构没变。
#   - 新增假设是：上侧边的 outward 行增量恒为 -1，下侧边恒为 +1。生成器沿用原题的四种固定边模板，保证该假设成立。
# placement_legality:
#   - 生成器继续采用 output-first。
#   - 但占位检测只把“激活边的射线”计入输出占位；未激活边只占自己的短线格子。
#   - 这样既能防止真实输出中的碰撞，也允许输入里同时出现 active / inactive 的外边框。
# visual_observability:
#   - 全局线索选用网格纵横比，这是输入中直接可见的整体属性。
#   - 为了让人类更容易从少量样例归纳，生成器把 tall / wide 两类尺寸拉开至少数格，不生成接近正方形的模糊案例。
# visual_cleanliness:
#   - 每个对象仍是“菱形 + 若干短对角段 + 少量射线”，图面结构与原题一致，不引入额外标记。
#   - 通过回溯放置避免对象、短线和真实射线彼此重叠。
# color_inference:
#   - 颜色不承载新规则，只延续原题的独立非背景色分配，因此不会引入额外颜色映射负担。


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


def _outer_line_info(pts: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]] | None:
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _activation_mode(h: int, w: int) -> str:
    return "top" if h > w else "bottom"


def _outward_is_active(outward: tuple[int, int], h: int, w: int) -> bool:
    mode = _activation_mode(h, w)
    return outward[0] < 0 if mode == "top" else outward[0] > 0


def main(grid: np.ndarray) -> np.ndarray:
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

        for oc in outer_comps:
            ocol = int(g[oc[0][0], oc[0][1]])
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

            if not _outward_is_active(outward, h, w):
                continue

            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = ocol
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
TALL_H_RANGE = (21, 24)
TALL_W_RANGE = (14, 19)
WIDE_H_RANGE = (20, 22)
WIDE_W_RANGE = (24, 30)
DIAMOND_COUNT_RANGE = (1, 3)
MIN_SEGMENT_LEN = 2
MIN_ACTIVE_RAY_CELLS = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

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


def _choose_dimensions() -> tuple[int, int]:
    mode = random.choice(("top", "bottom"))
    if mode == "top":
        h = random.randint(*TALL_H_RANGE)
        w = random.randint(*TALL_W_RANGE)
    else:
        h = random.randint(*WIDE_H_RANGE)
        w = random.randint(*WIDE_W_RANGE)
    if not (GRID_H_RANGE[0] <= h <= GRID_H_RANGE[1] and GRID_W_RANGE[0] <= w <= GRID_W_RANGE[1]):
        raise AssertionError("dimension choice escaped original bounds")
    return h, w


def _active_edge_names(h: int, w: int) -> tuple[str, str]:
    return ("TL", "TR") if _activation_mode(h, w) == "top" else ("BL", "BR")


def _sample_specs(diamond_count: int, active_edges: tuple[str, str]) -> list[tuple[str, ...]]:
    active_pool = list(active_edges)
    inactive_pool = [edge for edge in _EDGE_NAMES if edge not in active_edges]
    specs: list[tuple[str, ...]] = []
    for _ in range(diamond_count):
        chosen = random.sample(active_pool, random.randint(1, len(active_pool)))
        chosen += random.sample(inactive_pool, random.randint(1, len(inactive_pool)))
        specs.append(tuple(sorted(chosen)))
    return specs


def _candidate_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
    edges: tuple[str, ...],
    active_edges: set[str],
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    active_ray_cells = 0

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
        rays: list[tuple[int, int]] = []
        if edge in active_edges:
            rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
            if len(rays) < MIN_ACTIVE_RAY_CELLS:
                return None
            active_ray_cells += len(rays)

        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells):
            return None
        if any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "name": edge,
                "segment": segment,
                "rays": rays,
                "active": edge in active_edges,
                "weight": len(part_cells),
            }
        )

    if active_ray_cells == 0:
        return None

    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


def _search_placements(
    h: int,
    w: int,
    specs: list[tuple[str, ...]],
    active_edges: set[str],
) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            p
            for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edges, active_edges)) is not None
        ]
        random.shuffle(candidate_cache[edges])

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

    weighted_edges = sorted(
        ((ep["weight"], random.random(), pi, ei) for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h, w = _choose_dimensions()
    active_edges = set(_active_edge_names(h, w))
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _sample_specs(diamond_count, tuple(sorted(active_edges)))

    placements = _search_placements(h, w, specs, active_edges)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    active_edge_total = 0
    inactive_edge_total = 0

    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            if edge_part["active"]:
                _paint_cells(out, edge_part["rays"], color)
                ray_coords.extend(edge_part["rays"])
                active_edge_total += 1
            else:
                inactive_edge_total += 1

    if active_edge_total == 0 or inactive_edge_total == 0 or not ray_coords:
        return None
    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    rs, cs = zip(*ray_coords)
    rs_arr = np.array(rs)
    cs_arr = np.array(cs)
    inp[rs_arr, cs_arr] = (rs_arr + cs_arr + 1) % 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")

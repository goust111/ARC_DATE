# Variant of 80a900e0
# Dimension: D5 — 发射源对象形状 (src)
# Change: 源为8像素的空心菱形轮廓（5x5外接矩形中的对角线位置像素）... → 不同尺寸的菱形（3x3或7x7）
#
# Design analysis
# spec:
#   core_change:
#     old: every source object is the same 5x5 hollow diamond with 8 pixels
#     new: each source object is either a 3x3 hollow diamond (4 pixels) or a 7x7 hollow diamond (12 pixels)
#   preserved_parts:
#     - background is still checkerboard 0/1 with top-left = 1
#     - non-bg pixels still occupy the same checkerboard phase
#     - each occupied diamond edge still has one outer diagonal segment
#     - each segment still emits two outward diagonal rays from its two endpoints
#   concretization:
#     - size 3 diamond pixels are {(0,1),(1,0),(1,2),(2,1)} in its bbox
#     - size 7 diamond pixels are {(0,3),(1,2),(1,4),(2,1),(2,5),(3,0),(3,6),(4,1),(4,5),(5,2),(5,4),(6,3)}
#     - outer segments remain "one diagonal step outward from an edge", so their lengths become size-dependent:
#       2 cells for a 3x3 diamond, 4 cells for a 7x7 diamond
#   behavior_trace:
#     - original TR edge on a 5x5 diamond uses side [(0,2),(1,3),(2,4)] and outer segment [(-1,3),(0,4),(1,5)]
#     - variant size-3 TR edge uses side [(0,1),(1,2)] and outer segment [(-1,2),(0,3)]
#     - variant size-7 TR edge uses side [(0,3),(1,4),(2,5),(3,6)] and outer segment [(-1,4),(0,5),(1,6),(2,7)]
#     - the ray rule itself is unchanged; only the source geometry that determines where segments can exist is variable
#   complexity_calibration:
#     - only two size branches are allowed: 3 and 7. This is enough to be visually obvious but still learnable from 2-4 examples.
#     - multi-diamond instances are biased to include both sizes, so the size parameter is repeatedly visible.
#   hidden_assumption_audit:
#     - original `_is_diamond8` assumes len(comp)==8 and bbox span==4; this must become size-aware
#     - original generator hard-codes 5x5 offsets, edge offsets, and top-left parity for that single size
#     - original `even_toplefts` assumes all diamond offsets have even parity; size 3 and 7 need odd-parity anchors instead
#     - original adjacent-marker extraction is safe only if nearby markers do not merge across colors; here we extract outer components per color
#   placement_legality:
#     - generator places full output first (diamonds + segments + rays), rejecting any overlap
#     - the top-left parity is chosen per size so all non-bg pixels stay on the intended checkerboard phase
#     - segment colors are assigned with a small adjacency-aware greedy rule so touching segments do not share a color
#   visual_observability:
#     - the difference is directly visible in the input because the source outline itself is smaller or larger
#     - the output preserves that difference while extending rays from correspondingly shorter or longer outer segments
#   new_main_assumptions_checked_against_generator:
#     - diamonds are isolated non-bg components in their own colors: guaranteed by distinct diamond colors
#     - each outer segment is a single diagonal component of one color adjacent to exactly one diamond side: guaranteed by output-first placement
#     - same-color adjacent segments should not fuse into one accidental component: mitigated by adjacency-aware segment coloring

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


def _checkerboard_value(r: int, c: int) -> int:
    return (r + c + 1) % 2


def _diamond_outline_offsets(size: int) -> tuple[tuple[int, int], ...]:
    radius = (size - 1) // 2
    center = radius
    pts = []
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) == radius:
                pts.append((r, c))
    return tuple(pts)


DIAMOND_SIZES = (3, 7)
_DIAMOND_OFFSETS = {size: _diamond_outline_offsets(size) for size in DIAMOND_SIZES}
_EDGE_NAMES = ("TR", "TL", "BL", "BR")


def _build_edge_info(size: int) -> dict[str, dict[str, object]]:
    k = (size - 1) // 2
    return {
        "TR": {
            "segment": tuple((i - 1, k + i + 1) for i in range(k + 1)),
            "outward": (-1, 1),
            "slope": "diff",
        },
        "TL": {
            "segment": tuple((i - 1, k - i - 1) for i in range(k + 1)),
            "outward": (-1, -1),
            "slope": "sum",
        },
        "BL": {
            "segment": tuple((k + i + 1, i - 1) for i in range(k + 1)),
            "outward": (1, -1),
            "slope": "diff",
        },
        "BR": {
            "segment": tuple((k + i + 1, 2 * k - i + 1) for i in range(k + 1)),
            "outward": (1, 1),
            "slope": "sum",
        },
    }


_EDGE_INFO = {size: _build_edge_info(size) for size in DIAMOND_SIZES}


def _diamond_size(comp: list[tuple[int, int]]) -> int | None:
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    size_r = r1 - r0 + 1
    size_c = c1 - c0 + 1
    if size_r != size_c or size_r not in _DIAMOND_OFFSETS:
        return None
    norm = {(r - r0, c - c0) for r, c in comp}
    if len(comp) != len(_DIAMOND_OFFSETS[size_r]):
        return None
    return size_r if norm == set(_DIAMOND_OFFSETS[size_r]) else None


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
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]], int]] = []
    for col, pts in comps:
        size = _diamond_size(pts)
        if size is not None:
            diamonds.append((col, pts, size))

    out = g.copy()

    def bbox_key(item):
        _, pts, _ = item
        return _bbox(pts)

    for _, dpts, _ in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj_by_color: dict[int, set[tuple[int, int]]] = {}
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
                        adj_by_color.setdefault(v, set()).add((rr, cc))

        for ocol, adj_cells in adj_by_color.items():
            adj_mask = np.zeros((h, w), dtype=bool)
            for r, c in adj_cells:
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
                            out[rr, cc] = ocol
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
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400


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
    h: int, w: int, r0: int, c0: int, size: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_OFFSETS[size]]
    all_cells = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[size][edge]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells(
            (h, w), segment[-1], info["outward"]
        )
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "edge": edge,
                "segment": segment,
                "rays": rays,
                "weight": len(part_cells),
            }
        )
    return {
        "size": size,
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
    }


def _search_placements(h: int, w: int, specs: list[tuple[int, tuple[str, ...]]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: (specs[i][0], len(specs[i][1])), reverse=True)
    candidate_cache: dict[tuple[int, tuple[str, ...]], list[dict[str, object]]] = {}
    for spec in {specs[i] for i in order}:
        size, edges = spec
        anchor_parity = ((size - 1) // 2) % 2
        top_lefts = [
            (r, c)
            for r in range(h - size + 1)
            for c in range(w - size + 1)
            if (r + c) % 2 == anchor_parity
        ]
        random.shuffle(top_lefts)
        candidate_cache[spec] = [
            p
            for r0, c0 in top_lefts
            if (p := _candidate_placement(h, w, r0, c0, size, edges)) is not None
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


def _touches8(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in bset:
                    return True
    return False


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for pi, color in enumerate(diamond_colors):
        counts[color] += len(placements[pi]["diamond"])

    flat_edges: list[tuple[int, int, dict[str, object]]] = []
    for pi, placement in enumerate(placements):
        for ei, edge_part in enumerate(placement["edges"]):
            flat_edges.append((pi, ei, edge_part))

    conflicts: dict[tuple[int, int], set[tuple[int, int]]] = {(pi, ei): set() for pi, ei, _ in flat_edges}
    for i, (pi0, ei0, edge0) in enumerate(flat_edges):
        for pi1, ei1, edge1 in flat_edges[i + 1 :]:
            if _touches8(edge0["segment"], edge1["segment"]):
                conflicts[(pi0, ei0)].add((pi1, ei1))
                conflicts[(pi1, ei1)].add((pi0, ei0))

    weighted_edges = sorted(
        ((edge["weight"], random.random(), pi, ei) for pi, ei, edge in flat_edges),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(pl["edges"]) for pl in placements]
    for _, _, pi, ei in weighted_edges:
        used_by_neighbors = {
            segment_colors[npi][nei]
            for npi, nei in conflicts[(pi, ei)]
            if segment_colors[npi][nei] != 0
        }
        choices = [c for c in palette if c not in used_by_neighbors] or palette
        chosen = min(choices, key=lambda c: (counts[c], random.random()))
        segment_colors[pi][ei] = chosen
        counts[chosen] += placements[pi]["edges"][ei]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _sample_sizes(diamond_count: int) -> list[int]:
    if diamond_count == 1:
        return [random.choice(DIAMOND_SIZES)]
    sizes = [random.choice(DIAMOND_SIZES) for _ in range(diamond_count)]
    if random.random() < 0.75:
        sizes[0] = 3
        sizes[1] = 7
        random.shuffle(sizes)
    return sizes


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    sizes = _sample_sizes(diamond_count)
    specs = [
        (sizes[i], tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_DIAMOND)))))
        for i in range(diamond_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    for r, c in ray_coords:
        inp[r, c] = _checkerboard_value(r, c)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")

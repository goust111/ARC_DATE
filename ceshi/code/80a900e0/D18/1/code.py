# Variant of 80a900e0
# Dimension: D18 — 输入输出关系方向 (meta)
# Change: 输出先构建（output-first）：先放菱形和射线，再删除射线生成输入。解题方向为从输入推断并添加射线
#   → 反向任务：输入含射线，输出仅保留射线
#
# Design analysis
# ---------------
# variant_spec:
#   unchanged_geometry:
#     grid: 14-30 cols x 20-24 rows
#     background: checkerboard of colors 0 and 1; top-left is 1; bg[r,c]=(r+c+1)%2
#     diamond:
#       shape: same 8-pixel hollow diamond in a 5x5 bbox
#       count: 1-3
#       color: distinct non-bg color per diamond
#     outer_segment:
#       shape: same short diagonal segment on 2-4 diamond edges
#       length: 2-3 cells after boundary clipping
#       color: non-bg, different from the owning diamond
#     rays:
#       source: from both endpoints of each outer segment
#       direction: same outward diagonal as the original task
#       extent: contiguous until grid boundary
#   changed_io_direction:
#     input: full original-output figure = checkerboard + diamonds + outer segments + rays
#     output: checkerboard with only ray pixels preserved; diamonds and outer segments are deleted
#   behavior_trace:
#     original:
#       - generator built full figure as output
#       - generator erased ray cells to get input
#       - solver reconstructed missing rays from diamonds + segments
#     variant:
#       - generator directly builds the full figure as input
#       - expected output restores checkerboard on every diamond/segment cell, keeping only ray cells
#       - solver still uses diamonds + adjacent segments as anchors, but now extracts existing ray
#         pixels from the input instead of painting new ones into background
#   concrete_small_example:
#     - Suppose one TR outer segment has cells [(r-1,c+3),(r,c+4),(r+1,c+5)] and color 8.
#     - Original solver walked outward from the two endpoints and filled background 8s until edge.
#     - Variant solver walks outward from the same endpoints but records only the already-present
#       consecutive 8s. The segment cells themselves are not copied to output.
#   complexity_calibration:
#     - The diamond+segment scaffold is retained so the ray/source distinction is visually grounded.
#     - Multiple diamonds and 2-4 occupied edges per diamond remain, so examples are not trivial.
#     - Ray colors may repeat across different segments, but extraction is anchored locally by
#       adjacency to each diamond rather than by global color grouping.
#   implicit_assumptions_audit:
#     - `_top_k_colors(..., 2)` still assumes colors 0 and 1 remain the two most frequent colors.
#       Generator enforces this on the full input.
#     - `_is_diamond8` assumes diamond colors stay separate from segment/ray colors; generator keeps
#       every diamond color distinct and never reuses it for its segments.
#     - Adjacent non-bg pixels around a diamond must correspond only to outer segments, not rays.
#       This remains true because rays start one more outward step beyond the segment endpoints.
#     - While extracting a ray, `main()` now assumes the outward path is a contiguous run of the
#       segment color and that no unrelated object lies on that path. Generator guarantees all
#       figure cells are pairwise disjoint.
#   placement_legality:
#     - Candidate placements reject overlap among diamonds, segments, and full rays.
#     - The search places all diamonds jointly, so different objects cannot intersect.
#     - Samples with zero visible ray pixels are rejected because the D18 change would be invisible.
#   visual_observability:
#     - Input contains a complete decorated diamond figure; output removes the central source objects
#       and leaves only the emitted diagonals. This is visually different from the original task,
#       where output added pixels rather than deleting the source object.
#   cleanliness:
#     - No extra marks are introduced. Output contains only checkerboard plus thin diagonal rays,
#       which keeps the scene readable after the source objects disappear.

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
        endpoints = sorted(pts, key=lambda p: (p[0], p[1]))
        return ("rc_diff", endpoints)
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        endpoints = sorted(pts, key=lambda p: (p[0], -p[1]))
        return ("rc_sum", endpoints)
    return None


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


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
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = _checkerboard(h, w)

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        adj = []
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

        for oc in _cc8(adj_mask):
            ocol = int(g[oc[0][0], oc[0][1]])
            parsed = _segment_outward(oc, dset)
            if parsed is None:
                continue
            endpoints, outward = parsed
            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == ocol:
                    out[rr, cc] = ocol
                    rr += outward[0]
                    cc += outward[1]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Input contains the full decorated figure: checkerboard, diamonds, outer segments, and rays.
    Output keeps only the ray pixels and restores checkerboard everywhere else.
    """
    g = np.asarray(grid, dtype=int)
    return _extract_rays_only(g)


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
MIN_TOTAL_RAY_PIXELS = 6
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
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells(
            (h, w), segment[-1], info["outward"]
        )
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
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


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
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

    inp = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(inp, placement["diamond"], diamond_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = segment_colors[pi][ei]
            _paint_cells(inp, edge_part["segment"], color)
            _paint_cells(inp, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

    if len(ray_coords) < MIN_TOTAL_RAY_PIXELS or not _bg_counts_safe(inp):
        return None

    out = _checkerboard(h, w)
    for pi, placement in enumerate(placements):
        for ei, edge_part in enumerate(placement["edges"]):
            _paint_cells(out, edge_part["rays"], segment_colors[pi][ei])

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")

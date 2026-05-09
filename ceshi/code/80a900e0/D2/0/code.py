# Variant of 80a900e0
# Dimension: D2 — 射线发射点 (path)
# Change: 从外边框线段的两个端点发射，每条线段恰好发射2条射线... → 从线段每个像素发射

"""
Design analysis for the D2 variant.

a) Concrete behavior and trace
   - The input still contains:
     * a 0/1 checkerboard background with top-left = 1,
     * one to three 8-pixel hollow diamonds,
     * 3-pixel diagonal outer segments adjacent to chosen diamond edges.
   - The output change is only in how rays are emitted:
     * original: a 3-pixel segment emitted 2 rays, only from its two endpoints;
     * variant: the same 3-pixel segment emits 3 rays, one from each segment pixel.
   - Example on a TR segment with cells [(r-1,c+3), (r,c+4), (r+1,c+5)] and outward
     direction (-1,+1):
     * original output adds rays starting from the first and third segment pixels;
     * variant output adds those two rays plus one more parallel middle ray starting
       from the center segment pixel.
   - This affects exactly the ray-casting stage in both solver and generator. Diamond
     detection, outer-segment detection, color handling, and outward-direction inference
     remain unchanged.

b) Complexity calibration
   - If clipped 2-pixel segments were allowed, the variant could silently collapse back
     to the original behavior because "all pixels" would still equal "both endpoints".
   - To keep the D2 change visible, the generator only places full 3-pixel outer segments.
   - Other freedoms remain close to the source task: 1-3 diamonds, 2-4 occupied edges
     per diamond, same checkerboard background, same color palette.

c) Implicit-assumption audit for main()
   - `_top_k_colors`, `_components8_by_color`, `_is_diamond8`: still valid because the
     background and diamond structure are unchanged.
   - Adjacency collection around each diamond: still valid because outer segments remain
     8-neighbor adjacent diagonal components.
   - `_outer_line_info`: still valid because segments are still short diagonal collinear
     components.
   - Normal-selection logic: still valid because "away from the diamond" does not change.
   - Original endpoint loop assumed "the only legal emitters are the two extremes".
     That assumption is the one D2 explicitly breaks, so the emission loop must iterate
     over every pixel in the detected segment instead of only `endpoints[0]` and
     `endpoints[-1]`.
   - The write rule "paint only while the current cell is background, otherwise stop"
     remains safe because rays are still supposed to terminate on non-background objects.

d) Placement legality
   - The generator is still output-first.
   - For each chosen edge, it builds the 3-pixel segment, then unions the rays emitted
     from every segment pixel.
   - A candidate placement is rejected if any diamond pixel, segment pixel, or ray pixel
     overlaps another pixel from the same placement or from previously placed diamonds.
   - This guarantees that `main()` sees clean, separable objects and that each ray can be
     reconstructed without ambiguous collisions.

e) Visual observability
   - The new effect is a visibly denser band: each segment now produces a parallel bundle
     of 3 rays instead of 2.
   - Because the generator forbids clipped 2-pixel segments, every occupied edge exhibits
     at least one extra ray compared with the original task, so the D2 change is visible
     in every generated example.

f) Visual cleanliness
   - Dense rays increase clutter risk. The generator therefore keeps the original
     non-overlap discipline and rejects placements whose ray bundles collide.
   - This preserves clean, readable diagonal bands rather than tangled overlays.

g) Color inference
   - Colors are unchanged structurally: each ray bundle uses the color of its own outer
     segment. No new color mapping is introduced, so the rule remains inferable from
     spatial structure alone.

h) New solver assumptions introduced by the variant
   - Each outer component must still be a single straight diagonal segment.
   - Rays from different segments are assumed not to collide before reaching a boundary or
     an existing non-background obstacle.
   - The generator enforces both assumptions by construction.
"""

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
    # Returns (slope_key, points_sorted_along_line) or None if not diagonal collinear.
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _paint_ray(out: np.ndarray, start: tuple[int, int], direction: tuple[int, int], color: int, bg_colors: set[int]) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if int(out[r, c]) not in bg_colors:
            break
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: each outer diagonal segment emits one outward diagonal ray from every
    segment pixel, not just from its two endpoints.
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
            slope_key, segment_points = info

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

            for pr, pc in segment_points:
                _paint_ray(out, (pr, pc), outward, ocol, bg_colors)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
SEGMENT_LEN = 3
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


def _segment_bundle_rays(shape: tuple[int, int], segment: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    rays: list[tuple[int, int]] = []
    for pixel in segment:
        rays.extend(_ray_cells(shape, pixel, direction))
    return rays


def _candidate_placement(h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    edge_parts = []
    for edge in edges:
        info = _EDGE_INFO[edge]
        segment = [(r0 + dr, c0 + dc) for dr, dc in info["segment"]]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in segment):
            return None
        if len(segment) != SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])
        rays = _segment_bundle_rays((h, w), segment, info["outward"])
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
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs_arr = np.array(rs, dtype=int)
        cs_arr = np.array(cs, dtype=int)
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

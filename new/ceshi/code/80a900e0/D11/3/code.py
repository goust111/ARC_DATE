# Variant of 80a900e0
# Dimension: D11 — 变换/输出效果 (transform)
# Change: 射线像素覆盖（用外边框颜色覆盖背景像素），不改变已有非背景像素... → 射线端点处放置标记

# Design analysis for the D11 variant
# ----------------------------------
# source_task:
# - None. No extra reference task to read.
#
# concretized_spec:
# - Input keeps the original geometry:
#   - checkerboard background with colors 0/1 and top-left = 1
#   - 1-3 hollow 8-pixel diamonds in distinct non-background colors
#   - 2-4 short outer diagonal segments per diamond, adjacent to diamond edges
# - Output no longer draws the whole ray corridors.
# - Instead, for each outer segment, take its two endpoints, follow the same outward
#   diagonal directions as in the original task, and place one special marker color
#   at the boundary cell where each ray would terminate.
# - The marker color is fixed across the whole task: 9. It is reserved from the
#   generator's object palette so it appears only as the new output marker.
#
# a) behavior trace:
# - Original local behavior:
#   - Suppose an outer segment sits on the diamond's top-right edge.
#   - Its outward direction is (-1, +1).
#   - From each endpoint, the original solver walks cell by cell and paints every
#     checkerboard cell until reaching the boundary.
# - Variant local behavior:
#   - The same outward direction is inferred.
#   - The walk still conceptually follows the whole ray, but only the final in-bounds
#     cell on the boundary is changed, and it is changed to marker color 9.
# - Difference:
#   - The geometry and direction inference stay identical.
#   - Only the transform effect changes from "fill corridor" to "mark terminal point".
#
# b) complexity calibration:
# - A fixed marker color keeps the new rule learnable from a few examples.
# - Each grid still has multiple rays (at least 4 markers because each diamond has
#   2-4 segments and each segment emits 2 rays), so the new sparse behavior remains
#   visually obvious.
# - The generator still reserves full ray corridors, preventing ambiguous cases where
#   multiple objects block or overlap hypothetical rays.
#
# c) implicit-assumption audit against the original main():
# - `_top_k_colors(..., 2)` still works because markers are absent from the input;
#   the two most common colors remain the checkerboard colors 0 and 1.
# - Diamond detection via connected 8-pixel components is unchanged.
# - Adjacent outer-segment extraction is unchanged because the input still contains
#   only diamonds and short segments as non-background shapes.
# - The original loop wrote `ocol` onto every background cell. That relied on the old
#   "ray corridor becomes visible" rule. In the variant this would be wrong, so the
#   write loop is replaced by a terminal-search routine that only returns the boundary
#   endpoint and then writes the marker color there.
# - The new solver must not blindly mark "the last visited cell" if a ray is blocked
#   by another non-background pixel or exits immediately. It therefore checks the
#   input grid path and only returns a terminal when the path stays background all the
#   way to the boundary. The generator guarantees this condition for valid samples.
#
# d) placement legality:
# - Generator uses the original output-first/backtracking placement style.
# - Full hypothetical ray corridors are still reserved in `all_cells`, even though
#   they are no longer visible in the output. This guarantees:
#   - no overlaps between different diamonds/segments/ray corridors
#   - no object blocks another ray before the boundary
#   - every marker position is uniquely attributable to one ray
# - Rays of length 0 are rejected so every segment contributes visible boundary marks.
#
# e) visual observability:
# - Across examples, the same boundary-touching positions appear where the original
#   long rays would have ended, but only a single marker pixel is shown.
# - Because the marker color is constant and introduced only in the output, a human
#   can infer "special terminal mark" instead of "segment-colored extension".
#
# f) visual neatness:
# - The output is sparser than the original, not denser.
# - Objects remain separated because the hidden full ray corridors are still protected
#   during generation, so boundary marks do not sit on top of other structures.
#
# g) color-change inferability:
# - The only new color behavior is: terminal boundary points become color 9.
# - Segment colors stay segment-local and are not remapped in the output, avoiding
#   multi-branch color correspondences that would be hard to infer.
#
# post-change solver assumptions and generator guarantees:
# - Every relevant adjacent component is a clipped diagonal segment of length >= 2.
# - Every endpoint has at least one in-bounds background cell along its outward ray.
# - Every outward ray corridor remains background until the boundary.
# - Color 9 is never used by diamonds or segments, so marker semantics stay unique.

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


MARKER_COLOR = 9


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


def _terminal_boundary_cell(
    g: np.ndarray,
    bg_colors: set[int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int] | None:
    """Return the boundary cell where the ray terminates, or None if blocked / zero-length."""
    h, w = g.shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if int(g[r, c]) not in bg_colors:
            return None
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return (r, c)
        r, c = nr, nc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - infer the same outward diagonal rays as in the original task
    - do not paint the whole ray path
    - place marker color 9 only at each ray's boundary termination cell
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
        adj = []
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
                score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if score > best_score:
                    best_score = score
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            for start in (endpoints[0], endpoints[-1]):
                terminal = _terminal_boundary_cell(g, bg_colors, start, outward)
                if terminal is not None:
                    out[terminal] = MARKER_COLOR

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
EDGES_PER_DIAMOND = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = tuple(range(2, 9))
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

        ray_a = _ray_cells((h, w), segment[0], info["outward"])
        ray_b = _ray_cells((h, w), segment[-1], info["outward"])
        if not ray_a or not ray_b:
            return None

        rays = ray_a + ray_b
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "rays": rays,
                "terminals": [ray_a[-1], ray_b[-1]],
                "weight": len(segment),
            }
        )
    return {"diamond": diamond, "edges": edge_parts, "all_cells": frozenset(all_cells)}


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
    return [placement for placement in placed if placement is not None]


def _assign_segment_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[list[int]]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set and c != MARKER_COLOR]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted_edges = sorted(
        (
            (edge_part["weight"], random.random(), placement_i, edge_i)
            for placement_i, placement in enumerate(placements)
            for edge_i, edge_part in enumerate(placement["edges"])
        ),
        reverse=True,
    )
    segment_colors: list[list[int]] = [[0] * len(placement["edges"]) for placement in placements]
    for _, _, placement_i, edge_i in weighted_edges:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        segment_colors[placement_i][edge_i] = chosen
        counts[chosen] += placements[placement_i]["edges"][edge_i]["weight"]
    return segment_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((count for color, count in counts.items() if color not in (0, 1)), default=0)
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
    marker_coords: list[tuple[int, int]] = []
    for placement_i, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[placement_i])
        for edge_i, edge_part in enumerate(placement["edges"]):
            _paint_cells(out, edge_part["segment"], segment_colors[placement_i][edge_i])
            _paint_cells(out, edge_part["terminals"], MARKER_COLOR)
            marker_coords.extend(edge_part["terminals"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if marker_coords:
        rs, cs = zip(*marker_coords)
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

# Variant of 80a900e0
# Dimension: D5 — 发射源对象形状 (src)
# Change: 源为8像素的空心菱形轮廓（5x5外接矩形中的对角线位置像素）... → 十字形（加号形状）

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


# Design analysis for the D5 variant
# ----------------------------------
# spec_variant:
#   preserved:
#     - grid size, checkerboard background, color palette, object count, and output-first generation
#     - each outer marker still emits exactly two rays, one from each segment endpoint
#     - rays still stop when they hit a non-background cell or the grid boundary
#   changed_core:
#     - the source object is no longer the 8-pixel hollow diamond
#     - it is now a 9-pixel plus in a 5x5 bbox:
#         ..#..
#         ..#..
#         #####
#         ..#..
#         ..#..
#     - the four usable attachments are no longer the diamond's diagonal edges
#     - they are the plus's four arm tips: N / S / W / E
#   outer_segment_geometry:
#     - N/S arms use a horizontal 3-pixel segment one cell beyond the arm tip
#     - W/E arms use a vertical 3-pixel segment one cell beyond the arm tip
#     - when clipped by a boundary, a segment may shrink to 2 visible pixels
#   output_rule:
#     - a horizontal segment emits two vertical rays in the outward direction
#     - a vertical segment emits two horizontal rays in the outward direction
#     - therefore the four arm tips correspond to the four cardinal ray directions
#
# behavior trace on one arm:
#   original:
#     - a diagonal 3-pixel segment sits parallel to one diamond edge
#     - solver infers which of the two diagonal normals points inward to the diamond
#     - rays then travel along the opposite diagonal normal
#   variant:
#     - a top-arm segment is horizontal and sits directly above the top arm tip
#     - solver checks the two possible normals (up/down); only shifting the segment down overlaps the plus
#     - the outward direction is therefore up, so the left and right endpoints emit two upward rays
#   difference:
#     - edge-based diagonal reasoning is replaced by tip-based axial reasoning
#     - the solver can no longer rely on diagonal collinearity or diamond-edge overlap
#
# hidden assumptions audit:
#   - old _is_diamond8 assumed len==8 and exact diagonal offsets; invalid now, replaced by _is_plus9
#   - old _outer_line_info assumed diagonal line components; invalid now, replaced by row/column detection
#   - old normal choice used diagonal candidates; invalid now, replaced by cardinal candidates
#   - old generator only needed output-cell disjointness; now we also reserve a halo around each plus so a
#     foreign non-bg pixel cannot become "adjacent decoration" of the wrong plus in the input
#
# legality / observability:
#   - each training example contains 1-3 pluses, each with 2-4 occupied arms, so the direction categories repeat
#   - the plus shape is visually distinct from the original diamond and uses axis-aligned rays, making the D5
#     change observable from a few examples
#   - colors are still free non-background colors only; no rule depends on color ordering or numeric ids


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]):
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


_PLUS9_OFFSETS = {
    (0, 2),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (4, 2),
}


def _is_plus9(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 9:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _PLUS9_OFFSETS


def _outer_line_info(pts: list[tuple[int, int]]):
    # Returns (orientation, endpoints_sorted) for straight row/column segments.
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if len(set(rs)) == 1:
        endpoints = sorted(pts, key=lambda p: p[1])
        if all(endpoints[i + 1][1] - endpoints[i][1] == 1 for i in range(len(endpoints) - 1)):
            return ("row", endpoints)
        return None
    if len(set(cs)) == 1:
        endpoints = sorted(pts, key=lambda p: p[0])
        if all(endpoints[i + 1][0] - endpoints[i][0] == 1 for i in range(len(endpoints) - 1)):
            return ("col", endpoints)
        return None
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend straight rays outward from axis-aligned outer segments adjacent to plus shapes on a checkerboard.

    Input structure:
    - Checkerboard bg (two most common colors).
    - Plus-shaped sources (9 pixels in a 5x5 bbox) in non-bg colors.
    - Short horizontal/vertical segments in another non-bg color, each adjacent to one arm tip.

    Rule:
    - For each segment, infer which side faces the plus by shifting it along the candidate normal directions.
    - The outward direction is the opposite cardinal direction.
    - Cast one straight ray from each segment endpoint, painting bg cells with the segment's color until a
      non-bg cell or the grid boundary is reached.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    pluses: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_plus9(pts)]

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, ppts in sorted(pluses, key=bbox_key):
        pset = set(ppts)

        adj = []
        for r, c in ppts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in pset:
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
            orientation, endpoints = info

            if orientation == "row":
                normals = [(-1, 0), (1, 0)]
            else:
                normals = [(0, -1), (0, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                sc = sum(((r + n[0], c + n[1]) in pset) for r, c in oc)
                if sc > best_score:
                    best_score = sc
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            p0 = endpoints[0]
            p1 = endpoints[-1]
            for pr, pc in (p0, p1):
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
PLUS_COUNT_RANGE = (1, 3)
ARMS_PER_PLUS = (2, 4)
MIN_SEGMENT_LEN = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

_PLUS_COORDS = (
    (0, 2),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (4, 2),
)

_ARM_INFO = {
    "N": {"segment": ((-1, 1), (-1, 2), (-1, 3)), "outward": (-1, 0), "orientation": "row"},
    "S": {"segment": ((5, 1), (5, 2), (5, 3)), "outward": (1, 0), "orientation": "row"},
    "W": {"segment": ((1, -1), (2, -1), (3, -1)), "outward": (0, -1), "orientation": "col"},
    "E": {"segment": ((1, 5), (2, 5), (3, 5)), "outward": (0, 1), "orientation": "col"},
}
_ARM_NAMES = tuple(_ARM_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], orientation: str) -> list[tuple[int, int]]:
    if orientation == "row":
        return sorted(cells, key=lambda p: p[1])
    return sorted(cells, key=lambda p: p[0])


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


def _plus_halo(
    h: int,
    w: int,
    plus_cells: list[tuple[int, int]],
    allowed_adjacent: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    plus_set = set(plus_cells)
    for r, c in plus_cells:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in plus_set:
                halo.add((rr, cc))
    halo.difference_update(allowed_adjacent)
    return halo


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, arms: tuple[str, ...]
) -> dict[str, object] | None:
    plus = [(r0 + dr, c0 + dc) for dr, dc in _PLUS_COORDS]
    input_cells = set(plus)
    all_cells = set(plus)
    edge_parts = []

    for arm in arms:
        info = _ARM_INFO[arm]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["orientation"])
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        input_cells.update(segment)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})

    source_halo = _plus_halo(h, w, plus, input_cells)
    return {
        "plus": plus,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "input_cells": frozenset(input_cells),
        "source_halo": frozenset(source_halo),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4)]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for arms in {specs[i] for i in order}:
        positions = toplefts[:]
        random.shuffle(positions)
        candidate_cache[arms] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, arms)) is not None
        ]

    occupied_all: set[tuple[int, int]] = set()
    occupied_input: set[tuple[int, int]] = set()
    occupied_halo: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            all_cells = placement["all_cells"]
            input_cells = placement["input_cells"]
            source_halo = placement["source_halo"]
            if not occupied_all.isdisjoint(all_cells):
                continue
            if not occupied_input.isdisjoint(source_halo):
                continue
            if not occupied_halo.isdisjoint(input_cells):
                continue
            occupied_all.update(all_cells)
            occupied_input.update(input_cells)
            occupied_halo.update(source_halo)
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied_all.difference_update(all_cells)
            occupied_input.difference_update(input_cells)
            occupied_halo.difference_update(source_halo)
            placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_segment_colors(placements: list[dict[str, object]], plus_colors: list[int]) -> list[list[int]]:
    plus_set = set(plus_colors)
    palette = [c for c in COLOR_RANGE if c not in plus_set]
    counts = Counter({c: 0 for c in palette})
    for c in plus_colors:
        counts[c] += len(_PLUS_COORDS)

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
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    plus_count = random.randint(*PLUS_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_ARM_NAMES, random.randint(*ARMS_PER_PLUS))))
        for _ in range(plus_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    plus_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, plus_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["plus"], plus_colors[pi])
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

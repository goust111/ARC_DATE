# Variant of 80a900e0
# Dimension: D5 — 发射源对象形状 (src)
# Change: 源为8像素的空心菱形轮廓（5x5外接矩形中的对角线位置像素）... → 空心正方形轮廓

# Design analysis for the D5 variant
# ----------------------------------
# spec:
#   grid: keep the original 20-24 x 14-30 checkerboard canvas
#   background: unchanged; bg[r,c] = (r+c+1)%2 with colors 0/1
#   source:
#     old: 8-pixel hollow diamond in a 5x5 bbox
#     new: 16-pixel hollow square ring in a 5x5 bbox, axis-aligned
#   outer_segment:
#     old: 3-pixel diagonal segment outside one diamond edge
#     new: 3-pixel horizontal/vertical segment centered outside one square side
#     sides: T / L / B / R
#   ray_rule:
#     old: from the two diagonal-segment endpoints, cast diagonal rays along the outward normal
#     new: from the two axis-aligned segment endpoints, cast straight rays along the outward normal
#          horizontal segment -> vertical rays; vertical segment -> horizontal rays
#
# Behavior trace on one edge:
#   original top-right diamond edge:
#     diagonal segment / / / sits one step outside the diamond edge, and solver compares the
#     two diagonal normals to see which side points back into the diamond. Rays then go in the
#     opposite diagonal direction.
#   variant top square side:
#     a centered horizontal segment sits at row r0-1 above the square's top edge. The inward
#     normal is (+1, 0) because shifting the segment one step downward overlaps the square's
#     top side. Therefore the outward direction is (-1, 0), and the segment's two endpoints
#     emit two upward vertical rays.
#   key difference:
#     shape detection changes from diamond-outline recognition to square-ring recognition, and
#     the outer-line recognizer must accept horizontal/vertical segments instead of diagonals.
#
# Complexity calibration:
#   - The square ring uses 16 pixels, which is visually distinct from the original 8-pixel diamond.
#   - Outer segments stay short (3 pixels) so adjacent sides do not merge under 8-connectivity.
#   - Each square still chooses only 2-4 sides, preserving repeated local evidence without
#     introducing many branch types.
#
# Implicit-assumption audit of the original main():
#   - _is_diamond8 assumed len==8 and a diagonal-only 5x5 mask. This fails completely for squares,
#     so a dedicated square-ring recognizer is required.
#   - _outer_line_info assumed every outer component is diagonal-collinear. In the variant the
#     components are horizontal or vertical, so orientation parsing must be rewritten.
#   - The "best inward normal" scoring logic is still valid, but only if normals are taken from
#     the axis pair compatible with the segment orientation.
#   - Adjacent non-bg pixels are grouped with 8-connectivity. For squares, side segments would
#     incorrectly merge at corners if they touched there, so the generator centers each 3-pixel
#     segment and keeps a one-cell gap from neighboring sides.
#
# Placement legality:
#   - Generator constructs output-first exactly as before.
#   - Each placement stores all occupied output cells and a protection zone around the square ring.
#     Other objects are forbidden from entering that zone so the solver never mistakes a foreign
#     ray/segment for a square's own adjacent border.
#   - Every chosen side must generate at least one ray pixel, avoiding degenerate examples where
#     input and output would be identical for that side.
#
# Visual observability:
#   - Humans can now see a 5x5 hollow square instead of a diamond.
#   - Top/bottom segments clearly yield vertical rays; left/right segments yield horizontal rays.
#   - Because all selected sides are visible in the output, the changed geometry is inferable from
#     a few examples rather than hidden behind boundary-clipped no-op segments.

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


_SQUARE16_OFFSETS = {
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 4),
    (2, 0), (2, 4),
    (3, 0), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
}


def _is_square_ring16(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 16:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _SQUARE16_OFFSETS


def _outer_line_info(pts: list[tuple[int, int]]):
    # Returns (orientation, endpoints_sorted) or None if not horizontal/vertical collinear.
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all(r == rs[0] for r in rs):
        return ("horizontal", sorted(pts, key=lambda p: (p[1], p[0])))
    if all(c == cs[0] for c in cs):
        return ("vertical", sorted(pts, key=lambda p: (p[0], p[1])))
    return None


def _adjacent_components_by_color(
    g: np.ndarray,
    adj_cells: list[tuple[int, int]],
    bg_colors: set[int],
) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    for color in sorted({int(g[r, c]) for r, c in adj_cells if int(g[r, c]) not in bg_colors}):
        mask = np.zeros_like(g, dtype=bool)
        for r, c in adj_cells:
            if int(g[r, c]) == color:
                mask[r, c] = True
        comps.extend(_cc8(mask))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend straight rays outward from short axis-aligned border segments adjacent to hollow square rings.

    Input structure:
    - Checkerboard bg (two most common colors).
    - Hollow 5x5 square rings (16 pixels) in non-bg colors.
    - Short horizontal/vertical segments in other non-bg colors adjacent to square sides.

    Rule:
    - For each adjacent segment, find which normal points back into the square ring.
    - Emit rays from the two segment endpoints in the opposite direction until a non-bg pixel or
      the grid boundary is reached.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    squares: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_square_ring16(pts)]

    out = g.copy()

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, spts in sorted(squares, key=bbox_key):
        sset = set(spts)
        adj = []
        for r, c in spts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in sset:
                    if int(g[rr, cc]) not in bg_colors:
                        adj.append((rr, cc))

        if not adj:
            continue

        outer_comps = _adjacent_components_by_color(g, adj, bg_colors)

        for oc in outer_comps:
            ocol = int(g[oc[0][0], oc[0][1]])
            info = _outer_line_info(oc)
            if info is None:
                continue
            orientation, endpoints = info

            normals = [(-1, 0), (1, 0)] if orientation == "horizontal" else [(0, -1), (0, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                sc = sum(((r + n[0], c + n[1]) in sset) for r, c in oc)
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


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
SQUARE_COUNT_RANGE = (1, 3)
EDGES_PER_SQUARE = (2, 4)
MIN_SEGMENT_LEN = 3
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

_SQUARE_COORDS = tuple(sorted(_SQUARE16_OFFSETS))

_EDGE_INFO = {
    "T": {"segment": ((-1, 1), (-1, 2), (-1, 3)), "outward": (-1, 0), "axis": "h"},
    "L": {"segment": ((1, -1), (2, -1), (3, -1)), "outward": (0, -1), "axis": "v"},
    "B": {"segment": ((5, 1), (5, 2), (5, 3)), "outward": (1, 0), "axis": "h"},
    "R": {"segment": ((1, 5), (2, 5), (3, 5)), "outward": (0, 1), "axis": "v"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], axis: str) -> list[tuple[int, int]]:
    if axis == "h":
        return sorted(cells, key=lambda p: (p[1], p[0]))
    return sorted(cells, key=lambda p: (p[0], p[1]))


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


def _protection_zone(square: list[tuple[int, int]], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    zone = set(square)
    for r, c in square:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                zone.add((rr, cc))
    return frozenset(zone)


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]
) -> dict[str, object] | None:
    square = [(r0 + dr, c0 + dc) for dr, dc in _SQUARE_COORDS]
    all_cells = set(square)
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
        segment = _sort_segment(segment, info["axis"])
        rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells((h, w), segment[-1], info["outward"])
        if not rays:
            return None
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append({"segment": segment, "rays": rays, "weight": len(part_cells)})
    return {
        "square": square,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "protected": _protection_zone(square, (h, w)),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4)]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = toplefts[:]
        random.shuffle(positions)
        candidate_cache[edges] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edges)) is not None
        ]

    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int, occupied: frozenset[tuple[int, int]], protected: frozenset[tuple[int, int]]) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            guard = placement["protected"]
            if occupied.isdisjoint(cells) and protected.isdisjoint(cells) and occupied.isdisjoint(guard):
                placed[spec_i] = placement
                if backtrack(i + 1, occupied | cells, protected | guard):
                    return True
                placed[spec_i] = None
        return False

    if not backtrack(0, frozenset(), frozenset()):
        return None
    return [p for p in placed if p is not None]


def _assign_segment_colors(placements: list[dict[str, object]], square_colors: list[int]) -> list[list[int]]:
    square_set = set(square_colors)
    palette = [c for c in COLOR_RANGE if c not in square_set]
    counts = Counter({c: 0 for c in palette})
    for c in square_colors:
        counts[c] += len(_SQUARE_COORDS)

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
    square_count = random.randint(*SQUARE_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_EDGE_NAMES, random.randint(*EDGES_PER_SQUARE))))
        for _ in range(square_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    square_colors = random.sample(list(COLOR_RANGE), len(placements))
    segment_colors = _assign_segment_colors(placements, square_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["square"], square_colors[pi])
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
        rs, cs = np.array(rs), np.array(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")

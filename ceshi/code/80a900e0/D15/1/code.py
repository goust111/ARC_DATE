# Variant of 80a900e0
# Dimension: D15 — 菱形颜色与外边框颜色的关系 (meta)
# Change: 菱形和外边框使用不同的非背景色，同一菱形的多条外边框可以共享颜色... → 外边框颜色由所在边位置决定
#
# Design analysis
# ===============
# spec_variant:
#   background:
#     same_as_source: 0/1 checkerboard, top-left=1, all foreground stays on even parity cells
#   objects:
#     diamond:
#       same_as_source: 5x5 hollow diamond outline with 8 pixels
#       color: chosen from a pool disjoint from the outer-segment palette
#     outer_segment:
#       same_geometry: 3-pixel diagonal segment (2 if clipped), one step outside one diamond edge
#       new_color_rule:
#         TR: fixed color 2
#         TL: fixed color 8
#         BL: fixed color 4
#         BR: fixed color 6
#       note: this mapping is global for the whole task; it is not re-sampled per example
#   output_rule:
#     same_geometry: each visible outer segment emits two diagonal rays from its endpoints
#     new_semantics: ray color comes from the globally inferred edge-position mapping
#
# a) Concrete behavior trace:
#   Original logic reads one outer component, takes its own pixel color, and extends rays with that color.
#   Variant inputs still contain the same segment geometry, but now the segment color means "which side of the
#   diamond this segment belongs to". Example: if two diamonds both expose their TR edge, both TR segments are
#   red(2) even when the two diamonds have different outline colors. The solver therefore first classifies each
#   segment as TR/TL/BL/BR from slope + inward/outward normal, aggregates the observed global edge->color map,
#   and then paints rays using that map. On valid inputs this equals the segment's own color, but the solver is
#   now explicitly aware that color is position-derived rather than diamond-specific.
#
# b) Complexity calibration:
#   The new rule only works if humans can see repetition. To make D15 observable, generated examples use 2-3
#   diamonds and 3-4 occupied edges per diamond, and the union of edges always covers all four positions. This
#   keeps the task close to the original geometry while ensuring repeated support for the fixed mapping.
#
# c) Implicit assumptions audit for main():
#   - Background detection via the two most common colors still works only if 0 and 1 dominate; generator checks this.
#   - Diamond detection by shape stays valid because diamond colors remain distinct and never equal segment colors.
#   - The source solver assumed `ocol = component color = ray color`. That assumption is exactly what D15 changes.
#     The new solver replaces it with `edge_name -> majority observed color`, then uses that mapping to paint rays.
#   - Normal-direction inference still depends only on geometry, which is unchanged.
#
# d) Placement legality:
#   Generation remains output-first and keeps the original non-overlap rule on diamond pixels, segments, and full
#   ray footprints. Because repeated edge colors now appear across diamonds, the generator also verifies that the
#   solver can still separate local adjacent components around each diamond.
#
# e) Visual observability:
#   Fixed edge colors are globally constant and reused across all examples. Each generated example contains at
#   least two diamonds, all four edge positions appear somewhere, and multiple positions repeat across diamonds.
#   This gives the human solver visible evidence that "same side => same color".
#
# f) Visual cleanliness:
#   Geometry is unchanged and all placements remain disjoint, so the image stays as readable as the source task.
#   The only added structure is consistent color reuse across matching edge positions.
#
# g) Color-rule inferability:
#   Exactly four segment colors are used, one per edge position. Diamond colors come from a disjoint palette so
#   the observer is not forced to disentangle "segment color copied from diamond" from "segment color fixed by side".

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


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


_EDGE_TO_COLOR = {
    "TR": 2,
    "TL": 8,
    "BL": 4,
    "BR": 6,
}

_EDGE_FROM_ORIENTATION = {
    ("rc_diff", (-1, 1)): "TR",
    ("rc_sum", (-1, -1)): "TL",
    ("rc_diff", (1, -1)): "BL",
    ("rc_sum", (1, 1)): "BR",
}


def _classify_outer_component(
    g: np.ndarray,
    comp: list[tuple[int, int]],
    dset: set[tuple[int, int]],
) -> dict[str, object] | None:
    info = _outer_line_info(comp)
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
        score = sum(((r + n[0], c + n[1]) in dset) for r, c in comp)
        if score > best_score:
            best_score = score
            best_n = n
    if best_n is None or best_score <= 0:
        return None

    outward = (-best_n[0], -best_n[1])
    edge_name = _EDGE_FROM_ORIENTATION.get((slope_key, outward))
    if edge_name is None:
        return None

    return {
        "edge_name": edge_name,
        "outward": outward,
        "endpoints": (endpoints[0], endpoints[-1]),
        "observed_color": int(g[comp[0][0], comp[0][1]]),
    }


def _extract_structure(
    g: np.ndarray,
) -> tuple[set[int], list[tuple[int, list[tuple[int, int]]]], list[dict[str, object]]]:
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    outer_parts: list[dict[str, object]] = []
    for diamond_index, (_, dpts) in enumerate(sorted(diamonds, key=bbox_key)):
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
        for comp in _cc8(adj_mask):
            part = _classify_outer_component(g, comp, dset)
            if part is None:
                continue
            part["diamond_index"] = diamond_index
            outer_parts.append(part)

    return bg_colors, diamonds, outer_parts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from each outer segment.

    Variant-specific detail:
    - Outer-segment colors are globally tied to edge positions TR/TL/BL/BR.
    - The solver infers this edge->color mapping from all visible outer segments, then uses it to color rays.
    """
    g = np.asarray(grid, dtype=int)
    bg_colors, _diamonds, outer_parts = _extract_structure(g)
    out = g.copy()

    edge_color_votes: dict[str, Counter[int]] = {edge: Counter() for edge in _EDGE_TO_COLOR}
    for part in outer_parts:
        edge_color_votes[part["edge_name"]].update([part["observed_color"]])
    edge_to_color = {
        edge: int(votes.most_common(1)[0][0]) if votes else _EDGE_TO_COLOR[edge]
        for edge, votes in edge_color_votes.items()
    }

    for part in outer_parts:
        color = edge_to_color[part["edge_name"]]
        dr, dc = part["outward"]
        for pr, pc in part["endpoints"]:
            rr, cc = pr + dr, pc + dc
            while 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                if int(out[rr, cc]) in bg_colors:
                    out[rr, cc] = color
                else:
                    break
                rr += dr
                cc += dc

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400
MIN_SEGMENT_LEN = 2

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
_SEGMENT_COLORS = set(_EDGE_TO_COLOR.values())
_DIAMOND_COLOR_POOL = tuple(c for c in COLOR_RANGE if c not in _SEGMENT_COLORS)


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
        edge_parts.append({"edge_name": edge, "segment": segment, "rays": rays, "weight": len(part_cells)})
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


def _choose_specs(diamond_count: int) -> list[tuple[str, ...]]:
    for _ in range(200):
        specs = [
            tuple(sorted(random.sample(_EDGE_NAMES, random.randint(3, 4))))
            for _ in range(diamond_count)
        ]
        coverage = set().union(*map(set, specs))
        repeated_edges = sum(sum(edge in spec for spec in specs) >= 2 for edge in _EDGE_NAMES)
        if coverage == set(_EDGE_NAMES) and repeated_edges >= 2:
            return specs

    specs = [
        ("BL", "BR", "TL"),
        ("BL", "BR", "TR"),
        ("TL", "TR", "BR"),
    ]
    return specs[:diamond_count]


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _variant_observability_ok(inp: np.ndarray) -> bool:
    _bg_colors, diamonds, outer_parts = _extract_structure(inp)
    if len(diamonds) < 2:
        return False
    if not outer_parts:
        return False

    coverage = {part["edge_name"] for part in outer_parts}
    if coverage != set(_EDGE_NAMES):
        return False

    for part in outer_parts:
        if part["observed_color"] != _EDGE_TO_COLOR[part["edge_name"]]:
            return False

    repeated_edges = sum(sum(part["edge_name"] == edge for part in outer_parts) >= 2 for edge in _EDGE_NAMES)
    return repeated_edges >= 2


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.choice((2, 2, 3))
    specs = _choose_specs(diamond_count)

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(_DIAMOND_COLOR_POOL), len(placements))

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[pi])
        for edge_part in placement["edges"]:
            color = _EDGE_TO_COLOR[edge_part["edge_name"]]
            _paint_cells(out, edge_part["segment"], color)
            _paint_cells(out, edge_part["rays"], color)
            ray_coords.extend(edge_part["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs = np.asarray(rs)
        cs = np.asarray(cs)
        inp[rs, cs] = (rs + cs + 1) % 2

    if not _variant_observability_ok(inp):
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

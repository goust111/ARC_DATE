# Variant of 80a900e0
# Dimension: D2 — 射线发射点 (path)
# Change: 从外边框线段的两个端点发射，每条线段恰好发射2条射线... → 从菱形顶点直接发射
#
# Design analysis
# ===============
# source_task:
#   none
#
# variant_spec:
#   background:
#     unchanged: 0/1 checkerboard, top-left = 1
#   diamond:
#     unchanged: 8-pixel hollow diamond in a 5x5 bbox
#   outer_segment:
#     unchanged visually: diagonal 3-pixel guide segment adjacent to one diamond edge
#     role changed:
#       it no longer provides the ray start cells
#       it only identifies one activated diamond vertex, the ray direction, and the ray color
#   vertex_activation_rule:
#     We traverse the diamond clockwise using vertices T->R->B->L->T.
#     Each occupied edge activates the start vertex of that clockwise edge:
#       TR -> T, BR -> R, BL -> B, TL -> L
#     The ray direction is the outward normal of that edge:
#       TR -> NE, BR -> SE, BL -> SW, TL -> NW
#   output_rule:
#     For each outer segment, start from the activated diamond vertex.
#     Step along that edge's outward diagonal.
#     If the first stepped cell(s) belong to the same guide segment, skip over them:
#       the guide segment is a marker, not a blocker.
#     After leaving the marker, paint background cells with the segment color
#     until the grid boundary or another non-background pixel is reached.
#
# behavior_trace:
#   original_local_example:
#     TR segment outside a diamond emitted NE rays from both segment endpoints.
#     Those starts were outside the diamond already, so painting began immediately.
#   variant_local_example:
#     The same TR segment activates the top diamond vertex instead.
#     The first NE step lands on one segment cell; solver skips that marker cell and
#     continues painting from the next background cell onward.
#   visible_difference:
#     each segment now yields one vertex-anchored ray instead of two endpoint rays
#     the source is a diamond tip, so the topological relation is different and
#     the output geometry is visibly sparser than the original task
#
# complexity_calibration:
#   - We keep only 4 edge types and 4 corresponding vertex-direction cases.
#   - Each segment still has a strong spatial cue because it stays adjacent to one edge.
#   - To keep the new behavior visible, the generator requires every activated edge to
#     have at least one painted ray cell beyond the guide segment.
#
# implicit_assumption_audit:
#   - Original main() assumed "segment endpoints are the emission points".
#     That assumption is removed.
#   - Original main() used segment endpoints directly after determining outward normal.
#     In the variant, the segment is first mapped to one of the four diamond vertices.
#   - Original painting stopped on any non-bg cell, which would immediately block a
#     vertex-fired ray because the adjacent guide segment is non-bg.
#     The variant therefore explicitly skips cells belonging to the same outer segment
#     before it starts painting.
#   - Original code only needed the outward normal. The variant additionally needs the
#     diamond's 4 canonical vertices from the diamond bbox; generator guarantees every
#     detected diamond is still the exact 5x5 / 8-pixel template.
#
# placement_legality:
#   - Generator remains output-first.
#   - It places diamonds plus guide segments first conceptually, then computes the
#     vertex-fired ray cells for each occupied edge.
#   - Placements are rejected if any ray overlaps another object or another ray.
#   - Each occupied edge must produce at least one actual painted ray cell, otherwise
#     the sample is rejected as visually weak for this variant.
#
# visual_observability:
#   - The new rule is visible because the number and anchor points of rays change.
#   - Segments remain on the edges, while the extrapolated colored paths clearly line up
#     with a single diamond tip rather than two segment endpoints.
#   - The mapping is low-entropy: 4 edge positions -> 4 tip directions.
#
# cleanliness:
#   - No extra markers are introduced.
#   - Rays are kept disjoint from all non-background objects during generation.
#
# color_inference:
#   - Colors behave exactly as before: a guide segment's color is copied onto its ray.
#   - No hidden ordering by color id is used.

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

_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_VERTEX_OFFSETS = {
    "T": (0, 2),
    "R": (2, 4),
    "B": (4, 2),
    "L": (2, 0),
}

_OUTWARD_TO_ACTIVE_VERTEX = {
    (-1, 1): "T",   # TR edge
    (1, 1): "R",    # BR edge
    (1, -1): "B",   # BL edge
    (-1, -1): "L",  # TL edge
}

_OUTWARD_TO_TOUCHED_VERTICES = {
    (-1, 1): {"T", "R"},
    (1, 1): {"R", "B"},
    (1, -1): {"B", "L"},
    (-1, -1): {"L", "T"},
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


def _diamond_vertices(pts: list[tuple[int, int]]) -> dict[str, tuple[int, int]]:
    r0, c0, _, _ = _bbox(pts)
    return {name: (r0 + dr, c0 + dc) for name, (dr, dc) in _VERTEX_OFFSETS.items()}


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

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)
        vertices = _diamond_vertices(dpts)

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
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            ocol = int(g[oc[0][0], oc[0][1]])
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, _ = info

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
            active_name = _OUTWARD_TO_ACTIVE_VERTEX.get(outward)
            if active_name is None:
                continue

            shifted = {(r + best_n[0], c + best_n[1]) for r, c in oc}
            touched_vertices = {name for name, coord in vertices.items() if coord in shifted}
            if active_name not in touched_vertices:
                continue
            if not touched_vertices.issubset(_OUTWARD_TO_TOUCHED_VERTICES[outward]):
                continue

            vr, vc = vertices[active_name]
            ocset = set(oc)
            rr, cc = vr + outward[0], vc + outward[1]

            # The adjacent guide segment is a marker, not an obstacle.
            while 0 <= rr < h and 0 <= cc < w and (rr, cc) in ocset:
                rr += outward[0]
                cc += outward[1]

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
MIN_RAY_LEN = 1
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 300

_EDGE_INFO = {
    "TR": {
        "segment": ((-1, 3), (0, 4), (1, 5)),
        "outward": (-1, 1),
        "slope": "diff",
        "active_vertex": (0, 2),
    },
    "TL": {
        "segment": ((-1, 1), (0, 0), (1, -1)),
        "outward": (-1, -1),
        "slope": "sum",
        "active_vertex": (2, 0),
    },
    "BL": {
        "segment": ((3, -1), (4, 0), (5, 1)),
        "outward": (1, -1),
        "slope": "diff",
        "active_vertex": (4, 2),
    },
    "BR": {
        "segment": ((3, 5), (4, 4), (5, 3)),
        "outward": (1, 1),
        "slope": "sum",
        "active_vertex": (2, 4),
    },
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


def _ray_cells_from_vertex(
    shape: tuple[int, int],
    source: tuple[int, int],
    direction: tuple[int, int],
    guide_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = source[0] + dr, source[1] + dc
    while 0 <= r < h and 0 <= c < w and (r, c) in guide_cells:
        r += dr
        c += dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _halo8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


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
        guide_cells = set(segment)
        source = (r0 + info["active_vertex"][0], c0 + info["active_vertex"][1])
        rays = _ray_cells_from_vertex((h, w), source, info["outward"], guide_cells)
        if len(rays) < MIN_RAY_LEN:
            return None
        part_cells = segment + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        edge_parts.append(
            {
                "segment": segment,
                "rays": rays,
                "source": source,
                "weight": len(part_cells),
            }
        )

    return {
        "diamond": diamond,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "halo_cells": frozenset(_halo8(all_cells, (h, w))),
    }


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
    halo_counts: Counter[tuple[int, int]] = Counter()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            cells = placement["all_cells"]
            if occupied.isdisjoint(cells) and all(halo_counts[cell] == 0 for cell in cells):
                occupied.update(cells)
                for cell in placement["halo_cells"]:
                    halo_counts[cell] += 1
                placed[spec_i] = placement
                if backtrack(i + 1):
                    return True
                occupied.difference_update(cells)
                for cell in placement["halo_cells"]:
                    halo_counts[cell] -= 1
                    if halo_counts[cell] == 0:
                        del halo_counts[cell]
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
        (
            (ep["weight"], random.random(), pi, ei)
            for pi, pl in enumerate(placements)
            for ei, ep in enumerate(pl["edges"])
        ),
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

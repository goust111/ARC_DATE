# Combo variant of 80a900e0 (index 21)
# combo_spec: D5/0+D6/1+D14/1
# Name: 方形L标记条件激活
#
# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#   - Original task: a 5x5 hollow diamond has one short diagonal outer segment per chosen edge.
#     Each segment emits two diagonal rays from its endpoints, and every valid outer segment fires.
#   - D5 changes the source object to a 5x5 hollow square ring, so the recognizer can no longer rely
#     on the 8-pixel diamond mask. The source is now the 16-pixel square border.
#   - D6 changes the outer object from a collinear segment into a 3-pixel L marker. Direction is no
#     longer read from line slope; instead the marker bend and two tips determine a diagonal outward
#     vector. For example, the top-right marker cells {(r0-2,c0+4),(r0-1,c0+4),(r0-1,c0+5)} have
#     bend (r0-1,c0+4), tips above/right of the bend, and therefore outward direction (-1,+1).
#   - D14 is resolved here as a fixed orientation filter: only markers whose inferred outward row
#     delta is negative are active. So TL/TR markers above the square emit rays; BL/BR markers below
#     the square remain visible but do not extend.
#   - Net effect on one square carrying TR and BR markers:
#       * input: both markers are present near the two right-side corners;
#       * output: the TR marker grows two NE rays from its tips, while the BR marker stays unchanged.
#
# b) Complexity calibration:
#   - The square ring is visually distinct from the original diamond, so the source-shape change is
#     obvious from a couple of examples.
#   - Marker orientation still has only four discrete cases (TL/TR/BL/BR), keeping direction
#     inference learnable.
#   - Activation has only two observable classes, active-upward vs inactive-downward. The generator
#     forces both classes to appear in every object, so the conditioning signal is repeated locally.
#
# c) Implicit-assumption audit:
#   - Original _is_diamond8 assumes len==8 and a diagonal-only 5x5 pattern; invalid for this combo.
#   - Original _outer_line_info assumes outer components are collinear diagonal segments; invalid once
#     the outer objects become L markers.
#   - Original "collect pixels adjacent to the source and split them" logic is brittle here because a
#     square-corner L marker is not a single diagonal segment and may sit partly two steps away from
#     some square pixels. The combo solver instead detects all non-bg components globally, recognizes
#     squares and Ls by shape, then assigns each L to the unique square with the best 8-neighbor score.
#   - Original solver assumes every valid outer object fires. The combo solver adds an explicit
#     activation gate `outward[0] < 0`.
#
# d) Placement legality:
#   - Generator stays output-first.
#   - Each square placement stores: square cells, chosen marker cells, emitted ray cells for active
#     markers, and a protection halo around the input-time objects. Another square's input objects may
#     not enter that halo, preventing ambiguous ownership or same-color component merging in the input.
#   - Active markers must contribute at least a small positive ray length; otherwise the sample is
#     retried because the activation effect would be too weak to see.
#
# e) Visual observability:
#   - A human can see three simultaneous clues directly in the pixels: source is a hollow square, the
#     short outer glyphs are L-shaped instead of straight, and only the upward-facing Ls elongate.
#   - Activation does not depend on hidden metadata such as color id ordering. It depends only on the
#     visible bend orientation of each L.
#
# f) Visual cleanliness:
#   - Markers sit near the four square corners, which keeps the local motif compact and readable.
#   - Backtracking rejects overlaps between squares, markers, and real output rays; the protection halo
#     keeps different input objects from crowding into the same neighborhood.
#
# g) Color inference:
#   - Rays always reuse their own marker color, matching the original task's color logic.
#   - Colors carry no extra meaning for activation, so the new rule remains shape/orientation based.
#
# New solver assumptions checked against the generator:
#   - Square sources are exactly 16-pixel 5x5 hollow rings.
#   - Markers are exactly 3-cell 4-connected Ls in a 2x2 box.
#   - Each marker has one unique best square by 8-neighbor score because placements are separated by a
#     protection halo.
#   - Only top-facing markers are expected to fire, and the generator always includes both firing and
#     non-firing markers so the combo behavior is observable rather than degenerate.

from __future__ import annotations

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


_SQUARE16_OFFSETS = {
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 0), (1, 4),
    (2, 0), (2, 4),
    (3, 0), (3, 4),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
}

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _l_marker_info(pts: list[tuple[int, int]]) -> dict[str, object] | None:
    if len(pts) != 3:
        return None

    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 1 or (c1 - c0) != 1:
        return None

    norm = {(r - r0, c - c0) for r, c in pts}
    if len(norm) != 3 or any(p not in {(0, 0), (0, 1), (1, 0), (1, 1)} for p in norm):
        return None

    pset = set(pts)
    deg = {p: sum((p[0] + dr, p[1] + dc) in pset for dr, dc in _DIR4) for p in pset}
    bends = [p for p, d in deg.items() if d == 2]
    tips = [p for p, d in deg.items() if d == 1]
    if len(bends) != 1 or len(tips) != 2:
        return None

    bend = bends[0]
    arm_vecs = [(tr - bend[0], tc - bend[1]) for tr, tc in tips]
    if sorted(abs(dr) + abs(dc) for dr, dc in arm_vecs) != [1, 1]:
        return None
    if arm_vecs[0][0] * arm_vecs[1][0] + arm_vecs[0][1] * arm_vecs[1][1] != 0:
        return None

    outward = (arm_vecs[0][0] + arm_vecs[1][0], arm_vecs[0][1] + arm_vecs[1][1])
    if outward[0] == 0 or outward[1] == 0:
        return None

    return {"bend": bend, "tips": sorted(tips), "outward": outward}


def _adjacency_score(pts: list[tuple[int, int]], square_set: set[tuple[int, int]]) -> int:
    score = 0
    for r, c in pts:
        for rr, cc in neighbors8(r, c):
            if (rr, cc) in square_set:
                score += 1
    return score


def _marker_is_active(info: dict[str, object]) -> bool:
    outward = info["outward"]
    return int(outward[0]) < 0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from upward-facing L markers attached to hollow square rings on a checkerboard.

    Input structure:
    - Checkerboard background given by the two most common colors.
    - Hollow 5x5 square rings (16 pixels) in non-background colors.
    - Short 3-pixel L markers near square corners in other non-background colors.

    Rule:
    - Recognize each L marker by its bend and two arm tips, then infer its diagonal outward direction
      as the sum of the two unit bend-to-tip vectors.
    - Assign each marker to the unique neighboring square, but only let markers with upward outward
      direction emit rays. From each active tip, cast a diagonal ray in the marker color until a
      non-background pixel or the grid boundary stops it.

    Key insight:
    - Direction no longer comes from a line segment slope. The L shape itself encodes the ray
      direction, and activation is a second filter based only on whether that direction points upward.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)

    squares: list[tuple[int, list[tuple[int, int]]]] = []
    markers: list[tuple[int, list[tuple[int, int]], dict[str, object]]] = []
    for col, pts in comps:
        if _is_square_ring16(pts):
            squares.append((col, pts))
            continue
        info = _l_marker_info(pts)
        if info is not None:
            markers.append((col, pts, info))

    out = g.copy()

    def bbox_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
        _, pts = item
        return _bbox(pts)

    sorted_squares = sorted(squares, key=bbox_key)
    square_sets = [set(pts) for _, pts in sorted_squares]
    assignments: list[list[tuple[int, dict[str, object]]]] = [[] for _ in sorted_squares]

    for mcol, pts, info in markers:
        scores = [_adjacency_score(pts, sset) for sset in square_sets]
        best = max(scores, default=0)
        if best <= 0 or scores.count(best) != 1:
            continue
        assignments[scores.index(best)].append((mcol, info))

    for si in range(len(sorted_squares)):
        for mcol, info in assignments[si]:
            if not _marker_is_active(info):
                continue
            dr, dc = info["outward"]
            for sr, sc in info["tips"]:
                rr, cc = sr + dr, sc + dc
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = mcol
                    else:
                        break
                    rr += dr
                    cc += dc

    return out


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
SQUARE_COUNT_RANGE = (1, 3)
ACTIVE_PER_SQUARE = (1, 2)
INACTIVE_PER_SQUARE = (1, 2)
MIN_ACTIVE_RAY_CELLS = 2
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 500

_SQUARE_COORDS = tuple(sorted(_SQUARE16_OFFSETS))
_ACTIVE_EDGE_NAMES = ("TL", "TR")
_INACTIVE_EDGE_NAMES = ("BL", "BR")

_EDGE_INFO = {
    "TL": {"marker": ((-2, 0), (-1, -1), (-1, 0))},
    "TR": {"marker": ((-2, 4), (-1, 4), (-1, 5))},
    "BL": {"marker": ((4, -2), (4, -1), (5, -1))},
    "BR": {"marker": ((4, 5), (4, 6), (5, 5))},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


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


def _protection_zone(cells: list[tuple[int, int]], shape: tuple[int, int]) -> frozenset[tuple[int, int]]:
    h, w = shape
    zone = set(cells)
    frontier = set(cells)
    for _ in range(1):
        next_frontier: set[tuple[int, int]] = set()
        for r, c in frontier:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in zone:
                    zone.add((rr, cc))
                    next_frontier.add((rr, cc))
        frontier = next_frontier
    return frozenset(zone)


def _sample_specs(square_count: int) -> list[tuple[str, ...]]:
    specs: list[tuple[str, ...]] = []
    for _ in range(square_count):
        chosen = set(random.sample(list(_ACTIVE_EDGE_NAMES), random.randint(*ACTIVE_PER_SQUARE)))
        chosen.update(random.sample(list(_INACTIVE_EDGE_NAMES), random.randint(*INACTIVE_PER_SQUARE)))
        specs.append(tuple(sorted(chosen)))
    return specs


def _candidate_placement(h: int, w: int, r0: int, c0: int, edges: tuple[str, ...]) -> dict[str, object] | None:
    square = [(r0 + dr, c0 + dc) for dr, dc in _SQUARE_COORDS]
    if not all(0 <= r < h and 0 <= c < w for r, c in square):
        return None

    all_cells = set(square)
    input_cells = list(square)
    edge_parts = []
    active_ray_total = 0

    for edge in edges:
        marker = [(r0 + dr, c0 + dc) for dr, dc in _EDGE_INFO[edge]["marker"]]
        if not all(0 <= r < h and 0 <= c < w for r, c in marker):
            return None

        info = _l_marker_info(marker)
        if info is None:
            return None

        rays: list[tuple[int, int]] = []
        active = _marker_is_active(info)
        if active:
            rays = _ray_cells((h, w), info["tips"][0], info["outward"]) + _ray_cells(
                (h, w), info["tips"][-1], info["outward"]
            )
            if len(rays) < MIN_ACTIVE_RAY_CELLS:
                return None
            active_ray_total += len(rays)

        part_cells = marker + rays
        if len(set(part_cells)) != len(part_cells):
            return None
        if any(cell in all_cells for cell in part_cells):
            return None

        all_cells.update(part_cells)
        input_cells.extend(marker)
        edge_parts.append(
            {
                "name": edge,
                "marker": marker,
                "rays": rays,
                "active": active,
                "weight": len(part_cells),
            }
        )

    if active_ray_total == 0:
        return None

    return {
        "square": square,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "input_cells": tuple(input_cells),
        "protected": _protection_zone(input_cells, (h, w)),
    }


def _search_placements(h: int, w: int, specs: list[tuple[str, ...]]) -> list[dict[str, object]] | None:
    order = sorted(range(len(specs)), key=lambda i: len(specs[i]), reverse=True)
    toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4)]

    candidate_cache: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for edges in {specs[i] for i in order}:
        positions = toplefts[:]
        random.shuffle(positions)
        candidates = []
        for r0, c0 in positions:
            placement = _candidate_placement(h, w, r0, c0, edges)
            if placement is not None:
                candidates.append(placement)
        random.shuffle(candidates)
        candidate_cache[edges] = candidates

    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(
        i: int,
        occupied_output: frozenset[tuple[int, int]],
        protected_input: frozenset[tuple[int, int]],
    ) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            out_cells = placement["all_cells"]
            input_cells = placement["input_cells"]
            protected = placement["protected"]
            if occupied_output.isdisjoint(out_cells) and protected_input.isdisjoint(input_cells):
                placed[spec_i] = placement
                if backtrack(i + 1, occupied_output | out_cells, protected_input | protected):
                    return True
                placed[spec_i] = None
        return False

    if not backtrack(0, frozenset(), frozenset()):
        return None
    return [p for p in placed if p is not None]


def _assign_marker_colors(placements: list[dict[str, object]], square_colors: list[int]) -> list[list[int]]:
    square_set = set(square_colors)
    palette = [c for c in COLOR_RANGE if c not in square_set]
    counts = Counter({c: 0 for c in palette})
    for c in square_colors:
        counts[c] += len(_SQUARE_COORDS)

    weighted_edges = sorted(
        ((ep["weight"], random.random(), pi, ei) for pi, pl in enumerate(placements) for ei, ep in enumerate(pl["edges"])),
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
    square_count = random.randint(*SQUARE_COUNT_RANGE)
    specs = _sample_specs(square_count)

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    square_colors = random.sample(list(COLOR_RANGE), len(placements))
    marker_colors = _assign_marker_colors(placements, square_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    active_markers = 0
    inactive_markers = 0

    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["square"], square_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = marker_colors[pi][ei]
            _paint_cells(out, edge_part["marker"], color)
            if edge_part["active"]:
                _paint_cells(out, edge_part["rays"], color)
                ray_coords.extend(edge_part["rays"])
                active_markers += 1
            else:
                inactive_markers += 1

    if active_markers == 0 or inactive_markers == 0 or not ray_coords:
        return None
    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    rs, cs = zip(*ray_coords)
    rs_arr = np.array(rs)
    cs_arr = np.array(cs)
    inp[rs_arr, cs_arr] = (rs_arr + cs_arr + 1) % 2

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

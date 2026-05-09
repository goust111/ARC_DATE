# Combo variant of 80a900e0 (index 10)
# combo_spec: D5/1+D6/0
# Name: 十字加点标

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Original task: a hollow diamond has a short 3-pixel diagonal segment outside each chosen edge;
#      the segment endpoints emit two diagonal rays.
#    - D5 changes the source object into a 9-pixel plus in a 5x5 bbox. The four actionable sites
#      become the arm tips N/S/W/E, and the natural rays become cardinal instead of diagonal.
#    - D6 then compresses each 3-pixel outer segment into its center pixel only.
#    - Fused result: each chosen arm keeps only one point marker one step beyond the arm tip:
#          N marker at (-1,2), S at (5,2), W at (2,-1), E at (2,5) relative to the plus bbox.
#      The point is interpreted as the midpoint of the hidden 3-pixel segment that would have appeared
#      in the D5-only variant. Example: a top marker at (r0-1,c0+2) reconstructs virtual endpoints
#      (r0-1,c0+1) and (r0-1,c0+3), then both emit upward rays.
#    - Difference from source: shape reasoning switches from diamond edges to plus arm tips, and
#      orientation reasoning switches from visible segment slope to point-vs-arm relative position.
#
# b) Complexity calibration:
#    - The marker positions are limited to exactly four discrete arm-adjacent slots, so humans can
#      infer the mapping from a few examples.
#    - Each grid still has 1-3 pluses, each with 2-4 marked arms, so direction categories repeat.
#    - The visual change is obvious: diamonds become pluses and line segments disappear into dots.
#
# c) Implicit assumptions audit of original main():
#    - `_is_diamond8` no longer applies; replace it with `_is_plus9`.
#    - Collecting adjacent non-bg pixels and grouping them as line components is invalid because the
#      outer object is now a single pixel. The solver must inspect fixed candidate marker slots instead.
#    - Inferring outward direction from segment slope is invalid; outward direction is now arm-specific.
#    - Visible endpoints no longer exist; the solver must reconstruct virtual endpoints from the marker
#      and the hidden segment tangent.
#
# d) Placement legality:
#    - Generator stays output-first: place plus, marker, and full rays; reject overlaps immediately.
#    - Every selected marker must be in-bounds and must produce at least two visible ray cells in total,
#      preventing degenerate input==output cases.
#    - Besides output-cell disjointness, the input must reserve every plus's four candidate marker slots
#      so another object's input pixel cannot masquerade as an unchosen marker.
#
# e) Visual observability:
#    - Across examples, the same "marker above arm => two upward rays" pattern repeats on different colors
#      and locations. No rule depends on numeric color ids.
#
# f) Visual cleanliness:
#    - Rays are still straight and non-overlapping in the output-first construction.
#    - Input contains only pluses and isolated arm markers, so the geometry remains readable.
#
# g) Color inference:
#    - Each marker shares its color with its rays. Plus colors are distinct from source markers, preserving
#      the original "source object vs emitted color" separation without hidden color ordering.
#
# h) New solver assumptions and generator guarantees:
#    - Solver assumes genuine markers appear only in the four arm-adjacent slots around a plus bbox.
#      Generator reserves these slots so false positives are not introduced by other objects.
#    - Solver assumes any detected marker can emit unobstructed rays until a boundary or other non-bg
#      object. Generator verifies this by constructing the full output first and rejecting collisions.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
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


def _ray_cells(
    shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _marker_rays(
    shape: tuple[int, int], marker: tuple[int, int], tangent: tuple[int, int], outward: tuple[int, int]
) -> list[tuple[int, int]]:
    a = (marker[0] - tangent[0], marker[1] - tangent[1])
    b = (marker[0] + tangent[0], marker[1] + tangent[1])
    return _ray_cells(shape, a, outward) + _ray_cells(shape, b, outward)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend straight rays from single-pixel arm markers adjacent to plus shapes on a checkerboard.

    Input structure:
    - Checkerboard background using the two most frequent colors.
    - 9-pixel plus sources in 5x5 bounding boxes, each in a non-background color.
    - Zero or more single-pixel markers, each sitting just beyond one arm tip of a plus.

    Rule:
    - Treat each marker as the midpoint of a hidden 3-pixel segment orthogonal to that arm's ray direction.
    - Recover the segment's two virtual endpoints by moving one step along the marker's tangent.
    - From each virtual endpoint, cast a straight ray away from the plus, painting background cells with
      the marker's color until a non-background cell or the grid boundary is reached.

    Key insight:
    - The marker's absolute orientation is not encoded by its own shape; it is encoded by which arm of the
      plus it sits next to. The solver must combine the plus geometry with the point position to recover
      both the hidden segment orientation and the outward ray direction.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    pluses: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_plus9(pts)]

    out = g.copy()

    def bbox_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
        _, pts = item
        return _bbox(pts)

    for _, ppts in sorted(pluses, key=bbox_key):
        r0, c0, _, _ = _bbox(ppts)
        for arm in _ARM_NAMES:
            info = _ARM_INFO[arm]
            mr = r0 + info["marker"][0]
            mc = c0 + info["marker"][1]
            if not (0 <= mr < h and 0 <= mc < w):
                continue
            mcol = int(g[mr, mc])
            if mcol in bg_colors:
                continue

            tr, tc = info["tangent"]
            dr, dc = info["outward"]
            for sign in (-1, 1):
                pr = mr + sign * tr
                pc = mc + sign * tc
                rr, cc = pr + dr, pc + dc
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
PLUS_COUNT_RANGE = (1, 3)
ARMS_PER_PLUS = (2, 4)
MIN_RAY_CELLS = 2
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
    "N": {"marker": (-1, 2), "tangent": (0, 1), "outward": (-1, 0)},
    "S": {"marker": (5, 2), "tangent": (0, 1), "outward": (1, 0)},
    "W": {"marker": (2, -1), "tangent": (1, 0), "outward": (0, -1)},
    "E": {"marker": (2, 5), "tangent": (1, 0), "outward": (0, 1)},
}
_ARM_NAMES = tuple(_ARM_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _marker_slots(h: int, w: int, r0: int, c0: int) -> set[tuple[int, int]]:
    slots: set[tuple[int, int]] = set()
    for info in _ARM_INFO.values():
        rr = r0 + info["marker"][0]
        cc = c0 + info["marker"][1]
        if 0 <= rr < h and 0 <= cc < w:
            slots.add((rr, cc))
    return slots


def _candidate_placement(
    h: int, w: int, r0: int, c0: int, arms: tuple[str, ...]
) -> dict[str, object] | None:
    plus = [(r0 + dr, c0 + dc) for dr, dc in _PLUS_COORDS]
    input_cells = set(plus)
    all_cells = set(plus)
    edge_parts = []

    for arm in arms:
        info = _ARM_INFO[arm]
        marker = (r0 + info["marker"][0], c0 + info["marker"][1])
        if not (0 <= marker[0] < h and 0 <= marker[1] < w):
            return None
        rays = _marker_rays((h, w), marker, info["tangent"], info["outward"])
        if len(rays) < MIN_RAY_CELLS:
            return None
        part_cells = [marker] + rays
        if len(set(part_cells)) != len(part_cells) or any(cell in all_cells for cell in part_cells):
            return None
        all_cells.update(part_cells)
        input_cells.add(marker)
        edge_parts.append({"marker": marker, "rays": rays, "weight": len(part_cells)})

    return {
        "plus": plus,
        "edges": edge_parts,
        "all_cells": frozenset(all_cells),
        "input_cells": frozenset(input_cells),
        "marker_slots": frozenset(_marker_slots(h, w, r0, c0)),
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
    reserved_marker_slots: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[specs[spec_i]]:
            all_cells = placement["all_cells"]
            input_cells = placement["input_cells"]
            marker_slots = placement["marker_slots"]
            if not occupied_all.isdisjoint(all_cells):
                continue
            if not reserved_marker_slots.isdisjoint(input_cells):
                continue
            if not occupied_input.isdisjoint(marker_slots):
                continue
            occupied_all.update(all_cells)
            occupied_input.update(input_cells)
            reserved_marker_slots.update(marker_slots)
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied_all.difference_update(all_cells)
            occupied_input.difference_update(input_cells)
            reserved_marker_slots.difference_update(marker_slots)
            placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_marker_colors(placements: list[dict[str, object]], plus_colors: list[int]) -> list[list[int]]:
    plus_set = set(plus_colors)
    palette = [c for c in COLOR_RANGE if c not in plus_set]
    counts = Counter({c: 0 for c in palette})
    for c in plus_colors:
        counts[c] += len(_PLUS_COORDS)

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
    plus_count = random.randint(*PLUS_COUNT_RANGE)
    specs = [
        tuple(sorted(random.sample(_ARM_NAMES, random.randint(*ARMS_PER_PLUS))))
        for _ in range(plus_count)
    ]

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    plus_colors = random.sample(list(COLOR_RANGE), len(placements))
    marker_colors = _assign_marker_colors(placements, plus_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for pi, placement in enumerate(placements):
        _paint_cells(out, placement["plus"], plus_colors[pi])
        for ei, edge_part in enumerate(placement["edges"]):
            color = marker_colors[pi][ei]
            _paint_cells(out, [edge_part["marker"]], color)
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

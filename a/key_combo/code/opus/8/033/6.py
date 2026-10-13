from __future__ import annotations

import random
from typing import Callable

import numpy as np


BACKGROUND = 0
GREY = 5
PIXEL_COLORS = (1, 2, 3, 4)

MIN_GRID = 15
MAX_GRID = 19
MIN_ANCHORS = 3
MAX_ANCHORS = 5
MIN_PIXELS = 4
MAX_PIXELS = 10
MIN_ANCHOR_GAP = 5
MAX_GENERATION_ATTEMPTS = 500
MAX_PIXEL_PLACEMENT_ATTEMPTS = 80


Anchor = tuple[int, int, int, int]
Coord = tuple[int, int]


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _anchor_gap(a: Anchor, b: Anchor) -> int:
    return max(_axis_gap(a[0], a[2], b[0], b[2]), _axis_gap(a[1], a[3], b[1], b[3]))


def _rect_distances(cell: Coord, anchor: Anchor) -> tuple[int, int]:
    r, c = cell
    r0, c0, r1, c1 = anchor
    row_gap = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    col_gap = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return max(row_gap, col_gap), row_gap + col_gap


def _expanded_bbox(anchor: Anchor, layers: int, shape: tuple[int, int]) -> Anchor:
    h, w = shape
    r0, c0, r1, c1 = anchor
    return (
        max(0, r0 - layers),
        max(0, c0 - layers),
        min(h - 1, r1 + layers),
        min(w - 1, c1 + layers),
    )


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_anchors(grid: np.ndarray) -> list[Anchor]:
    anchors: list[Anchor] = []
    for comp in _components_of_color(grid, GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        anchor = (min(rows), min(cols), max(rows), max(cols))
        assert len(comp) == 4, "grey anchors must have area 4"
        assert anchor[2] - anchor[0] == 1 and anchor[3] - anchor[1] == 1, "grey anchors must be solid 2x2 blocks"
        anchors.append(anchor)
    anchors.sort(key=lambda box: (box[0], box[1]))
    return anchors


def _assignment_details(cell: Coord, anchors: list[Anchor]) -> list[tuple[int, int, int, int, int]]:
    details = []
    for idx, anchor in enumerate(anchors):
        cheb, manhattan = _rect_distances(cell, anchor)
        details.append((cheb, manhattan, anchor[0], anchor[1], idx))
    details.sort()
    return details


def _winner_anchor(cell: Coord, anchors: list[Anchor]) -> int:
    return _assignment_details(cell, anchors)[0][4]


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    anchors = _find_anchors(g)
    assert MIN_ANCHORS <= len(anchors) <= MAX_ANCHORS, "expected 3-5 grey anchors"

    pixel_mask = (g != BACKGROUND) & (g != GREY)
    pixels = sorted(tuple(map(int, coord)) for coord in np.argwhere(pixel_mask))
    assert MIN_PIXELS <= len(pixels) <= MAX_PIXELS, "expected 4-10 singleton pixels"

    assignments: dict[Coord, int] = {}
    per_anchor: list[list[Coord]] = [[] for _ in anchors]
    for cell in pixels:
        idx = _winner_anchor(cell, anchors)
        assignments[cell] = idx
        per_anchor[idx].append(cell)

    stats = []
    for idx, anchor in enumerate(anchors):
        cells = sorted(per_anchor[idx])
        assert cells, "each anchor must absorb at least one singleton"
        ranked = sorted(cells, key=lambda cell: (_rect_distances(cell, anchor)[0], cell[0], cell[1]))
        nearest = ranked[0]
        closest_dist = _rect_distances(nearest, anchor)[0]
        layers = len(cells)
        color = int(g[nearest])
        region = _expanded_bbox(anchor, layers, (h, w))
        stats.append(
            {
                "anchor": anchor,
                "count": layers,
                "color": color,
                "nearest_cell": nearest,
                "closest_dist": closest_dist,
                "region": region,
                "cells": cells,
            }
        )

    cleared = g.copy()
    for r, c in pixels:
        cleared[r, c] = BACKGROUND

    owner_grid = np.full((h, w), -1, dtype=int)
    out = cleared.copy()
    for r in range(h):
        for c in range(w):
            if int(cleared[r, c]) != BACKGROUND:
                continue
            claimants = []
            for idx, info in enumerate(stats):
                rr0, cc0, rr1, cc1 = info["region"]
                if rr0 <= r <= rr1 and cc0 <= c <= cc1:
                    claimants.append(idx)
            if not claimants:
                continue
            owner = max(
                claimants,
                key=lambda idx: (
                    stats[idx]["count"],
                    -stats[idx]["closest_dist"],
                    -stats[idx]["anchor"][0],
                    -stats[idx]["anchor"][1],
                ),
            )
            owner_grid[r, c] = owner
            out[r, c] = stats[owner]["color"]

    return {
        "anchors": anchors,
        "pixels": pixels,
        "assignments": assignments,
        "per_anchor": per_anchor,
        "stats": stats,
        "owner_grid": owner_grid,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assign each colored singleton to its nearest grey anchor and grow a count-sized halo around that anchor.

    Input structure:
    - The background is black.
    - There are 3-5 solid grey 2x2 anchor blocks.
    - Every other non-black cell is a single colored pixel, separated from the anchors and from each other.

    Rule:
    - For each colored singleton, measure its distance to each grey anchor by Chebyshev distance to the nearest cell of the 2x2 block.
    - Assign the singleton to the anchor with smallest Chebyshev distance; ties use smaller Manhattan distance to the block, then the earlier anchor top-left in reading order.
    - For each anchor, let N be its number of assigned singletons, and let C be the color of its assigned singleton with smallest Chebyshev distance to that anchor, breaking equal distances by pixel reading order.
    - Clear all original singleton pixels, then expand the anchor's 2x2 bounding box outward by N cells on every side, clipped to the grid, and recolor black cells in that region to C.
    - Grey anchor cells stay grey, and overlapping halo cells belong to the anchor with larger N; if N ties, the anchor whose nearest assigned singleton is closer wins, then the earlier anchor top-left wins.

    Key insight:
    - Counting absorbed pixels does not produce a separate symbol: it directly becomes the halo thickness, so assignment, counting, color selection, and overlap priority all matter.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _place_anchor_pair(h: int, w: int, gap: int, vertical: bool) -> list[Anchor] | None:
    if vertical:
        max_r = h - gap - 2
        if max_r < 0:
            return None
        r0 = random.randint(0, max_r)
        c0 = random.randint(0, w - 2)
        a = (r0, c0, r0 + 1, c0 + 1)
        b = (r0 + gap, c0, r0 + gap + 1, c0 + 1)
        return [a, b]
    max_c = w - gap - 2
    if max_c < 0:
        return None
    r0 = random.randint(0, h - 2)
    c0 = random.randint(0, max_c)
    a = (r0, c0, r0 + 1, c0 + 1)
    b = (r0, c0 + gap, r0 + 1, c0 + gap + 1)
    return [a, b]


def _place_anchors(h: int, w: int, anchor_count: int, gap: int) -> list[Anchor] | None:
    pair = _place_anchor_pair(h, w, gap, vertical=bool(random.getrandbits(1)))
    if pair is None:
        return None

    anchors = list(pair)
    candidates = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if len(anchors) == anchor_count:
            break
        anchor = (r, c, r + 1, c + 1)
        if all(_anchor_gap(anchor, other) >= MIN_ANCHOR_GAP for other in anchors):
            anchors.append(anchor)
    if len(anchors) != anchor_count:
        return None
    anchors.sort(key=lambda box: (box[0], box[1]))
    return anchors


def _choose_counts(anchor_count: int, mode: str) -> list[int] | None:
    remaining = anchor_count - 2
    if mode == "count_priority":
        pair_options = [(4, 1), (1, 4), (3, 2), (2, 3), (4, 2), (2, 4)]
    else:
        max_equal = (MAX_PIXELS - remaining) // 2
        if max_equal < 3:
            return None
        pair_options = [(value, value) for value in range(3, max_equal + 1)]
    random.shuffle(pair_options)

    for p0, p1 in pair_options:
        rest = [random.randint(1, 2) for _ in range(remaining)]
        counts = [p0, p1, *rest]
        if not (MIN_PIXELS <= sum(counts) <= MAX_PIXELS):
            continue
        if len(set(counts)) == 1:
            continue
        random.shuffle(rest)
        return [p0, p1, *rest]
    return None


def _choose_nearest_distances(anchor_count: int, mode: str) -> list[int]:
    dists = [random.choice((2, 3)) for _ in range(anchor_count)]
    if mode == "distance_priority":
        dists[0], dists[1] = random.choice(((2, 3), (3, 2)))
    elif mode == "order_priority":
        same = random.choice((2, 3))
        dists[0], dists[1] = same, same
    else:
        dists[0], dists[1] = random.choice((2, 3)), random.choice((2, 3))
    return dists


def _cell_far_from_anchors(cell: Coord, anchors: list[Anchor]) -> bool:
    return min(_rect_distances(cell, anchor)[0] for anchor in anchors) >= 2


def _cell_far_from_pixels(cell: Coord, pixels: dict[Coord, int]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in pixels)


def _candidate_cells(
    shape: tuple[int, int],
    anchors: list[Anchor],
    used_pixels: dict[Coord, int],
    target_anchor: int,
    exact_dist: int | None,
    predicate: Callable[[Coord, list[tuple[int, int, int, int, int]]], bool] | None,
) -> list[Coord]:
    h, w = shape
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in used_pixels:
                continue
            if not _cell_far_from_anchors(cell, anchors):
                continue
            if not _cell_far_from_pixels(cell, used_pixels):
                continue
            details = _assignment_details(cell, anchors)
            if details[0][4] != target_anchor:
                continue
            if exact_dist is not None and details[0][0] != exact_dist:
                continue
            if predicate is not None and not predicate(cell, details):
                continue
            cells.append(cell)
    random.shuffle(cells)
    return cells


def _place_pixels(
    shape: tuple[int, int],
    anchors: list[Anchor],
    counts: list[int],
    nearest_dists: list[int],
    assignment_mode: str,
) -> dict[Coord, int] | None:
    used_pixels: dict[Coord, int] = {}

    pair_colors = random.sample(list(PIXEL_COLORS), 2)
    nearest_colors = [pair_colors[0], pair_colors[1]]
    nearest_colors.extend(random.choice(PIXEL_COLORS) for _ in range(len(anchors) - 2))

    if assignment_mode == "manhattan":
        tie_anchor = random.choice((0, 1))
        tie_predicate = lambda _cell, details, anchor=tie_anchor: any(
            cheb == details[0][0] and manhattan > details[0][1]
            for cheb, manhattan, _r0, _c0, idx in details[1:]
            if idx != anchor
        )
    elif assignment_mode == "order":
        tie_anchor = 0
        tie_predicate = lambda _cell, details, anchor=tie_anchor: any(
            cheb == details[0][0] and manhattan == details[0][1]
            for cheb, manhattan, _r0, _c0, idx in details[1:]
            if idx != anchor
        )
    else:
        tie_anchor = None
        tie_predicate = None

    for idx in range(len(anchors)):
        predicate = tie_predicate if idx == tie_anchor else None
        cells = _candidate_cells(shape, anchors, used_pixels, idx, nearest_dists[idx], predicate)
        if not cells:
            return None
        used_pixels[cells[0]] = nearest_colors[idx]

    for idx, count in enumerate(counts):
        for extra_i in range(count - 1):
            exact_dist = nearest_dists[idx] + 1 + (extra_i % 2)
            cells = _candidate_cells(shape, anchors, used_pixels, idx, exact_dist, None)
            if not cells:
                cells = _candidate_cells(shape, anchors, used_pixels, idx, None, None)
            if not cells:
                return None
            used_pixels[cells[0]] = random.choice(PIXEL_COLORS)

    return used_pixels


def _exclusive_overlap_cells(stats: list[dict], owner_grid: np.ndarray, a: int, b: int) -> list[Coord]:
    h, w = owner_grid.shape
    cells: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if owner_grid[r, c] < 0:
                continue
            in_a = stats[a]["region"][0] <= r <= stats[a]["region"][2] and stats[a]["region"][1] <= c <= stats[a]["region"][3]
            in_b = stats[b]["region"][0] <= r <= stats[b]["region"][2] and stats[b]["region"][1] <= c <= stats[b]["region"][3]
            if not (in_a and in_b):
                continue
            in_others = False
            for idx, info in enumerate(stats):
                if idx in (a, b):
                    continue
                rr0, cc0, rr1, cc1 = info["region"]
                if rr0 <= r <= rr1 and cc0 <= c <= cc1:
                    in_others = True
                    break
            if not in_others:
                cells.append((r, c))
    return cells


def _validate_generated(grid: np.ndarray, mode: str, assignment_mode: str) -> bool:
    try:
        analysis = _analyze(grid)
    except AssertionError:
        return False

    if not np.array_equal(main(grid), analysis["output"]):
        return False

    counts = [info["count"] for info in analysis["stats"]]
    if len(set(counts)) == 1:
        return False

    for pixel in analysis["pixels"]:
        if min(_rect_distances(pixel, anchor)[0] for anchor in analysis["anchors"]) < 2:
            return False

    if len({info["color"] for info in analysis["stats"]}) < 2:
        return False

    if assignment_mode == "manhattan":
        ok = False
        for pixel, idx in analysis["assignments"].items():
            details = _assignment_details(pixel, analysis["anchors"])
            if details[0][4] != idx:
                continue
            if any(cheb == details[0][0] and manhattan > details[0][1] for cheb, manhattan, _r0, _c0, other in details[1:] if other != idx):
                ok = True
                break
        if not ok:
            return False
    elif assignment_mode == "order":
        ok = False
        for pixel, idx in analysis["assignments"].items():
            details = _assignment_details(pixel, analysis["anchors"])
            if details[0][4] != idx:
                continue
            if any(cheb == details[0][0] and manhattan == details[0][1] for cheb, manhattan, _r0, _c0, other in details[1:] if other != idx):
                ok = True
                break
        if not ok:
            return False

    pair_cells = _exclusive_overlap_cells(analysis["stats"], analysis["owner_grid"], 0, 1)
    if not pair_cells:
        return False

    info0 = analysis["stats"][0]
    info1 = analysis["stats"][1]
    if info0["color"] == info1["color"]:
        return False

    owner = analysis["owner_grid"][pair_cells[0]]
    if mode == "count_priority":
        expected = 0 if info0["count"] > info1["count"] else 1
        if info0["count"] == info1["count"] or owner != expected:
            return False
    elif mode == "distance_priority":
        if info0["count"] != info1["count"]:
            return False
        if info0["closest_dist"] == info1["closest_dist"]:
            return False
        expected = 0 if info0["closest_dist"] < info1["closest_dist"] else 1
        if owner != expected:
            return False
    else:
        if info0["count"] != info1["count"]:
            return False
        if info0["closest_dist"] != info1["closest_dist"]:
            return False
        if owner != 0:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    anchor_count = random.randint(MIN_ANCHORS, MAX_ANCHORS)
    mode = random.choice(("count_priority", "distance_priority", "order_priority"))
    assignment_mode = random.choice(("none", "manhattan", "order"))
    gap = 7 if assignment_mode == "order" or mode == "order_priority" else random.choice((6, 7))

    anchors = _place_anchors(h, w, anchor_count, gap)
    if anchors is None:
        return None

    counts = _choose_counts(anchor_count, mode)
    if counts is None:
        return None

    nearest_dists = _choose_nearest_distances(anchor_count, mode)
    pixels = None
    for _ in range(MAX_PIXEL_PLACEMENT_ATTEMPTS):
        pixels = _place_pixels((h, w), anchors, counts, nearest_dists, assignment_mode)
        if pixels is not None:
            break
    if pixels is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for r0, c0, r1, c1 in anchors:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    for (r, c), color in pixels.items():
        grid[r, c] = color

    return grid if _validate_generated(grid, mode, assignment_mode) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid halo-count ARC input")

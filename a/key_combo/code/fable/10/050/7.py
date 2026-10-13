from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
BAR_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
CASES = [
    ("top", "TL"),
    ("top", "TR"),
    ("bottom", "BL"),
    ("bottom", "BR"),
    ("left", "TL"),
    ("left", "BL"),
    ("right", "TR"),
    ("right", "BR"),
]

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _rectangle_from_markers(grid: np.ndarray) -> BBox:
    yellow = np.argwhere(grid == YELLOW)
    assert len(yellow) == 4, "expected exactly four yellow markers"

    rows = sorted({int(r) for r, _c in yellow})
    cols = sorted({int(c) for _r, c in yellow})
    assert len(rows) == 2 and len(cols) == 2, "markers must share exactly two rows and two cols"
    assert {(int(r), int(c)) for r, c in yellow} == {
        (rows[0], cols[0]),
        (rows[0], cols[1]),
        (rows[1], cols[0]),
        (rows[1], cols[1]),
    }, "yellow markers must be the four outer-corner clues"

    r0, c0 = rows[0] + 1, cols[0] + 1
    r1, c1 = rows[1] - 1, cols[1] - 1
    assert r1 > r0 and c1 > c0, "hidden rectangle must be at least 2x2"
    return r0, c0, r1, c1


def _boundary_cycle(rect: BBox) -> list[Coord]:
    r0, c0, r1, c1 = rect
    cycle: list[Coord] = []
    for c in range(c0, c1 + 1):
        cycle.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cycle.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cycle.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cycle.append((r, c0))
    assert len(cycle) == 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    return cycle


def _corners(rect: BBox) -> dict[str, Coord]:
    r0, c0, r1, c1 = rect
    return {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BR": (r1, c1),
        "BL": (r1, c0),
    }


def _ordered_runs(values: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for value in values:
        if not runs or value != runs[-1][-1] + 1:
            runs.append([value])
        else:
            runs[-1].append(value)
    return runs


def _find_ruler(grid: np.ndarray, bg: int) -> tuple[int, list[list[Coord]], str]:
    colors = sorted(set(int(v) for v in np.unique(grid)) - {bg, YELLOW})
    assert len(colors) == 1, "expected exactly one ruler color"
    color = colors[0]
    coords = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
    rows = sorted({r for r, _c in coords})
    cols = sorted({c for _r, c in coords})
    h, w = grid.shape

    if len(rows) == 1:
        row = rows[0]
        assert row in (0, h - 1), "horizontal ruler must lie on top or bottom border"
        runs = _ordered_runs(sorted(cols))
        assert len(runs) == 2 and runs[1][0] - runs[0][-1] == 2, "horizontal ruler must have one-cell gap"
        segments = [[(row, c) for c in run] for run in runs]
        return color, segments, "horizontal"

    if len(cols) == 1:
        col = cols[0]
        assert col in (0, w - 1), "vertical ruler must lie on left or right border"
        runs = _ordered_runs(sorted(rows))
        assert len(runs) == 2 and runs[1][0] - runs[0][-1] == 2, "vertical ruler must have one-cell gap"
        segments = [[(r, col) for r in run] for run in runs]
        return color, segments, "vertical"

    raise AssertionError("ruler must be a straight border bar")


def _cyclic_path(cycle: list[Coord], start: Coord, clockwise: bool) -> list[Coord]:
    idx = cycle.index(start)
    step = 1 if clockwise else -1
    return [cycle[(idx + step * k) % len(cycle)] for k in range(len(cycle))]


def _closest_corner(corners: dict[str, Coord], ruler_cells: list[Coord]) -> str:
    dist_by_corner = {
        name: min(_manhattan(corner, cell) for cell in ruler_cells)
        for name, corner in corners.items()
    }
    best = min(dist_by_corner.values())
    winners = [name for name, dist in dist_by_corner.items() if dist == best]
    assert len(winners) == 1, "nearest boundary corner must be unique"
    return winners[0]


def _segment_order(segments: list[list[Coord]], boundary: list[Coord]) -> tuple[list[Coord], list[Coord]]:
    dists = [
        min(_manhattan(cell, boundary_cell) for cell in segment for boundary_cell in boundary)
        for segment in segments
    ]
    assert dists[0] != dists[1], "near and far ruler segments must be distinguishable"
    near_idx = 0 if dists[0] < dists[1] else 1
    return segments[near_idx], segments[1 - near_idx]


def _first_edge_lengths(rect: BBox, corner_name: str) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if corner_name in ("TL", "BR"):
        return width, height
    return height, width


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    rect = _rectangle_from_markers(g)
    boundary = _boundary_cycle(rect)
    corners = _corners(rect)
    bar_color, segments, orientation = _find_ruler(g, bg)
    ruler_cells = [cell for segment in segments for cell in segment]
    corner_name = _closest_corner(corners, ruler_cells)
    near_segment, far_segment = _segment_order(segments, boundary)
    clockwise = _cyclic_path(boundary, corners[corner_name], clockwise=True)
    counterclockwise = _cyclic_path(boundary, corners[corner_name], clockwise=False)

    near_len = len(near_segment)
    far_len = len(far_segment)
    assert near_len >= 2 and far_len >= 2, "both ruler pieces must contribute visibly"
    assert near_len != far_len, "gap split must affect direction assignment"
    assert near_len + far_len < len(boundary), "combined ruler length must stay below perimeter"

    return {
        "bg": bg,
        "rect": rect,
        "boundary": boundary,
        "corners": corners,
        "corner_name": corner_name,
        "bar_color": bar_color,
        "segments": segments,
        "ruler_cells": ruler_cells,
        "orientation": orientation,
        "near_segment": near_segment,
        "far_segment": far_segment,
        "near_len": near_len,
        "far_len": far_len,
        "clockwise": clockwise,
        "counterclockwise": counterclockwise,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Wrap the two pieces of a broken border ruler onto the invisible rectangle implied by four yellow corner markers.

    Input structure:
    - The background is black.
    - Four yellow pixels sit just outside the diagonal directions of a hidden axis-aligned rectangle; the rectangle itself is not drawn.
    - One non-yellow straight ruler bar lies on a grid border and is split into two segments by a single-cell gap.

    Rule:
    - Recover the hidden rectangle from the four yellow markers and enumerate its full one-cell-thick boundary.
    - Find the unique rectangle corner closest to the ruler bar.
    - Measure the ruler segment nearer to the rectangle and paint that many boundary cells clockwise from the chosen corner.
    - Measure the farther ruler segment and paint that many boundary cells counterclockwise from the same corner.
    - Use the ruler color for both wrapped segments, remove the original ruler, keep the yellow markers, and leave the rest black.

    Key insight:
    - The yellow clues define only geometry, while the broken bar provides both the total transferable paint and the split that assigns clockwise versus counterclockwise boundary lengths.
    """
    info = _analyze(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(info["bg"])
    bar_color = int(info["bar_color"])

    for r, c in info["ruler_cells"]:
        out[r, c] = bg

    for r, c in info["clockwise"][: int(info["near_len"])]:
        out[r, c] = bar_color
    # Count the shared corner only once so the moved paint exactly matches the ruler pixel count.
    for r, c in info["counterclockwise"][1 : int(info["far_len"]) + 1]:
        out[r, c] = bar_color

    return out


def _candidate_lengths(perimeter: int, available_span: int, cw_edge: int, ccw_edge: int) -> list[tuple[int, int]]:
    max_total = min(perimeter - 1, available_span - 1)
    pairs = []
    for near_len in range(2, max_total - 1):
        for far_len in range(2, max_total - near_len + 1):
            if near_len == far_len:
                continue
            total = near_len + far_len
            if total >= perimeter:
                continue
            pairs.append((near_len, far_len))

    if not pairs:
        return []

    wrapping = [pair for pair in pairs if pair[0] > cw_edge or pair[1] > ccw_edge]
    if wrapping and random.random() < 0.7:
        return wrapping
    return pairs


def _place_ruler(
    grid: np.ndarray,
    rect: BBox,
    side: str,
    corner_name: str,
    bar_color: int,
    near_len: int,
    far_len: int,
) -> None:
    r0, c0, r1, c1 = rect
    extent = near_len + far_len + 1
    h, w = grid.shape

    if (side, corner_name) == ("top", "TL"):
        row = 0
        start = c0 - extent
        gap = start + far_len
        grid[row, start:gap] = bar_color
        grid[row, gap + 1 : c0] = bar_color
    elif (side, corner_name) == ("top", "TR"):
        row = 0
        start = c1 + 1
        gap = start + near_len
        grid[row, start:gap] = bar_color
        grid[row, gap + 1 : start + extent] = bar_color
    elif (side, corner_name) == ("bottom", "BL"):
        row = h - 1
        start = c0 - extent
        gap = start + far_len
        grid[row, start:gap] = bar_color
        grid[row, gap + 1 : c0] = bar_color
    elif (side, corner_name) == ("bottom", "BR"):
        row = h - 1
        start = c1 + 1
        gap = start + near_len
        grid[row, start:gap] = bar_color
        grid[row, gap + 1 : start + extent] = bar_color
    elif (side, corner_name) == ("left", "TL"):
        col = 0
        start = r0 - extent
        gap = start + far_len
        grid[start:gap, col] = bar_color
        grid[gap + 1 : r0, col] = bar_color
    elif (side, corner_name) == ("left", "BL"):
        col = 0
        start = r1 + 1
        gap = start + near_len
        grid[start:gap, col] = bar_color
        grid[gap + 1 : start + extent, col] = bar_color
    elif (side, corner_name) == ("right", "TR"):
        col = w - 1
        start = r0 - extent
        gap = start + far_len
        grid[start:gap, col] = bar_color
        grid[gap + 1 : r0, col] = bar_color
    elif (side, corner_name) == ("right", "BR"):
        col = w - 1
        start = r1 + 1
        gap = start + near_len
        grid[start:gap, col] = bar_color
        grid[gap + 1 : start + extent, col] = bar_color
    else:
        raise AssertionError("unsupported ruler placement case")


def _make_example() -> np.ndarray | None:
    h = random.randint(15, 30)
    w = random.randint(15, 30)

    rect_h = random.randint(3, min(9, h - 8))
    rect_w = random.randint(3, min(9, w - 8))
    r0 = random.randint(3, h - rect_h - 4)
    c0 = random.randint(3, w - rect_w - 4)
    rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    corners = _corners(rect)

    side, corner_name = random.choice(CASES)
    chosen_corner = corners[corner_name]
    perimeter = len(_boundary_cycle(rect))
    cw_edge, ccw_edge = _first_edge_lengths(rect, corner_name)

    if side == "top":
        available_span = chosen_corner[1] if corner_name == "TL" else w - 1 - chosen_corner[1]
    elif side == "bottom":
        available_span = chosen_corner[1] if corner_name == "BL" else w - 1 - chosen_corner[1]
    elif side == "left":
        available_span = chosen_corner[0] if corner_name == "TL" else h - 1 - chosen_corner[0]
    else:
        available_span = chosen_corner[0] if corner_name == "TR" else h - 1 - chosen_corner[0]

    pairs = _candidate_lengths(perimeter, available_span, cw_edge, ccw_edge)
    if not pairs:
        return None

    near_len, far_len = random.choice(pairs)
    bar_color = random.choice(BAR_COLORS)

    grid = np.full((h, w), BLACK, dtype=int)
    r1, c1 = rect[2], rect[3]
    grid[r0 - 1, c0 - 1] = YELLOW
    grid[r0 - 1, c1 + 1] = YELLOW
    grid[r1 + 1, c0 - 1] = YELLOW
    grid[r1 + 1, c1 + 1] = YELLOW
    _place_ruler(grid, rect, side, corner_name, bar_color, near_len, far_len)

    info = _analyze(grid)
    if info["corner_name"] != corner_name:
        return None
    if int(info["near_len"]) != near_len or int(info["far_len"]) != far_len:
        return None

    expected = set(info["clockwise"][:near_len]) | set(info["counterclockwise"][1 : far_len + 1])
    wrong_swapped = set(info["clockwise"][:far_len]) | set(info["counterclockwise"][1 : near_len + 1])
    if expected == wrong_swapped:
        return None

    return grid


def _validate_example(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        out = main(grid)
    except AssertionError:
        return False

    bar_color = int(info["bar_color"])
    bg = int(info["bg"])
    expected_bar = set(info["clockwise"][: int(info["near_len"])]) | set(
        info["counterclockwise"][1 : int(info["far_len"]) + 1]
    )
    actual_bar = {(int(r), int(c)) for r, c in np.argwhere(out == bar_color)}
    if actual_bar != expected_bar:
        return False
    if len(actual_bar) != sum(len(segment) for segment in info["segments"]):
        return False

    for r, c in info["ruler_cells"]:
        if out[r, c] != bg:
            return False

    yellow_in = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    yellow_out = {(int(r), int(c)) for r, c in np.argwhere(out == YELLOW)}
    if yellow_in != yellow_out:
        return False

    for r, c in info["boundary"]:
        if (r, c) not in expected_bar and out[r, c] != BLACK:
            return False

    if int(info["near_len"]) <= 1 or int(info["far_len"]) <= 1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _make_example()
        if grid is not None and _validate_example(grid):
            return grid
    raise RuntimeError("failed to generate a valid ruler-wrap puzzle")

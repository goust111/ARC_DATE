from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
LINE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_GRID = 10
MAX_GRID = 28
MIN_LENGTH = 4
MAX_LENGTH = 12
MAX_SEGMENTS = 5
GRID_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _trace_segment(
    grid: np.ndarray, start: tuple[int, int], bg: int
) -> tuple[list[tuple[int, int]], int, int, tuple[int, int]]:
    h, w = grid.shape
    r0, c0 = start
    next_cells = [
        (rr, cc)
        for rr, cc in _neighbors4((h, w), r0, c0)
        if int(grid[rr, cc]) != int(bg)
    ]
    assert len(next_cells) == 1, "grey endpoint must have exactly one foreground neighbor"

    dr = int(next_cells[0][0] - r0)
    dc = int(next_cells[0][1] - c0)
    cells: list[tuple[int, int]] = []
    rr, cc = r0, c0
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != int(bg):
        cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc

    assert len(cells) >= 3, "each conveyor must be a straight segment of length at least 3"
    cell_set = set(cells)
    for idx, (r, c) in enumerate(cells):
        fg_neighbors = [
            (rr, cc)
            for rr, cc in _neighbors4((h, w), r, c)
            if int(grid[rr, cc]) != int(bg)
        ]
        expected_degree = 1 if idx in {0, len(cells) - 1} else 2
        assert len(fg_neighbors) == expected_degree, "segments must be isolated straight paths"
        assert all(nei in cell_set for nei in fg_neighbors), "segment may not branch or touch another object"

    line_color = int(grid[cells[-1]])
    assert line_color != GREY, "the non-grey end must keep the conveyor color"

    marker_count = sum(int(grid[r, c]) != line_color for r, c in cells[1:-1])
    target = cells[marker_count]
    return cells, line_color, marker_count, target


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each grey endpoint forward along its colored segment by the number of off-color markers on that segment.

    Input structure:
    - The background is black.
    - The foreground consists of separate horizontal or vertical one-pixel-thick segments, each with its own line color.
    - One endpoint of each segment is grey, while the opposite endpoint keeps the segment's line color.
    - Some interior pixels on a segment may be recolored into non-grey marker pixels that differ from that segment's line color.

    Rule:
    - For each segment, start at the grey endpoint and follow the segment toward the opposite endpoint.
    - Count that segment's marker pixels: every interior pixel whose color is neither grey nor the segment's own line color.
    - Move the grey endpoint forward by exactly that many cells along the segment, repainting the old endpoint back to the segment color.
    - Marker pixels remain unchanged unless the grey pixel's final landing cell overwrites one of them.

    Key insight:
    - The grey pixel is the moving object, the endpoint orientation gives its travel direction, and the number of off-color interior markers encodes the travel distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    grey_starts = [tuple(map(int, pos)) for pos in np.argwhere(g == GREY)]
    assert grey_starts, "expected at least one grey conveyor endpoint"

    covered: set[tuple[int, int]] = set()
    out = g.copy()
    for start in grey_starts:
        cells, line_color, _marker_count, target = _trace_segment(g, start, bg)
        for cell in cells:
            assert cell not in covered, "segments must not overlap"
            covered.add(cell)
        out[start] = line_color
        out[target] = GREY

    all_foreground = {tuple(map(int, pos)) for pos in np.argwhere(g != bg)}
    assert covered == all_foreground, "all foreground pixels must belong to a grey-anchored segment"
    return out


def _balanced_binary(count: int, a, b) -> list:
    major, minor = (a, b) if random.random() < 0.5 else (b, a)
    values = [major] * ((count + 1) // 2) + [minor] * (count // 2)
    random.shuffle(values)
    return values


def _expanded(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _marker_map(length: int, line_color: int, land_on_marker: bool) -> dict[int, int]:
    interior = list(range(1, length - 1))
    assert interior, "segments must have an interior"

    if land_on_marker or len(interior) == 1:
        marker_count = random.randint(1, len(interior))
        positions = [marker_count]
        others = [idx for idx in interior if idx != marker_count]
        positions.extend(random.sample(others, marker_count - 1))
    else:
        assert len(interior) >= 2, "landing on a line-color cell needs at least two interior cells"
        marker_count = random.randint(1, len(interior) - 1)
        candidates = [idx for idx in interior if idx != marker_count]
        positions = random.sample(candidates, marker_count)

    palette = [color for color in MARKER_COLORS if color != line_color]
    active_colors = random.sample(palette, k=random.randint(1, min(3, len(palette))))
    return {int(idx): int(random.choice(active_colors)) for idx in positions}


def _propose_segment(
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    line_color: int,
    orientation: str,
    sign: int,
    land_on_marker: bool,
) -> dict[str, object] | None:
    max_len = min(MAX_LENGTH, w if orientation == "h" else h)
    if max_len < MIN_LENGTH:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        length = random.randint(MIN_LENGTH, max_len)
        if orientation == "h":
            row = random.randrange(h)
            start_col = random.randint(0, w - length) if sign > 0 else random.randint(length - 1, w - 1)
            cells = [(row, start_col + sign * k) for k in range(length)]
        else:
            col = random.randrange(w)
            start_row = random.randint(0, h - length) if sign > 0 else random.randint(length - 1, h - 1)
            cells = [(start_row + sign * k, col) for k in range(length)]

        if any(cell in forbidden for cell in cells):
            continue

        markers = _marker_map(length, line_color, land_on_marker)
        return {
            "cells": cells,
            "line_color": int(line_color),
            "markers": markers,
        }
    return None


def _paint_segment(grid: np.ndarray, spec: dict[str, object]) -> None:
    cells = spec["cells"]
    line_color = int(spec["line_color"])
    markers = spec["markers"]
    assert isinstance(cells, list)
    assert isinstance(markers, dict)

    for idx, (r, c) in enumerate(cells):
        if idx == 0:
            grid[r, c] = GREY
        elif idx in markers:
            grid[r, c] = int(markers[idx])
        else:
            grid[r, c] = line_color


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    area = h * w

    max_segments = min(MAX_SEGMENTS, len(LINE_COLORS), 1 + area // 120)
    min_segments = 1 if max_segments == 1 else 2
    segment_count = random.randint(min_segments, max_segments)

    line_colors = random.sample(LINE_COLORS, segment_count)
    orientations = (
        _balanced_binary(segment_count, "h", "v")
        if segment_count > 1
        else [random.choice(["h", "v"])]
    )
    signs = (
        _balanced_binary(segment_count, -1, 1)
        if segment_count > 1
        else [random.choice([-1, 1])]
    )
    landing_modes = (
        _balanced_binary(segment_count, False, True)
        if segment_count > 1
        else [random.choice([False, True])]
    )

    grid = np.full((h, w), BG, dtype=int)
    forbidden: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []
    for line_color, orientation, sign, land_on_marker in zip(
        line_colors, orientations, signs, landing_modes
    ):
        spec = _propose_segment(h, w, forbidden, line_color, orientation, sign, land_on_marker)
        if spec is None:
            return None
        specs.append(spec)
        forbidden.update(_expanded(spec["cells"], (h, w)))

    random.shuffle(specs)
    for spec in specs:
        _paint_segment(grid, spec)

    vals, counts = np.unique(grid, return_counts=True)
    mode = int(vals[int(np.argmax(counts))])
    if mode != BG or int(np.sum(counts == counts.max())) != 1:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GRID_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("generate_input failed after repeated placement attempts")

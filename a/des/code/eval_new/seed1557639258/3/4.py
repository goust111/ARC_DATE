from __future__ import annotations

import random
import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

STRIP_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE)
GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
RIGHT_END_TURNS = (2, 4, 6)
VERTICAL_END_TURNS = (3, 5)
MIN_COL_GAP = 2
STRIP_MARGIN = 1
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _ordered_path_from_start(path_mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    path = {tuple(map(int, rc)) for rc in np.argwhere(path_mask)}
    assert start in path

    ordered = [start]
    visited = {start}
    prev: tuple[int, int] | None = None
    cur = start

    while True:
        nxts = [
            (rr, cc)
            for rr, cc in _neighbors4(cur[0], cur[1], h, w)
            if (rr, cc) in path and (rr, cc) != prev
        ]
        if not nxts:
            break
        assert len(nxts) == 1, f"path must be a single chain, got branch at {cur}"
        nxt = nxts[0]
        assert nxt not in visited, f"path must not self-intersect, repeated {nxt}"
        ordered.append(nxt)
        visited.add(nxt)
        prev, cur = cur, nxt

    assert len(visited) == len(path), "path traversal must cover the whole grey polyline"
    return ordered


def _turn_indices(path: list[tuple[int, int]]) -> list[int]:
    turns: list[int] = []
    for i in range(1, len(path) - 1):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        d1 = (r1 - r0, c1 - c0)
        d2 = (r2 - r1, c2 - c1)
        if d1 != d2:
            turns.append(i)
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """Trace the maroon-marked grey polyline and paint each pre-turn segment with the color of its next turn.

    Input structure:
    - The background is black.
    - One grey 4-connected single-pixel-wide polyline runs from one grid edge to a different edge.
    - A single maroon pixel sits orthogonally adjacent to exactly one path endpoint and marks the traversal start.
    - The top row or bottom row contains an instruction strip of distinct colored pixels; its length equals the number of path turns.

    Rule:
    - Start from the path endpoint touching the maroon marker and follow the unique grey path to the other endpoint.
    - Enumerate turns in traversal order; recolor the first turn with the first strip color, the second turn with the second strip color, and so on.
    - Recolor each straight segment before a turn with that upcoming turn's color, including the start segment.
    - Leave the last straight segment after the final turn grey, erase the instruction strip to black, and keep the maroon marker.

    Key insight:
    - The strip is indexed by walk order rather than spatial position, and each color applies backward to the segment that leads into its turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_cells = np.argwhere(g == MAROON)
    assert len(marker_cells) == 1, f"expected one maroon start marker, got {len(marker_cells)}"
    marker = tuple(map(int, marker_cells[0]))

    strip_cells = np.argwhere((g != BLACK) & (g != GREY) & (g != MAROON))
    assert len(strip_cells) >= 1, "expected a non-empty instruction strip"
    strip_rows = {int(r) for r, _c in strip_cells}
    assert len(strip_rows) == 1, f"expected one instruction row, got {strip_rows}"
    strip_row = next(iter(strip_rows))
    assert strip_row in {0, h - 1}, f"instruction strip must be on top or bottom row, got {strip_row}"
    strip_cells = strip_cells[np.argsort(strip_cells[:, 1])]
    strip_colors = [int(g[int(r), int(c)]) for r, c in strip_cells]
    assert len(strip_colors) == len(set(strip_colors)), "instruction colors must be distinct"

    path_mask = g == GREY
    path_cells = np.argwhere(path_mask)
    assert len(path_cells) >= 3, "path must contain at least three cells"

    degrees: dict[tuple[int, int], int] = {}
    endpoints: list[tuple[int, int]] = []
    for r, c in path_cells:
        rc = (int(r), int(c))
        deg = sum(
            1
            for rr, cc in _neighbors4(rc[0], rc[1], h, w)
            if bool(path_mask[rr, cc])
        )
        degrees[rc] = deg
        assert deg in {1, 2}, f"path cell {rc} has invalid degree {deg}"
        if deg == 1:
            endpoints.append(rc)

    assert len(endpoints) == 2, f"expected exactly two path endpoints, got {endpoints}"
    start_candidates = [
        endpoint
        for endpoint in endpoints
        if marker in _neighbors4(endpoint[0], endpoint[1], h, w)
    ]
    assert len(start_candidates) == 1, f"maroon marker must identify one endpoint, got {start_candidates}"
    start = start_candidates[0]

    path = _ordered_path_from_start(path_mask, start)
    turns = _turn_indices(path)
    assert len(turns) == len(strip_colors), "instruction strip length must match the number of turns"

    out = g.copy()
    for r, c in strip_cells:
        out[int(r), int(c)] = BLACK

    paint_start = 0
    for turn_idx, color in zip(turns, strip_colors):
        for i in range(paint_start, turn_idx + 1):
            rr, cc = path[i]
            out[rr, cc] = int(color)
        paint_start = turn_idx + 1

    return out


def _sample_increasing(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    values = list(range(lo, hi + 1))
    if len(values) < count:
        return None
    for _ in range(80):
        picked = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _sample_rows(count: int, h: int) -> list[int]:
    rows = [random.randint(1, h - 2)]
    while len(rows) < count:
        nxt = random.randint(1, h - 2)
        if nxt != rows[-1]:
            rows.append(nxt)
    return rows


def _append_segment(path: list[tuple[int, int]], target: tuple[int, int]) -> None:
    r0, c0 = path[-1]
    r1, c1 = target
    assert r0 == r1 or c0 == c1
    dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    r = r0
    c = c0
    while (r, c) != (r1, c1):
        r += dr
        c += dc
        path.append((r, c))


def _build_path(h: int, w: int, strip_side: str, end_style: str, turn_count: int) -> list[tuple[int, int]] | None:
    assert strip_side in {"top", "bottom"}
    if end_style == "right":
        m = turn_count // 2
        cols = _sample_increasing(m, 2, w - 3, MIN_COL_GAP)
        if cols is None:
            return None
        rows = _sample_rows(m + 1, h)
        path = [(rows[0], 0)]
        for i, col in enumerate(cols):
            _append_segment(path, (rows[i], col))
            _append_segment(path, (rows[i + 1], col))
        _append_segment(path, (rows[-1], w - 1))
        return path

    end_row = 0 if strip_side == "bottom" else h - 1
    m = (turn_count + 1) // 2
    cols = _sample_increasing(m, 2, w - 2, MIN_COL_GAP)
    if cols is None:
        return None
    rows = _sample_rows(m, h)
    path = [(rows[0], 0)]
    for i, col in enumerate(cols):
        _append_segment(path, (rows[i], col))
        if i + 1 < len(rows):
            _append_segment(path, (rows[i + 1], col))
        else:
            _append_segment(path, (end_row, col))
    return path


def _marker_location(
    endpoint: tuple[int, int],
    next_cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[int, int] | None:
    r, c = endpoint
    candidates = []
    for rr, cc in _neighbors4(r, c, h, w):
        if (rr, cc) == next_cell or (rr, cc) in occupied:
            continue
        candidates.append((rr, cc))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_strip(grid: np.ndarray, colors: list[int], strip_side: str) -> None:
    h, w = grid.shape
    row = 0 if strip_side == "top" else h - 1
    start_lo = STRIP_MARGIN
    start_hi = w - len(colors) - STRIP_MARGIN
    start_col = random.randint(start_lo, start_hi)
    for i, color in enumerate(colors):
        grid[row, start_col + i] = int(color)


def _build_candidate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)

    strip_side = random.choice(["top", "bottom"])
    end_style = random.choice(["right", "vertical"])
    if end_style == "right":
        turn_count = random.choice(RIGHT_END_TURNS)
    else:
        turn_count = random.choice(VERTICAL_END_TURNS)

    if w < max(8, turn_count + 6) or h < 8:
        return None

    path = _build_path(h, w, strip_side, end_style, turn_count)
    if path is None:
        return None

    if len(set(path)) != len(path):
        return None

    turns = _turn_indices(path)
    if len(turns) != turn_count:
        return None

    start_from_head = random.choice([True, False])
    if start_from_head:
        start_endpoint = path[0]
        next_cell = path[1]
    else:
        start_endpoint = path[-1]
        next_cell = path[-2]

    marker = _marker_location(start_endpoint, next_cell, set(path), h, w)
    if marker is None:
        return None

    strip_colors = random.sample(STRIP_COLORS, turn_count)
    _place_strip(grid, strip_colors, strip_side)

    for r, c in path:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = GREY

    if grid[marker] != BLACK:
        return None
    grid[marker] = MAROON

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")

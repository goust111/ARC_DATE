from __future__ import annotations

import random
from collections import deque

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _occupied_rows(grid: np.ndarray, bg: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if np.any(grid[r] != bg)]


def _marker_color(grid: np.ndarray, bg: int) -> int:
    rows = _occupied_rows(grid, bg)
    assert rows, "expected at least one occupied row"

    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    candidates = []
    for color in colors:
        per_row = [int(np.count_nonzero(grid[r] == color)) for r in rows]
        if all(count == 1 for count in per_row):
            candidates.append(color)

    assert candidates, "expected a color that appears exactly once in each occupied row"
    totals = {color: int(np.count_nonzero(grid == color)) for color in candidates}
    best = min(candidates, key=lambda color: (totals[color], color))
    return int(best)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each occupied row to the left or right edge using the marker pixel
    as a direction cue, then erase the marker.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    marker = _marker_color(g, bg)

    out = np.full((h, w), bg, dtype=int)
    for r in range(h):
        row_values = [int(v) for v in g[r] if int(v) != bg]
        if not row_values:
            continue

        marker_positions = [i for i, v in enumerate(row_values) if v == marker]
        assert len(marker_positions) == 1, "each occupied row must contain exactly one marker"
        marker_idx = int(marker_positions[0])

        body = [v for i, v in enumerate(row_values) if i != marker_idx]
        if not body:
            continue

        goes_left = 2 * marker_idx + 1 < len(row_values)
        if goes_left:
            out[r, : len(body)] = body
        else:
            out[r, w - len(body) :] = body

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (8, 14)
GRID_WIDTH_RANGE = (12, 20)
ROW_COUNT_RANGE = (3, 8)
RUN_LENGTH_RANGE = (3, 8)
SIDE_MARGIN = 1
TOP_BOTTOM_MARGIN = 1
MAX_SHIFT_PER_ROW = 2
MAX_ATTEMPTS = 400


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _marker_index(length: int, direction: str) -> int:
    assert length >= 3
    if direction == "left":
        return random.randint(0, (length - 2) // 2)
    if direction == "right":
        return random.randint((length + 1) // 2, length - 1)
    raise AssertionError(direction)


def _row_directions(n_rows: int) -> list[str]:
    n_left = n_rows // 2
    if n_rows % 2 == 1 and random.random() < 0.5:
        n_left += 1
    directions = ["left"] * n_left + ["right"] * (n_rows - n_left)
    random.shuffle(directions)
    assert "left" in directions and "right" in directions
    return directions


def _pick_row_starts(width: int, lengths: list[int]) -> list[int] | None:
    max_starts = [width - SIDE_MARGIN - length for length in lengths]
    if any(limit < SIDE_MARGIN for limit in max_starts):
        return None

    starts = [random.randint(SIDE_MARGIN, max_starts[0])]
    for i in range(1, len(lengths)):
        prev_start = starts[-1]
        prev_len = lengths[i - 1]
        cur_len = lengths[i]

        overlap_lo = max(SIDE_MARGIN, prev_start - cur_len + 1)
        overlap_hi = min(max_starts[i], prev_start + prev_len - 1)
        if overlap_lo > overlap_hi:
            return None

        local_lo = max(overlap_lo, prev_start - MAX_SHIFT_PER_ROW)
        local_hi = min(overlap_hi, prev_start + MAX_SHIFT_PER_ROW)
        if local_lo <= local_hi:
            start = random.randint(local_lo, local_hi)
        else:
            start = random.randint(overlap_lo, overlap_hi)
        starts.append(int(start))

    return starts


def _expected_output(
    shape: tuple[int, int],
    bg: int,
    body_color: int,
    rows: list[int],
    lengths: list[int],
    directions: list[str],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    for r, length, direction in zip(rows, lengths, directions):
        count = length - 1
        if direction == "left":
            out[r, :count] = body_color
        else:
            out[r, w - count :] = body_color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    max_rows = min(ROW_COUNT_RANGE[1], height - 2 * TOP_BOTTOM_MARGIN)
    if max_rows < ROW_COUNT_RANGE[0]:
        return None
    n_rows = random.randint(ROW_COUNT_RANGE[0], max_rows)
    top = random.randint(TOP_BOTTOM_MARGIN, height - TOP_BOTTOM_MARGIN - n_rows)
    rows = list(range(top, top + n_rows))

    max_length = min(RUN_LENGTH_RANGE[1], width - 2 * SIDE_MARGIN)
    if max_length < RUN_LENGTH_RANGE[0]:
        return None
    lengths = [random.randint(RUN_LENGTH_RANGE[0], max_length) for _ in rows]
    starts = _pick_row_starts(width, lengths)
    if starts is None:
        return None

    bg, body_color, marker_color = random.sample(range(10), 3)
    directions = _row_directions(n_rows)

    grid = np.full((height, width), bg, dtype=int)
    for r, start, length, direction in zip(rows, starts, lengths, directions):
        marker_at = _marker_index(length, direction)
        grid[r, start : start + length] = body_color
        grid[r, start + marker_at] = marker_color

    fg_mask = grid != bg
    if not _is_connected(fg_mask):
        return None
    if int(fg_mask.sum()) * 2 >= grid.size:
        return None

    expected = _expected_output(grid.shape, bg, body_color, rows, lengths, directions)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

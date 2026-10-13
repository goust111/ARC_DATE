from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
PINK = 6
ORANGE = 7
TEAL = 8

STAMP_OFFSETS = (
    (-1, 0, RED),
    (1, 0, TEAL),
    (0, -1, ORANGE),
    (0, 1, PINK),
)

GEN_ATTEMPTS = 400


def _find_pair_rows(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return (gap, row, left_col, right_col) for mirrored blue-marker rows."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    mid = width // 2
    pairs: list[tuple[int, int, int, int]] = []

    for r in range(height):
        cols = np.flatnonzero(g[r] == BLUE)
        assert len(cols) in (0, 2), "each active row must contain exactly two blue pixels"
        if len(cols) == 0:
            continue
        c0, c1 = map(int, cols)
        assert c0 + c1 == 2 * mid, "blue pixels must mirror across the vertical center axis"
        pairs.append((c1 - c0 - 1, int(r), c0, c1))

    assert pairs, "expected at least one mirrored marker pair"
    return pairs


def _apply_stamp(out: np.ndarray, source: np.ndarray, r: int, c: int) -> None:
    height, width = out.shape
    for dr, dc, color in STAMP_OFFSETS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width and int(source[nr, nc]) == BG:
            out[nr, nc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the unique closest mirrored pair with two directional crosses and connect all other mirrored pairs.

    Input structure:
    - The grid uses black background (0) and blue foreground markers (1).
    - Exactly 5 to 7 rows contain a mirrored pair of blue pixels across the vertical center axis.
    - Each active row has a distinct horizontal gap between its two blue pixels, and the closest row is unique.

    Rule:
    - Find every row with exactly two mirrored blue pixels and measure the number of empty cells between them.
    - Identify the unique row with the strictly smallest gap.
    - On that closest row, keep the blue pixels and stamp their orthogonal neighbors: up red (2), down teal (8), left orange (7), right pink (6), skipping out-of-bounds cells.
    - On every other mirrored-pair row, fill the whole horizontal span between the two blue pixels with blue.

    Key insight:
    - The decision is controlled only by relative gap size: the closest mirrored pair branches into local directional decoration, while every wider pair becomes a plain horizontal connection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    pairs = _find_pair_rows(g)
    min_gap = min(gap for gap, _r, _c0, _c1 in pairs)
    closest = [pair for pair in pairs if pair[0] == min_gap]
    assert len(closest) == 1, "closest mirrored pair must be unique"
    _gap, closest_row, left_col, right_col = closest[0]

    for gap, r, c0, c1 in pairs:
        if r == closest_row:
            _apply_stamp(out, g, r, c0)
            _apply_stamp(out, g, r, c1)
        else:
            out[r, c0 : c1 + 1] = BLUE

    return out


def _sample_non_adjacent_rows(height: int, count: int) -> list[int]:
    row_sets = []
    for start in (1, 2):
        rows = list(range(start, height - 1, 2))
        if len(rows) >= count:
            row_sets.append(rows)
    assert row_sets, "height must allow non-adjacent active rows"
    pool = random.choice(row_sets)
    return sorted(random.sample(pool, count))


def _build_case() -> np.ndarray:
    pair_count = random.randint(5, 7)

    min_height = max(13, 2 * pair_count + 1)
    height = random.randint(min_height, min(23, min_height + 6))

    min_width = max(15, 2 * pair_count + 5)
    width_choices = [width for width in range(min_width, 26, 2)]
    assert width_choices, "width choices must stay within ARC limits"
    width = random.choice(width_choices)
    mid = width // 2

    rows = _sample_non_adjacent_rows(height, pair_count)
    distances = random.sample(list(range(2, mid)), pair_count)
    random.shuffle(distances)

    grid = np.zeros((height, width), dtype=int)
    for r, dist in zip(rows, distances):
        grid[r, mid - dist] = BLUE
        grid[r, mid + dist] = BLUE
    return grid


def _validate_case(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    height, width = grid.shape
    if max(height, width) > 30 or width % 2 == 0:
        return False
    if not set(map(int, np.unique(grid))).issubset({BG, BLUE}):
        return False

    try:
        pairs = _find_pair_rows(grid)
    except AssertionError:
        return False

    if not 5 <= len(pairs) <= 7:
        return False

    gaps = [gap for gap, _r, _c0, _c1 in pairs]
    if len(set(gaps)) != len(gaps) or min(gaps) < 3:
        return False

    rows = sorted(r for _gap, r, _c0, _c1 in pairs)
    if any(rows[i + 1] - rows[i] < 2 for i in range(len(rows) - 1)):
        return False

    if any(c0 <= 0 or c1 >= width - 1 for _gap, _r, c0, c1 in pairs):
        return False

    out = main(grid)
    min_gap = min(gaps)
    closest = [pair for pair in pairs if pair[0] == min_gap]
    if len(closest) != 1:
        return False
    _gap, stamp_row, left_col, right_col = closest[0]

    allowed_changes: set[tuple[int, int]] = set()
    for gap, r, c0, c1 in pairs:
        if r == stamp_row:
            if not np.all(out[r, [c0, c1]] == BLUE):
                return False
            if np.any(out[r, c0 + 1 : c1] == BLUE):
                return False
            for c in (c0, c1):
                for dr, dc, color in STAMP_OFFSETS:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if grid[nr, nc] != BG:
                        return False
                    if int(out[nr, nc]) != color:
                        return False
                    allowed_changes.add((nr, nc))
        else:
            if not np.all(out[r, c0 : c1 + 1] == BLUE):
                return False
            for c in range(c0 + 1, c1):
                allowed_changes.add((r, c))

    diff = np.argwhere(out != grid)
    if any((int(r), int(c)) not in allowed_changes for r, c in diff):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if _validate_case(grid):
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")

from __future__ import annotations

import random

import numpy as np


BLACK = 0
CORNER = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
MAX_INTERNAL = 29
MAX_GENERATION_ATTEMPTS = 300


def _runs(values: np.ndarray) -> list[tuple[int, int, int, int]]:
    assert values.ndim == 1 and values.size > 0, "expected a non-empty barcode"
    allowed = set(COLORS)
    vals = [int(v) for v in values]
    assert all(v in allowed for v in vals), "barcodes use colored non-black, non-corner cells"

    runs: list[tuple[int, int, int, int]] = []
    start = 0
    color = vals[0]
    for i, value in enumerate(vals[1:], start=1):
        if value != color:
            runs.append((start, i - 1, color, i - start))
            start = i
            color = value
    runs.append((start, len(vals) - 1, color, len(vals) - start))
    return runs


def _winner(row_color: int, row_len: int, col_color: int, col_len: int) -> int:
    if row_color == col_color:
        return row_color
    if row_len > col_len:
        return row_color
    if col_len > row_len:
        return col_color
    return BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """Project top and left color barcodes into crossing regions, resolving conflicts by segment length.

    Input structure:
    - The top row and left column are fully covered by colored runs, with a single grey corner marker at the upper left.
    - The remaining interior cells are black.
    - Top-row runs define vertical column bands; left-column runs define horizontal row bands.

    Rule:
    - Remove the top row and left column.
    - Fill each row-band and column-band crossing rectangle from the two projected run colors.
    - Matching colors fill with that color; different colors choose the longer run's color; equal-length different-color runs cancel to black.

    Key insight:
    The color of each interior rectangle is decided by the two distant barcode segments that project into it, and conflict ownership depends on segment length rather than local pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    H, W = g.shape
    assert H >= 2 and W >= 2, "expected a corner plus non-empty interior"
    assert int(g[0, 0]) == CORNER, "expected a grey upper-left corner marker"
    assert np.all(g[1:, 1:] == BLACK), "interior must start black"

    col_runs = _runs(g[0, 1:])
    row_runs = _runs(g[1:, 0])

    out = np.full((H - 1, W - 1), BLACK, dtype=int)
    for r0, r1, row_color, row_len in row_runs:
        for c0, c1, col_color, col_len in col_runs:
            color = _winner(row_color, row_len, col_color, col_len)
            out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _expected_from_segments(
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> np.ndarray:
    height = sum(length for _color, length in row_segments)
    width = sum(length for _color, length in col_segments)
    out = np.full((height, width), BLACK, dtype=int)

    r0 = 0
    for row_color, row_len in row_segments:
        c0 = 0
        for col_color, col_len in col_segments:
            color = _winner(row_color, row_len, col_color, col_len)
            out[r0 : r0 + row_len, c0 : c0 + col_len] = color
            c0 += col_len
        r0 += row_len
    return out


def _render_input(
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> np.ndarray:
    height = 1 + sum(length for _color, length in row_segments)
    width = 1 + sum(length for _color, length in col_segments)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, 0] = CORNER

    c = 1
    for color, length in col_segments:
        grid[0, c : c + length] = color
        c += length

    r = 1
    for color, length in row_segments:
        grid[r : r + length, 0] = color
        r += length
    return grid


def _assign_colors(count: int, fixed: dict[int, int], rng: random.Random) -> list[int] | None:
    colors: list[int | None] = [None] * count
    for idx, color in fixed.items():
        colors[idx] = int(color)

    for idx in range(count):
        if colors[idx] is not None:
            if idx and colors[idx - 1] == colors[idx]:
                return None
            continue

        forbidden = set()
        if idx:
            forbidden.add(int(colors[idx - 1]))
        if idx + 1 in fixed:
            forbidden.add(int(fixed[idx + 1]))
        choices = [color for color in COLORS if color not in forbidden]
        if not choices:
            return None
        colors[idx] = rng.choice(choices)

    if any(colors[i] == colors[i - 1] for i in range(1, count)):
        return None
    return [int(color) for color in colors if color is not None]


def _assign_lengths(count: int, fixed: dict[int, int], rng: random.Random) -> list[int]:
    lengths = [0] * count
    for idx, length in fixed.items():
        lengths[idx] = int(length)
    for idx in range(count):
        if lengths[idx] == 0:
            lengths[idx] = rng.randint(1, 6)
    return lengths


def _branch_counts(
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
) -> dict[str, int]:
    counts = {"same": 0, "row_longer": 0, "col_longer": 0, "cancel": 0, "same_equal": 0}
    for row_color, row_len in row_segments:
        for col_color, col_len in col_segments:
            if row_color == col_color:
                counts["same"] += 1
                if row_len == col_len:
                    counts["same_equal"] += 1
            elif row_len > col_len:
                counts["row_longer"] += 1
            elif col_len > row_len:
                counts["col_longer"] += 1
            else:
                counts["cancel"] += 1
    return counts


def _priority_output(
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    mode: str,
) -> np.ndarray:
    height = sum(length for _color, length in row_segments)
    width = sum(length for _color, length in col_segments)
    out = np.full((height, width), BLACK, dtype=int)

    r0 = 0
    for row_color, row_len in row_segments:
        c0 = 0
        for col_color, col_len in col_segments:
            if mode == "row":
                color = row_color
            elif mode == "col":
                color = col_color
            elif mode == "same":
                color = row_color if row_color == col_color else BLACK
            else:
                raise ValueError(mode)
            out[r0 : r0 + row_len, c0 : c0 + col_len] = color
            c0 += col_len
        r0 += row_len
    return out


def _semantic_checks(
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    expected: np.ndarray,
) -> bool:
    counts = _branch_counts(row_segments, col_segments)
    required = ("same", "same_equal", "row_longer", "col_longer", "cancel")
    if any(counts[key] == 0 for key in required):
        return False
    if BLACK not in set(map(int, expected.ravel())):
        return False
    if len(set(map(int, expected.ravel())) - {BLACK}) < 3:
        return False
    for mode in ("row", "col", "same"):
        if np.array_equal(expected, _priority_output(row_segments, col_segments, mode)):
            return False
    return True


def _sample_segments(rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    n_rows = rng.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    n_cols = rng.randint(MIN_SEGMENTS, MAX_SEGMENTS)

    short = rng.randint(1, 3)
    mid = rng.randint(short + 1, min(short + 3, 5))
    long = rng.randint(mid + 1, 6)

    r_same, r_cancel, r_long = rng.sample(range(n_rows), 3)
    c_same, c_cancel, c_long = rng.sample(range(n_cols), 3)

    shared, row_cancel, col_cancel, row_long, col_long = rng.sample(COLORS, 5)
    row_fixed_colors = {r_same: shared, r_cancel: row_cancel, r_long: row_long}
    col_fixed_colors = {c_same: shared, c_cancel: col_cancel, c_long: col_long}

    row_colors = _assign_colors(n_rows, row_fixed_colors, rng)
    col_colors = _assign_colors(n_cols, col_fixed_colors, rng)
    if row_colors is None or col_colors is None:
        return None

    row_lengths = _assign_lengths(n_rows, {r_same: mid, r_cancel: short, r_long: long}, rng)
    col_lengths = _assign_lengths(n_cols, {c_same: mid, c_cancel: short, c_long: long}, rng)
    if sum(row_lengths) > MAX_INTERNAL or sum(col_lengths) > MAX_INTERNAL:
        return None

    row_segments = list(zip(row_colors, row_lengths))
    col_segments = list(zip(col_colors, col_lengths))
    return row_segments, col_segments


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sampled = _sample_segments(rng)
    if sampled is None:
        return None
    row_segments, col_segments = sampled

    expected = _expected_from_segments(row_segments, col_segments)
    if not _semantic_checks(row_segments, col_segments, expected):
        return None

    inp = _render_input(row_segments, col_segments)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-barcode input")

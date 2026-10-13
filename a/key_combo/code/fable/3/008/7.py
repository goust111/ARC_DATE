from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

GEN_ATTEMPTS = 500
GAP_CHOICES = (4, 5, 6, 7, 8)
EVEN_GAPS = (4, 6, 8)
ODD_GAPS = (5, 7)


def _red_segments(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return left-anchored red segments as (row, length)."""
    assert np.all((grid == BLACK) | (grid == RED)), "input must contain only black and red"

    segments: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] == RED)
        if cols.size == 0:
            continue

        length = int(cols.size)
        assert length >= 2, "each red segment must have length at least two"
        assert np.array_equal(cols, np.arange(length)), "red segments must be contiguous from the left edge"
        segments.append((int(r), length))

    assert len(segments) >= 2, "expected multiple red source segments"
    lengths = [length for _row, length in segments]
    assert len(set(lengths)) == len(lengths), "red segment lengths must be distinct"
    return segments


def _nearest_indices(row: int, seed_rows: list[int]) -> list[int]:
    dists = [abs(int(row) - int(seed_row)) for seed_row in seed_rows]
    best = min(dists)
    return [i for i, dist in enumerate(dists) if dist == best]


def _render_from_segments(shape: tuple[int, int], segments: list[tuple[int, int]]) -> np.ndarray:
    height, width = shape
    seed_rows = [row for row, _length in segments]
    out = np.full((height, width), BLACK, dtype=int)

    for row, length in segments:
        out[row, :length] = RED

    red_rows = set(seed_rows)
    for r in range(height):
        if r in red_rows:
            continue

        owners = _nearest_indices(r, seed_rows)
        if len(owners) != 1:
            continue

        owner = owners[0]
        seed_row, seed_length = segments[owner]
        dist = abs(r - seed_row)
        if r < seed_row:
            fill_width = seed_length + dist
            assert fill_width <= width, "green wedge must fit inside the grid"
            out[r, :fill_width] = GREEN
        else:
            fill_width = seed_length - dist
            if fill_width > 0:
                out[r, :fill_width] = BLUE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nearest-owned triangular wedges from multiple left-edge red segments.

    Input structure:
    - The grid has a black background.
    - Several horizontal red segments start at the left edge, each on a different row.
    - The red segments are contiguous and have distinct lengths.

    Rule:
    - Keep every red segment unchanged.
    - Each non-red row belongs to the vertically nearest red segment; rows tied between two nearest segments remain black.
    - Rows above their owning segment are filled green from the left edge with width segment length plus row distance.
    - Rows below their owning segment are filled blue from the left edge with width segment length minus row distance; nonpositive widths leave the row black.

    Key insight:
    - A single red segment's expanding and shrinking wedge rule is clipped by nearest-segment territories, with equal-distance rows acting as blank separators.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids are at most 30 by 30"

    segments = _red_segments(g)
    return _render_from_segments(g.shape, segments)


def _row_labels(height: int, seed_rows: list[int]) -> list[tuple[str, int | None]]:
    labels: list[tuple[str, int | None]] = []
    red_to_index = {row: i for i, row in enumerate(seed_rows)}
    for r in range(height):
        if r in red_to_index:
            labels.append(("red", red_to_index[r]))
            continue

        owners = _nearest_indices(r, seed_rows)
        if len(owners) != 1:
            labels.append(("tie", None))
        else:
            labels.append(("owner", owners[0]))
    return labels


def _max_green_extras(height: int, rows: list[int]) -> list[int]:
    extras = [0 for _ in rows]
    for r, (kind, owner) in enumerate(_row_labels(height, rows)):
        if kind != "owner" or owner is None:
            continue
        if r < rows[owner]:
            extras[owner] = max(extras[owner], rows[owner] - r)
    return extras


def _choose_layout(rng: random.Random) -> tuple[int, list[int], int] | None:
    for _ in range(200):
        count = rng.randint(3, 5)
        top_margin = rng.randint(2, 6)
        bottom_margin = rng.randint(2, 7)
        gaps = [rng.choice(GAP_CHOICES) for _ in range(count - 1)]

        if not any(gap % 2 == 0 for gap in gaps):
            gaps[rng.randrange(len(gaps))] = rng.choice(EVEN_GAPS)
        if not any(gap % 2 == 1 for gap in gaps):
            gaps[rng.randrange(len(gaps))] = rng.choice(ODD_GAPS)

        height = top_margin + sum(gaps) + bottom_margin + 1
        if height > 30:
            continue

        rows = [top_margin]
        for gap in gaps:
            rows.append(rows[-1] + gap)
        return height, rows, bottom_margin
    return None


def _choose_lengths(
    rng: random.Random,
    width: int,
    rows: list[int],
    max_green_extra: list[int],
    bottom_margin: int,
) -> list[int] | None:
    uppers = [min(12, width - extra) for extra in max_green_extra]
    uppers[-1] = min(uppers[-1], bottom_margin)
    if any(upper < 2 for upper in uppers):
        return None

    order = sorted(range(len(rows)), key=lambda i: uppers[i])
    assigned: list[int | None] = [None for _ in rows]
    used: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        choices = list(range(2, uppers[idx] + 1))
        rng.shuffle(choices)
        for value in choices:
            if value in used:
                continue
            assigned[idx] = value
            used.add(value)
            if backtrack(pos + 1):
                return True
            used.remove(value)
            assigned[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(value) for value in assigned if value is not None]


def _expected_row(grid_width: int, color: int | None, fill_width: int = 0) -> np.ndarray:
    row = np.full(grid_width, BLACK, dtype=int)
    if color is not None and fill_width > 0:
        row[:fill_width] = color
    return row


def _passes_semantic_checks(inp: np.ndarray, expected: np.ndarray, segments: list[tuple[int, int]]) -> bool:
    height, width = inp.shape
    rows = [row for row, _length in segments]
    labels = _row_labels(height, rows)

    has_top_green = False
    has_between_blue = False
    has_between_green = False
    has_tie_blank = False
    has_blue_fill = False
    has_width_exhausted_blank = False

    for r, (kind, owner) in enumerate(labels):
        if kind == "red":
            assert owner is not None
            seed_row, seed_length = segments[owner]
            if not np.array_equal(expected[r], _expected_row(width, RED, seed_length)):
                return False
            if seed_row != r:
                return False
            continue

        if kind == "tie":
            if not np.all(expected[r] == BLACK):
                return False
            has_tie_blank = True
            continue

        assert owner is not None
        seed_row, seed_length = segments[owner]
        dist = abs(r - seed_row)

        if r < seed_row:
            fill_width = seed_length + dist
            if fill_width > width:
                return False
            if not np.array_equal(expected[r], _expected_row(width, GREEN, fill_width)):
                return False
            if r < rows[0]:
                has_top_green = True
            if owner > 0 and rows[owner - 1] < r < seed_row:
                has_between_green = True
        else:
            fill_width = seed_length - dist
            if fill_width > 0:
                if not np.array_equal(expected[r], _expected_row(width, BLUE, fill_width)):
                    return False
                has_blue_fill = True
                if owner + 1 < len(rows) and seed_row < r < rows[owner + 1]:
                    has_between_blue = True
            else:
                if not np.all(expected[r] == BLACK):
                    return False
                has_width_exhausted_blank = True

    colors = set(map(int, np.unique(expected)))
    has_all_output_colors = {BLACK, BLUE, RED, GREEN}.issubset(colors)
    return (
        has_all_output_colors
        and has_top_green
        and has_between_blue
        and has_between_green
        and has_tie_blank
        and has_blue_fill
        and has_width_exhausted_blank
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layout = _choose_layout(rng)
    if layout is None:
        return None
    height, rows, bottom_margin = layout

    max_green_extra = _max_green_extras(height, rows)
    width_min = max(8, max(max_green_extra) + len(rows) + 1)
    if width_min > 30:
        return None
    width = rng.randint(width_min, 30)

    lengths = _choose_lengths(rng, width, rows, max_green_extra, bottom_margin)
    if lengths is None:
        return None

    segments = list(zip(rows, lengths))
    inp = np.full((height, width), BLACK, dtype=int)
    for row, length in segments:
        inp[row, :length] = RED

    expected = _render_from_segments(inp.shape, segments)
    if not _passes_semantic_checks(inp, expected, segments):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")

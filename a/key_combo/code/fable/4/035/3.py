from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GRAY = 5
RECTANGLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400


def _mode(grid: np.ndarray) -> int:
    """Return the most frequent color, as in arc_utils.get_bg."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _full_span_bars(grid: np.ndarray, background: int) -> tuple[list[int], list[int]]:
    """Locate full-width rows and full-height columns in (row, column) coordinates."""
    rows = [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] != background))]
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] != background))]
    return rows, cols


def _decode_rectangles(
    grid: np.ndarray,
    background: int,
    bar_rows: list[int],
    bar_cols: list[int],
) -> list[tuple[int, int, int, int, int]]:
    """Decode (color, r0, c0, r1, c1), combining the two kinds of cross-section."""
    colors = sorted(set(map(int, np.unique(grid))) - {background, GRAY})
    assert colors, "expected at least one colored cross-section"

    rectangles: list[tuple[int, int, int, int, int]] = []
    for color in colors:
        # Horizontal bars reveal columns only.  Every occurrence must report the
        # same complete, contiguous column interval.
        horizontal_ranges: list[tuple[int, int]] = []
        for r in bar_rows:
            cols = np.flatnonzero(grid[r, :] == color)
            if cols.size:
                c0, c1 = int(cols[0]), int(cols[-1])
                assert np.array_equal(cols, np.arange(c0, c1 + 1)), (
                    "a horizontal cross-section must be contiguous"
                )
                horizontal_ranges.append((c0, c1))
        assert horizontal_ranges, "each color needs a horizontal cross-section"
        assert len(set(horizontal_ranges)) == 1, (
            "same-color horizontal cross-sections must give one column range"
        )
        c0, c1 = horizontal_ranges[0]

        # Vertical bars reveal rows only.  This is deliberately decoded
        # separately: neither orientation supplies the other pair of bounds.
        vertical_ranges: list[tuple[int, int]] = []
        for c in bar_cols:
            rows = np.flatnonzero(grid[:, c] == color)
            if rows.size:
                r0, r1 = int(rows[0]), int(rows[-1])
                assert np.array_equal(rows, np.arange(r0, r1 + 1)), (
                    "a vertical cross-section must be contiguous"
                )
                vertical_ranges.append((r0, r1))
        assert vertical_ranges, "each color needs a vertical cross-section"
        assert len(set(vertical_ranges)) == 1, (
            "same-color vertical cross-sections must give one row range"
        )
        r0, r1 = vertical_ranges[0]
        assert r0 < r1 and c0 < c1, "the reconstructed object must be a rectangle"

        # Check that the color consists of all and only the intersections of
        # this hidden rectangle with every bar it crosses.
        projected = np.zeros(grid.shape, dtype=bool)
        for r in bar_rows:
            if r0 <= r <= r1:
                projected[r, c0 : c1 + 1] = True
        for c in bar_cols:
            if c0 <= c <= c1:
                projected[r0 : r1 + 1, c] = True
        assert np.array_equal(grid == color, projected), (
            "colored pixels must be the complete two-axis projection of one rectangle"
        )
        rectangles.append((color, r0, c0, r1, c1))

    # With no stated depth order, overlapping hidden rectangles would have an
    # ambiguous visible color.  Reject such inputs rather than inventing one.
    occupied = np.zeros(grid.shape, dtype=bool)
    for _color, r0, c0, r1, c1 in rectangles:
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        assert not bool(region.any()), "hidden rectangles must not overlap"
        region[:] = True
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct solid hidden rectangles from their horizontal and vertical bar cross-sections.

    Input structure:
    - A dominant background contains gray rows spanning the full width and gray columns spanning the full height, with unequal gaps.
    - Continuous colored segments replace gray where never-drawn solid rectangles cross those bars; every color occurs on at least one bar of each orientation.
    - Apart from the bars and their colored segments, all cells are background.

    Rule:
    - For each color, horizontal-bar segments determine one column interval and vertical-bar segments determine one row interval.
    - Fill the solid rectangle formed by those two intervals, then restore every full-span row and column to pure gray above all rectangles.

    Key insight:
    - Each visible segment is only a cross-section: the horizontal and vertical projections must be combined to recover both axes of an otherwise invisible rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    background = _mode(g)
    assert background != GRAY, "gray bars must differ from the background"

    bar_rows, bar_cols = _full_span_bars(g, background)
    assert len(bar_rows) >= 2 and len(bar_cols) >= 2, (
        "expected several horizontal and vertical full-span bars"
    )

    bar_mask = np.zeros(g.shape, dtype=bool)
    bar_mask[bar_rows, :] = True
    bar_mask[:, bar_cols] = True
    assert np.all(g[bar_mask] != background), "bars must span the complete grid"
    assert np.all(g[~bar_mask] == background), "foreground may occur only on bars"
    assert np.all((g == GRAY) <= bar_mask), "gray may occur only on full-span bars"

    rectangles = _decode_rectangles(g, background, bar_rows, bar_cols)

    out = np.full(g.shape, background, dtype=int)
    for color, r0, c0, r1, c1 in rectangles:
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    out[bar_rows, :] = GRAY
    out[:, bar_cols] = GRAY
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _sample_bars(length: int, rng: random.Random) -> list[int] | None:
    """Sample 3--4 bars with visibly unequal gaps and margins at both ends."""
    count = rng.randint(3, 4)
    for _ in range(80):
        gaps = [rng.randint(4, 7) for _ in range(count - 1)]
        if len(set(gaps)) == 1:
            continue
        span = sum(gaps)
        max_start = length - 3 - span
        if max_start < 2:
            continue
        start = rng.randint(2, max_start)
        bars = [start]
        for gap in gaps:
            bars.append(bars[-1] + gap)
        return bars
    return None


def _axis_intervals(length: int, bars: list[int], multiplicity: int) -> list[tuple[int, int]]:
    """Enumerate off-bar intervals containing exactly one or at least two bars."""
    intervals: list[tuple[int, int]] = []
    bar_set = set(bars)
    for lo in range(1, length - 2):
        if lo in bar_set:
            continue
        for hi in range(lo + 2, min(length - 2, lo + 12) + 1):
            if hi in bar_set:
                continue
            inside = sum(lo <= p <= hi for p in bars)
            if (multiplicity == 1 and inside == 1) or (multiplicity == 2 and inside >= 2):
                intervals.append((lo, hi))
    return intervals


def _separated(candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    """Require at least one background row or column between colored rectangles."""
    r0, c0, r1, c1 = candidate
    for q0, d0, q1, d1 in placed:
        if not (r1 + 1 < q0 or q1 + 1 < r0 or c1 + 1 < d0 or d1 + 1 < c0):
            return False
    return True


def _place_rectangles(
    height: int,
    width: int,
    bar_rows: list[int],
    bar_cols: list[int],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    """Place counterexample profiles: 1xmany and manyx1, plus varied extras."""
    profiles: list[tuple[int, int]] = [(1, 2), (2, 1)]
    extras = ((1, 1), (2, 2), (1, 2), (2, 1))
    profiles.extend(rng.choice(extras) for _ in range(count - 2))
    rng.shuffle(profiles)

    row_options = {m: _axis_intervals(height, bar_rows, m) for m in (1, 2)}
    col_options = {m: _axis_intervals(width, bar_cols, m) for m in (1, 2)}
    if any(not opts for opts in (*row_options.values(), *col_options.values())):
        return None

    placed: list[tuple[int, int, int, int]] = []
    for row_mult, col_mult in profiles:
        choices: list[tuple[int, int, int, int]] = []
        for _ in range(1200):
            r0, r1 = rng.choice(row_options[row_mult])
            c0, c1 = rng.choice(col_options[col_mult])
            candidate = (r0, c0, r1, c1)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if 12 <= area <= 100 and (r1 - r0) != (c1 - c0) and _separated(candidate, placed):
                choices.append(candidate)
                if len(choices) >= 40:
                    break
        if not choices:
            return None
        placed.append(rng.choice(choices))
    return placed


def _render_pair(
    shape: tuple[int, int],
    bar_rows: list[int],
    bar_cols: list[int],
    rectangles: list[tuple[int, int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Render the clue grid and an oracle output independently of main()."""
    inp = np.full(shape, BACKGROUND, dtype=int)
    inp[bar_rows, :] = GRAY
    inp[:, bar_cols] = GRAY

    expected = np.full(shape, BACKGROUND, dtype=int)
    for color, r0, c0, r1, c1 in rectangles:
        expected[r0 : r1 + 1, c0 : c1 + 1] = color
        for r in bar_rows:
            if r0 <= r <= r1:
                inp[r, c0 : c1 + 1] = color
        for c in bar_cols:
            if c0 <= c <= c1:
                inp[r0 : r1 + 1, c] = color

    expected[bar_rows, :] = GRAY
    expected[:, bar_cols] = GRAY
    return inp, expected


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    bar_rows: list[int],
    bar_cols: list[int],
    rectangles: list[tuple[int, int, int, int, int]],
) -> bool:
    """Check rule semantics, axis necessity, uniqueness, and non-degeneracy."""
    if inp.ndim != 2 or max(inp.shape) > 30 or inp.shape != expected.shape:
        return False
    found_rows, found_cols = _full_span_bars(inp, BACKGROUND)
    if found_rows != bar_rows or found_cols != bar_cols:
        return False
    if len(set(np.diff(bar_rows))) == 1 or len(set(np.diff(bar_cols))) == 1:
        return False
    if np.array_equal(inp, expected):
        return False

    actual_profiles: set[tuple[int, int]] = set()
    occupied = np.zeros(inp.shape, dtype=bool)
    for color, r0, c0, r1, c1 in rectangles:
        h_rows = [r for r in bar_rows if r0 <= r <= r1]
        v_cols = [c for c in bar_cols if c0 <= c <= c1]
        if not h_rows or not v_cols:
            return False
        actual_profiles.add((1 if len(h_rows) == 1 else 2, 1 if len(v_cols) == 1 else 2))

        # Horizontal evidence reaches both column boundaries but cannot reveal
        # either row boundary; vertical evidence has the complementary role.
        horizontal = np.zeros(inp.shape, dtype=bool)
        horizontal[h_rows, c0 : c1 + 1] = True
        vertical = np.zeros(inp.shape, dtype=bool)
        vertical[r0 : r1 + 1, v_cols] = True
        if not (np.any(horizontal[:, c0]) and np.any(horizontal[:, c1])):
            return False
        if not (np.any(vertical[r0, :]) and np.any(vertical[r1, :])):
            return False
        if r0 >= min(h_rows) or r1 <= max(h_rows):
            return False
        if c0 >= min(v_cols) or c1 <= max(v_cols):
            return False
        if not np.array_equal(inp == color, horizontal | vertical):
            return False

        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if bool(region.any()):
            return False
        region[:] = True
        if not bool(np.any(expected[r0 : r1 + 1, c0 : c1 + 1] == color)):
            return False

    if (1, 2) not in actual_profiles or (2, 1) not in actual_profiles:
        return False
    if set(map(int, np.unique(inp))) - {BACKGROUND, GRAY} != {r[0] for r in rectangles}:
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return np.array_equal(solved, expected)


def _generation_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(20, 30)
    width = rng.randint(20, 30)
    bar_rows = _sample_bars(height, rng)
    bar_cols = _sample_bars(width, rng)
    if bar_rows is None or bar_cols is None:
        return None

    rectangle_count = rng.randint(2, 4)
    boxes = _place_rectangles(height, width, bar_rows, bar_cols, rectangle_count, rng)
    if boxes is None:
        return None
    colors = rng.sample(RECTANGLE_COLORS, rectangle_count)
    rectangles = [(color, *box) for color, box in zip(colors, boxes)]

    inp, expected = _render_pair((height, width), bar_rows, bar_cols, rectangles)
    if _mode(inp) != BACKGROUND:
        return None
    if not _validate_generated(inp, expected, bar_rows, bar_cols, rectangles):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, semantically validated hidden-rectangle instance."""
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generation_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-rectangle input")

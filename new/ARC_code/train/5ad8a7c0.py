from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the innermost mirrored marker pair(s) by filling their horizontal span.

    Input structure
    - The grid has one background color and one foreground color.
    - Each foreground row contains exactly two foreground pixels mirrored across the vertical center.

    Rule
    - Find the foreground row pair(s) with the smallest horizontal gap between the two markers.
    - Fill every pixel between those two markers in each selected row with the foreground color.
    - Leave all other pixels unchanged.

    Key insight
    - Wider marker pairs are context; the closest marker pair(s) show the unresolved horizontal segment.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    foreground = colors_present(g, exclude={bg})
    assert len(foreground) == 1, "expected exactly one foreground color"
    color = foreground[0]

    pairs = []
    w = g.shape[1]
    for r, row in enumerate(g):
        cols = np.flatnonzero(row == color)
        assert len(cols) in (0, 2), "foreground rows must contain exactly two markers"
        if len(cols) == 2:
            c0, c1 = map(int, cols)
            assert c0 + c1 == w - 1, "marker pairs must be horizontally mirrored"
            pairs.append((c1 - c0, r, c0, c1))

    assert pairs, "expected at least one foreground marker pair"
    min_gap = min(gap for gap, _r, _c0, _c1 in pairs)

    out = g.copy()
    for gap, r, c0, c1 in pairs:
        if gap == min_gap:
            out[r, c0 : c1 + 1] = color
    return out

# --------------- generator (att2, o5x) ---------------
# Canvas and palette.
HEIGHT_RANGE = (4, 8)
WIDTH_CHOICES = (6, 8, 10, 12)
FOREGROUND_COLOR = 2

# Gap-level and row sampling.
MAX_GAP_LEVELS = 3
SINGLE_LEVEL_MIN_ROWS = 2
MIN_GAP_ROW_RANGE = (1, 2)

# Rejection loop.
GENERATION_ATTEMPTS = 300


def _cols_for_gap(width: int, gap: int) -> tuple[int, int]:
    c0 = (width - 1 - gap) // 2
    return c0, width - 1 - c0


def _sample_gap_levels(width: int) -> list[int]:
    all_gaps = list(range(width - 1, 0, -2))
    level_count = random.randint(1, min(MAX_GAP_LEVELS, len(all_gaps)))
    return sorted(random.sample(all_gaps, level_count))


def _sample_row_gaps(height: int, chosen_gaps: list[int]) -> list[int | None]:
    """Return a shuffled per-row gap plan; None means the row is empty."""
    min_gap = chosen_gaps[0]
    non_min_gaps = chosen_gaps[1:]
    min_rows = SINGLE_LEVEL_MIN_ROWS if len(chosen_gaps) == 1 else random.randint(*MIN_GAP_ROW_RANGE)
    row_gaps: list[int | None] = [min_gap] * min_rows + non_min_gaps
    extras: list[int | None] = [None] + non_min_gaps
    row_gaps += [random.choice(extras) for _ in range(height - len(row_gaps))]
    random.shuffle(row_gaps)
    return row_gaps


def _render_case(height: int, width: int, row_gaps: list[int | None], min_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Render the input grid and the expected solver output for oracle checking."""
    grid = np.zeros((height, width), dtype=int)
    expected = grid.copy()

    for r, gap in enumerate(row_gaps):
        if gap is None:
            continue
        c0, c1 = _cols_for_gap(width, gap)
        grid[r, [c0, c1]] = FOREGROUND_COLOR
        if gap == min_gap:
            expected[r, c0 : c1 + 1] = FOREGROUND_COLOR
        else:
            expected[r, [c0, c1]] = FOREGROUND_COLOR
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.choice(WIDTH_CHOICES)
    chosen_gaps = _sample_gap_levels(width)
    row_gaps = _sample_row_gaps(height, chosen_gaps)
    grid, expected = _render_case(height, width, row_gaps, chosen_gaps[0])

    try:
        return grid if np.array_equal(main(grid), expected) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5ad8a7c0 input")

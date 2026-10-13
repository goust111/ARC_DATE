from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def _minimal_period(g: np.ndarray, bg: int) -> int:
    width = g.shape[1]
    coords = np.argwhere(g != bg)
    assert coords.size > 0, "expected at least one foreground pixel"
    assert np.any(coords[:, 1] == 0), "foreground motif must start at the left edge"

    for period in range(1, width):
        tile_colors: dict[tuple[int, int], int] = {}
        ok = True
        for r, c in coords:
            key = (int(r), int(c) % period)
            color = int(g[r, c])
            if key in tile_colors and tile_colors[key] != color:
                ok = False
                break
            tile_colors[key] = color
        if ok:
            assert int(coords[:, 1].max()) >= period, "expected at least a partial repeated motif"
            return period

    raise AssertionError("expected a consistent horizontal period smaller than the grid width")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the partially shown horizontal motif by repeating its minimal column period across the grid.

    Input structure
    The input is a rectangular grid with a dominant background color and a
    foreground motif drawn from the left edge across part of the rows.

    Rule
    Find the smallest column period for which all visible foreground pixels agree
    by row and column residue, then repeat that period across the full width.

    Key insight
    The background cells to the right are not empty final cells; they are the
    missing later copies of the same minimal horizontal tile already visible at
    the left.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    height, width = g.shape
    assert width > 1, "expected a grid wide enough to repeat a motif"

    period = _minimal_period(g, bg)
    tile = np.full((height, period), bg, dtype=int)
    for r, c in np.argwhere(g != bg):
        tile[int(r), int(c) % period] = int(g[r, c])

    out = np.empty_like(g)
    for c in range(width):
        out[:, c] = tile[:, c % period]
    return out

# --------------- generator (o5x) ---------------

# Canvas/tile shape.
GRID_HEIGHT, GRID_WIDTH, BACKGROUND = 5, 15, 0
PERIOD_BOUNDS, ACTIVE_ROWS_BOUNDS = (3, 6), (2, 3)
# Palette and input-prefix visibility.
COLOR_VALUES, COLOR_COUNT_BOUNDS = range(1, 10), (2, 3)
CUTOFF_EXTRA_BOUNDS = (1, 2)
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300

def _gen_column_colors(period: int, n_colors: int) -> list[int]:
    colors = random.sample(COLOR_VALUES, n_colors)
    random.shuffle(colors)

    if n_colors == 2:
        first_count = period // 2
        if period % 2 and random.choice([False, True]):
            first_count += 1
        word = [colors[0]] * first_count + [colors[1]] * (period - first_count)
    else:
        base = period // 3
        counts = [base, base, base]
        for i in random.sample(range(3), period - 3 * base):
            counts[i] += 1
        rest = [colors[1]] * counts[1] + [colors[2]] * counts[2]
        random.shuffle(rest)
        word = [colors[0]] * counts[0] + rest

    if random.choice([False, True]):
        word.reverse()
    return word

def _gen_nested_column_sets(period: int, active_rows: int) -> list[set[int]]:
    """Return top-to-bottom nested foreground column sets for active rows."""
    row_sets = [set(range(period))]
    for _ in range(active_rows - 1):
        lower = row_sets[-1]
        if len(lower) == 1:
            count = 1
        else:
            count = random.randint(1, len(lower) - 1)
        row_sets.append(set(random.sample(sorted(lower), count)))
    row_sets.reverse()
    return row_sets

def _build_tile(period: int, active_rows: int, row_sets: list[set[int]], column_colors: list[int]) -> np.ndarray:
    tile = np.full((GRID_HEIGHT, period), BACKGROUND, dtype=int)
    first_active = GRID_HEIGHT - active_rows
    for offset, cols in enumerate(row_sets):
        r = first_active + offset
        for c in cols:
            tile[r, c] = column_colors[c]
    return tile

def _make_input_prefix(out: np.ndarray, period: int) -> np.ndarray:
    """Hide columns after the sampled cutoff while leaving one full repeat visible."""
    cutoff = random.randint(
        period + CUTOFF_EXTRA_BOUNDS[0],
        min(2 * period + CUTOFF_EXTRA_BOUNDS[1], GRID_WIDTH - 1),
    )
    inp = out.copy()
    inp[:, cutoff:] = BACKGROUND
    return inp

def _gen_attempt() -> np.ndarray | None:
    period = random.randint(*PERIOD_BOUNDS)
    active_rows = random.randint(*ACTIVE_ROWS_BOUNDS)
    row_sets = _gen_nested_column_sets(period, active_rows)
    column_colors = _gen_column_colors(period, random.randint(*COLOR_COUNT_BOUNDS))

    tile = _build_tile(period, active_rows, row_sets, column_colors)
    out = np.empty((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    for c in range(GRID_WIDTH):
        out[:, c] = tile[:, c % period]
    inp = _make_input_prefix(out, period)
    try:
        pred = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(pred, out):
        return None
    return inp

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid d8c310e9 input")

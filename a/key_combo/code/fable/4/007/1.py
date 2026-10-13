from __future__ import annotations

from collections import Counter
import random

import numpy as np


RED = 2
GRAY = 5
ARC_COLORS = tuple(range(10))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _minimal_column_period(mask: np.ndarray) -> int:
    width = int(mask.shape[1])
    for period in range(1, width // 2 + 1):
        if np.array_equal(mask[:, period:], mask[:, :-period]):
            return period
    raise AssertionError("visible grey pattern must contain at least two column periods")


def _right_edge_red_runs(grid: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    height, width = map(int, grid.shape)
    for r in range(height):
        cols = np.flatnonzero(grid[r] == RED)
        if cols.size == 0:
            continue
        start = int(cols[0])
        end = int(cols[-1])
        assert end == width - 1, "red runs must touch the right edge"
        assert np.array_equal(cols, np.arange(start, width)), "each red run must be contiguous"
        runs.append((r, start, width - start))
    assert runs, "expected at least one red run"
    return runs


def _nearest_extended_barrier(row_tile: np.ndarray, before_col: int) -> int | None:
    period = int(row_tile.shape[0])
    for c in range(before_col - 1, -1, -1):
        if bool(row_tile[c % period]):
            return c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each right-edge red run left until the nearest barrier in the virtual grey period stops it.

    Input structure:
    - The background is the dominant color; red (2) runs are horizontal and attached to the right edge.
    - Grey (5) pixels appear only in a left prefix of columns and form a truncated column-periodic barrier pattern.
    - The visible grey prefix contains at least two complete repeats of its unique minimal column period.

    Rule:
    - Infer the minimal column period from the visible grey pixels and mentally repeat it across the whole width.
    - For each red run, find the nearest same-row grey barrier strictly to the left of the run in that virtual repetition.
    - Move the run so its left cell is one column right of that barrier, or to column 0 if the repeated row has no barrier.
    - Preserve only the originally drawn grey pixels and the moved red runs; virtual grey cells remain background.

    Key insight:
    - The blocking object is not fully drawn: the stopping column is usually determined by an invisible periodic extension.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert height <= 30 and width <= 30, "ARC grids must fit inside 30x30"

    bg = _background(g)
    assert bg not in (RED, GRAY), "background must be distinct from red and grey"
    assert set(map(int, np.unique(g))) <= {bg, RED, GRAY}, "unexpected colors"

    grey_mask = g == GRAY
    grey_cols = np.flatnonzero(np.any(grey_mask, axis=0))
    assert grey_cols.size > 0, "expected a visible grey pattern"
    visible_width = int(grey_cols[-1]) + 1
    assert visible_width < width, "grey pattern must be visibly truncated before the right edge"
    assert not np.any(grey_mask[:, visible_width:]), "grey pixels must stay in the visible left prefix"

    visible_mask = grey_mask[:, :visible_width]
    period = _minimal_column_period(visible_mask)
    assert visible_width >= 2 * period, "visible grey prefix must show at least two periods"
    tile = visible_mask[:, :period]

    out = np.full(g.shape, bg, dtype=int)
    out[grey_mask] = GRAY

    for r, old_start, length in _right_edge_red_runs(g):
        barrier = _nearest_extended_barrier(tile[r], old_start)
        new_start = 0 if barrier is None else barrier + 1
        assert new_start + length <= width, "slid run must stay in bounds"
        out[r, new_start : new_start + length] = RED

    return out


# ---------------- generator ----------------

GEN_ATTEMPTS = 500
TILE_ATTEMPTS = 200
MIN_HEIGHT, MAX_HEIGHT = 9, 18
MIN_WIDTH, MAX_WIDTH = 18, 30
PERIOD_RANGE = (3, 6)
RUN_LENGTHS = (2, 3, 4, 5)


def _tile_has_period(tile: np.ndarray, period: int) -> bool:
    return np.array_equal(tile[:, period:], tile[:, :-period])


def _make_period_tile(height: int, period: int) -> tuple[np.ndarray, list[int]] | None:
    for _ in range(TILE_ATTEMPTS):
        tile = np.zeros((height, period), dtype=bool)
        empty_count = random.randint(1, max(1, min(3, height // 4)))
        empty_rows = sorted(random.sample(range(height), empty_count))
        active_rows = [r for r in range(height) if r not in empty_rows]
        if len(active_rows) < 4:
            continue

        for c in range(period):
            tile[random.choice(active_rows), c] = True

        for r in active_rows:
            if not tile[r].any():
                tile[r, random.randrange(period)] = True
            for c in range(period):
                if random.random() < 0.22:
                    tile[r, c] = True

        if not np.all(tile.any(axis=0)):
            continue
        if any(_tile_has_period(tile, smaller) for smaller in range(1, period)):
            continue
        if sum(1 for r in active_rows if tile[r].any() and not tile[r].all()) < 3:
            continue
        return tile, empty_rows
    return None


def _valid_run_options(
    tile: np.ndarray,
    row: int,
    width: int,
    visible_width: int,
    require_hidden_barrier: bool,
) -> list[tuple[int, int, int | None, int]]:
    period = int(tile.shape[1])
    row_tile = tile[row]
    options: list[tuple[int, int, int | None, int]] = []
    for length in RUN_LENGTHS:
        if length >= width:
            continue
        old_start = width - length
        if any(bool(row_tile[c % period]) for c in range(old_start, width)):
            continue
        barrier = _nearest_extended_barrier(row_tile, old_start)
        if require_hidden_barrier and (barrier is None or barrier < visible_width):
            continue
        new_start = 0 if barrier is None else barrier + 1
        if new_start >= old_start:
            continue
        if any(bool(row_tile[c % period]) for c in range(new_start, new_start + length)):
            continue
        options.append((length, old_start, barrier, new_start))
    return options


def _render_expected(
    grid: np.ndarray,
    bg: int,
    tile: np.ndarray,
    red_specs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full(grid.shape, bg, dtype=int)
    out[grid == GRAY] = GRAY
    period = int(tile.shape[1])
    width = int(grid.shape[1])
    for row, length in red_specs:
        old_start = width - length
        barrier = _nearest_extended_barrier(tile[row], old_start)
        new_start = 0 if barrier is None else barrier + 1
        out[row, new_start : new_start + length] = RED
    return out


def _render_visible_only(grid: np.ndarray, bg: int, red_specs: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(grid.shape, bg, dtype=int)
    out[grid == GRAY] = GRAY
    width = int(grid.shape[1])
    for row, length in red_specs:
        old_start = width - length
        grey_left = np.flatnonzero((grid[row, :old_start] == GRAY))
        new_start = 0 if grey_left.size == 0 else int(grey_left[-1]) + 1
        out[row, new_start : new_start + length] = RED
    return out


def _choose_red_rows(
    tile: np.ndarray,
    width: int,
    visible_width: int,
    empty_rows: list[int],
) -> list[tuple[int, int, bool]] | None:
    hidden_rows: list[tuple[int, tuple[int, int, int | None, int]]] = []
    for r in range(tile.shape[0]):
        if not tile[r].any():
            continue
        options = _valid_run_options(tile, r, width, visible_width, True)
        if options:
            hidden_rows.append((r, random.choice(options)))

    empty_candidates: list[tuple[int, tuple[int, int, int | None, int]]] = []
    for r in empty_rows:
        options = _valid_run_options(tile, r, width, visible_width, False)
        if options:
            empty_candidates.append((r, random.choice(options)))

    if not hidden_rows or not empty_candidates:
        return None

    random.shuffle(hidden_rows)
    random.shuffle(empty_candidates)

    selected: list[tuple[int, int, bool]] = []
    for r, opt in hidden_rows[: random.randint(1, min(3, len(hidden_rows)))]:
        selected.append((r, opt[0], True))
    selected.append((empty_candidates[0][0], empty_candidates[0][1][0], False))

    remaining_empty = empty_candidates[1:]
    if remaining_empty and random.random() < 0.35:
        r, opt = remaining_empty[0]
        selected.append((r, opt[0], False))

    rows = [r for r, _length, _hidden in selected]
    if len(rows) != len(set(rows)):
        return None
    random.shuffle(selected)
    return selected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    period = random.randint(*PERIOD_RANGE)
    made = _make_period_tile(height, period)
    if made is None:
        return None
    tile, empty_rows = made

    repeat_count = random.choice((2, 3))
    visible_width = period * repeat_count
    min_width = max(MIN_WIDTH, visible_width + period + max(RUN_LENGTHS) + 1)
    if min_width > MAX_WIDTH:
        return None
    width = random.randint(min_width, MAX_WIDTH)

    red_rows = _choose_red_rows(tile, width, visible_width, empty_rows)
    if red_rows is None:
        return None

    bg_choices = [c for c in ARC_COLORS if c not in (RED, GRAY)]
    bg = random.choice(bg_choices)
    grid = np.full((height, width), bg, dtype=int)
    for c in range(visible_width):
        grid[tile[:, c % period], c] = GRAY

    red_specs: list[tuple[int, int]] = []
    has_hidden = False
    has_empty = False
    for row, length, hidden in red_rows:
        start = width - length
        assert np.all(grid[row, start:] == bg)
        grid[row, start:] = RED
        red_specs.append((row, length))
        has_hidden |= hidden
        has_empty |= not hidden

    expected = _render_expected(grid, bg, tile, red_specs)
    visible_only = _render_visible_only(grid, bg, red_specs)

    if not has_hidden or not has_empty:
        return None
    if np.array_equal(expected, visible_only):
        return None
    if np.array_equal(expected, grid):
        return None
    if _minimal_column_period(grid[:, :visible_width] == GRAY) != period:
        return None
    if _background(grid) != bg:
        return None
    if np.any((expected == GRAY) & ~(grid == GRAY)):
        return None
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid phantom-period-barrier input")

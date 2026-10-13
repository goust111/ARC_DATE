from __future__ import annotations

import numpy as np


CONTROL_COLOR = 7
BACKGROUND = 0
WALL_COLORS = (2, 3, 4, 8)
NOISE_COLORS = (1, 5, 6, 9)
PERIOD_CHOICES = (3, 4)
MAX_GENERATION_ATTEMPTS = 600


def _axis_period(grid: np.ndarray, axis: int) -> int:
    length = int(grid.shape[axis])
    for period in range(1, length + 1):
        ok = True
        for idx in range(length):
            ref = np.take(grid, idx % period, axis=axis)
            cur = np.take(grid, idx, axis=axis)
            if not np.array_equal(cur, ref):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("unreachable")


def _small_period_candidates(grid: np.ndarray, max_period: int = 4) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates: list[tuple[int, int]] = []
    for row_period in range(1, min(max_period, height) + 1):
        if row_period >= height:
            continue
        for col_period in range(1, min(max_period, width) + 1):
            if col_period >= width:
                continue
            tile = g[:row_period, :col_period]
            if np.array_equal(g, tile[np.arange(height) % row_period][:, np.arange(width) % col_period]):
                candidates.append((row_period, col_period))
    return candidates


def _unique_small_period(grid: np.ndarray, max_period: int = 4) -> tuple[int, int] | None:
    candidates = _small_period_candidates(grid, max_period=max_period)
    if not candidates:
        return None
    min_area = min(rp * cp for rp, cp in candidates)
    smallest = [(rp, cp) for rp, cp in candidates if rp * cp == min_area]
    if len(smallest) != 1:
        return None
    return smallest[0]


def _compress_with_sets(
    grid: np.ndarray, removed_rows: set[int], removed_cols: set[int]
) -> np.ndarray:
    keep_rows = [r for r in range(grid.shape[0]) if r not in removed_rows]
    keep_cols = [c for c in range(grid.shape[1]) if c not in removed_cols]
    assert keep_rows and keep_cols, "compression must leave a non-empty grid"
    return grid[np.ix_(keep_rows, keep_cols)].copy()


def _marked_sets(grid: np.ndarray) -> tuple[set[int], set[int]]:
    marked_rows = {int(r) for r in np.flatnonzero(grid[:, 0] == CONTROL_COLOR)}
    marked_cols = {int(c) for c in np.flatnonzero(grid[0, :] == CONTROL_COLOR)}
    marked_rows.discard(0)
    marked_cols.discard(0)
    return marked_rows, marked_cols


def _sample_without_small_period(
    rng: np.random.Generator, length: int, forbidden: np.ndarray
) -> np.ndarray:
    out = rng.choice(NOISE_COLORS, size=length, replace=True)
    while (
        np.any(out == forbidden)
        or _axis_period(out.reshape(1, -1), axis=1) in PERIOD_CHOICES
        or len(set(map(int, out.tolist()))) < 2
    ):
        out = rng.choice(NOISE_COLORS, size=length, replace=True)
    return out


def _make_tile(rng: np.random.Generator, row_period: int, col_period: int) -> np.ndarray:
    color_count = int(rng.integers(3, min(4, row_period * col_period) + 1))
    colors = rng.choice(WALL_COLORS, size=color_count, replace=False)
    for _ in range(500):
        tile = rng.choice(colors, size=(row_period, col_period), replace=True)
        flat = tile.reshape(-1)
        forced_positions = rng.choice(tile.size, size=color_count, replace=False)
        flat[forced_positions] = colors
        tile = flat.reshape(row_period, col_period)
        if len(set(map(int, tile.reshape(-1).tolist()))) < 3:
            continue
        if _axis_period(tile, axis=0) == row_period and _axis_period(tile, axis=1) == col_period:
            return tile
    raise AssertionError("failed to build a tile with the intended minimal period")


def _paint_disturbances(
    rng: np.random.Generator,
    body: np.ndarray,
    clean: np.ndarray,
    disturbed_rows: list[int],
    disturbed_cols: list[int],
    clean_rows: list[int],
    clean_cols: list[int],
) -> None:
    for row_idx, body_r in enumerate(disturbed_rows):
        reference = clean[row_idx % clean.shape[0], :]
        body[body_r, clean_cols] = _sample_without_small_period(rng, len(clean_cols), reference)
    for col_idx, body_c in enumerate(disturbed_cols):
        reference = clean[:, col_idx % clean.shape[1]]
        column = _sample_without_small_period(rng, len(clean_rows), reference)
        body[clean_rows, body_c] = column
    for body_r in disturbed_rows:
        for body_c in disturbed_cols:
            body[body_r, body_c] = int(rng.choice(NOISE_COLORS))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete the control-marked interference rows and columns, then return the compact periodic wallpaper.

    Input structure:
    - Row 0 and column 0 are control borders rather than part of the wallpaper.
    - Orange control pixels in the top border mark interference columns; orange control pixels in the left border mark interference rows.
    - Inside the body, the unmarked rows and columns form a perfect repeating wallpaper once the marked lines are removed.

    Rule:
    - Read row 0 to collect the marked column indices and read column 0 to collect the marked row indices.
    - Delete row 0, column 0, every marked body row, and every marked body column.
    - Preserve the relative order of the remaining cells and return the compressed subgrid.

    Key insight:
    - The orange controls mark which lines must be excluded, not which lines should be kept; periodicity reappears only after those marked disturbances are removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2, "expected a 2D grid with control borders"
    assert int(g[0, 0]) == BACKGROUND, "top-left control corner must be black"
    assert np.all(np.isin(g[0, :], (BACKGROUND, CONTROL_COLOR))), "top border must be pure control markers"
    assert np.all(np.isin(g[:, 0], (BACKGROUND, CONTROL_COLOR))), "left border must be pure control markers"
    assert not np.any(g[1:, 1:] == CONTROL_COLOR), "control color must stay on the borders"

    marked_rows, marked_cols = _marked_sets(g)
    out = _compress_with_sets(g, {0, *marked_rows}, {0, *marked_cols})

    period = _unique_small_period(out, max_period=max(PERIOD_CHOICES))
    assert period is not None, "remaining wallpaper must admit a unique small period"
    assert period[0] in PERIOD_CHOICES and period[1] in PERIOD_CHOICES, "expected a 3x3/4x4-style wallpaper period"
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        row_period = int(rng.choice(PERIOD_CHOICES))
        col_period = int(rng.choice(PERIOD_CHOICES))
        row_repeats = int(rng.integers(2, 6))
        col_repeats = int(rng.integers(2, 6))
        clean = np.tile(_make_tile(rng, row_period, col_period), (row_repeats, col_repeats))

        disturbed_row_count = int(rng.integers(2, 4))
        disturbed_col_count = int(rng.integers(2, 4))
        body_h = int(clean.shape[0] + disturbed_row_count)
        body_w = int(clean.shape[1] + disturbed_col_count)
        if max(body_h + 1, body_w + 1) > 30:
            continue

        disturbed_rows = sorted(rng.choice(body_h, size=disturbed_row_count, replace=False).tolist())
        disturbed_cols = sorted(rng.choice(body_w, size=disturbed_col_count, replace=False).tolist())
        clean_rows = [r for r in range(body_h) if r not in disturbed_rows]
        clean_cols = [c for c in range(body_w) if c not in disturbed_cols]

        body = np.empty((body_h, body_w), dtype=int)
        body[np.ix_(clean_rows, clean_cols)] = clean
        _paint_disturbances(rng, body, clean, disturbed_rows, disturbed_cols, clean_rows, clean_cols)

        inp = np.zeros((body_h + 1, body_w + 1), dtype=int)
        inp[1:, 1:] = body
        inp[0, np.array(disturbed_cols) + 1] = CONTROL_COLOR
        inp[np.array(disturbed_rows) + 1, 0] = CONTROL_COLOR

        if _unique_small_period(clean, max_period=max(PERIOD_CHOICES)) != (row_period, col_period):
            continue
        if not np.array_equal(main(inp), clean):
            continue

        marked_rows = {r + 1 for r in disturbed_rows}
        marked_cols = {c + 1 for c in disturbed_cols}
        semantic_ok = True

        for row_idx in marked_rows:
            kept_bad_row = _compress_with_sets(inp, ({0} | marked_rows) - {row_idx}, {0, *marked_cols})
            if _unique_small_period(kept_bad_row, max_period=max(PERIOD_CHOICES)) is not None:
                semantic_ok = False
                break
        if not semantic_ok:
            continue

        for col_idx in marked_cols:
            kept_bad_col = _compress_with_sets(inp, {0, *marked_rows}, ({0} | marked_cols) - {col_idx})
            if _unique_small_period(kept_bad_col, max_period=max(PERIOD_CHOICES)) is not None:
                semantic_ok = False
                break
        if not semantic_ok:
            continue

        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")

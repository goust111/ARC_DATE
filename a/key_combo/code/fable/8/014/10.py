from __future__ import annotations

import random

import numpy as np


Model = tuple[tuple[int, int, int, int], int]


def _candidate_model(grid: np.ndarray, row_period: int, col_period: int, shift: int) -> Model | None:
    """Return a fully observed skew-periodic model, or None on contradiction."""
    h, w = grid.shape
    mapping: dict[tuple[int, int], int] = {}

    for r in range(h):
        row_key = r % row_period
        skew = (shift * r) % col_period
        for c in range(w):
            key = (row_key, (c - skew) % col_period)
            value = int(grid[r, c])
            old = mapping.get(key)
            if old is None:
                mapping[key] = value
            elif old != value:
                return None

    if len(mapping) != row_period * col_period:
        return None

    score = (row_period * col_period, row_period + col_period, row_period, col_period)
    return score, shift


def _infer_model(grid: np.ndarray) -> tuple[int, int, int, dict[tuple[int, int], int]]:
    """Find the unique simplest model grid[r,c] = T[r mod pr, (c - s*r) mod pc]."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    best_score: tuple[int, int, int, int] | None = None
    best_models: list[tuple[int, int, int, dict[tuple[int, int], int]]] = []

    for row_period in range(1, h + 1):
        for col_period in range(1, w + 1):
            for shift in range(col_period):
                candidate = _candidate_model(g, row_period, col_period, shift)
                if candidate is None:
                    continue
                score, _ = candidate
                mapping: dict[tuple[int, int], int] = {}
                for r in range(h):
                    row_key = r % row_period
                    skew = (shift * r) % col_period
                    for c in range(w):
                        mapping[(row_key, (c - skew) % col_period)] = int(g[r, c])

                if best_score is None or score < best_score:
                    best_score = score
                    best_models = [(row_period, col_period, shift, mapping)]
                elif score == best_score:
                    best_models.append((row_period, col_period, shift, mapping))

    assert best_models, "expected at least one periodic extrapolation model"
    assert len(best_models) == 1, "period model must be uniquely determined"
    return best_models[0]


def _render_extended(
    mapping: dict[tuple[int, int], int],
    row_period: int,
    col_period: int,
    shift: int,
    height: int,
    width: int,
    border: int,
) -> np.ndarray:
    out = np.zeros((height + 2 * border, width + 2 * border), dtype=int)
    for r in range(out.shape[0]):
        src_r = r - border
        row_key = src_r % row_period
        skew = (shift * src_r) % col_period
        for c in range(out.shape[1]):
            src_c = c - border
            out[r, c] = mapping[(row_key, (src_c - skew) % col_period)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend the cut-out periodic pattern by two cells on every side.

    Input structure:
    - The whole grid is a cropped window from a 2D periodic color pattern.
    - The pattern repeats with a row period, a column period, and a fixed
      horizontal phase shift applied from one row to the next.

    Rule:
    - Keep the input grid unchanged in the center of the output.
    - Increase both height and width by 4 so each side gains a two-cell border.
    - Fill every new border cell by continuing the same skew-periodic pattern,
      not by copying the nearest edge row or column.

    Key insight:
    - The visible fragment comes from residue classes of one diagonal periodic
      lattice, so border cells must be inferred by the lattice coordinates
      `(r mod pr, (c - s*r) mod pc)` rather than by edge padding or wraparound.
    """
    g = np.asarray(grid, dtype=int)
    row_period, col_period, shift, mapping = _infer_model(g)
    return _render_extended(mapping, row_period, col_period, shift, *g.shape, border=2)


# --------------- generator (o5x) ---------------

MAX_GENERATION_ATTEMPTS = 400
MAX_INPUT_SIZE = 26
MIN_ROW_PERIOD = 2
MAX_ROW_PERIOD = 5
MIN_COL_PERIOD = 3
MAX_COL_PERIOD = 6
PALETTE = tuple(range(10))


def _random_period_size(period: int) -> int:
    choices = [size for size in range(2 * period + 1, MAX_INPUT_SIZE + 1) if size % period != 0]
    assert choices
    return random.choice(choices)


def _random_tile(row_period: int, col_period: int) -> np.ndarray:
    color_count = random.randint(3, min(6, len(PALETTE)))
    colors = random.sample(PALETTE, color_count)
    tile = np.array(
        [[random.choice(colors) for _ in range(col_period)] for _ in range(row_period)],
        dtype=int,
    )

    if len(set(map(int, tile.ravel()))) < 3:
        return _random_tile(row_period, col_period)
    for row in tile:
        if len(set(map(int, row))) < 2:
            return _random_tile(row_period, col_period)
    return tile


def _render_window(
    tile: np.ndarray,
    shift: int,
    height: int,
    width: int,
    row_offset: int,
    col_offset: int,
) -> np.ndarray:
    row_period, col_period = tile.shape
    grid = np.zeros((height, width), dtype=int)
    for r in range(height):
        abs_r = r + row_offset
        row_key = abs_r % row_period
        skew = (shift * abs_r) % col_period
        for c in range(width):
            abs_c = c + col_offset
            grid[r, c] = tile[row_key, (abs_c - skew) % col_period]
    return grid


def _edge_padded(grid: np.ndarray) -> np.ndarray:
    return np.pad(grid, 2, mode="edge")


def _wrapped(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.zeros((h + 4, w + 4), dtype=int)
    for r in range(h + 4):
        for c in range(w + 4):
            out[r, c] = grid[(r - 2) % h, (c - 2) % w]
    return out


def _gen_attempt() -> np.ndarray | None:
    row_period = random.randint(MIN_ROW_PERIOD, MAX_ROW_PERIOD)
    col_period = random.randint(MIN_COL_PERIOD, MAX_COL_PERIOD)
    shift = random.randint(1, col_period - 1)

    height = _random_period_size(row_period)
    width = _random_period_size(col_period)
    row_offset = random.randrange(row_period)
    col_offset = random.randrange(col_period)
    tile = _random_tile(row_period, col_period)

    inp = _render_window(tile, shift, height, width, row_offset, col_offset)
    expected = _render_window(tile, shift, height + 4, width + 4, row_offset - 2, col_offset - 2)

    if np.array_equal(expected, _edge_padded(inp)):
        return None
    if np.array_equal(expected, _wrapped(inp)):
        return None

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
    raise RuntimeError("failed to generate a valid input for puzzle 10")

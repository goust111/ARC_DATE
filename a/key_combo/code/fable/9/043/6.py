from __future__ import annotations

from itertools import combinations

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

INTERIOR_COLORS = np.array([BLACK, PINK, ORANGE, TEAL], dtype=int)
ALLOWED_COLORS = np.array([BLACK, RED, YELLOW, PINK, ORANGE, TEAL], dtype=int)


def _parse_beams(g: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    h, w = map(int, g.shape)
    coords = np.argwhere(g == color)
    assert len(coords) > 0, "expected at least one beam marker of each color"
    actual = {(int(r), int(c)) for r, c in coords}
    assert all(r in (0, h - 1) or c in (0, w - 1) for r, c in actual), "markers must stay on the border"
    assert not any((r, c) in actual for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))), "corners cannot be markers"

    row_beams = [r for r in range(1, h - 1) if g[r, 0] == color and g[r, w - 1] == color]
    col_beams = [c for c in range(1, w - 1) if g[0, c] == color and g[h - 1, c] == color]

    expected = {(r, 0) for r in row_beams} | {(r, w - 1) for r in row_beams}
    expected |= {(0, c) for c in col_beams} | {(h - 1, c) for c in col_beams}
    assert actual == expected, "each marker must belong to one opposite-edge pair of its own color"
    return row_beams, col_beams


def _beam_count_mask(shape: tuple[int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    h, w = shape
    row_mask = np.zeros(h, dtype=bool)
    col_mask = np.zeros(w, dtype=bool)
    row_mask[rows] = True
    col_mask[cols] = True
    counts = row_mask[:, None].astype(int) + col_mask[None, :].astype(int)
    counts[[0, h - 1], :] = 0
    counts[:, [0, w - 1]] = 0
    return counts


def _transform_from_layout(
    g: np.ndarray,
    yellow_rows: list[int],
    yellow_cols: list[int],
    red_rows: list[int],
    red_cols: list[int],
) -> np.ndarray:
    out = g.copy()
    yellow_odd = (_beam_count_mask(g.shape, yellow_rows, yellow_cols) % 2) == 1
    red_odd = (_beam_count_mask(g.shape, red_rows, red_cols) % 2) == 1

    black_mask = g == BLACK
    teal_mask = g == TEAL
    pink_mask = g == PINK
    orange_mask = g == ORANGE

    out[yellow_odd & black_mask] = TEAL
    out[yellow_odd & teal_mask] = BLACK
    out[red_odd & pink_mask] = ORANGE
    out[red_odd & orange_mask] = PINK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap black/cyan along yellow beams and pink/orange along red beams, with same-color crossings canceling by parity.

    Input structure
    - The strict interior uses only black, teal, pink, and orange.
    - Yellow and red marker pixels appear only on the border, each in aligned opposite-edge pairs that select full interior rows or columns.
    - Both colors can have several row beams and column beams, so same-color and cross-color intersections may occur.

    Rule
    - Keep every border marker unchanged.
    - Each yellow beam toggles black <-> teal on its strict interior cells; each red beam toggles pink <-> orange on its strict interior cells.
    - A cell affected by two beams of the same color is toggled twice and therefore returns to its original color; the two beam colors operate on disjoint color pairs and do not interfere.

    Key insight
    - The border markers define row and column involutions rather than paint regions: only odd yellow coverage matters for black/teal, and only odd red coverage matters for pink/orange.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 5 <= h <= 30 and 5 <= w <= 30, "expected an ARC-sized grid with a strict interior"
    assert np.all(np.isin(g, ALLOWED_COLORS)), "unexpected color outside the puzzle palette"
    assert not np.any(np.isin(g[1:-1, 1:-1], [RED, YELLOW])), "marker colors cannot appear in the strict interior"

    yellow_rows, yellow_cols = _parse_beams(g, YELLOW)
    red_rows, red_cols = _parse_beams(g, RED)
    assert yellow_rows or yellow_cols, "expected at least one yellow beam"
    assert red_rows or red_cols, "expected at least one red beam"
    return _transform_from_layout(g, yellow_rows, yellow_cols, red_rows, red_cols)


GRID_HEIGHT_RANGE = (11, 19)
GRID_WIDTH_RANGE = (11, 19)
BEAM_COUNT_RANGE = (1, 3)
MIN_MARKER_GAP = 2
BLOCK_SPAN_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400


def _sample_spaced_indices(limit: int, count: int, rng: np.random.Generator) -> list[int] | None:
    slots = range(1, limit - 1)
    valid = [combo for combo in combinations(slots, count) if all(b - a >= MIN_MARKER_GAP for a, b in zip(combo, combo[1:]))]
    if not valid:
        return None
    choice = valid[int(rng.integers(len(valid)))]
    return list(choice)


def _sample_axis_split(limit: int, rng: np.random.Generator) -> tuple[list[int], list[int]] | None:
    interior = limit - 2
    max_total = (interior + 1) // MIN_MARKER_GAP
    feasible = [
        (a, b)
        for a in range(BEAM_COUNT_RANGE[0], BEAM_COUNT_RANGE[1] + 1)
        for b in range(BEAM_COUNT_RANGE[0], BEAM_COUNT_RANGE[1] + 1)
        if a + b <= max_total
    ]
    if not feasible:
        return None
    first_count, second_count = feasible[int(rng.integers(len(feasible)))]
    merged = _sample_spaced_indices(limit, first_count + second_count, rng)
    if merged is None:
        return None
    merged = list(rng.permutation(merged))
    first = sorted(int(v) for v in merged[:first_count])
    second = sorted(int(v) for v in merged[first_count:])
    return first, second


def _fill_blocky_grid(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    h, w = shape
    grid = np.empty((h, w), dtype=int)
    r = 0
    while r < h:
        block_h = int(rng.integers(BLOCK_SPAN_RANGE[0], BLOCK_SPAN_RANGE[1]))
        c = 0
        while c < w:
            block_w = int(rng.integers(BLOCK_SPAN_RANGE[0], BLOCK_SPAN_RANGE[1]))
            color = int(rng.choice(INTERIOR_COLORS))
            grid[r : min(h, r + block_h), c : min(w, c + block_w)] = color
            c += block_w
        r += block_h
    return grid


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(mask)]


def _pick_unused(mask: np.ndarray, used: set[tuple[int, int]], rng: np.random.Generator) -> tuple[int, int] | None:
    choices = [coord for coord in _mask_coords(mask) if coord not in used]
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _force_semantic_witnesses(
    grid: np.ndarray,
    yellow_counts: np.ndarray,
    red_counts: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    used: set[tuple[int, int]] = set()
    yellow_odd = (yellow_counts % 2) == 1
    red_odd = (red_counts % 2) == 1
    yellow_cancel = yellow_counts == 2
    red_cancel = red_counts == 2
    mixed_odd = yellow_odd & red_odd
    yellow_none = yellow_counts == 0
    red_none = red_counts == 0

    requirements = [
        (yellow_odd, BLACK),
        (yellow_odd, TEAL),
        (yellow_cancel, BLACK),
        (red_odd, PINK),
        (red_odd, ORANGE),
        (red_cancel, PINK),
        (mixed_odd, BLACK),
        (mixed_odd, PINK),
        (yellow_odd, PINK),
        (red_odd, BLACK),
        (yellow_none, TEAL),
        (red_none, ORANGE),
    ]
    for mask, color in requirements:
        coord = _pick_unused(mask, used, rng)
        if coord is None:
            return False
        used.add(coord)
        grid[coord] = color

    return True


def _validate_semantics(
    inp: np.ndarray,
    out: np.ndarray,
    yellow_rows: list[int],
    yellow_cols: list[int],
    red_rows: list[int],
    red_cols: list[int],
) -> bool:
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(out), inp):
        return False

    yellow_counts = _beam_count_mask(inp.shape, yellow_rows, yellow_cols)
    red_counts = _beam_count_mask(inp.shape, red_rows, red_cols)
    yellow_odd = (yellow_counts % 2) == 1
    red_odd = (red_counts % 2) == 1

    if not np.any((yellow_counts == 2) & np.isin(inp, [BLACK, TEAL]) & (out == inp)):
        return False
    if not np.any((red_counts == 2) & np.isin(inp, [PINK, ORANGE]) & (out == inp)):
        return False
    if not np.any(yellow_odd & (inp == BLACK) & (out == TEAL)):
        return False
    if not np.any(yellow_odd & (inp == TEAL) & (out == BLACK)):
        return False
    if not np.any(red_odd & (inp == PINK) & (out == ORANGE)):
        return False
    if not np.any(red_odd & (inp == ORANGE) & (out == PINK)):
        return False
    if not np.any(yellow_odd & red_odd & np.isin(inp, [BLACK, TEAL])):
        return False
    if not np.any(yellow_odd & red_odd & np.isin(inp, [PINK, ORANGE])):
        return False
    if len(np.unique(inp[1:-1, 1:-1])) < 4:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] + 1))
    w = int(rng.integers(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] + 1))

    row_split = _sample_axis_split(h, rng)
    col_split = _sample_axis_split(w, rng)
    if row_split is None or col_split is None:
        return None
    yellow_rows, red_rows = row_split
    yellow_cols, red_cols = col_split

    grid = _fill_blocky_grid((h, w), rng)
    grid[np.ix_(yellow_rows, [0, w - 1])] = YELLOW
    grid[np.ix_(red_rows, [0, w - 1])] = RED
    grid[np.ix_([0, h - 1], yellow_cols)] = YELLOW
    grid[np.ix_([0, h - 1], red_cols)] = RED

    yellow_counts = _beam_count_mask(grid.shape, yellow_rows, yellow_cols)
    red_counts = _beam_count_mask(grid.shape, red_rows, red_cols)
    if not _force_semantic_witnesses(grid, yellow_counts, red_counts, rng):
        return None

    expected = _transform_from_layout(grid, yellow_rows, yellow_cols, red_rows, red_cols)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _validate_semantics(grid, expected, yellow_rows, yellow_cols, red_rows, red_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

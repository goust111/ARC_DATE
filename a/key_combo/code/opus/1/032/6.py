from __future__ import annotations

import itertools
import random

import numpy as np


FILLED_COLOR = 2
BLANK_COLOR = 0
STRIDE = 4
CENTER_OFFSET = 1
GUTTER_OFFSET = 3

# The stated 3x3-with-gutters geometry caps the macro height at 7 under ARC's 30x30 limit.
MACRO_ROWS = 7
MACRO_COL_OPTIONS = (5, 6)
MAX_GENERATION_ATTEMPTS = 400


def _stable_desc_order(values: list[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=int)
    return np.argsort(-arr, kind="stable")


def _count_holes(seq: np.ndarray | tuple[int, ...] | list[int]) -> int:
    values = [int(v) for v in seq]
    holes = 0
    i = 0
    while i < len(values):
        if values[i] == 1:
            i += 1
            continue
        j = i
        while j < len(values) and values[j] == 0:
            j += 1
        if i > 0 and j < len(values) and values[i - 1] == 1 and values[j] == 1:
            holes += 1
        i = j
    return holes


def _parse_macro_grid(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    assert h % STRIDE == STRIDE - 1 and w % STRIDE == STRIDE - 1, "expected 4r-1 by 4c-1 macro-grid geometry"

    center_rows = np.arange(CENTER_OFFSET, h, STRIDE)
    center_cols = np.arange(CENTER_OFFSET, w, STRIDE)
    gutter_rows = np.arange(GUTTER_OFFSET, h, STRIDE)
    gutter_cols = np.arange(GUTTER_OFFSET, w, STRIDE)
    assert center_rows.size >= 2 and center_cols.size >= 2, "expected at least a 2x2 macro-grid"

    blank = int(g[gutter_rows[0], gutter_cols[0]])
    assert np.all(g[gutter_rows, :] == blank), "expected uniform blank gutter rows"
    assert np.all(g[:, gutter_cols] == blank), "expected uniform blank gutter columns"

    filled = int(g[0, 0])
    assert filled != blank, "expected distinct frame and blank colors"

    variable = np.zeros_like(g, dtype=bool)
    variable[np.ix_(center_rows, center_cols)] = True

    structural = np.ones_like(g, dtype=bool)
    structural[gutter_rows, :] = False
    structural[:, gutter_cols] = False
    structural &= ~variable
    assert np.all(g[structural] == filled), "expected fixed filled 3x3 macro-cell shells"

    centers = g[np.ix_(center_rows, center_cols)]
    assert np.all((centers == blank) | (centers == filled)), "expected binary center pixels"
    return g, filled, blank, centers == filled, center_rows, center_cols, variable


def _hole_counts(centers: np.ndarray) -> list[int]:
    return [_count_holes(centers[:, c]) for c in range(centers.shape[1])]


def _fill_counts(centers: np.ndarray) -> list[int]:
    return [int(centers[:, c].sum()) for c in range(centers.shape[1])]


def _render_grid_from_centers(centers: np.ndarray, filled: int = FILLED_COLOR, blank: int = BLANK_COLOR) -> np.ndarray:
    n_rows, n_cols = centers.shape
    h = STRIDE * n_rows - 1
    w = STRIDE * n_cols - 1
    grid = np.full((h, w), blank, dtype=int)

    non_gutter = (np.arange(h)[:, None] % STRIDE != GUTTER_OFFSET) & (np.arange(w)[None, :] % STRIDE != GUTTER_OFFSET)
    grid[non_gutter] = filled

    center_rows = np.arange(CENTER_OFFSET, h, STRIDE)
    center_cols = np.arange(CENTER_OFFSET, w, STRIDE)
    grid[np.ix_(center_rows, center_cols)] = np.where(centers, filled, blank)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder macro-columns by how many enclosed blank runs appear in their center-pixel sequences.

    Input structure:
    - The grid is a rectangular macro-grid of red 3x3 cells separated by one-pixel black gutter rows and columns.
    - Every macro-cell differs only at its center pixel, which is either filled red or blank black.

    Rule:
    - Read each macro-column's center pixels from top to bottom as a binary sequence.
    - Count enclosed holes: maximal blank runs whose immediate neighbors above and below are both filled.
    - Stable-sort the macro-columns by hole count from larger to smaller, moving only the center-pixel columns and leaving the red shells and black gutters unchanged.

    Key insight:
    - The sort key is not the number of filled centers; it is the number of one-dimensional enclosed blank gaps between filled centers.
    """
    g, filled, blank, centers, center_rows, center_cols, _variable = _parse_macro_grid(grid)
    order = _stable_desc_order(_hole_counts(centers))

    out = g.copy()
    out[np.ix_(center_rows, center_cols)] = np.where(centers[:, order], filled, blank)
    return out


def _build_sequence_library() -> dict[int, dict[int, list[tuple[int, ...]]]]:
    library: dict[int, dict[int, list[tuple[int, ...]]]] = {}
    for bits in itertools.product((0, 1), repeat=MACRO_ROWS):
        holes = _count_holes(bits)
        fills = sum(bits)
        library.setdefault(holes, {}).setdefault(fills, []).append(bits)
    return library


SEQUENCE_LIBRARY = _build_sequence_library()

# Profiles are specified as (hole_count, fill_count). They intentionally create
# counterexamples to the old "sort by filled centers" shortcut.
PROFILE_SPECS = {
    5: (
        ((3, 4), (2, 4), (1, 6), (1, 2), (0, 4)),
        ((3, 4), (2, 5), (2, 4), (1, 2), (0, 4)),
        ((3, 4), (2, 4), (1, 4), (0, 7), (0, 4)),
        ((3, 4), (2, 5), (1, 4), (1, 6), (0, 4)),
    ),
    6: (
        ((3, 4), (2, 5), (2, 4), (1, 6), (1, 2), (0, 4)),
        ((3, 4), (2, 4), (2, 5), (1, 4), (0, 7), (0, 4)),
        ((3, 4), (2, 4), (1, 6), (1, 2), (0, 7), (0, 4)),
        ((3, 4), (2, 5), (2, 4), (1, 4), (1, 6), (0, 4)),
    ),
}


def _sample_columns(specs: tuple[tuple[int, int], ...]) -> list[tuple[int, ...]]:
    columns: list[tuple[int, ...]] = []
    used: set[tuple[int, ...]] = set()
    for holes, fills in specs:
        candidates = [seq for seq in SEQUENCE_LIBRARY[holes][fills] if seq not in used]
        assert candidates, f"no unused sequence for hole={holes}, fill={fills}"
        seq = random.choice(candidates)
        columns.append(seq)
        used.add(seq)
    return columns


def _has_same_fill_different_holes(hole_counts: list[int], fill_counts: list[int]) -> bool:
    for i in range(len(hole_counts)):
        for j in range(i + 1, len(hole_counts)):
            if fill_counts[i] == fill_counts[j] and hole_counts[i] != hole_counts[j]:
                return True
    return False


def _tie_order_defeats_fill(hole_counts: list[int], fill_counts: list[int]) -> bool:
    for hole in sorted(set(hole_counts), reverse=True):
        tied = [i for i, value in enumerate(hole_counts) if value == hole]
        if len(tied) < 2:
            continue
        if len({fill_counts[i] for i in tied}) < 2:
            continue
        fill_sorted = sorted(tied, key=lambda idx: -fill_counts[idx])
        if tied != fill_sorted:
            return True
    return False


def _interesting_input_columns(columns: list[tuple[int, ...]]) -> bool:
    centers = np.asarray(columns, dtype=int).T.astype(bool)
    hole_counts = _hole_counts(centers)
    fill_counts = _fill_counts(centers)
    hole_order = _stable_desc_order(hole_counts)
    fill_order = _stable_desc_order(fill_counts)

    if len(set(hole_counts)) != 4:
        return False
    if 0 not in hole_counts or max(hole_counts) < 3:
        return False
    if np.array_equal(hole_order, np.arange(len(columns))):
        return False
    if np.array_equal(hole_order, fill_order):
        return False
    if hole_order[0] == fill_order[0]:
        return False
    if not _has_same_fill_different_holes(hole_counts, fill_counts):
        return False
    if not _tie_order_defeats_fill(hole_counts, fill_counts):
        return False
    return True


def _choose_input_columns(sorted_columns: list[tuple[int, ...]]) -> np.ndarray | None:
    indices = list(range(len(sorted_columns)))
    perms = list(itertools.permutations(indices))
    random.shuffle(perms)
    for perm in perms:
        columns = [sorted_columns[i] for i in perm]
        if _interesting_input_columns(columns):
            return np.asarray(columns, dtype=int).T.astype(bool)
    return None


def _semantic_expected(centers: np.ndarray) -> np.ndarray:
    hole_order = _stable_desc_order(_hole_counts(centers))
    return centers[:, hole_order]


def _validate_example(inp: np.ndarray) -> bool:
    try:
        g, filled, blank, centers, center_rows, center_cols, variable = _parse_macro_grid(inp)
    except AssertionError:
        return False

    n_rows, n_cols = centers.shape
    if n_rows != MACRO_ROWS or n_cols not in MACRO_COL_OPTIONS:
        return False

    hole_counts = _hole_counts(centers)
    fill_counts = _fill_counts(centers)
    hole_order = _stable_desc_order(hole_counts)
    fill_order = _stable_desc_order(fill_counts)

    if len(set(hole_counts)) != 4:
        return False
    if 0 not in hole_counts or max(hole_counts) < 3:
        return False
    if np.array_equal(hole_order, np.arange(n_cols)):
        return False
    if np.array_equal(hole_order, fill_order):
        return False
    if not _has_same_fill_different_holes(hole_counts, fill_counts):
        return False
    if not _tie_order_defeats_fill(hole_counts, fill_counts):
        return False

    expected_centers = _semantic_expected(centers)
    expected = g.copy()
    expected[np.ix_(center_rows, center_cols)] = np.where(expected_centers, filled, blank)

    try:
        out = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(out, expected):
        return False

    try:
        out_g, _out_filled, _out_blank, out_centers, _out_rows, _out_cols, out_variable = _parse_macro_grid(out)
    except AssertionError:
        return False

    if not np.array_equal(out_centers, expected_centers):
        return False
    if not np.array_equal(g[~variable], out_g[~out_variable]):
        return False
    if np.array_equal(expected_centers, centers[:, fill_order]):
        return False

    for hole in sorted(set(hole_counts), reverse=True):
        tied_input = [i for i, value in enumerate(hole_counts) if value == hole]
        tied_output = [idx for idx in hole_order if hole_counts[idx] == hole]
        if tied_input != tied_output:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_cols = random.choice(MACRO_COL_OPTIONS)
    specs = random.choice(PROFILE_SPECS[n_cols])
    sorted_columns = _sample_columns(specs)
    centers = _choose_input_columns(sorted_columns)
    if centers is None:
        return None

    inp = _render_grid_from_centers(centers)
    if not _validate_example(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid macro-hole-sorting input")

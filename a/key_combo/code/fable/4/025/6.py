from __future__ import annotations

import numpy as np


PLACEHOLDER = 5
PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _pick_cells(cells: list[tuple[int, int]], count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    order = rng.permutation(len(cells))
    return [cells[int(idx)] for idx in order[:count]]


def _sample_spaced_indices(
    limit: int,
    count: int,
    rng: np.random.Generator,
    *,
    margin: int = 1,
    min_gap: int = 2,
) -> list[int] | None:
    candidates = np.arange(margin, limit - margin, dtype=int)
    if len(candidates) < count:
        return None

    for _ in range(200):
        picked = sorted(int(v) for v in rng.choice(candidates, size=count, replace=False))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _build_controls(
    rows: list[int],
    cols: list[int],
    rng: np.random.Generator,
) -> set[tuple[int, int]]:
    row_order = list(rows)
    col_order = list(cols)
    rng.shuffle(row_order)
    rng.shuffle(col_order)

    controls: set[tuple[int, int]] = set()
    for i, row in enumerate(row_order):
        controls.add((int(row), int(col_order[i % len(col_order)])))

    used_cols = {c for _, c in controls}
    for col in col_order:
        if col in used_cols:
            continue
        controls.add((int(row_order[int(rng.integers(len(row_order)))]), int(col)))

    all_cells = [(int(r), int(c)) for r in rows for c in cols]
    extras = [cell for cell in all_cells if cell not in controls]
    max_extra = max(0, len(extras) - 1)
    if max_extra > 0:
        extra_count = int(rng.integers(0, min(3, max_extra) + 1))
        for cell in _pick_cells(extras, extra_count, rng):
            controls.add(cell)
    return controls


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal same-color row/column intersections by recoloring only the qualifying grey placeholders.

    Input structure:
    - The grid has one background color, several colored control pixels, and grey placeholder pixels.
    - Control pixels of the same color can occupy multiple rows and multiple columns.
    - Each grey placeholder is arranged so it matches at most one color's row/column product.

    Rule:
    - For every non-background, non-grey color, collect the rows containing that color's control pixels and the columns containing that color's control pixels.
    - A grey placeholder becomes color C exactly when its row belongs to C's row set and its column belongs to C's column set.
    - Grey placeholders with no such same-color row/column match are erased to the background; all control pixels stay unchanged.

    Key insight:
    - The visible control pixels define invisible row keys and column keys, and the grey cells only reveal the intersections where both keys come from the same color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) not in {bg, PLACEHOLDER})

    rows_by_color: dict[int, set[int]] = {}
    cols_by_color: dict[int, set[int]] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        rows_by_color[color] = {int(r) for r, _ in coords}
        cols_by_color[color] = {int(c) for _, c in coords}

    out = g.copy()
    for r0, c0 in np.argwhere(g == PLACEHOLDER):
        r = int(r0)
        c = int(c0)
        matches = [
            color
            for color in colors
            if r in rows_by_color[color] and c in cols_by_color[color]
        ]
        assert len(matches) <= 1, "each placeholder should match at most one color"
        out[r, c] = int(matches[0]) if matches else int(bg)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    color_count = int(rng.integers(2, 4))
    row_counts = [int(rng.integers(2, 4)) for _ in range(color_count)]
    col_counts = [int(rng.integers(2, 4)) for _ in range(color_count)]
    total_rows = int(sum(row_counts))
    total_cols = int(sum(col_counts))

    min_h = max(10, 2 * total_rows + 1)
    min_w = max(10, 2 * total_cols + 1)
    max_h = min(29, min_h + 8)
    max_w = min(29, min_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    height = int(rng.integers(min_h, max_h + 1))
    width = int(rng.integers(min_w, max_w + 1))
    all_rows = _sample_spaced_indices(height, total_rows, rng)
    all_cols = _sample_spaced_indices(width, total_cols, rng)
    if all_rows is None or all_cols is None:
        return None

    rng.shuffle(all_rows)
    rng.shuffle(all_cols)

    bg = int(rng.choice([color for color in PALETTE if color != PLACEHOLDER]))
    color_pool = [color for color in PALETTE if color not in {bg, PLACEHOLDER}]
    colors = [int(color) for color in rng.choice(color_pool, size=color_count, replace=False)]

    rows_by_color: dict[int, list[int]] = {}
    cols_by_color: dict[int, list[int]] = {}
    row_start = 0
    col_start = 0
    for color, row_count, col_count in zip(colors, row_counts, col_counts):
        rows_by_color[color] = sorted(int(v) for v in all_rows[row_start : row_start + row_count])
        cols_by_color[color] = sorted(int(v) for v in all_cols[col_start : col_start + col_count])
        row_start += row_count
        col_start += col_count

    grid = np.full((height, width), bg, dtype=int)
    controls_by_color: dict[int, set[tuple[int, int]]] = {}
    chosen_valid: dict[int, list[tuple[int, int]]] = {}

    for color in colors:
        controls = _build_controls(rows_by_color[color], cols_by_color[color], rng)
        options = [
            (r, c)
            for r in rows_by_color[color]
            for c in cols_by_color[color]
            if (r, c) not in controls
        ]
        if not options:
            return None
        controls_by_color[color] = controls
        for r, c in controls:
            grid[r, c] = int(color)

        valid_count = int(rng.integers(1, min(3, len(options)) + 1))
        chosen_valid[color] = _pick_cells(options, valid_count, rng)
        for r, c in chosen_valid[color]:
            grid[r, c] = PLACEHOLDER

    occupied = {
        (int(r), int(c))
        for r, c in np.argwhere(grid != bg)
    }
    used_rows = {row for rows in rows_by_color.values() for row in rows}
    used_cols = {col for cols in cols_by_color.values() for col in cols}

    row_only: list[tuple[int, int]] = []
    col_only: list[tuple[int, int]] = []
    mismatch: list[tuple[int, int]] = []
    isolated: list[tuple[int, int]] = []

    for color in colors:
        for r in rows_by_color[color]:
            for c in range(width):
                cell = (int(r), int(c))
                if c in cols_by_color[color] or cell in occupied:
                    continue
                row_only.append(cell)

        for c in cols_by_color[color]:
            for r in range(height):
                cell = (int(r), int(c))
                if r in rows_by_color[color] or cell in occupied:
                    continue
                col_only.append(cell)

    for color_a in colors:
        for color_b in colors:
            if color_a == color_b:
                continue
            for r in rows_by_color[color_a]:
                for c in cols_by_color[color_b]:
                    cell = (int(r), int(c))
                    if cell not in occupied:
                        mismatch.append(cell)

    for r in range(height):
        if r in used_rows:
            continue
        for c in range(width):
            cell = (int(r), int(c))
            if c not in used_cols and cell not in occupied:
                isolated.append(cell)

    invalid_cells: set[tuple[int, int]] = set()

    mandatory_pools = [row_only, col_only, mismatch if color_count >= 2 else isolated]
    for pool in mandatory_pools:
        candidates = [cell for cell in pool if cell not in invalid_cells]
        if not candidates:
            return None
        invalid_cells.add(candidates[int(rng.integers(len(candidates)))])

    extra_pools = row_only + col_only + mismatch + isolated
    extra_candidates = [cell for cell in extra_pools if cell not in invalid_cells]
    extra_total = int(rng.integers(0, min(5, len(extra_candidates)) + 1))
    for cell in _pick_cells(extra_candidates, extra_total, rng):
        invalid_cells.add(cell)

    for r, c in invalid_cells:
        grid[r, c] = PLACEHOLDER

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None

    total_valid = 0
    for color, cells in chosen_valid.items():
        for r, c in cells:
            total_valid += 1
            if int(solved[r, c]) != int(color):
                return None
    if total_valid == 0:
        return None

    if not invalid_cells:
        return None
    for r, c in invalid_cells:
        if int(solved[r, c]) != int(bg):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

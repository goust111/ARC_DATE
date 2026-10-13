from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
MARK_COLORS = {BLUE, RED, GREEN}
ALL_COLORS = list(range(10))

ODD_SIZES = [9, 11, 13, 15, 17]
BACKGROUND_CHOICES = [0, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _symmetry_flags(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    row_has_fg = np.any(g != bg, axis=1)
    col_has_fg = np.any(g != bg, axis=0)
    row_sym = row_has_fg & np.all(g == g[:, ::-1], axis=1)
    col_sym = col_has_fg & np.all(g == g[::-1, :], axis=0)
    return row_sym.astype(bool), col_sym.astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect foreground symmetry in each row and each column, then mark the
    corresponding center cells.

    A row is marked red when it contains at least one non-background pixel and
    the row equals its left-right reversal. A column is marked blue when it
    contains at least one non-background pixel and the column equals its
    top-bottom reversal. If the unique row-center and column-center marker
    coincide, that pixel becomes green.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    row_sym, col_sym = _symmetry_flags(g, bg)

    out = g.copy()
    mid_r = g.shape[0] // 2
    mid_c = g.shape[1] // 2

    out[row_sym, mid_c] = RED
    out[mid_r, col_sym] = BLUE
    if bool(row_sym[mid_r] and col_sym[mid_c]):
        out[mid_r, mid_c] = GREEN

    return out


def _split_nonempty(items: list[int]) -> tuple[list[int], list[int]]:
    assert len(items) >= 2
    shuffled = items[:]
    random.shuffle(shuffled)
    cut = random.randint(1, len(shuffled) - 1)
    return shuffled[:cut], shuffled[cut:]


def _choose_palette(bg: int) -> list[int]:
    pool = [color for color in ALL_COLORS if color not in MARK_COLORS and color != bg]
    random.shuffle(pool)
    return pool[: random.randint(2, min(4, len(pool)))]


def _other_color(palette: list[int], color: int) -> int:
    choices = [candidate for candidate in palette if candidate != color]
    return random.choice(choices)


def _assign_targets(extra_ids: list[int], center_is_symmetric: bool) -> dict[int, bool]:
    assert extra_ids
    order = extra_ids[:]
    random.shuffle(order)
    targets: dict[int, bool] = {}

    targets[order[0]] = not center_is_symmetric
    if len(order) >= 2:
        targets[order[1]] = center_is_symmetric
    for idx in order[2:]:
        targets[idx] = random.choice([False, True])
    return targets


def _paint_symmetric_row(grid: np.ndarray, row: int, pair_cols: list[int], palette: list[int]) -> None:
    width = grid.shape[1]
    for col in pair_cols:
        color = random.choice(palette)
        grid[row, col] = color
        grid[row, width - 1 - col] = color


def _paint_asymmetric_row(grid: np.ndarray, row: int, pair_cols: list[int], palette: list[int]) -> None:
    width = grid.shape[1]
    bad_col = random.choice(pair_cols)
    style = random.choice(["missing", "mismatch"])

    for col in pair_cols:
        if col == bad_col:
            color = random.choice(palette)
            if style == "missing":
                target_col = col if random.random() < 0.5 else width - 1 - col
                grid[row, target_col] = color
            else:
                other = _other_color(palette, color)
                grid[row, col] = color
                grid[row, width - 1 - col] = other
        elif random.random() < 0.7:
            color = random.choice(palette)
            grid[row, col] = color
            grid[row, width - 1 - col] = color


def _paint_symmetric_col(grid: np.ndarray, col: int, top_rows: list[int], palette: list[int]) -> None:
    height = grid.shape[0]
    for row in top_rows:
        color = random.choice(palette)
        grid[row, col] = color
        grid[height - 1 - row, col] = color


def _paint_asymmetric_col(grid: np.ndarray, col: int, top_rows: list[int], palette: list[int]) -> None:
    height = grid.shape[0]
    bad_row = random.choice(top_rows)
    style = random.choice(["missing", "mismatch"])

    for row in top_rows:
        if row == bad_row:
            color = random.choice(palette)
            if style == "missing":
                target_row = row if random.random() < 0.5 else height - 1 - row
                grid[target_row, col] = color
            else:
                other = _other_color(palette, color)
                grid[row, col] = color
                grid[height - 1 - row, col] = other
        elif random.random() < 0.7:
            color = random.choice(palette)
            grid[row, col] = color
            grid[height - 1 - row, col] = color


def _apply_random_flips(
    grid: np.ndarray, true_rows: set[int], true_cols: set[int]
) -> tuple[np.ndarray, set[int], set[int]]:
    h, w = grid.shape
    out = grid
    rows = set(true_rows)
    cols = set(true_cols)

    if random.random() < 0.5:
        out = np.flipud(out).copy()
        rows = {h - 1 - row for row in rows}
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
        cols = {w - 1 - col for col in cols}

    return out, rows, cols


def _counts_ok(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(value): int(count) for value, count in zip(values, counts)}
    bg_count = count_map[bg]
    return all(color == bg or count < bg_count for color, count in count_map.items())


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(ODD_SIZES)
    width = random.choice(ODD_SIZES)
    mid_r = height // 2
    mid_c = width // 2

    top_half = list(range(mid_r))
    left_half = list(range(mid_c))
    if len(top_half) < 2 or len(left_half) < 2:
        return None

    bg = random.choice(BACKGROUND_CHOICES)
    palette = _choose_palette(bg)
    grid = np.full((height, width), bg, dtype=int)

    row_pool, col_row_pool = _split_nonempty(top_half)
    row_col_pool, col_pool = _split_nonempty(left_half)

    row_activity_rows = random.sample(row_pool, random.randint(1, min(2, len(row_pool))))
    col_activity_rows = random.sample(col_row_pool, random.randint(1, min(2, len(col_row_pool))))
    row_pair_cols = random.sample(row_col_pool, random.randint(1, min(2, len(row_col_pool))))
    col_activity_cols = random.sample(col_pool, random.randint(1, min(2, len(col_pool))))

    center_row_symmetric, center_col_symmetric = random.choice(
        [(False, False), (False, True), (True, False), (True, True)]
    )

    row_targets = {mid_r: center_row_symmetric}
    row_targets.update(_assign_targets(row_activity_rows, center_row_symmetric))
    col_targets = {mid_c: center_col_symmetric}
    col_targets.update(_assign_targets(col_activity_cols, center_col_symmetric))

    for row, symmetric in row_targets.items():
        if symmetric:
            _paint_symmetric_row(grid, row, row_pair_cols, palette)
        else:
            _paint_asymmetric_row(grid, row, row_pair_cols, palette)

    all_col_targets = {mid_c: center_col_symmetric}
    all_col_targets.update({col: col_targets[col] for col in col_activity_cols})
    for col, symmetric in all_col_targets.items():
        if symmetric:
            _paint_symmetric_col(grid, col, col_activity_rows, palette)
        else:
            _paint_asymmetric_col(grid, col, col_activity_rows, palette)

    expected_true_rows = {row for row, symmetric in row_targets.items() if symmetric}
    expected_true_cols = {col for col, symmetric in all_col_targets.items() if symmetric}

    grid, expected_true_rows, expected_true_cols = _apply_random_flips(
        grid, expected_true_rows, expected_true_cols
    )

    if not _counts_ok(grid, bg):
        return None

    actual_row_sym, actual_col_sym = _symmetry_flags(grid, bg)
    actual_true_rows = {idx for idx, flag in enumerate(actual_row_sym) if flag}
    actual_true_cols = {idx for idx, flag in enumerate(actual_col_sym) if flag}

    if actual_true_rows != expected_true_rows:
        return None
    if actual_true_cols != expected_true_cols:
        return None

    if not ({False, True} <= set(row_targets.values())):
        return None
    if not ({False, True} <= set(all_col_targets.values())):
        return None

    out = main(grid)
    center = int(out[grid.shape[0] // 2, grid.shape[1] // 2])
    expected_center = (
        GREEN
        if center_row_symmetric and center_col_symmetric
        else RED
        if center_row_symmetric
        else BLUE
        if center_col_symmetric
        else bg
    )
    if center != expected_center:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-marker input")

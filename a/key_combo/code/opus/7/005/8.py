import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREEN = 3
MASK = 6
TEAL = 8
FG_COLORS = (RED, GREEN)
MAX_GENERATION_ATTEMPTS = 400


def _mirror_col(col: int, width: int) -> int:
    return int(width - 1 - col)


def _foreground_columns(grid: np.ndarray) -> set[int]:
    cols = set()
    for c in range(grid.shape[1]):
        if np.any(grid[:, c] != BG):
            cols.add(int(c))
    return cols


def _visible_foreground_columns(grid: np.ndarray) -> set[int]:
    cols = set()
    for c in range(grid.shape[1]):
        if np.any(np.isin(grid[:, c], FG_COLORS)):
            cols.add(int(c))
    return cols


def _complete_masks(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    height, width = out.shape
    for r in range(height):
        for c in range(width):
            if out[r, c] != MASK:
                continue
            mc = _mirror_col(c, width)
            mate = int(out[r, mc])
            assert mate != MASK, "masked cells must have a visible mirror mate"
            out[r, c] = mate
    return out


def _fill_foreground_columns(grid: np.ndarray, selected_cols: Iterable[int] | None = None) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    cols = _foreground_columns(out) if selected_cols is None else {int(c) for c in selected_cols}
    for c in cols:
        mask = out[:, c] == BG
        out[mask, c] = TEAL
    return out


def _tile_2x2(grid: np.ndarray) -> np.ndarray:
    return np.tile(np.asarray(grid, dtype=int), (2, 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the hidden half-symmetric pixels, cyan-fill every foreground column, then tile the result 2x2.

    Input structure:
    - The background is black.
    - Red and green pixels form a pattern with left-right mirror symmetry around the vertical axis.
    - Some right-half pixels are replaced by pink masks that hide the mirrored value at that location.

    Rule:
    - For every pink cell, copy the color from its left-right mirror partner.
    - After all masks are resolved, mark every column that contains at least one non-background pixel as a foreground column.
    - In each foreground column, recolor the remaining black cells to cyan while keeping existing foreground colors unchanged.
    - Output a 2x2 tiling of that completed cyan-filled tile.

    Key insight:
    - Some columns look empty before completion because their only foreground pixels were masked, so the masks must be resolved before deciding which columns get the cyan full-column fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    completed = _complete_masks(g)
    tile = _fill_foreground_columns(completed)
    out = _tile_2x2(tile)
    return np.asarray(out, dtype=int)


def _interval_rows(height: int) -> list[int]:
    covered = set()
    segment_count = random.randint(1, min(3, height - 1))
    for _ in range(segment_count):
        length = random.randint(1, min(4, height - 1))
        start = random.randint(0, height - length)
        for r in range(start, start + length):
            covered.add(r)
    if not covered or len(covered) == height:
        return []
    return sorted(int(r) for r in covered)


def _add_column_pattern(grid: np.ndarray, left_col: int) -> bool:
    rows = _interval_rows(grid.shape[0])
    if not rows:
        return False
    width = grid.shape[1]
    right_col = _mirror_col(left_col, width)
    for r in rows:
        color = random.choice(FG_COLORS)
        grid[r, left_col] = color
        grid[r, right_col] = color
    return True


def _add_center_column_pattern(grid: np.ndarray) -> None:
    height, width = grid.shape
    if width % 2 == 0 or random.random() < 0.5:
        return
    center = width // 2
    rows = _interval_rows(height)
    if not rows:
        return
    for r in rows:
        grid[r, center] = random.choice(FG_COLORS)


def _build_symmetric_grid(height: int, width: int) -> np.ndarray | None:
    strict_left_width = width // 2
    if strict_left_width < 3:
        return None

    grid = np.zeros((height, width), dtype=int)
    max_pairs = min(5, strict_left_width - 1)
    if max_pairs < 2:
        return None
    pair_count = random.randint(2, max_pairs)
    left_cols = sorted(random.sample(range(strict_left_width), pair_count))

    for col in left_cols:
        if not _add_column_pattern(grid, col):
            return None

    if sum(np.any(np.isin(grid, FG_COLORS), axis=0)) < 4:
        return None

    _add_center_column_pattern(grid)
    return grid


def _mask_grid(completed: np.ndarray) -> tuple[np.ndarray, set[int]] | tuple[None, None]:
    _height, width = completed.shape
    strict_left_width = width // 2
    right_cols = [
        c
        for c in range(strict_left_width, width)
        if c != _mirror_col(c, width) and np.any(np.isin(completed[:, c], FG_COLORS))
    ]
    if len(right_cols) < 2:
        return None, None

    activation_count = random.randint(1, min(3, len(right_cols) - 1))
    activation_cols = set(random.sample(right_cols, activation_count))
    remaining = [c for c in right_cols if c not in activation_cols]
    distractor_col = random.choice(remaining)

    inp = np.array(completed, copy=True)

    for c in activation_cols:
        rows = np.where(np.isin(inp[:, c], FG_COLORS))[0]
        if len(rows) == 0:
            return None, None
        inp[rows, c] = MASK

    distractor_rows = np.where(np.isin(inp[:, distractor_col], FG_COLORS))[0]
    if len(distractor_rows) < 2:
        return None, None
    keep_count = random.randint(1, len(distractor_rows) - 1)
    keep_rows = set(random.sample(list(map(int, distractor_rows)), keep_count))
    for r in distractor_rows:
        if int(r) not in keep_rows:
            inp[int(r), distractor_col] = MASK

    for c in remaining:
        if c == distractor_col or random.random() < 0.45:
            continue
        rows = [int(r) for r in np.where(np.isin(inp[:, c], FG_COLORS))[0]]
        if len(rows) <= 1:
            continue
        sample_size = random.randint(1, len(rows) - 1)
        for r in random.sample(rows, sample_size):
            inp[r, c] = MASK

    return inp, activation_cols


def _force_both_foreground_colors(completed: np.ndarray) -> bool:
    if np.any(completed == RED) and np.any(completed == GREEN):
        return True

    height, width = completed.shape
    reps = []
    for r, c in np.argwhere(np.isin(completed, FG_COLORS)):
        rr = int(r)
        cc = int(c)
        if cc <= _mirror_col(cc, width):
            reps.append((rr, cc))
    if len(reps) < 2:
        return False

    random.shuffle(reps)
    for color, (r, c) in zip(FG_COLORS, reps[:2]):
        mc = _mirror_col(c, width)
        completed[r, c] = color
        completed[r, mc] = color
    return np.any(completed == RED) and np.any(completed == GREEN)


def _semantic_checks(inp: np.ndarray, completed: np.ndarray, activation_cols: set[int]) -> bool:
    height, width = inp.shape
    if not (inp.ndim == 2 and max(inp.shape) <= 15):
        return False
    if not np.all(completed == np.fliplr(completed)):
        return False
    if np.any(inp[:, : width // 2] == MASK):
        return False
    if width % 2 == 1 and np.any(inp[:, width // 2] == MASK):
        return False

    for r, c in np.argwhere(inp == MASK):
        if completed[r, c] not in FG_COLORS:
            return False
        if inp[r, _mirror_col(int(c), width)] == MASK:
            return False

    visible_cols = _visible_foreground_columns(inp)
    completed_cols = _visible_foreground_columns(completed)
    if not activation_cols:
        return False
    if not activation_cols.issubset(completed_cols - visible_cols):
        return False
    if completed_cols == visible_cols:
        return False
    if len({c for c in range(width) if np.any(inp[:, c] == MASK) and c in visible_cols}) == 0:
        return False

    tile = _fill_foreground_columns(completed)
    if not np.any(tile == TEAL):
        return False
    if not np.any(tile == BG):
        return False

    expected = _tile_2x2(tile)
    if not np.array_equal(main(inp), expected):
        return False

    wrong_tile = _fill_foreground_columns(completed, visible_cols)
    if np.array_equal(_tile_2x2(wrong_tile), expected):
        return False

    if np.array_equal(_tile_2x2(completed), expected):
        return False

    hidden_only = {
        c
        for c in range(width)
        if c in completed_cols and c not in visible_cols and np.any(inp[:, c] == MASK)
    }
    if hidden_only != activation_cols:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(7, 14)
    width = random.randint(7, 15)
    completed = _build_symmetric_grid(height, width)
    if completed is None:
        return None

    if not _force_both_foreground_colors(completed):
        return None

    inp, activation_cols = _mask_grid(completed)
    if inp is None or activation_cols is None:
        return None

    if not _semantic_checks(inp, completed, activation_cols):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return np.asarray(grid, dtype=int)
    raise RuntimeError("failed to generate a valid input for puzzle 8")

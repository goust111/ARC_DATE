from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PLACEHOLDER = 5
TILE_COLORS = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 500


def _placeholder_start(grid: np.ndarray) -> int:
    placeholder_cols = np.all(grid == PLACEHOLDER, axis=0)
    assert np.any(placeholder_cols), "expected a placeholder suffix"

    start = int(np.argmax(placeholder_cols))
    assert start > 0, "placeholder suffix must start after visible columns"
    assert np.all(placeholder_cols[start:]), "placeholder must be a suffix"
    assert not np.any(placeholder_cols[:start]), "placeholder must not appear before suffix"
    return start


def _minimal_horizontal_period(grid: np.ndarray, width: int) -> int:
    assert width > 0
    visible = np.asarray(grid[:, :width], dtype=int)
    for period in range(1, width + 1):
        matches = True
        for c in range(width - period):
            if not np.array_equal(visible[:, c], visible[:, c + period]):
                matches = False
                break
        if matches:
            return period
    raise AssertionError("failed to infer horizontal period")


def _complete_placeholder(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    cut = _placeholder_start(g)
    period = _minimal_horizontal_period(g, cut)

    completed = g.copy()
    for c in range(cut, g.shape[1]):
        completed[:, c] = g[:, c % period]
    return completed, period, cut


def _apply_gravity(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)

    for c in range(w):
        column = g[:, c]
        stack = column[column != BACKGROUND]
        if stack.size:
            out[h - stack.size :, c] = stack
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the hidden horizontal tile repetition, then drop every foreground pixel straight down.

    Input structure:
    - The grid has a black background, a visible left region tiled by a sparse repeating pattern, and a grey placeholder suffix on the right.
    - The visible pattern uses foreground colors plus black gaps, and the placeholder suffix covers full columns.
    - The left/right boundary is aligned to a whole number of horizontal tile periods.

    Rule:
    - Infer the smallest horizontal period from the visible non-placeholder columns.
    - Replace every grey placeholder cell by continuing that horizontal period across the hidden suffix.
    - After completion, process each column independently: keep the top-to-bottom order of non-black pixels, but let them fall to the bottom of the column.

    Key insight:
    - The grey suffix is not preserved; it hides a continuation of the same periodic columns, and only after restoring those columns can the final gravity compaction be applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    completed, _, _ = _complete_placeholder(g)
    return _apply_gravity(completed)


def _build_tile(tile_h: int, tile_w: int, colors: tuple[int, ...]) -> np.ndarray | None:
    area = tile_h * tile_w
    min_bg = max(1, int(np.ceil(area * 0.4)))
    max_bg = min(area - 1, int(np.floor(area * 0.6)))
    if min_bg > max_bg:
        return None

    bg_count = random.randint(min_bg, max_bg)
    tile = np.zeros((tile_h, tile_w), dtype=int)

    cells = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    random.shuffle(cells)

    for r, c in cells[bg_count:]:
        tile[r, c] = random.choice(colors)

    if len(set(map(int, tile[tile != BACKGROUND].flat))) < 2:
        return None

    if np.any(np.all(tile == BACKGROUND, axis=0)):
        return None

    if np.any(np.all(tile == BACKGROUND, axis=1)):
        return None

    return tile


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    tile_h, tile_w = tile.shape
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = tile[r % tile_h, c % tile_w]
    return out


def _floating_column_count(grid: np.ndarray, start_col: int) -> int:
    h, w = grid.shape
    count = 0
    for c in range(start_col, w):
        if any(grid[r, c] != BACKGROUND and grid[r + 1, c] == BACKGROUND for r in range(h - 1)):
            count += 1
    return count


def _gravity_changes(grid: np.ndarray) -> int:
    fallen = _apply_gravity(grid)
    return sum(
        1
        for c in range(grid.shape[1])
        if not np.array_equal(grid[:, c], fallen[:, c])
    )


def _gen_attempt() -> np.ndarray | None:
    tile_h = random.randint(3, 4)
    tile_w = random.randint(3, 4)
    row_reps = random.randint(3, 4)
    visible_reps = random.randint(2, 4)
    hidden_reps = random.randint(1, 2)
    hidden_extra = random.randint(0, tile_w - 1)
    colors = tuple(random.sample(TILE_COLORS, random.randint(2, 3)))

    tile = _build_tile(tile_h, tile_w, colors)
    if tile is None:
        return None

    h = tile_h * row_reps
    cut = tile_w * visible_reps
    hidden_w = hidden_reps * tile_w + hidden_extra
    w = cut + hidden_w

    if max(h, w) > 30:
        return None

    placeholder_ratio = hidden_w / w
    if not (0.25 <= placeholder_ratio <= 0.5):
        return None

    full = _tile_grid(tile, (h, w))

    if _minimal_horizontal_period(full, cut) != tile_w:
        return None

    if not np.any(full[:, cut:] != BACKGROUND):
        return None

    if _floating_column_count(full, cut) < 3:
        return None

    if _gravity_changes(full) < 3:
        return None

    inp = full.copy()
    inp[:, cut:] = PLACEHOLDER
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for ARC task 9")

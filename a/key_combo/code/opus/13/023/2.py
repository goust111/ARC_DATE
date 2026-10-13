from __future__ import annotations

import random

import numpy as np


INPUT_COLORS = (0, 1, 2, 3, 4)
REPAINT_COLOR = 5
GRID_WIDTH = 5
GRID_HEIGHT_RANGE = (6, 8)
MAX_ATTEMPTS = 300


def _uniform_row_mask(grid: np.ndarray) -> np.ndarray:
    return np.all(grid == grid[:, :1], axis=1)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    uniform_mask = _uniform_row_mask(g)
    scale = int(uniform_mask.sum())

    if scale == 0:
        return np.zeros((0, g.shape[1]), dtype=int)

    output_rows: list[np.ndarray] = []
    for is_uniform, row in zip(uniform_mask, g):
        if bool(is_uniform):
            output_rows.append(np.full((1, g.shape[1]), REPAINT_COLOR, dtype=int))
        else:
            output_rows.append(np.repeat(row[None, :], scale, axis=0))
    return np.vstack(output_rows)


def main(grid: np.ndarray) -> np.ndarray:
    """Count uniform rows, repaint them gray, and vertically repeat only the non-uniform rows by that count.

    Input structure:
    - A small rectangular grid uses black, blue, red, green, and yellow cells.
    - Some rows are single-color rows whose cells all match each other.
    - The remaining rows are mixed-color rows containing at least two colors.

    Rule:
    - Count the single-color rows and call that number N.
    - Process rows from top to bottom without changing their order.
    - Replace each single-color row by one gray row of the same width.
    - Replace each mixed-color row by N identical copies of that row stacked vertically.

    Key insight:
    - The single-color rows are self-referential controls: their count determines the scale factor for the
      mixed rows, while those same control rows are exempt from scaling and only get recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _transform(g)


def _make_uniform_colors(count: int) -> list[int]:
    colors = [random.choice(INPUT_COLORS) for _ in range(count)]
    if count >= 2 and len(set(colors)) == 1:
        idx = random.randrange(count)
        alternatives = [color for color in INPUT_COLORS if color != colors[idx]]
        colors[idx] = random.choice(alternatives)
    return colors


def _make_mixed_row() -> np.ndarray:
    mode = random.randrange(3)

    if mode == 0:
        left_color, right_color = random.sample(INPUT_COLORS, 2)
        split = random.randint(1, GRID_WIDTH - 1)
        row = [left_color] * split + [right_color] * (GRID_WIDTH - split)
    elif mode == 1:
        color_a, color_b = random.sample(INPUT_COLORS, 2)
        phase = random.randrange(2)
        row = [color_a if (c + phase) % 2 == 0 else color_b for c in range(GRID_WIDTH)]
    else:
        color_a, color_b, color_c = random.sample(INPUT_COLORS, 3)
        row = [random.choice((color_a, color_b)) for _ in range(GRID_WIDTH)]
        specials = random.sample(range(GRID_WIDTH), 2)
        row[specials[0]] = color_c
        row[specials[1]] = color_b

    if len(set(row)) < 2:
        return _make_mixed_row()
    return np.array(row, dtype=int)


def _build_expected(rows: list[np.ndarray], uniform_flags: list[bool], uniform_count: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for row, is_uniform in zip(rows, uniform_flags):
        if is_uniform:
            pieces.append(np.full((1, GRID_WIDTH), REPAINT_COLOR, dtype=int))
        else:
            pieces.append(np.repeat(row[None, :], uniform_count, axis=0))
    return np.vstack(pieces)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    valid_uniform_counts = [count for count in range(2, min(4, height - 2) + 1)]
    uniform_count = random.choice(valid_uniform_counts)

    uniform_rows = set(random.sample(range(height), uniform_count))
    uniform_colors = _make_uniform_colors(uniform_count)

    rows: list[np.ndarray] = []
    uniform_flags: list[bool] = []
    uniform_idx = 0
    mixed_patterns = set()

    for r in range(height):
        if r in uniform_rows:
            row = np.full(GRID_WIDTH, uniform_colors[uniform_idx], dtype=int)
            uniform_idx += 1
            uniform_flags.append(True)
        else:
            row = _make_mixed_row()
            mixed_patterns.add(tuple(map(int, row)))
            uniform_flags.append(False)
        rows.append(row)

    grid = np.vstack(rows)
    if int(_uniform_row_mask(grid).sum()) != uniform_count:
        return None
    if len(rows) - uniform_count < 2:
        return None
    if len(set(map(int, grid.reshape(-1)))) < 3:
        return None
    if len(mixed_patterns) == 1 and len(mixed_patterns) < len(rows) - uniform_count:
        return None

    expected = _build_expected(rows, uniform_flags, uniform_count)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

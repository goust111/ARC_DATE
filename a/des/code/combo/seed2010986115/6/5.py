from __future__ import annotations

import random

import numpy as np


BG = 0
GRID = 5
CELL = 3
STEP = CELL + 1
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 300


def _pattern(bg: int, size: int, row_color: int, col_color: int) -> np.ndarray:
    if row_color == col_color:
        return np.full((size, size), row_color, dtype=int)

    patch = np.full((size, size), bg, dtype=int)
    mid = size // 2
    patch[mid, :] = row_color
    patch[:, mid] = col_color
    patch[mid, mid] = row_color
    return patch


def _detect_grid(g: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, int]:
    best: tuple[int, np.ndarray, np.ndarray, int] | None = None
    best_score = -1

    for color in np.unique(g):
        color = int(color)
        row_lines = np.flatnonzero(np.all(g == color, axis=1))
        col_lines = np.flatnonzero(np.all(g == color, axis=0))
        if row_lines.size < 3 or col_lines.size < 3:
            continue
        row_steps = np.diff(row_lines)
        col_steps = np.diff(col_lines)
        if row_steps.size == 0 or col_steps.size == 0:
            continue
        if not np.all(row_steps == row_steps[0]) or not np.all(col_steps == col_steps[0]):
            continue
        if int(row_steps[0]) != int(col_steps[0]):
            continue

        step = int(row_steps[0])
        cell_size = step - 1
        if cell_size <= 0 or cell_size % 2 == 0:
            continue

        score = int(row_lines.size + col_lines.size)
        if score > best_score:
            best_score = score
            best = (color, row_lines, col_lines, cell_size)

    assert best is not None, "failed to detect the grey cell grid"
    return best


def _cell_patch(g: np.ndarray, row_lines: np.ndarray, col_lines: np.ndarray, i: int, j: int) -> np.ndarray:
    r0 = int(row_lines[i]) + 1
    r1 = int(row_lines[i + 1])
    c0 = int(col_lines[j]) + 1
    c1 = int(col_lines[j + 1])
    return g[r0:r1, c0:c1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the row/column marker colors from the header cells, then fill each empty
    interior cell with the pattern implied by that row-color/column-color pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _grid_color, row_lines, col_lines, cell_size = _detect_grid(g)
    bg = int(_cell_patch(g, row_lines, col_lines, 0, 0)[cell_size // 2, cell_size // 2])

    row_count = int(row_lines.size - 1)
    col_count = int(col_lines.size - 1)
    assert row_count >= 2 and col_count >= 2

    mid = cell_size // 2
    row_colors = []
    for i in range(1, row_count):
        patch = _cell_patch(g, row_lines, col_lines, i, 0)
        row_color = int(patch[mid, mid])
        assert row_color != bg
        row_colors.append(row_color)

    col_colors = []
    for j in range(1, col_count):
        patch = _cell_patch(g, row_lines, col_lines, 0, j)
        col_color = int(patch[mid, mid])
        assert col_color != bg
        col_colors.append(col_color)

    out = g.copy()
    for i, row_color in enumerate(row_colors, start=1):
        for j, col_color in enumerate(col_colors, start=1):
            patch = _cell_patch(out, row_lines, col_lines, i, j)
            if np.all(patch == bg):
                patch[:, :] = _pattern(bg, cell_size, row_color, col_color)

    return out


def _full_expected(row_colors: list[int], col_colors: list[int]) -> np.ndarray:
    total_rows = len(row_colors) + 1
    total_cols = len(col_colors) + 1
    height = total_rows * STEP + 1
    width = total_cols * STEP + 1

    grid = np.full((height, width), BG, dtype=int)
    for r in range(0, height, STEP):
        grid[r, :] = GRID
    for c in range(0, width, STEP):
        grid[:, c] = GRID

    for i, color in enumerate(row_colors, start=1):
        grid[i * STEP + CELL // 2 + 1, CELL // 2 + 1] = color
    for j, color in enumerate(col_colors, start=1):
        grid[CELL // 2 + 1, j * STEP + CELL // 2 + 1] = color

    for i, row_color in enumerate(row_colors, start=1):
        for j, col_color in enumerate(col_colors, start=1):
            r0 = i * STEP + 1
            c0 = j * STEP + 1
            grid[r0 : r0 + CELL, c0 : c0 + CELL] = _pattern(BG, CELL, row_color, col_color)

    return grid


def _gen_markers(nrows: int, ncols: int) -> tuple[list[int], list[int]] | None:
    pool_size = random.randint(2, min(6, len(PALETTE)))
    pool = random.sample(PALETTE, pool_size)

    row_colors = [random.choice(pool) for _ in range(nrows)]
    col_colors = [random.choice(pool) for _ in range(ncols)]

    shared = random.choice(pool)
    row_colors[random.randrange(nrows)] = shared
    col_colors[random.randrange(ncols)] = shared

    if len(set(row_colors + col_colors)) == 1:
        alt = random.choice([c for c in pool if c != shared])
        if random.random() < 0.5:
            row_colors[random.randrange(nrows)] = alt
        else:
            col_colors[random.randrange(ncols)] = alt

    equal_positions = [(r, c) for r in range(nrows) for c in range(ncols) if row_colors[r] == col_colors[c]]
    diff_positions = [(r, c) for r in range(nrows) for c in range(ncols) if row_colors[r] != col_colors[c]]
    if not equal_positions or not diff_positions:
        return None

    return row_colors, col_colors


def _sample_examples(
    row_colors: list[int],
    col_colors: list[int],
) -> set[tuple[int, int]] | None:
    nrows = len(row_colors)
    ncols = len(col_colors)
    all_positions = [(r, c) for r in range(nrows) for c in range(ncols)]
    equal_positions = [pos for pos in all_positions if row_colors[pos[0]] == col_colors[pos[1]]]
    diff_positions = [pos for pos in all_positions if row_colors[pos[0]] != col_colors[pos[1]]]
    if not equal_positions or not diff_positions:
        return None

    examples = {
        random.choice(equal_positions),
        random.choice(diff_positions),
    }

    reserved_blank: set[tuple[int, int]] = set()
    equal_left = [pos for pos in equal_positions if pos not in examples]
    diff_left = [pos for pos in diff_positions if pos not in examples]
    if equal_left:
        reserved_blank.add(random.choice(equal_left))
    if diff_left:
        reserved_blank.add(random.choice(diff_left))

    free_positions = [pos for pos in all_positions if pos not in examples and pos not in reserved_blank]
    max_examples = len(all_positions) - max(1, len(reserved_blank))
    if max_examples < len(examples):
        return None
    target_count = random.randint(len(examples), max_examples)
    extra_count = target_count - len(examples)
    if extra_count > len(free_positions):
        return None
    if extra_count:
        examples.update(random.sample(free_positions, extra_count))

    if len(examples) == len(all_positions):
        return None
    return examples


def _gen_attempt() -> np.ndarray | None:
    total_rows = random.randint(4, 6)
    total_cols = random.randint(4, 6)
    nrows = total_rows - 1
    ncols = total_cols - 1

    markers = _gen_markers(nrows, ncols)
    if markers is None:
        return None
    row_colors, col_colors = markers

    example_positions = _sample_examples(row_colors, col_colors)
    if example_positions is None:
        return None

    full = _full_expected(row_colors, col_colors)
    inp = full.copy()
    for r in range(nrows):
        for c in range(ncols):
            if (r, c) in example_positions:
                continue
            r0 = (r + 1) * STEP + 1
            c0 = (c + 1) * STEP + 1
            inp[r0 : r0 + CELL, c0 : c0 + CELL] = BG

    if np.array_equal(inp, full):
        return None
    if not np.array_equal(main(inp), full):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

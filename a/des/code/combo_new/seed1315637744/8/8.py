from __future__ import annotations

import random

import numpy as np


BLACK = 0
SEPARATOR = 5
LEGEND_COLORS = (1, 2, 3, 4, 7)  # blue, red, green, yellow, orange
PATTERN_COLORS = tuple(color for color in range(10) if color not in {BLACK, SEPARATOR})

HALF_HEIGHT_RANGE = (6, 14)
HALF_WIDTH_RANGE = (6, 14)
BRANCHES = (1, 2, 3, 4, 5, "gt5")
GT5_DIFF_RANGE = (6, 8)
SHAPE_TYPES = ("rect", "hline", "vline", "el", "plus")
MIN_SHAPES = 3
MAX_SHAPES = 6
MAX_SHAPE_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 400


def _center_separator_column(top: np.ndarray) -> int:
    h, w = top.shape
    assert h > 0 and w >= 3 and w % 2 == 1, "top area must have two equal halves and one separator column"
    sep = w // 2
    sep_color = int(top[0, sep])
    assert sep_color != BLACK and np.all(top[:, sep] == sep_color), "separator column must be solid and non-black"
    return sep


def _output_from_halves(left: np.ndarray, right: np.ndarray, legend: np.ndarray) -> np.ndarray:
    assert left.shape == right.shape, "halves must have identical shapes"
    diff = left != right
    out = np.zeros_like(left)
    diff_count = int(np.count_nonzero(diff))
    if diff_count == 0:
        return out
    color = int(legend[min(diff_count, 5) - 1])
    out[diff] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the source and noisy halves, then mark each differing position using the legend color for the difference count.

    Input structure:
    - The grid has a black-background top area split into two equal halves by one solid grey separator column.
    - The left half is a multicolor source pattern and the right half is the same-sized noisy copy with a few pixel edits.
    - The bottom row is a color legend whose first five cells encode the count-to-color mapping.

    Rule:
    - Discard the bottom legend row and compare the left and right halves pixel by pixel.
    - Count the total number of differing positions D and create a black output grid the size of one half.
    - Mark every differing position with the legend color at index D, clamped so counts above five use the fifth legend color.

    Key insight:
    - The task is not to repair the noisy copy; it is to turn the set of mismatch coordinates into a mask whose color is chosen by reading the bottom legend as a difference-count lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 5, "expected a 2D grid with a legend row"

    top = g[:-1]
    legend = g[-1]
    assert legend.shape[0] >= 5, "legend row must expose at least five cells"

    sep = _center_separator_column(top)
    left = top[:, :sep]
    right = top[:, sep + 1 :]
    return _output_from_halves(left, right, legend[:5])


def _shape_cells() -> list[tuple[int, int]]:
    kind = random.choice(SHAPE_TYPES)
    if kind == "rect":
        height = random.randint(1, 3)
        width = random.randint(1, 4)
        return [(r, c) for r in range(height) for c in range(width)]

    if kind == "hline":
        width = random.randint(2, 5)
        return [(0, c) for c in range(width)]

    if kind == "vline":
        height = random.randint(2, 5)
        return [(r, 0) for r in range(height)]

    if kind == "el":
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        orientation = random.randint(0, 3)
        if orientation == 0:
            cells = {(r, 0) for r in range(height)} | {(height - 1, c) for c in range(width)}
        elif orientation == 1:
            cells = {(r, width - 1) for r in range(height)} | {(height - 1, c) for c in range(width)}
        elif orientation == 2:
            cells = {(r, 0) for r in range(height)} | {(0, c) for c in range(width)}
        else:
            cells = {(r, width - 1) for r in range(height)} | {(0, c) for c in range(width)}
        return sorted(cells)

    arm = random.randint(1, 2)
    size = 2 * arm + 1
    center = arm
    cells = {(center, c) for c in range(size)} | {(r, center) for r in range(size)}
    return sorted(cells)


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        blocked[r0:r1, c0:c1] = True


def _place_pattern(height: int, width: int, min_pixels: int) -> np.ndarray | None:
    n_shapes = random.randint(MIN_SHAPES, MAX_SHAPES)
    palette_size = random.randint(2, min(5, len(PATTERN_COLORS)))
    palette = random.sample(list(PATTERN_COLORS), k=palette_size)

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    used_colors: set[int] = set()
    placed = 0

    for _ in range(MAX_SHAPE_ATTEMPTS):
        if placed >= n_shapes:
            break
        rel_cells = _shape_cells()
        rel_rows = [r for r, _ in rel_cells]
        rel_cols = [c for _, c in rel_cells]
        shape_h = max(rel_rows) + 1
        shape_w = max(rel_cols) + 1
        if shape_h > height or shape_w > width:
            continue

        top = random.randint(0, height - shape_h)
        left = random.randint(0, width - shape_w)
        cells = [(top + r, left + c) for r, c in rel_cells]
        if any(blocked[r, c] for r, c in cells):
            continue

        color = random.choice(palette)
        for r, c in cells:
            grid[r, c] = color
        _mark_blocked(blocked, cells)
        used_colors.add(color)
        placed += 1

    non_black = int(np.count_nonzero(grid))
    max_pixels = (height * width * 3) // 5
    if placed < MIN_SHAPES:
        return None
    if non_black < max(min_pixels, 6) or non_black > max_pixels:
        return None
    if len(used_colors) < 2:
        return None
    return grid


def _choose_diff_count(height: int, width: int) -> int:
    branch = random.choice(BRANCHES)
    if branch != "gt5":
        return int(branch)
    upper = min(GT5_DIFF_RANGE[1], height * width)
    assert upper >= GT5_DIFF_RANGE[0], "half-grid must fit the >5 branch"
    return random.randint(GT5_DIFF_RANGE[0], upper)


def _spread_positions(height: int, width: int, count: int) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    attempts = 0
    while len(positions) < count and attempts < 600:
        pos = (random.randrange(height), random.randrange(width))
        attempts += 1
        if pos in used:
            continue
        if any(max(abs(pos[0] - r), abs(pos[1] - c)) <= 1 for r, c in positions):
            continue
        positions.append(pos)
        used.add(pos)

    if len(positions) < count:
        remaining = [(r, c) for r in range(height) for c in range(width) if (r, c) not in used]
        random.shuffle(remaining)
        positions.extend(remaining[: count - len(positions)])
    assert len(positions) == count, "must sample the requested number of difference positions"
    return positions


def _apply_noise(source: np.ndarray, diff_positions: list[tuple[int, int]]) -> np.ndarray:
    noisy = source.copy()
    for r, c in diff_positions:
        current = int(source[r, c])
        if current == BLACK:
            noisy[r, c] = random.choice(PATTERN_COLORS)
            continue

        if random.random() < 0.5:
            noisy[r, c] = BLACK
        else:
            choices = [color for color in PATTERN_COLORS if color != current]
            noisy[r, c] = random.choice(choices)
    return noisy


def _assemble_input(source: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    height, width = source.shape
    grid = np.zeros((height + 1, 2 * width + 1), dtype=int)
    grid[:height, :width] = source
    grid[:height, width] = SEPARATOR
    grid[:height, width + 1 :] = noisy
    grid[height, : len(LEGEND_COLORS)] = LEGEND_COLORS
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HALF_HEIGHT_RANGE)
    width = random.randint(*HALF_WIDTH_RANGE)
    diff_count = _choose_diff_count(height, width)

    source = _place_pattern(height, width, min_pixels=diff_count)
    if source is None:
        return None

    diff_positions = _spread_positions(height, width, diff_count)
    noisy = _apply_noise(source, diff_positions)
    if int(np.count_nonzero(source != noisy)) != diff_count:
        return None

    grid = _assemble_input(source, noisy)
    expected = _output_from_halves(source, noisy, np.array(LEGEND_COLORS, dtype=int))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")

from __future__ import annotations

import random

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _parse_grid(grid: np.ndarray) -> tuple[int, list[int], int, tuple[int, int], dict[int, tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 3, "grid is too small for this task"

    bg = _background_color(g)

    code_colors: list[int] = []
    for c in range(w):
        color = int(g[0, c])
        if color == bg:
            break
        code_colors.append(color)

    assert len(code_colors) >= 2, "expected at least two code colors"
    assert len(code_colors) == len(set(code_colors)), "code colors must be distinct"
    assert len(code_colors) < w, "top code row must end before the right edge"
    assert np.all(g[0, len(code_colors) :] == bg), "top row must be code colors followed by background"

    separator_color = int(g[1, 0])
    assert separator_color != bg, "separator row must be non-background"
    assert np.all(g[1, :] == separator_color), "second row must be a full-width separator"
    assert separator_color not in code_colors, "separator color must not appear in the code row"

    lower = g[2:, :]
    lower_colors = set(map(int, np.unique(lower)))
    allowed_lower = set(code_colors) | {bg, separator_color}
    assert lower_colors <= allowed_lower, "lower region contains unexpected colors"

    start_cells = np.argwhere(lower == separator_color)
    assert len(start_cells) == 1, "expected exactly one start marker"
    start = (int(start_cells[0, 0] + 2), int(start_cells[0, 1]))

    targets: dict[int, tuple[int, int]] = {}
    for color in code_colors:
        coords = np.argwhere(lower == int(color))
        assert len(coords) == 1, f"expected exactly one marker for color {color}"
        targets[int(color)] = (int(coords[0, 0] + 2), int(coords[0, 1]))

    return bg, code_colors, separator_color, start, targets


def _paint_leg(
    out: np.ndarray,
    bg: int,
    start: tuple[int, int],
    target: tuple[int, int],
    color: int,
) -> None:
    r, c = start
    tr, tc = target

    col_step = 1 if tc >= c else -1
    for cc in range(c, tc + col_step, col_step):
        if int(out[r, cc]) == bg:
            out[r, cc] = int(color)

    row_step = 1 if tr >= r else -1
    for rr in range(r, tr + row_step, row_step):
        if int(out[rr, tc]) == bg:
            out[rr, tc] = int(color)


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, code_colors, _separator_color, start, targets = _parse_grid(g)
    out = g.copy()

    current = start
    for color in code_colors:
        target = targets[int(color)]
        _paint_leg(out, bg, current, target, int(color))
        current = target

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top code row as an itinerary and connect the start to same-colored markers in that exact order.

    Input structure:
    - The first row begins with distinct non-background code colors and then switches to background.
    - The second row is a full-width separator of one color; below it is a black background region.
    - In the lower region, each code color appears exactly once as a single-pixel marker, and one extra single pixel in the separator color marks the start.

    Rule:
    - Start from the separator-colored marker and visit the lower markers in the left-to-right order given by the top row.
    - For each trip, move horizontally along the current row until the target column, then vertically along that column until the target row.
    - Recolor only background cells on that path with the target marker's color; preserve the code row, separator row, all marker pixels, and any earlier path cells.

    Key insight:
    - The top row is not a palette but an ordered route description, while the start is identified by matching the separator row's color.
    """
    return _oracle_output(grid)


def _greedy_nearest_order(
    start: tuple[int, int],
    targets: dict[int, tuple[int, int]],
) -> list[int]:
    remaining = set(targets)
    order: list[int] = []
    current = start
    while remaining:
        color = min(
            remaining,
            key=lambda value: (
                abs(targets[value][0] - current[0]) + abs(targets[value][1] - current[1]),
                targets[value][0],
                targets[value][1],
                value,
            ),
        )
        order.append(int(color))
        current = targets[color]
        remaining.remove(color)
    return order


def _choose_points(h: int, w: int, count: int, min_gap: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(2, h) for c in range(w)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) >= min_gap for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _sample_layout() -> np.ndarray | None:
    code_len = random.randint(2, 5)
    h = random.randint(max(7, code_len + 4), 18)
    w = random.randint(code_len + 3, 20)

    colors = random.sample(NON_BLACK_COLORS, code_len + 1)
    separator_color = int(colors[0])
    code_colors = [int(color) for color in colors[1:]]

    lower_area = (h - 2) * w
    min_gap = 3 if lower_area >= (code_len + 1) * 14 else 2
    points = _choose_points(h, w, code_len + 1, min_gap)
    if points is None:
        return None

    start = points[0]
    marker_positions = {color: points[i + 1] for i, color in enumerate(code_colors)}

    if _greedy_nearest_order(start, marker_positions) == code_colors:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :code_len] = np.array(code_colors, dtype=int)
    grid[1, :] = separator_color
    grid[start] = separator_color
    for color, pos in marker_positions.items():
        grid[pos] = int(color)

    out = _oracle_output(grid)
    added = int(np.count_nonzero((grid == BLACK) & (out != BLACK)))
    if added < max(4, code_len + 1):
        return None
    if added > ((h - 2) * w) // 3:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

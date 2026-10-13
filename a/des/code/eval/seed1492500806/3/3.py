from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)

CLOCKWISE_DIRS = (
    (-1, 0),   # up
    (-1, 1),   # up-right
    (0, 1),    # right
    (1, 1),    # down-right
    (1, 0),    # down
    (1, -1),   # down-left
    (0, -1),   # left
    (-1, -1),  # up-left
)

CELL_SIZES = (7, 9, 11, 13, 15)
LAYOUTS = ((2, 2), (2, 3), (3, 2))

LEGEND_BOX = 15
LEGEND_ICON = 5
LEGEND_TILE_W = 17
LEGEND_TILE_H = LEGEND_BOX + 1 + LEGEND_ICON
LEGEND_GAP = 1
LEGEND_ROWS = 4
LEGEND_COLS = 2

MAX_GENERATION_ATTEMPTS = 300


def _legend_dimensions() -> tuple[int, int]:
    height = LEGEND_ROWS * LEGEND_TILE_H + (LEGEND_ROWS - 1) * LEGEND_GAP
    width = LEGEND_COLS * LEGEND_TILE_W + (LEGEND_COLS - 1) * LEGEND_GAP
    return height, width


def _infer_separator_color(grid: np.ndarray) -> tuple[int, list[int]]:
    best: tuple[tuple[int, int], int, list[int]] | None = None
    for color in map(int, np.unique(grid)):
        rows = np.flatnonzero(np.all(grid == color, axis=1)).tolist()
        if len(rows) < 3:
            continue
        diffs = np.diff(rows)
        regularity = Counter(map(int, diffs)).most_common(1)[0][1] if len(diffs) else 0
        score = (len(rows), regularity)
        if best is None or score > best[0]:
            best = (score, color, rows)
    assert best is not None, "could not infer separator color"
    return best[1], best[2]


def _cell_boxes(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int, int, int]], int]:
    sep_color, sep_rows = _infer_separator_color(grid)
    puzzle_bottom = int(sep_rows[-1])
    top = grid[: puzzle_bottom + 1, :]
    sep_cols = np.flatnonzero(np.all(top == sep_color, axis=0)).tolist()

    assert len(sep_rows) >= 3 and len(sep_cols) >= 3
    row_steps = {int(x) for x in np.diff(sep_rows)}
    col_steps = {int(x) for x in np.diff(sep_cols)}
    assert len(row_steps) == 1 and len(col_steps) == 1
    step = row_steps.pop()
    assert step == col_steps.pop() and step >= 2
    cell_size = step - 1
    assert cell_size % 2 == 1

    boxes: list[tuple[int, int, int, int]] = []
    for r0, r1 in zip(sep_rows, sep_rows[1:]):
        for c0, c1 in zip(sep_cols, sep_cols[1:]):
            boxes.append((int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)))

    interior_colors: list[int] = []
    for r0, c0, r1, c1 in boxes:
        interior_colors.extend(map(int, grid[r0 : r1 + 1, c0 : c1 + 1].ravel()))
    background = Counter(interior_colors).most_common(1)[0][0]
    assert background != sep_color
    return sep_color, int(background), boxes, puzzle_bottom


def _pointer_direction(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    center: tuple[int, int],
    color: int,
) -> int:
    r0, c0, r1, c1 = box
    cr, cc = center

    actual = {
        (int(r0 + dr), int(c0 + dc))
        for dr, dc in np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    }

    lengths: list[int] = []
    for drow, dcol in CLOCKWISE_DIRS:
        length = 0
        rr = cr + drow
        cc2 = cc + dcol
        while r0 <= rr <= r1 and c0 <= cc2 <= c1 and int(grid[rr, cc2]) == color:
            length += 1
            rr += drow
            cc2 += dcol
        lengths.append(length)

    hits = [idx for idx, length in enumerate(lengths) if length > 0]
    assert len(hits) == 1, "each clock must have exactly one pointer direction"
    direction = hits[0]

    expected = {(cr, cc)}
    drow, dcol = CLOCKWISE_DIRS[direction]
    for step in range(1, lengths[direction] + 1):
        expected.add((cr + drow * step, cc + dcol * step))
    assert actual == expected, "pointer pixels must form one straight segment from the center"
    return direction


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each colored pointer direction and fill that many clipped square layers in its cell.
    Grey separators and the bottom legend band are preserved.
    """
    g = np.asarray(grid, dtype=int)
    _sep_color, bg, boxes, puzzle_bottom = _cell_boxes(g)
    assert puzzle_bottom < g.shape[0] - 1, "expected a legend band below the puzzle grid"

    out = g.copy()
    for box in boxes:
        r0, c0, r1, c1 = box
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        color = int(g[cr, cc])
        assert color != bg

        direction = _pointer_direction(g, box, (cr, cc), color)
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

        rr0 = max(r0, cr - direction)
        rr1 = min(r1, cr + direction)
        cc0 = max(c0, cc - direction)
        cc1 = min(c1, cc + direction)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color

    return out


def _draw_pointer(grid: np.ndarray, center: tuple[int, int], direction: int, length: int, color: int) -> None:
    cr, cc = center
    grid[cr, cc] = color
    drow, dcol = CLOCKWISE_DIRS[direction]
    for step in range(1, length + 1):
        grid[cr + drow * step, cc + dcol * step] = color


def _draw_legend(grid: np.ndarray, start_row: int) -> None:
    grid[start_row:, 0] = MAROON

    for level in range(8):
        tile_row = level // LEGEND_COLS
        tile_col = level % LEGEND_COLS
        tile_r0 = start_row + tile_row * (LEGEND_TILE_H + LEGEND_GAP)
        tile_c0 = tile_col * (LEGEND_TILE_W + LEGEND_GAP)

        center_r = tile_r0 + LEGEND_BOX // 2
        center_c = tile_c0 + 1 + LEGEND_BOX // 2
        r0 = center_r - level
        r1 = center_r + level
        c0 = center_c - level
        c1 = center_c + level
        grid[r0 : r1 + 1, c0 : c1 + 1] = MAROON

        icon_center = (tile_r0 + LEGEND_BOX + 1 + LEGEND_ICON // 2, center_c)
        _draw_pointer(grid, icon_center, level, 2, MAROON)


def _generate_case() -> np.ndarray:
    cell_size = random.choice(CELL_SIZES)
    n_rows, n_cols = random.choice(LAYOUTS)
    n_cells = n_rows * n_cols

    puzzle_height = n_rows * cell_size + (n_rows + 1)
    puzzle_width = n_cols * cell_size + (n_cols + 1)
    legend_height, legend_width = _legend_dimensions()
    height = puzzle_height + legend_height
    width = max(puzzle_width, legend_width)

    grid = np.full((height, width), BLACK, dtype=int)

    for idx in range(n_rows + 1):
        grid[idx * (cell_size + 1), :] = GREY
    for idx in range(n_cols + 1):
        grid[:puzzle_height, idx * (cell_size + 1)] = GREY

    colors = random.sample(PALETTE, n_cells)
    directions = random.sample(range(8), n_cells)
    random.shuffle(colors)
    random.shuffle(directions)

    cell_index = 0
    for row in range(n_rows):
        for col in range(n_cols):
            r0 = row * (cell_size + 1) + 1
            c0 = col * (cell_size + 1) + 1
            r1 = r0 + cell_size - 1
            c1 = c0 + cell_size - 1
            center = ((r0 + r1) // 2, (c0 + c1) // 2)

            direction = directions[cell_index]
            color = colors[cell_index]
            length = random.randint(1, min(3, cell_size // 2))
            _draw_pointer(grid, center, direction, length, color)
            cell_index += 1

    _draw_legend(grid, puzzle_height)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")

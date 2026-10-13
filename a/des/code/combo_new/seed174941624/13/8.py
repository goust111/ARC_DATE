import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
DATA_COLORS = (1, 2, 3, 4, 6, 7, 8)
CELL_SIZE = 3
LINE_SIZE = 1
STRIDE = CELL_SIZE + LINE_SIZE
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = np.argwhere(mask)
    if points.size == 0:
        raise ValueError("empty mask")
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_board(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], list[int], list[int]]:
    best = None
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        mask = grid == color
        r0, c0, r1, c1 = _bbox(mask)
        rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == color)]
        cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == color)]
        if len(rows) < 2 or len(cols) < 2:
            continue
        row_steps = [b - a for a, b in zip(rows, rows[1:])]
        col_steps = [b - a for a, b in zip(cols, cols[1:])]
        if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
            continue
        if row_steps[0] != STRIDE or col_steps[0] != STRIDE:
            continue
        expected = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r in rows:
            expected[r - r0, :] = True
        for c in cols:
            expected[:, c - c0] = True
        if not np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], expected):
            continue
        score = expected.sum()
        if best is None or score > best[0]:
            best = (score, color, (r0, c0, r1, c1), rows, cols)
    if best is None:
        raise ValueError("failed to detect grey board")
    return best[1], best[2], best[3], best[4]


def _find_legend(grid: np.ndarray, bg: int, board_color: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color in (bg, board_color):
            continue
        mask = grid == color
        r0, c0, r1, c1 = _bbox(mask)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if np.array_equal(mask[r0 : r1 + 1, c0 : c1 + 1], expected):
            candidates.append((h * w, color, (r0, c0, r1, c1)))
    if not candidates:
        raise ValueError("failed to detect legend frame")
    candidates.sort()
    _area, color, bbox = candidates[0]
    return color, bbox


def _cell_ranges(lines: list[int]) -> list[tuple[int, int]]:
    return [(lines[i] + 1, lines[i + 1]) for i in range(len(lines) - 1)]


def _uniform_cell_color(cell: np.ndarray, ignore: Iterable[int]) -> int | None:
    ignored = set(int(v) for v in ignore)
    values = [int(v) for v in np.unique(cell) if int(v) not in ignored]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("cell contains multiple data colors")
    color = values[0]
    if not np.all((cell == color) | np.isin(cell, list(ignored))):
        raise ValueError("cell is not uniform")
    return color


def _legend_priority(grid: np.ndarray, legend_bbox: tuple[int, int, int, int], bg: int, frame_color: int) -> dict[int, int]:
    r0, c0, r1, c1 = legend_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    priority: list[int] = []
    for rr in range(interior.shape[0]):
        row = [int(v) for v in interior[rr] if int(v) not in (bg, frame_color)]
        if len(row) != 1:
            raise ValueError("legend rows must contain exactly one priority color")
        priority.append(row[0])
    if len(priority) != len(set(priority)):
        raise ValueError("legend colors must be unique")
    return {color: idx for idx, color in enumerate(priority)}


def _winner(a: int, b: int, priority: dict[int, int]) -> int:
    if a == b:
        return a
    return a if priority[a] < priority[b] else b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each non-template cell of a gridded board using its row color, column color, and a priority legend.

    Input structure:
    - A grey line lattice partitions the board into equal 3x3 cells.
    - The leftmost cell column stores solid row-template colors, and the top cell row stores solid column-template colors.
    - The top-left template-intersection cell is background, while the remaining interior cells start empty.
    - A maroon framed legend near the lower-right lists all relevant colors as a vertical single-pixel column, ordered from highest to lowest priority.

    Rule:
    - Keep the template row, the template column, the grey lattice, and the maroon legend unchanged.
    - For every non-template cell, read its row-template color A and column-template color B.
    - If A and B match, fill the whole 3x3 cell with that color.
    - Otherwise, compare A and B using the legend ordering and fill the cell with whichever color appears higher in the legend.

    Key insight:
    - The board acts like a color cross-table, but conflicting row/column colors are not resolved locally; the corner legend provides the global priority order that decides every mismatch.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    board_color, _board_bbox, row_lines, col_lines = _find_board(g, bg)
    legend_color, legend_bbox = _find_legend(g, bg, board_color)
    priority = _legend_priority(g, legend_bbox, bg, legend_color)

    row_ranges = _cell_ranges(row_lines)
    col_ranges = _cell_ranges(col_lines)
    row_templates: list[int] = []
    col_templates: list[int] = []

    for row_idx in range(1, len(row_ranges)):
        rs, re = row_ranges[row_idx]
        cs, ce = col_ranges[0]
        color = _uniform_cell_color(g[rs:re, cs:ce], ignore=(bg, board_color, legend_color))
        if color is None:
            raise ValueError("missing row template color")
        row_templates.append(color)

    for col_idx in range(1, len(col_ranges)):
        rs, re = row_ranges[0]
        cs, ce = col_ranges[col_idx]
        color = _uniform_cell_color(g[rs:re, cs:ce], ignore=(bg, board_color, legend_color))
        if color is None:
            raise ValueError("missing column template color")
        col_templates.append(color)

    out = g.copy()
    for row_idx, a in enumerate(row_templates, start=1):
        for col_idx, b in enumerate(col_templates, start=1):
            rs, re = row_ranges[row_idx]
            cs, ce = col_ranges[col_idx]
            out[rs:re, cs:ce] = _winner(a, b, priority)
    return out


def _has_branch_coverage(row_colors: list[int], col_colors: list[int], priority: dict[int, int]) -> bool:
    seen_equal = False
    seen_row_win = False
    seen_col_win = False
    for a in row_colors:
        for b in col_colors:
            if a == b:
                seen_equal = True
            elif priority[a] < priority[b]:
                seen_row_win = True
            else:
                seen_col_win = True
    return seen_equal and seen_row_win and seen_col_win


def _choose_templates(row_count: int, col_count: int, colors: list[int], rng: random.Random) -> tuple[list[int], list[int]]:
    priority = {color: idx for idx, color in enumerate(colors)}
    required = set(colors)
    for _ in range(200):
        row_colors = [rng.choice(colors) for _ in range(row_count)]
        col_colors = [rng.choice(colors) for _ in range(col_count)]
        if set(row_colors) | set(col_colors) != required:
            continue
        if _has_branch_coverage(row_colors, col_colors, priority):
            return row_colors, col_colors
    raise RuntimeError("failed to sample non-degenerate templates")


def _paint_cell(grid: np.ndarray, top: int, left: int, cell_r: int, cell_c: int, color: int) -> None:
    rs = top + cell_r * STRIDE + LINE_SIZE
    cs = left + cell_c * STRIDE + LINE_SIZE
    grid[rs : rs + CELL_SIZE, cs : cs + CELL_SIZE] = color


def _draw_board(grid: np.ndarray, top: int, left: int, rows: int, cols: int) -> None:
    board_h = rows * STRIDE + LINE_SIZE
    board_w = cols * STRIDE + LINE_SIZE
    for offset in range(0, board_h, STRIDE):
        grid[top + offset, left : left + board_w] = GREY
    for offset in range(0, board_w, STRIDE):
        grid[top : top + board_h, left + offset] = GREY


def _draw_legend(grid: np.ndarray, top: int, left: int, priority_colors: list[int]) -> None:
    height = len(priority_colors) + 2
    width = 3
    grid[top, left : left + width] = MAROON
    grid[top + height - 1, left : left + width] = MAROON
    grid[top : top + height, left] = MAROON
    grid[top : top + height, left + width - 1] = MAROON
    for idx, color in enumerate(priority_colors, start=1):
        grid[top + idx, left + 1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(3, 6)
    cols = rng.randint(3, 6)
    row_count = rows - 1
    col_count = cols - 1
    max_colors = min(len(DATA_COLORS), row_count + col_count - 1, 5)
    color_count = rng.randint(2, max_colors)
    priority_colors = rng.sample(list(DATA_COLORS), color_count)
    row_colors, col_colors = _choose_templates(row_count, col_count, priority_colors, rng)

    board_h = rows * STRIDE + LINE_SIZE
    board_w = cols * STRIDE + LINE_SIZE
    legend_h = color_count + 2
    legend_w = 3

    top_pad = rng.randint(0, 1)
    left_pad = rng.randint(0, 1)
    gap = rng.randint(1, 2)
    right_pad = rng.randint(0, 1)
    bottom_pad = rng.randint(0, 1)

    board_top = top_pad
    board_left = left_pad
    legend_left = board_left + board_w + gap
    max_lift = min(2, board_h - legend_h)
    legend_top = board_top + board_h - legend_h - rng.randint(0, max_lift)

    height = max(board_top + board_h, legend_top + legend_h) + bottom_pad
    width = legend_left + legend_w + right_pad
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    _draw_board(grid, board_top, board_left, rows, cols)
    _draw_legend(grid, legend_top, legend_left, priority_colors)

    for idx, color in enumerate(row_colors, start=1):
        _paint_cell(grid, board_top, board_left, idx, 0, color)
    for idx, color in enumerate(col_colors, start=1):
        _paint_cell(grid, board_top, board_left, 0, idx, color)

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        try:
            _ = main(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("generate_input failed after repeated attempts")

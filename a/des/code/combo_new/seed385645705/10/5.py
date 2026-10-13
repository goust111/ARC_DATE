import random
from collections import Counter

import numpy as np


BLACK = 0
GENERATION_COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
ORANGE = 7

GRID_CELL_ROWS_RANGE = (2, 4)
GRID_CELL_COLS_RANGE = (2, 4)
CELL_SIZE_RANGE = (3, 6)
OUTER_MARGIN_RANGE = (0, 2)

MAJORITY_PATTERNS = (
    (1,),
    (2,),
    (2, 1),
    (3, 1),
    (2, 1, 1),
    (3, 2),
    (4, 1),
    (3, 1, 1),
)
TIE_TEMPLATES = (
    (0, 1),
    (0, 1, 0, 1),
    (0, 1, 2, 0, 1),
)

MAX_GENERATION_ATTEMPTS = 300


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(int(x) for x in coords[0])
    stack = [start]
    seen = {start}

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _find_grid_structure(
    grid: np.ndarray,
) -> tuple[int, tuple[int, int, int, int], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    candidates = []

    for color in np.unique(g):
        color = int(color)
        if color == BLACK:
            continue

        mask = g == color
        if not _is_connected(mask):
            continue

        r0, c0, r1, c1 = _bbox(mask)
        full_rows = [r for r in range(r0, r1 + 1) if np.all(mask[r, c0 : c1 + 1])]
        full_cols = [c for c in range(c0, c1 + 1) if np.all(mask[r0 : r1 + 1, c])]
        cell_rows = sum(1 for a, b in zip(full_rows, full_rows[1:]) if b - a > 1)
        cell_cols = sum(1 for a, b in zip(full_cols, full_cols[1:]) if b - a > 1)
        if cell_rows == 0 or cell_cols == 0:
            continue

        candidates.append(
            (
                cell_rows * cell_cols,
                len(full_rows) + len(full_cols),
                int(mask.sum()),
                color,
                (r0, c0, r1, c1),
                full_rows,
                full_cols,
            )
        )

    assert candidates, "failed to find the rectilinear partition grid"
    _, _, _, color, bbox, full_rows, full_cols = max(candidates)
    return color, bbox, full_rows, full_cols


def _vote_color(cell: np.ndarray, grid_color: int) -> int | None:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    scan_index = 0

    for r in range(cell.shape[0]):
        for c in range(cell.shape[1]):
            value = int(cell[r, c])
            if value not in (BLACK, grid_color):
                counts[value] += 1
                first_seen.setdefault(value, scan_index)
            scan_index += 1

    if not counts:
        return None

    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    winners.sort(key=lambda color: first_seen[color])
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each orange-bounded cell by the majority interior color.

    Input structure:
    - Background is black.
    - A connected orange 1-pixel rectilinear grid forms an outer frame plus full
      internal horizontal and vertical dividers.
    - The dividers partition the frame interior into separate rectangular cells.
    - Each cell contains a few scattered colored pixels or may be entirely black.

    Rule:
    - Detect the orange grid and iterate over every cell interior between
      consecutive horizontal and vertical divider lines.
    - Count the non-black colors inside each cell.
    - If several colors tie for the maximum count, choose the one whose first
      occurrence appears earliest in row-major scan order within that cell.
    - Replace only the black pixels of that cell with the chosen color; keep
      existing colored pixels and all orange grid pixels unchanged. Empty cells
      stay black.

    Key insight:
    - The orange network defines independent voting regions, and each region's
      fill color is chosen locally by majority with row-major tie-breaking.
    """
    g = np.asarray(grid, dtype=int)
    grid_color, _bbox_unused, h_lines, v_lines = _find_grid_structure(g)
    out = g.copy()

    for top, bottom in zip(h_lines, h_lines[1:]):
        if bottom - top <= 1:
            continue
        for left, right in zip(v_lines, v_lines[1:]):
            if right - left <= 1:
                continue

            cell = g[top + 1 : bottom, left + 1 : right]
            fill_color = _vote_color(cell, grid_color)
            if fill_color is None:
                continue

            out_cell = out[top + 1 : bottom, left + 1 : right]
            out_cell[out_cell == BLACK] = fill_color

    return out


def _sample_sizes(count: int, max_total: int) -> list[int] | None:
    low, high = CELL_SIZE_RANGE
    min_total = count * low
    max_feasible = min(count * high, max_total)
    if min_total > max_feasible:
        return None

    target = random.randint(min_total, max_feasible)
    sizes = [low] * count
    remaining = target - min_total
    while remaining > 0:
        idx = random.randrange(count)
        if sizes[idx] < high:
            sizes[idx] += 1
            remaining -= 1

    random.shuffle(sizes)
    return sizes


def _line_positions(start: int, sizes: list[int]) -> list[int]:
    positions = [start]
    cursor = start
    for size in sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _apply_votes(
    grid: np.ndarray,
    expected: np.ndarray,
    coords: list[tuple[int, int]],
    placed: list[tuple[tuple[int, int], int]],
    fill_color: int | None,
) -> None:
    for (r, c), color in placed:
        grid[r, c] = color
        expected[r, c] = color

    if fill_color is None:
        return

    for r, c in coords:
        if expected[r, c] == BLACK:
            expected[r, c] = fill_color


def _populate_majority_cell(
    grid: np.ndarray,
    expected: np.ndarray,
    coords: list[tuple[int, int]],
) -> None:
    pattern = random.choice(MAJORITY_PATTERNS)
    colors = random.sample(GENERATION_COLOR_POOL, len(pattern))
    selected = random.sample(coords, sum(pattern))
    random.shuffle(selected)

    placed: list[tuple[tuple[int, int], int]] = []
    cursor = 0
    for color, count in zip(colors, pattern):
        for _ in range(count):
            placed.append((selected[cursor], color))
            cursor += 1

    _apply_votes(grid, expected, coords, placed, colors[0])


def _populate_tie_cell(
    grid: np.ndarray,
    expected: np.ndarray,
    coords: list[tuple[int, int]],
) -> None:
    template = random.choice(TIE_TEMPLATES)
    color_count = max(template) + 1
    colors = random.sample(GENERATION_COLOR_POOL, color_count)
    selected = sorted(random.sample(coords, len(template)))
    placed = [(coord, colors[idx]) for coord, idx in zip(selected, template)]
    _apply_votes(grid, expected, coords, placed, colors[0])


def _populate_cell(
    grid: np.ndarray,
    expected: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    mode: str,
) -> None:
    coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if mode == "empty":
        return
    if mode == "tie":
        _populate_tie_cell(grid, expected, coords)
        return
    _populate_majority_cell(grid, expected, coords)


def _gen_attempt() -> np.ndarray | None:
    cell_rows = random.randint(*GRID_CELL_ROWS_RANGE)
    cell_cols = random.randint(*GRID_CELL_COLS_RANGE)
    top = random.randint(*OUTER_MARGIN_RANGE)
    bottom = random.randint(*OUTER_MARGIN_RANGE)
    left = random.randint(*OUTER_MARGIN_RANGE)
    right = random.randint(*OUTER_MARGIN_RANGE)

    max_height_total = 30 - top - bottom - (cell_rows + 1)
    max_width_total = 30 - left - right - (cell_cols + 1)
    heights = _sample_sizes(cell_rows, max_height_total)
    widths = _sample_sizes(cell_cols, max_width_total)
    if heights is None or widths is None:
        return None

    h_lines = _line_positions(top, heights)
    v_lines = _line_positions(left, widths)
    height = bottom + h_lines[-1] + 1
    width = right + v_lines[-1] + 1
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)

    for r in h_lines:
        grid[r, v_lines[0] : v_lines[-1] + 1] = ORANGE
        expected[r, v_lines[0] : v_lines[-1] + 1] = ORANGE
    for c in v_lines:
        grid[h_lines[0] : h_lines[-1] + 1, c] = ORANGE
        expected[h_lines[0] : h_lines[-1] + 1, c] = ORANGE

    cells = []
    for top_line, bottom_line in zip(h_lines, h_lines[1:]):
        for left_line, right_line in zip(v_lines, v_lines[1:]):
            cells.append((top_line + 1, bottom_line - 1, left_line + 1, right_line - 1))

    modes = ["majority"] * len(cells)
    special = random.sample(range(len(cells)), 2)
    modes[special[0]] = "tie"
    modes[special[1]] = "empty"

    remaining = [idx for idx, mode in enumerate(modes) if mode == "majority"]
    if len(cells) >= 6 and remaining and random.random() < 0.4:
        modes[random.choice(remaining)] = "tie"
        remaining = [idx for idx, mode in enumerate(modes) if mode == "majority"]
    if len(cells) >= 8 and remaining and random.random() < 0.25:
        modes[random.choice(remaining)] = "empty"

    for (r0, r1, c0, c1), mode in zip(cells, modes):
        _populate_cell(grid, expected, r0, r1, c0, c1, mode)

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")

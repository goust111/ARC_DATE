import random
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

CLOCKWISE_DIRS = ("up", "right", "down", "left")
DIR_INDEX = {name: idx for idx, name in enumerate(CLOCKWISE_DIRS)}
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8)

TABLE_ROW_CHOICES = (2, 3, 4)
TABLE_COL_CHOICES = (2, 3, 4)
CELL_SIZE_CHOICES = (5, 6, 7)
MAX_GENERATION_ATTEMPTS = 400

BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]

PAIR_DIRS = tuple(combinations(CLOCKWISE_DIRS, 2))


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(separators):
        if start <= sep - 1:
            spans.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _legend_mapping(cell: np.ndarray) -> dict[str, int] | None:
    if int(np.count_nonzero(cell != BLACK)) != 5:
        return None

    grey_coords = np.argwhere(cell == GREY)
    if len(grey_coords) != 1:
        return None

    r, c = map(int, grey_coords[0])
    h, w = cell.shape
    if not (0 < r < h - 1 and 0 < c < w - 1):
        return None

    diagonals = [cell[r - 1, c - 1], cell[r - 1, c + 1], cell[r + 1, c - 1], cell[r + 1, c + 1]]
    if any(int(value) != BLACK for value in diagonals):
        return None

    mapping = {
        "up": int(cell[r - 1, c]),
        "down": int(cell[r + 1, c]),
        "left": int(cell[r, c - 1]),
        "right": int(cell[r, c + 1]),
    }
    colors = tuple(mapping.values())
    if any(color in {BLACK, GREY, MAROON} for color in colors):
        return None
    if len(set(colors)) != 4:
        return None
    return mapping


def _direction_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    if mr < r0:
        return "up"
    if mr > r1:
        return "down"
    if mc < c0:
        return "left"
    if mc > c1:
        return "right"

    center_r = 0.5 * (r0 + r1)
    center_c = 0.5 * (c0 + c1)
    dr = mr - center_r
    dc = mc - center_c
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _resolve_color(directions: list[str], mapping: dict[str, int]) -> int:
    unique_dirs: list[str] = []
    for direction in directions:
        if direction not in unique_dirs:
            unique_dirs.append(direction)
    assert 1 <= len(unique_dirs) <= 2, "expected one or two distinct marker directions"
    if len(unique_dirs) == 1:
        return mapping[unique_dirs[0]]
    lead = min(unique_dirs, key=lambda name: DIR_INDEX[name])
    return mapping[lead]


def _solve_cell(cell: np.ndarray, mapping: dict[str, int]) -> np.ndarray:
    figure_mask = (cell != BLACK) & (cell != GREY) & (cell != MAROON)
    figure_coords = np.argwhere(figure_mask)
    assert len(figure_coords) > 0, "every non-legend cell must contain a figure"

    marker_coords = [tuple(map(int, coord)) for coord in np.argwhere(cell == MAROON)]
    assert 1 <= len(marker_coords) <= 2, "every non-legend cell must contain one or two maroon markers"

    rows = figure_coords[:, 0]
    cols = figure_coords[:, 1]
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
    directions = [_direction_from_marker(marker, bbox) for marker in marker_coords]
    final_color = _resolve_color(directions, mapping)

    out = cell.copy()
    out[cell == MAROON] = BLACK
    out[figure_mask] = final_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each partitioned sprite by reading a global direction-to-color legend and interpreting outside maroon markers.

    Input structure:
    - The background is black.
    - Full grey separator rows and columns partition the canvas into rectangular cells.
    - One special cell contains a 3x3 legend: a grey center, four colored orthogonal neighbors, and black diagonals.
    - Every other cell contains one monochrome figure plus one or two maroon marker pixels placed outside the figure bbox on a cardinal side.

    Rule:
    - Read the legend cell as a mapping from direction to color: up, down, left, and right.
    - For each non-legend cell, determine the direction of each maroon marker relative to the figure bbox.
    - A single marker directly chooses its legend color; two different markers choose the color whose direction comes earlier in clockwise order up→right→down→left.
    - Repaint the whole figure with that chosen color and erase the maroon markers, while leaving the legend and separators unchanged.

    Key insight:
    - The maroon pixels are not part of the figure; they encode directions that must be interpreted through the separate legend cell before the figure can be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    assert row_spans and col_spans, "expected at least one cell"

    cells: list[tuple[int, int, int, int, np.ndarray, dict[str, int] | None]] = []
    legend_count = 0
    legend_mapping: dict[str, int] | None = None
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            mapping = _legend_mapping(cell)
            if mapping is not None:
                legend_count += 1
                legend_mapping = mapping
            cells.append((r0, r1, c0, c1, cell, mapping))

    assert legend_count == 1 and legend_mapping is not None, "expected exactly one legend cell"

    out = g.copy()
    for r0, r1, c0, c1, cell, mapping in cells:
        if mapping is not None:
            continue
        out[r0 : r1 + 1, c0 : c1 + 1] = _solve_cell(cell, legend_mapping)
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _random_shape(rng: random.Random) -> np.ndarray:
    mask = BASE_SHAPES[rng.randrange(len(BASE_SHAPES))].copy()
    mask = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return _trim_mask(mask)


def _build_table_grid(heights: list[int], widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    total_h = sum(heights) + len(heights) - 1
    total_w = sum(widths) + len(widths) - 1
    grid = np.full((total_h, total_w), BLACK, dtype=int)

    row_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, height in enumerate(heights):
        row_spans.append((cursor, cursor + height - 1))
        cursor += height
        if idx != len(heights) - 1:
            grid[cursor, :] = GREY
            cursor += 1

    col_spans: list[tuple[int, int]] = []
    cursor = 0
    for idx, width in enumerate(widths):
        col_spans.append((cursor, cursor + width - 1))
        cursor += width
        if idx != len(widths) - 1:
            grid[:, cursor] = GREY
            cursor += 1

    return grid, row_spans, col_spans


def _place_legend(cell: np.ndarray, colors: tuple[int, int, int, int], rng: random.Random) -> None:
    h, w = cell.shape
    center_r = rng.randrange(1, h - 1)
    center_c = rng.randrange(1, w - 1)
    cell[center_r, center_c] = GREY
    cell[center_r - 1, center_c] = colors[0]
    cell[center_r, center_c + 1] = colors[1]
    cell[center_r + 1, center_c] = colors[2]
    cell[center_r, center_c - 1] = colors[3]


def _marker_counts(num_cells: int, rng: random.Random) -> list[int]:
    counts = [1, 2]
    while len(counts) < num_cells:
        counts.append(1 if rng.random() < 0.5 else 2)
    rng.shuffle(counts)
    return counts


def _place_figure_and_markers(
    cell: np.ndarray,
    figure_color: int,
    marker_dirs: tuple[str, ...],
    rng: random.Random,
) -> None:
    h, w = cell.shape
    mask = _random_shape(rng)
    sh, sw = mask.shape
    assert h >= sh + 2 and w >= sw + 2, "cell too small for figure placement"

    top = rng.randrange(1, h - sh)
    left = rng.randrange(1, w - sw)
    cell[top : top + sh, left : left + sw][mask] = figure_color

    r0, c0 = top, left
    r1, c1 = top + sh - 1, left + sw - 1
    for direction in marker_dirs:
        if direction == "up":
            row = r0 - 1
            col = rng.randrange(c0, c1 + 1)
        elif direction == "down":
            row = r1 + 1
            col = rng.randrange(c0, c1 + 1)
        elif direction == "left":
            row = rng.randrange(r0, r1 + 1)
            col = c0 - 1
        else:
            row = rng.randrange(r0, r1 + 1)
            col = c1 + 1
        cell[row, col] = MAROON


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.choice(TABLE_ROW_CHOICES)
    n_cols = rng.choice(TABLE_COL_CHOICES)
    heights = [rng.choice(CELL_SIZE_CHOICES) for _ in range(n_rows)]
    widths = [rng.choice(CELL_SIZE_CHOICES) for _ in range(n_cols)]
    total_h = sum(heights) + n_rows - 1
    total_w = sum(widths) + n_cols - 1
    if total_h > 30 or total_w > 30:
        return None

    grid, row_spans, col_spans = _build_table_grid(heights, widths)
    cells = [(ri, ci) for ri in range(n_rows) for ci in range(n_cols)]
    legend_cell = rng.choice(cells)
    pattern_cells = [cell for cell in cells if cell != legend_cell]
    if len(pattern_cells) < 2:
        return None

    legend_colors = tuple(rng.sample(list(NON_SPECIAL_COLORS), 4))
    figure_palette = [color for color in NON_SPECIAL_COLORS if color not in legend_colors]
    if not figure_palette:
        return None

    marker_counts = _marker_counts(len(pattern_cells), rng)

    for ri, ci in cells:
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        cell = grid[r0 : r1 + 1, c0 : c1 + 1]
        if (ri, ci) == legend_cell:
            _place_legend(cell, legend_colors, rng)
            continue

        count = marker_counts.pop()
        if count == 1:
            marker_dirs = (rng.choice(CLOCKWISE_DIRS),)
        else:
            marker_dirs = rng.choice(PAIR_DIRS)
        figure_color = rng.choice(figure_palette)
        _place_figure_and_markers(cell, figure_color, marker_dirs, rng)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

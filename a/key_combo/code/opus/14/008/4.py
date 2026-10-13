import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR = 5
PATTERN_COLORS = (1, 2, 3, 4, 7)
GEN_ATTEMPTS = 600


def _separator_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEPARATOR)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEPARATOR)]


def _intervals(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *map(int, separators), int(size)]
    spans = []
    for left, right in zip(bounds, bounds[1:]):
        start = left + 1
        end = right
        if start < end:
            spans.append((start, end))
    return spans


def _extract_pattern(cell: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int]:
    coords = np.argwhere((cell != BG) & (cell != SEPARATOR))
    assert coords.size > 0, "every cell must contain a pattern"
    colors = {int(cell[r, c]) for r, c in coords}
    assert len(colors) == 1, "every cell must contain exactly one monochrome pattern"
    key = tuple(sorted((int(r), int(c)) for r, c in coords))
    return key, next(iter(colors))


def _analyze_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[dict]]]:
    row_separators = _separator_indices(grid, axis=0)
    col_separators = _separator_indices(grid, axis=1)
    row_spans = _intervals(grid.shape[0], row_separators)
    col_spans = _intervals(grid.shape[1], col_separators)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected a separator-defined cell grid"

    cells: list[list[dict]] = []
    for row_index, (r0, r1) in enumerate(row_spans):
        row_cells = []
        for col_index, (c0, c1) in enumerate(col_spans):
            pattern, color = _extract_pattern(grid[r0:r1, c0:c1])
            row_cells.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "pattern": pattern,
                    "color": color,
                }
            )
        cells.append(row_cells)
    return row_spans, col_spans, cells


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, cells = _analyze_grid(g)

    counts = Counter(cell["pattern"] for row in cells for cell in row)
    unique_patterns = [pattern for pattern, count in counts.items() if count == 1]
    assert len(unique_patterns) == 1, "expected exactly one unique pattern mask"
    unique_pattern = unique_patterns[0]

    out = g.copy()

    for row in cells:
        for cell in row:
            if cell["pattern"] != unique_pattern:
                continue
            view = out[cell["r0"] : cell["r1"], cell["c0"] : cell["c1"]]
            view[(view != BG) & (view != SEPARATOR)] = BG

    for row_index, row in enumerate(cells):
        for col_index in range(len(row) - 1):
            left = row[col_index]
            right = row[col_index + 1]
            if left["pattern"] != right["pattern"]:
                continue
            assert left["color"] == right["color"], "same pattern should use one color"
            sep_col = col_spans[col_index][1]
            out[row_spans[row_index][0] : row_spans[row_index][1], sep_col] = left["color"]

    for row_index in range(len(cells) - 1):
        for col_index in range(len(cells[0])):
            top = cells[row_index][col_index]
            bottom = cells[row_index + 1][col_index]
            if top["pattern"] != bottom["pattern"]:
                continue
            assert top["color"] == bottom["color"], "same pattern should use one color"
            sep_row = row_spans[row_index][1]
            out[sep_row, col_spans[col_index][0] : col_spans[col_index][1]] = top["color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the single outlier cell pattern and recolor separator segments between adjacent matching cells.

    Input structure:
    - Black background cells are partitioned by complete one-pixel grey separator rows and columns into a regular cell grid.
    - Each cell contains exactly one small monochrome 4-connected pattern drawn in one non-grey, non-black color.
    - Most cell patterns repeat exactly by translation within their own cell coordinates, while one cell has a unique mask.

    Rule:
    - Use the full grey separator rows and columns to recover the cell boundaries.
    - Compare the non-black pattern pixels inside each cell by their coordinates relative to that cell's top-left corner.
    - Erase the only cell whose relative pattern mask appears once, turning its pattern pixels black.
    - For every horizontally or vertically adjacent pair of cells with the same relative mask, recolor only their shared separator segment to that pattern's color, keeping separator intersections grey.

    Key insight:
    - The grey separators serve two roles at once: they define which local patterns should be compared, and they are also the channels that become colored only between adjacent equal-pattern cells.
    """
    return _transform(grid)


def _neighbors4(coord: tuple[int, int], height: int, width: int) -> list[tuple[int, int]]:
    r, c = coord
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _random_connected_pattern(height: int, width: int) -> tuple[tuple[int, int], ...]:
    target_area = random.randint(2, min(5, height * width))
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set(_neighbors4(start, height, width))

    while len(cells) < target_area:
        if not frontier:
            frontier = {
                nbr
                for cell in cells
                for nbr in _neighbors4(cell, height, width)
                if nbr not in cells
            }
        new_cell = random.choice(tuple(frontier))
        frontier.discard(new_cell)
        cells.add(new_cell)
        for nbr in _neighbors4(new_cell, height, width):
            if nbr not in cells:
                frontier.add(nbr)

    return tuple(sorted(cells))


def _pattern_catalog(height: int, width: int, count: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    patterns = []
    while len(patterns) < count:
        pattern = _random_connected_pattern(height, width)
        if pattern in seen:
            continue
        seen.add(pattern)
        patterns.append(pattern)
    return patterns


def _count_options(total_cells: int) -> list[tuple[int, ...]]:
    options: set[tuple[int, ...]] = set()

    def rec(remaining: int, parts_left: int, prefix: tuple[int, ...]) -> None:
        if parts_left == 1:
            if remaining >= 2:
                options.add(tuple(sorted(prefix + (remaining,), reverse=True)))
            return
        for part in range(2, remaining - 2 * (parts_left - 1) + 1):
            rec(remaining - part, parts_left - 1, prefix + (part,))

    for repeated_types in (2, 3):
        remaining = total_cells - 1
        if remaining < 2 * repeated_types:
            continue
        rec(remaining, repeated_types, ())

    return sorted(options)


def _layout_ok(layout: np.ndarray, unique_id: int) -> bool:
    rows, cols = layout.shape
    horizontal = 0
    vertical = 0
    for r in range(rows):
        for c in range(cols - 1):
            if layout[r, c] == layout[r, c + 1] != unique_id:
                horizontal += 1
    for r in range(rows - 1):
        for c in range(cols):
            if layout[r, c] == layout[r + 1, c] != unique_id:
                vertical += 1

    if horizontal == 0 or vertical == 0:
        return False
    if horizontal + vertical < 2 or horizontal + vertical > 6:
        return False

    unique_pos = tuple(map(int, np.argwhere(layout == unique_id)[0]))
    neighbors = _neighbors4(unique_pos, rows, cols)
    if not neighbors:
        return False
    if not any(layout[r, c] != unique_id for r, c in neighbors):
        return False

    return True


def _sample_layout(rows: int, cols: int, repeated_counts: tuple[int, ...]) -> np.ndarray | None:
    unique_id = len(repeated_counts)
    bag = []
    for pattern_id, count in enumerate(repeated_counts):
        bag.extend([pattern_id] * int(count))
    bag.append(unique_id)
    bag = np.array(bag, dtype=int)

    for _ in range(300):
        random.shuffle(bag)
        layout = bag.reshape(rows, cols)
        if _layout_ok(layout, unique_id):
            return layout.copy()
    return None


def _render_input(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    layout: np.ndarray,
    patterns: list[tuple[tuple[int, int], ...]],
    colors: list[int],
) -> np.ndarray:
    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    grid = np.full((height, width), BG, dtype=int)

    row_spans = []
    cursor = 0
    for r in range(rows):
        row_spans.append((cursor, cursor + cell_h))
        cursor += cell_h
        if r < rows - 1:
            grid[cursor, :] = SEPARATOR
            cursor += 1

    col_spans = []
    cursor = 0
    for c in range(cols):
        col_spans.append((cursor, cursor + cell_w))
        cursor += cell_w
        if c < cols - 1:
            grid[:, cursor] = SEPARATOR
            cursor += 1

    for r in range(rows):
        for c in range(cols):
            pattern_id = int(layout[r, c])
            color = int(colors[pattern_id])
            for pr, pc in patterns[pattern_id]:
                grid[row_spans[r][0] + pr, col_spans[c][0] + pc] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    rows = 3
    cols = random.choice((3, 4))
    cell_h = random.choice((3, 4))
    cell_w = random.choice((3, 4))
    total_cells = rows * cols

    repeated_counts = random.choice(_count_options(total_cells))
    repeated_type_count = len(repeated_counts)
    pattern_count = repeated_type_count + 1
    colors = random.sample(PATTERN_COLORS, pattern_count)
    patterns = _pattern_catalog(cell_h, cell_w, pattern_count)

    layout = _sample_layout(rows, cols, repeated_counts)
    if layout is None:
        return None

    grid = _render_input(rows, cols, cell_h, cell_w, layout, patterns, colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None

    unique_id = repeated_type_count
    unique_pos = tuple(map(int, np.argwhere(layout == unique_id)[0]))
    row_spans = _intervals(grid.shape[0], _separator_indices(grid, axis=0))
    col_spans = _intervals(grid.shape[1], _separator_indices(grid, axis=1))
    ur0, ur1 = row_spans[unique_pos[0]]
    uc0, uc1 = col_spans[unique_pos[1]]
    if np.any(out[ur0:ur1, uc0:uc1] != BG):
        return None

    changed_separator = (grid == SEPARATOR) & (out != SEPARATOR)
    if changed_separator.sum() == 0:
        return None

    horizontal_changes = 0
    vertical_changes = 0
    for r in range(rows):
        for c in range(cols - 1):
            if layout[r, c] == layout[r, c + 1] != unique_id:
                sep_col = col_spans[c][1]
                if np.all(out[row_spans[r][0] : row_spans[r][1], sep_col] == colors[int(layout[r, c])]):
                    horizontal_changes += 1
    for r in range(rows - 1):
        for c in range(cols):
            if layout[r, c] == layout[r + 1, c] != unique_id:
                sep_row = row_spans[r][1]
                if np.all(out[sep_row, col_spans[c][0] : col_spans[c][1]] == colors[int(layout[r, c])]):
                    vertical_changes += 1

    if horizontal_changes == 0 or vertical_changes == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

import random

import numpy as np


PATH_COLOR = 5
BACKGROUND = 0
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _most_common_color(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _palette_from_bottom_row(grid, bg):
    bottom = grid[-1]
    indices = np.flatnonzero(bottom != bg)
    if len(indices) == 0:
        return []

    runs = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))

    start, end = max(runs, key=lambda pair: pair[1] - pair[0])
    return [int(x) for x in bottom[start : end + 1]]


def _neighbors4(cell):
    r, c = cell
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _collect_component(grid, start, seen):
    stack = [start]
    seen.add(start)
    cells = []
    while stack:
        cell = stack.pop()
        cells.append(cell)
        for nb in _neighbors4(cell):
            r, c = nb
            if (
                0 <= r < grid.shape[0] - 1
                and 0 <= c < grid.shape[1]
                and nb not in seen
                and grid[r, c] == PATH_COLOR
            ):
                seen.add(nb)
                stack.append(nb)
    return cells


def _count_turns(cells):
    if len(cells) < 3:
        return 0

    cell_set = set(cells)
    adjacency = {
        cell: [nb for nb in _neighbors4(cell) if nb in cell_set] for cell in cell_set
    }
    endpoints = [cell for cell, nbs in adjacency.items() if len(nbs) == 1]
    if not endpoints:
        return 0

    ordered = []
    seen = set()
    prev = None
    cur = endpoints[0]
    while cur is not None and cur not in seen:
        ordered.append(cur)
        seen.add(cur)
        next_cells = [nb for nb in adjacency[cur] if nb != prev]
        if not next_cells:
            break
        prev, cur = cur, next_cells[0]

    directions = []
    for (r1, c1), (r2, c2) in zip(ordered, ordered[1:]):
        directions.append((r2 - r1, c2 - c1))
    return sum(a != b for a, b in zip(directions, directions[1:]))


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each grey polyline according to its number of turns.

    Input structure: The background is the most common color. The bottom row
    contains a contiguous palette strip of distinct non-background colors.
    Above it are several disconnected one-pixel-wide grey paths made only of
    horizontal and vertical segments, each with two endpoints.
    Rule: For every grey path, trace the path from one endpoint to the other
    and count how many times the step direction changes. Use that turn count as
    a zero-based index into the bottom palette strip, then recolor the whole
    path with the selected palette color. The palette row is copied unchanged.
    Key insight: The bottom strip is a lookup table, and the lookup key is the
    number of corners in each grey path rather than the path's size or position.
    """
    out = grid.copy()
    bg = _most_common_color(grid)
    palette = _palette_from_bottom_row(grid, bg)
    seen = set()

    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1]):
            cell = (r, c)
            if grid[r, c] == PATH_COLOR and cell not in seen:
                cells = _collect_component(grid, cell, seen)
                turns = _count_turns(cells)
                color = palette[turns]
                for rr, cc in cells:
                    out[rr, cc] = color
    return out


def _path_cells(start, first_dir, turns, lengths, side_choices):
    cells = [start]
    r, c = start
    dir_idx = first_dir
    for segment_index, length in enumerate(lengths):
        dr, dc = DIRS[dir_idx]
        for _ in range(length):
            r += dr
            c += dc
            cells.append((r, c))
        if segment_index < turns:
            dir_idx = (dir_idx + side_choices[segment_index]) % 4
    return cells


def _is_valid_path(cells, height, width, occupied):
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False

    for r, c in cells:
        if not (0 <= r < height - 1 and 0 <= c < width):
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return False

    adjacency_counts = []
    for cell in cells:
        adjacency_counts.append(sum(nb in cell_set for nb in _neighbors4(cell)))
    return adjacency_counts.count(1) == 2 and all(x in (1, 2) for x in adjacency_counts)


def _make_random_path(height, width, turns, occupied):
    for _ in range(500):
        lengths = [random.randint(2, 5) for _ in range(turns + 1)]
        side_choices = [random.choice((-1, 1)) for _ in range(turns)]
        start = (random.randrange(height - 1), random.randrange(width))
        first_dir = random.randrange(4)
        cells = _path_cells(start, first_dir, turns, lengths, side_choices)
        if _is_valid_path(cells, height, width, occupied):
            return cells
    return None


def generate_input() -> np.ndarray:
    for _ in range(100):
        height = random.randint(14, 30)
        width = random.randint(14, 30)
        colors = PALETTE_COLORS[:]
        random.shuffle(colors)
        palette_len = random.randint(3, min(7, width, len(colors)))
        palette = colors[:palette_len]

        grid = np.full((height, width), BACKGROUND, dtype=int)
        palette_start = random.randint(0, width - palette_len)
        grid[-1, palette_start : palette_start + palette_len] = palette

        object_count = random.randint(2, 5)
        turn_counts = [random.randrange(palette_len) for _ in range(object_count)]
        if object_count >= 3:
            turn_counts[0] = 0
            turn_counts[1] = random.randrange(1, palette_len)
        random.shuffle(turn_counts)

        occupied = set()
        placed = []
        for turns in turn_counts:
            cells = _make_random_path(height, width, turns, occupied)
            if cells is None:
                break
            placed.append(cells)
            occupied.update(cells)
        else:
            for cells in placed:
                for r, c in cells:
                    grid[r, c] = PATH_COLOR
            return grid

    height = width = 24
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[-1, 1:6] = [1, 2, 3, 4, 6]
    fallback_paths = [
        [(2, 2), (2, 3), (2, 4), (2, 5)],
        [(6, 2), (6, 3), (6, 4), (7, 4), (8, 4)],
        [(12, 3), (12, 4), (13, 4), (14, 4), (14, 5), (14, 6)],
    ]
    for cells in fallback_paths:
        for r, c in cells:
            grid[r, c] = PATH_COLOR
    return grid

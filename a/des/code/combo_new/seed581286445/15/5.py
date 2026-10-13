from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
LEGEND_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

GEN_RETRIES = 400
PARTITION_RETRIES = 80


def neighbors4(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = True
            comp: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                comp.append((int(r), int(c)))
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            out.append(comp)
    return out


def bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    color_mask = (~np.isin(grid, [BLACK, GREY, MAROON]))
    candidates: list[tuple[int, int, dict[int, int], set[tuple[int, int]]]] = []

    for comp in components(color_mask):
        rows = sorted(r for r, _ in comp)
        cols = {c for _, c in comp}
        if len(comp) != 5 or len(cols) != 1:
            continue
        if rows != list(range(rows[0], rows[0] + 5)):
            continue

        col = next(iter(cols))
        mapping: dict[int, int] = {}
        legend_cells = set(comp)
        ok = True
        for row in rows:
            run = 0
            cur = col - 1
            while cur >= 0 and int(grid[row, cur]) == MAROON:
                legend_cells.add((int(row), int(cur)))
                run += 1
                cur -= 1
            if run == 0 or run in mapping:
                ok = False
                break
            mapping[run] = int(grid[row, col])

        if ok and sorted(mapping) == [1, 2, 3, 4, 5] and len(set(mapping.values())) == 5:
            candidates.append((rows[0], int(col), mapping, legend_cells))

    assert candidates, "expected one vertical color legend with maroon row markers"
    _top, _col, mapping, legend_cells = min(candidates, key=lambda item: (item[0], item[1]))

    legend_mask = np.zeros_like(grid, dtype=bool)
    for row, col in legend_cells:
        legend_mask[row, col] = True
    return mapping, legend_mask


def _room_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    rooms: list[list[tuple[int, int]]] = []
    for comp in components(grid != GREY):
        if any(r in (0, height - 1) or c in (0, width - 1) for r, c in comp):
            continue
        colors = {int(grid[r, c]) for r, c in comp}
        if colors.issubset({BLACK, MAROON}):
            rooms.append(comp)
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with the legend color selected by its maroon-seed count.

    Input structure:
    - Black is the background outside a connected Grey wall structure.
    - The Grey walls split the interior into one or more enclosed 4-connected rooms made of Black cells, and each room contains 1 to 5 Maroon seed pixels.
    - A separate legend in the upper-left area is a 5-row vertical color column; each legend row has a horizontal Maroon marker run on its left whose length identifies the row's count.

    Rule:
    - Read the legend by counting the contiguous Maroon marker run to the left of each color cell, producing a visible count-to-color mapping for counts 1 through 5.
    - Find every enclosed room as a non-Grey connected component that does not touch the grid border.
    - Count the Maroon seeds inside each room, then replace every Black and Maroon cell in that room with the legend color mapped from that seed count.
    - Keep the Grey walls, the legend, and the exterior background unchanged.

    Key insight:
    - Maroon does not directly specify the output color; it is only a count cue, and the actual fill color is obtained through the legend's row-wise count markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    mapping, _legend_mask = _find_legend(g)
    rooms = _room_components(g)
    assert rooms, "expected at least one enclosed room"

    out = g.copy()
    for comp in rooms:
        seed_count = sum(int(g[row, col]) == MAROON for row, col in comp)
        assert 1 <= seed_count <= 5, "each room must contain 1..5 maroon seeds"
        fill_color = int(mapping[seed_count])
        for row, col in comp:
            out[row, col] = fill_color
    return out


def _cell_origin(top: int, left: int, cell_h: int, cell_w: int, row: int, col: int) -> tuple[int, int]:
    return top + 1 + row * (cell_h + 1), left + 1 + col * (cell_w + 1)


def _coarse_partition(cell_rows: int, cell_cols: int, room_count: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(cell_rows) for c in range(cell_cols)]
    seeds = random.sample(cells, room_count)

    labels = -np.ones((cell_rows, cell_cols), dtype=int)
    sizes = [0] * room_count
    for room_id, (row, col) in enumerate(seeds):
        labels[row, col] = room_id
        sizes[room_id] = 1

    unassigned = {cell for cell in cells if cell not in seeds}
    while unassigned:
        frontier: list[tuple[int, int, tuple[int, ...]]] = []
        for row, col in unassigned:
            neighbors = {
                int(labels[nr, nc])
                for nr, nc in neighbors4(row, col)
                if 0 <= nr < cell_rows and 0 <= nc < cell_cols and int(labels[nr, nc]) >= 0
            }
            if neighbors:
                frontier.append((int(row), int(col), tuple(sorted(neighbors))))
        if not frontier:
            return None

        row, col, room_ids = random.choice(frontier)
        smallest = min(sizes[room_id] for room_id in room_ids)
        choices = [room_id for room_id in room_ids if sizes[room_id] <= smallest + 1]
        room_id = random.choice(choices)
        labels[row, col] = room_id
        sizes[room_id] += 1
        unassigned.remove((row, col))

    return labels


def _has_irregular_room(labels: np.ndarray) -> bool:
    for room_id in range(int(labels.max()) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(labels == room_id)]
        r0, c0, r1, c1 = bbox(coords)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return True
    return False


def _sample_room_counts(room_count: int) -> list[int]:
    counts = list(range(1, 6))
    if room_count <= 5:
        if random.random() < 0.5:
            picked = random.sample(counts, room_count)
        else:
            picked = [random.choice(counts) for _ in range(room_count)]
    else:
        picked = counts + [random.choice(counts) for _ in range(room_count - 5)]
    random.shuffle(picked)
    if room_count > 1 and len(set(picked)) == 1:
        picked[0] = 1 + (picked[0] % 5)
    return picked


def _draw_wall_structure(
    grid: np.ndarray,
    labels: np.ndarray,
    top: int,
    left: int,
    cell_h: int,
    cell_w: int,
) -> None:
    cell_rows, cell_cols = labels.shape
    build_h = cell_rows * cell_h + cell_rows + 1
    build_w = cell_cols * cell_w + cell_cols + 1

    grid[top : top + build_h, left : left + build_w] = GREY
    for row in range(cell_rows):
        for col in range(cell_cols):
            r0, c0 = _cell_origin(top, left, cell_h, cell_w, row, col)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = BLACK

    for row in range(cell_rows):
        for col in range(cell_cols):
            if col + 1 < cell_cols and int(labels[row, col]) == int(labels[row, col + 1]):
                r0, c0 = _cell_origin(top, left, cell_h, cell_w, row, col)
                grid[r0 : r0 + cell_h, c0 + cell_w] = BLACK
            if row + 1 < cell_rows and int(labels[row, col]) == int(labels[row + 1, col]):
                r0, c0 = _cell_origin(top, left, cell_h, cell_w, row, col)
                grid[r0 + cell_h, c0 : c0 + cell_w] = BLACK


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    legend_top = random.randint(0, 2)
    legend_left = random.randint(0, 2)
    gap = random.randint(1, 3)

    cell_rows = random.randint(2, 4)
    cell_cols = random.randint(2, 4)
    cell_h = random.randint(3, 4)
    cell_w = random.randint(3, 4)

    total_cells = cell_rows * cell_cols
    max_rooms = min(6, total_cells - 1)
    if max_rooms < 2:
        return None
    room_count = random.randint(2, max_rooms)

    build_h = cell_rows * cell_h + cell_rows + 1
    build_w = cell_cols * cell_w + cell_cols + 1
    build_top = legend_top + 5 + gap
    min_height = build_top + build_h
    min_width = max(legend_left + 6, build_w)
    if min_height > 30 or min_width > 30:
        return None

    grid_h = random.randint(min_height, 30)
    grid_w = random.randint(min_width, 30)
    build_left = random.randint(0, grid_w - build_w)

    labels = None
    for _ in range(PARTITION_RETRIES):
        candidate = _coarse_partition(cell_rows, cell_cols, room_count)
        if candidate is None:
            continue
        if _has_irregular_room(candidate) or random.random() < 0.25:
            labels = candidate
            break
    if labels is None:
        return None

    palette_colors = random.sample(LEGEND_COLOR_POOL, 5)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    color_col = legend_left + 5
    for count, color in enumerate(palette_colors, start=1):
        row = legend_top + count - 1
        grid[row, color_col - count : color_col] = MAROON
        grid[row, color_col] = color

    _draw_wall_structure(grid, labels, build_top, build_left, cell_h, cell_w)

    rooms = sorted(_room_components(grid), key=bbox)
    if len(rooms) != room_count:
        return None

    room_counts = _sample_room_counts(room_count)
    out = grid.copy()
    for comp, seed_count in zip(rooms, room_counts, strict=False):
        if len(comp) < seed_count:
            return None
        for row, col in random.sample(comp, seed_count):
            grid[row, col] = MAROON
        fill_color = palette_colors[seed_count - 1]
        for row, col in comp:
            out[row, col] = fill_color

    try:
        if not np.array_equal(main(grid), out):
            return None
    except AssertionError:
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_attempt()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate a valid input")

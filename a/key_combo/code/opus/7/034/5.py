from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
COLOR_CYCLE = (1, 2, 3, 6, 7, 8)
COLOR_INDEX = {color: idx for idx, color in enumerate(COLOR_CYCLE)}
GRID_SIZES = (15, 17)
SOURCE_SIZE_RANGE = (6, 12)
SOURCE_COLOR_COUNT_RANGE = (2, 3)
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _shift_color(color: int, distance: int) -> int:
    return COLOR_CYCLE[(COLOR_INDEX[int(color)] + int(distance)) % len(COLOR_CYCLE)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add the 180-degree reflected copy of the left pattern, recoloring it by Manhattan-distance offsets from the center marker.

    Input structure:
    - The grid has odd dimensions on a black background with a unique yellow center marker.
    - The center column is a grey divider except for the yellow center cell.
    - A connected multicolor source pattern lies entirely to the left of the divider, while the right half is otherwise empty.

    Rule:
    - Keep the input unchanged.
    - For each non-black, non-yellow, non-grey source pixel on the left half, rotate it 180 degrees around the center marker.
    - Compute that pixel's Manhattan distance to the center and advance its color through the cycle blue, red, green, pink, orange, teal by that distance.
    - Paint the recolored reflected pixel only if the target cell is still black; never overwrite existing pixels.

    Key insight:
    - The yellow center simultaneously defines where the reflected copy goes and how far each source pixel shifts around the color cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h % 2 == 1 and w % 2 == 1, "expected odd dimensions"

    cr, cc = h // 2, w // 2
    assert int(g[cr, cc]) == YELLOW, "expected a yellow center marker"

    out = g.copy()
    rows, cols = np.indices(g.shape)
    source_mask = (cols < cc) & (g != BG) & (g != YELLOW) & (g != GREY)
    source_coords = np.argwhere(source_mask)
    assert source_coords.size > 0, "expected a left-side source pattern"

    for r, c in source_coords:
        color = int(g[r, c])
        assert color in COLOR_INDEX, "source colors must come from the visible color cycle"
        rr = 2 * cr - int(r)
        rc = 2 * cc - int(c)
        assert 0 <= rr < h and 0 <= rc < w, "reflection must stay in bounds"
        if int(out[rr, rc]) != BG:
            continue
        distance = abs(cr - int(r)) + abs(cc - int(c))
        out[rr, rc] = _shift_color(color, distance)

    return out


def _allowed_cells(n: int) -> list[tuple[int, int]]:
    cr = cc = n // 2
    cells = []
    for r in range(n):
        for c in range(1, cc):
            if abs(r - cr) + abs(c - cc) <= cc:
                cells.append((r, c))
    return cells


def _grow_shape(n: int) -> set[tuple[int, int]] | None:
    cr = cc = n // 2
    allowed = set(_allowed_cells(n))
    target_size = random.randint(*SOURCE_SIZE_RANGE)
    seed_pool = [
        (r, c)
        for r, c in allowed
        if abs(r - cr) <= 3 and cc - 5 <= c <= cc - 1
    ]
    if not seed_pool:
        return None

    cells = {random.choice(seed_pool)}
    while len(cells) < target_size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in allowed and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None

        frontier = sorted(set(frontier))
        scored = []
        for nr, nc in frontier:
            dist = _manhattan((nr, nc), (cr, cc))
            center_bias = abs(nr - cr) + 2 * abs(nc - (cc - 2))
            scored.append((dist, center_bias, random.random(), (nr, nc)))
        scored.sort()
        pick_pool = [item[-1] for item in scored[: min(4, len(scored))]]
        cells.add(random.choice(pick_pool))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    distances = {_manhattan((r, c), (cr, cc)) for r, c in cells}
    if len(set(rows)) < 2 or len(set(cols)) < 2:
        return None
    if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 1:
        return None
    if len(distances) < 3:
        return None

    return cells


def _partition_sizes(total: int, parts: int, min_each: int) -> list[int] | None:
    if total < parts * min_each:
        return None
    sizes = [min_each] * parts
    for _ in range(total - parts * min_each):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _color_shape(cells: set[tuple[int, int]], n: int) -> dict[tuple[int, int], int] | None:
    cr = cc = n // 2
    color_count = random.randint(*SOURCE_COLOR_COUNT_RANGE)
    colors = random.sample(COLOR_CYCLE, color_count)
    sizes = _partition_sizes(len(cells), color_count, 2)
    if sizes is None:
        return None

    cells_list = list(cells)
    schemes = [
        lambda rc: (rc[0], rc[1]),
        lambda rc: (rc[1], rc[0]),
        lambda rc: (_manhattan(rc, (cr, cc)), rc[0], rc[1]),
        lambda rc: (abs(rc[0] - cr), rc[1], rc[0]),
    ]
    ordered = sorted(cells_list, key=random.choice(schemes))

    coloring: dict[tuple[int, int], int] = {}
    start = 0
    for color, size in zip(colors, sizes):
        for cell in ordered[start : start + size]:
            coloring[cell] = color
        start += size

    if len(set(coloring.values())) != color_count:
        return None
    return coloring


def _build_grid(n: int, coloring: dict[tuple[int, int], int]) -> np.ndarray:
    grid = np.full((n, n), BG, dtype=int)
    cr = cc = n // 2
    grid[:, cc] = GREY
    grid[cr, cc] = YELLOW
    for (r, c), color in coloring.items():
        grid[r, c] = color
    return grid


def _reflected_targets(coloring: dict[tuple[int, int], int], n: int) -> list[tuple[int, int, int]]:
    cr = cc = n // 2
    targets = []
    for (r, c), color in coloring.items():
        rr = 2 * cr - r
        rc = 2 * cc - c
        dist = abs(cr - r) + abs(cc - c)
        targets.append((rr, rc, _shift_color(color, dist)))
    return targets


def _valid_sample(grid: np.ndarray, coloring: dict[tuple[int, int], int]) -> bool:
    n = grid.shape[0]
    cr = cc = n // 2
    targets = _reflected_targets(coloring, n)
    target_coords = {(r, c) for r, c, _color in targets}
    target_colors = {color for _r, _c, color in targets}
    distances = {_manhattan((r, c), (cr, cc)) for r, c in coloring}

    if any(c <= cc for _r, c in target_coords):
        return False
    if len(target_colors) < 2:
        return False
    if len(distances) < 3:
        return False
    if np.count_nonzero(grid[:, cc] == GREY) != n - 1:
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out, main(grid.copy())):
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    cells = _grow_shape(n)
    if cells is None:
        return None
    coloring = _color_shape(cells, n)
    if coloring is None:
        return None
    grid = _build_grid(n, coloring)
    if not _valid_sample(grid, coloring):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")

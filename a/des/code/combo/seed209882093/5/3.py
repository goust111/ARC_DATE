from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
PAIR_COLORS = (1, 2, 3, 4, 6, 8, 9)

MAX_GENERATION_ATTEMPTS = 400
PAIR_ATTEMPTS = 250
MIN_LEG = 2


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _find_marker(grid: np.ndarray) -> tuple[Coord, Coord, bool, set[Coord]]:
    orange_pts = np.argwhere(grid == ORANGE)
    assert orange_pts.shape[0] == 1, "expected a unique orange wind arm"
    orange = (int(orange_pts[0, 0]), int(orange_pts[0, 1]))

    h, w = grid.shape
    grey_neighbors = []
    for nr, nc in _neighbors4(*orange):
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
            grey_neighbors.append((nr, nc))
    assert len(grey_neighbors) == 1, "orange arm must touch the unique grey center"
    center = grey_neighbors[0]

    marker_cells = {center, orange}
    vertical_first = False
    for nr, nc in _neighbors4(*center):
        assert 0 <= nr < h and 0 <= nc < w, "wind marker must fit fully in-bounds"
        value = int(grid[nr, nc])
        if (nr, nc) == orange:
            vertical_first = nr != center[0]
            continue
        assert value == GREY, "wind marker must be a grey plus with one orange arm"
        marker_cells.add((nr, nc))

    assert len(marker_cells) == 5, "wind marker must occupy exactly five cells"
    assert int(np.count_nonzero(grid == GREY)) == 4, "grey pixels must belong only to the wind marker"
    assert int(np.count_nonzero(grid == ORANGE)) == 1, "orange pixels must belong only to the wind marker"
    return center, orange, vertical_first, marker_cells


def _path_cells(a: Coord, b: Coord, horizontal_first: bool) -> list[Coord]:
    ar, ac = a
    br, bc = b
    assert ar != br and ac != bc, "each pair must require a true L path"

    cells: list[Coord] = []
    if horizontal_first:
        for c in _inclusive_range(ac, bc):
            cells.append((ar, c))
        for r in _inclusive_range(ar + (1 if br > ar else -1), br):
            cells.append((r, bc))
    else:
        for r in _inclusive_range(ar, br):
            cells.append((r, ac))
        for c in _inclusive_range(ac + (1 if bc > ac else -1), bc):
            cells.append((br, c))
    return cells


def _collect_pairs(grid: np.ndarray) -> list[tuple[int, Coord, Coord]]:
    colors = [
        int(color)
        for color in np.unique(grid)
        if int(color) not in (BLACK, GREY, ORANGE)
    ]
    pairs: list[tuple[int, Coord, Coord]] = []
    for color in colors:
        coords = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
        assert len(coords) == 2, "each non-marker foreground color must appear exactly twice"
        a, b = sorted(coords)
        pairs.append((color, a, b))

    non_marker = int(np.count_nonzero((grid != BLACK) & (grid != GREY) & (grid != ORANGE)))
    assert non_marker == 2 * len(pairs), "all non-marker colors must form unique pairs"
    assert pairs, "expected at least one colored pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color pixel pair by an L path whose bend order is controlled by the orange arm of a grey wind marker.

    The input contains:
    - a black background,
    - one grey plus marker with exactly one orange arm endpoint,
    - several colors that each appear exactly twice.

    If the orange arm points up or down, every pair is connected by going
    vertical first and horizontal second. If the orange arm points left or
    right, every pair is connected by going horizontal first and vertical
    second.

    For determinism, each pair is ordered top-to-bottom (row-major order)
    before drawing its path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _center, _orange, vertical_first, marker_cells = _find_marker(g)
    horizontal_first = not vertical_first
    pairs = _collect_pairs(g)

    out = g.copy()
    for color, a, b in pairs:
        path = _path_cells(a, b, horizontal_first=horizontal_first)
        endpoints = {a, b}
        for r, c in path:
            if (r, c) in marker_cells:
                raise AssertionError("pair paths may not overwrite the wind marker")
            if (r, c) not in endpoints:
                assert int(g[r, c]) == BLACK, "path interior must be drawn over black background"
            out[r, c] = color

    return out


def _sample_canvas(pair_count: int) -> tuple[int, int]:
    base = 10 + 2 * pair_count
    height = random.randint(base, base + 8)
    width = random.randint(base, base + 8)
    if random.random() < 0.35:
        width += random.randint(2, 5)
    elif random.random() < 0.35:
        height += random.randint(2, 5)
    return min(height, 28), min(width, 28)


def _marker_cells(center: Coord, orange_dir: Coord) -> set[Coord]:
    r, c = center
    dr, dc = orange_dir
    return {
        center,
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    }


def _place_marker(grid: np.ndarray, center: Coord, orange_dir: Coord) -> set[Coord]:
    r, c = center
    cells = _marker_cells(center, orange_dir)
    for rr, cc in cells:
        grid[rr, cc] = GREY
    grid[r + orange_dir[0], c + orange_dir[1]] = ORANGE
    return cells


def _sample_pair(
    height: int,
    width: int,
    occupied: set[Coord],
    horizontal_first: bool,
) -> tuple[Coord, Coord, list[Coord]] | None:
    for _ in range(PAIR_ATTEMPTS):
        ar = random.randint(0, height - 1 - MIN_LEG)
        ac = random.randint(0, width - 1)

        max_down = height - 1 - ar
        if max_down < MIN_LEG:
            continue
        dr = random.randint(MIN_LEG, max_down)

        max_left = ac
        max_right = width - 1 - ac
        horizontal_options = []
        if max_left >= MIN_LEG:
            horizontal_options.extend(range(-max_left, -MIN_LEG + 1))
        if max_right >= MIN_LEG:
            horizontal_options.extend(range(MIN_LEG, max_right + 1))
        if not horizontal_options:
            continue
        dc = random.choice(horizontal_options)

        a = (ar, ac)
        b = (ar + dr, ac + dc)
        path = _path_cells(a, b, horizontal_first=horizontal_first)
        if any(cell in occupied for cell in path):
            continue
        return a, b, path
    return None


def _expected_output(
    inp: np.ndarray,
    pair_specs: list[tuple[int, Coord, Coord]],
    horizontal_first: bool,
) -> np.ndarray:
    out = inp.copy()
    for color, a, b in pair_specs:
        for r, c in _path_cells(a, b, horizontal_first=horizontal_first):
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 5)
    height, width = _sample_canvas(pair_count)
    if min(height, width) < 7:
        return None

    grid = np.full((height, width), BLACK, dtype=int)

    marker_center = (
        random.randint(1, height - 2),
        random.randint(1, width - 2),
    )
    orange_dir = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
    marker = _place_marker(grid, marker_center, orange_dir)
    horizontal_first = orange_dir[0] == 0

    colors = random.sample(PAIR_COLORS, pair_count)
    occupied = set(marker)
    pair_specs: list[tuple[int, Coord, Coord]] = []

    for color in colors:
        sampled = _sample_pair(height, width, occupied, horizontal_first=horizontal_first)
        if sampled is None:
            return None
        a, b, path = sampled
        occupied.update(path)
        grid[a] = color
        grid[b] = color
        pair_specs.append((color, a, b))

    expected = _expected_output(grid, pair_specs, horizontal_first=horizontal_first)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wind-marker L-path input")

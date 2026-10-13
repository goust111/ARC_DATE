from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

BG = 0
SOURCE = 1
TRAIL = 2
TOP = 3
SEP = 5

GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 160


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(v) for v in np.unique(np.asarray(grid, dtype=int)))
    if exclude is None:
        return colors
    banned = {int(v) for v in exclude}
    return [color for color in colors if color not in banned]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != bg]
    assert len(rows) == 1, "expected exactly one solid non-background separator row"
    sep = int(rows[0])
    assert grid.shape[0] == 2 * sep + 1, "separator must split equal-height halves"
    return sep


def _bottom_coords_of_color(grid: np.ndarray, sep: int, color: int) -> list[Coord]:
    return [(int(r + sep + 1), int(c)) for r, c in np.argwhere(grid[sep + 1 :] == color)]


def _follow_trail(grid: np.ndarray, sep: int, trail_color: int, source: Coord) -> tuple[Coord, set[Coord]]:
    h, w = grid.shape
    adjacent = [
        (nr, nc)
        for nr, nc in _neighbors4(*source)
        if sep < nr < h and 0 <= nc < w and int(grid[nr, nc]) == trail_color
    ]
    assert len(adjacent) == 1, "each source must touch exactly one trail pixel"

    prev = source
    cur = adjacent[0]
    trail = {cur}
    while True:
        nxt = [
            (nr, nc)
            for nr, nc in _neighbors4(*cur)
            if (nr, nc) != prev and sep < nr < h and 0 <= nc < w and int(grid[nr, nc]) == trail_color
        ]
        assert len(nxt) <= 1, "trail must be an unbranched 4-connected path"
        if not nxt:
            return cur, trail
        prev, cur = cur, nxt[0]
        assert cur not in trail, "trail cannot loop"
        trail.add(cur)


def main(grid: np.ndarray) -> np.ndarray:
    """Move each mirrored bottom source along its trail, then only redraw moved endpoints inside the separator-adjacent window.

    Input structure:
    - A solid grey separator row splits the grid into equal-height top and bottom halves on a black background.
    - The top half contains green marker pixels, and the bottom half contains blue source pixels plus red 4-connected trails.
    - The blue sources are exactly the bottom pixels whose reflections across the separator match the green markers.

    Rule:
    - Identify the blue source color by reflecting bottom candidates into the top half and matching the green marker set.
    - For each source, follow its unique red trail to the other trail endpoint, then count how many such endpoints exist.
    - Clear all old markers, sources, and trails; keep the separator; place blue pixels only at endpoint positions lying in the first N rows below the separator, where N is the endpoint count.
    - Reflect each endpoint across the separator and place a green pixel only if that reflected location lies in the first N rows above the separator.

    Key insight:
    - The top markers are not mirrored from the original sources but from the moved endpoints, and the number of endpoints simultaneously determines how much of both halves is allowed to stay visible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    sep = _separator_row(g, bg)
    sep_color = int(g[sep, 0])

    top_colors = _colors_present(g[:sep], exclude={bg})
    assert len(top_colors) == 1, "expected exactly one top marker color"
    top_color = int(top_colors[0])
    top_coords = {tuple(map(int, rc)) for rc in np.argwhere(g[:sep] == top_color)}
    assert top_coords, "expected at least one top marker"

    bottom_colors = _colors_present(g[sep + 1 :], exclude={bg})
    assert len(bottom_colors) == 2, "expected bottom source color and trail color"
    assert top_color not in bottom_colors, "top marker color must differ from bottom colors"

    source_candidates: list[int] = []
    for color in bottom_colors:
        mirrored = {(2 * sep - r, c) for r, c in _bottom_coords_of_color(g, sep, int(color))}
        if mirrored == top_coords:
            source_candidates.append(int(color))
    assert len(source_candidates) == 1, "exactly one bottom color must mirror the top markers"
    source_color = int(source_candidates[0])
    trail_color = next(int(color) for color in bottom_colors if int(color) != source_color)

    source_coords = _bottom_coords_of_color(g, sep, source_color)
    all_trail = set(_bottom_coords_of_color(g, sep, trail_color))
    endpoints: list[Coord] = []
    used_trail: set[Coord] = set()
    for source in source_coords:
        endpoint, trail = _follow_trail(g, sep, trail_color, source)
        assert not (used_trail & trail), "different sources cannot share trail pixels"
        used_trail |= trail
        endpoints.append(endpoint)

    assert used_trail == all_trail, "every trail pixel must belong to exactly one source path"
    assert len(set(endpoints)) == len(endpoints), "distinct sources must end at distinct endpoints"

    n_endpoints = len(endpoints)
    upper_min = max(0, sep - n_endpoints)
    lower_max = min(g.shape[0] - 1, sep + n_endpoints)

    out = np.full_like(g, bg)
    out[sep, :] = sep_color
    for r, c in endpoints:
        if sep < r <= lower_max:
            out[r, c] = source_color
        mirror_r = 2 * sep - r
        if upper_min <= mirror_r < sep:
            out[mirror_r, c] = top_color
    return out


def _sample_columns(width: int, count: int, min_gap: int = 2) -> list[int] | None:
    candidates = list(range(1, width - 1))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _segment(a: Coord, b: Coord) -> list[Coord] | None:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    return None


def _polyline_path(points: list[Coord]) -> list[Coord] | None:
    cells: list[Coord] = []
    for i in range(len(points) - 1):
        seg = _segment(points[i], points[i + 1])
        if seg is None:
            return None
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _is_simple_trail(source: Coord, trail_cells: list[Coord]) -> bool:
    if not trail_cells:
        return False
    trail_set = set(trail_cells)
    if len(trail_set) != len(trail_cells):
        return False
    if len(trail_set) == 1:
        return sum(nb in trail_set for nb in _neighbors4(*source)) == 1

    seen = {trail_cells[0]}
    stack = [trail_cells[0]]
    while stack:
        cur = stack.pop()
        for nb in _neighbors4(*cur):
            if nb in trail_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if seen != trail_set:
        return False

    degrees = [sum(nb in trail_set for nb in _neighbors4(*cell)) for cell in trail_cells]
    if any(deg > 2 for deg in degrees):
        return False
    if sum(deg == 1 for deg in degrees) != 2:
        return False
    return sum(nb in trail_set for nb in _neighbors4(*source)) == 1


def _too_close_to_existing(source: Coord, trail_cells: list[Coord], occupied: set[Coord]) -> bool:
    for cell in [source, *trail_cells]:
        if cell in occupied:
            return True
        for nb in _neighbors4(*cell):
            if nb in occupied:
                return True
    return False


def _far_from_points(cell: Coord, others: list[Coord], min_manhattan: int = 2) -> bool:
    return all(abs(cell[0] - r) + abs(cell[1] - c) >= min_manhattan for r, c in others)


def _candidate_polylines(source: Coord, endpoint: Coord, sep: int, shape: tuple[int, int]) -> list[list[Coord]]:
    h, w = shape
    sr, sc = source
    er, ec = endpoint
    polylines: list[list[Coord]] = []

    if sr == er or sc == ec:
        polylines.append([source, endpoint])
    else:
        polylines.append([source, (sr, ec), endpoint])
        polylines.append([source, (er, sc), endpoint])

    mid_cols = list(range(1, w - 1))
    random.shuffle(mid_cols)
    for mid_c in mid_cols[:4]:
        if mid_c not in (sc, ec):
            polylines.append([source, (sr, mid_c), (er, mid_c), endpoint])

    mid_rows = list(range(sep + 1, h))
    random.shuffle(mid_rows)
    for mid_r in mid_rows[:4]:
        if mid_r not in (sr, er):
            polylines.append([source, (mid_r, sc), (mid_r, ec), endpoint])

    random.shuffle(polylines)
    return polylines


def _build_expected(shape: tuple[int, int], sep: int, endpoints: list[Coord]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[sep, :] = SEP
    n = len(endpoints)
    upper_min = max(0, sep - n)
    lower_max = min(shape[0] - 1, sep + n)
    for r, c in endpoints:
        if sep < r <= lower_max:
            out[r, c] = SOURCE
        mirror_r = 2 * sep - r
        if upper_min <= mirror_r < sep:
            out[mirror_r, c] = TOP
    return out


def _attempt_input() -> np.ndarray | None:
    half = random.randint(6, 8)
    height = 2 * half + 1
    sep = half
    max_sources = min(5, half)
    n_sources = random.randint(3, max_sources)
    width = random.randint(max(13, 2 * n_sources + 6), 19)

    endpoint_cols = _sample_columns(width, n_sources, min_gap=2)
    if endpoint_cols is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[sep, :] = SEP

    occupied_bottom: set[Coord] = set()
    source_coords: list[Coord] = []
    endpoints: list[Coord] = []

    for endpoint_c in endpoint_cols:
        placed = False
        for _ in range(PATH_ATTEMPTS):
            endpoint_r = sep + random.randint(1, n_sources)
            endpoint = (endpoint_r, endpoint_c)

            source_r_choices = list(range(sep + 1, height))
            random.shuffle(source_r_choices)
            source_r_choices.sort(key=lambda r: (r <= endpoint_r, random.random()))
            source_r = source_r_choices[0]
            source_c = random.randint(1, width - 2)
            source = (source_r, source_c)

            mirror_source = (2 * sep - source_r, source_c)
            if source == endpoint or not _in_bounds(grid.shape, *mirror_source):
                continue
            if mirror_source[0] >= sep:
                continue
            if not _far_from_points(source, source_coords, min_manhattan=2):
                continue
            if not _far_from_points(mirror_source, [(2 * sep - r, c) for r, c in source_coords], min_manhattan=2):
                continue

            for polyline in _candidate_polylines(source, endpoint, sep, grid.shape):
                full_path = _polyline_path(polyline)
                if full_path is None or len(full_path) < 2:
                    continue
                trail_cells = full_path[1:]
                if trail_cells[-1] != endpoint:
                    continue
                if any(not _in_bounds(grid.shape, r, c) or r <= sep for r, c in trail_cells):
                    continue
                if not _is_simple_trail(source, trail_cells):
                    continue
                if _too_close_to_existing(source, trail_cells, occupied_bottom):
                    continue

                source_coords.append(source)
                endpoints.append(endpoint)
                occupied_bottom.add(source)
                occupied_bottom.update(trail_cells)
                grid[source] = SOURCE
                for cell in trail_cells:
                    grid[cell] = TRAIL
                grid[mirror_source] = TOP
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    expected = _build_expected(grid.shape, sep, endpoints)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

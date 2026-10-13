import itertools
import random

import numpy as np


BACKGROUND = 0
START_COLOR = 5
STATION_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
POINT_RETRIES = 120
ORDER_RETRIES = 160


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _single_position(grid: np.ndarray, color: int, name: str) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) == 1, f"expected exactly one {name}"
    return tuple(map(int, coords[0]))


def _edge_sequences(shape: tuple[int, int]) -> list[tuple[str, list[tuple[int, int]]]]:
    height, width = shape
    return [
        ("top", [(0, c) for c in range(width)]),
        ("bottom", [(height - 1, c) for c in range(width)]),
        ("left", [(r, 0) for r in range(height)]),
        ("right", [(r, width - 1) for r in range(height)]),
    ]


def _find_strip(grid: np.ndarray, bg: int) -> tuple[str, list[tuple[int, int]], list[int]]:
    counts = {int(color): int(np.count_nonzero(grid == color)) for color in np.unique(grid)}
    candidates: list[tuple[str, list[tuple[int, int]], list[int]]] = []

    for edge_name, coords in _edge_sequences(grid.shape):
        values = [int(grid[r, c]) for r, c in coords]
        i = 0
        while i < len(coords):
            if values[i] in (bg, START_COLOR):
                i += 1
                continue
            j = i
            while j < len(coords) and values[j] not in (bg, START_COLOR):
                j += 1
            run_coords = coords[i:j]
            run_colors = values[i:j]
            if (
                len(run_coords) >= 2
                and len(set(run_colors)) == len(run_colors)
                and all(counts.get(color, 0) == 2 for color in run_colors)
            ):
                candidates.append((edge_name, run_coords, run_colors))
            i = j

    assert len(candidates) == 1, "expected exactly one border itinerary strip"
    return candidates[0]


def _leg_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    cells: list[tuple[int, int]] = []

    dc = 1 if c1 > c0 else -1
    if c0 != c1:
        for c in range(c0 + dc, c1 + dc, dc):
            cells.append((r0, c))

    dr = 1 if r1 > r0 else -1
    if r0 != r1:
        for r in range(r0 + dr, r1 + dr, dr):
            cells.append((r, c1))

    return cells


def _paint_leg(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    bg: int,
    color: int,
) -> None:
    for r, c in _leg_cells(start, end):
        if int(out[r, c]) == bg:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Visit the colored station pixels in the order written on the border strip, painting each horizontal-then-vertical hop with the destination color.

    Input structure:
    - A black background contains one grey single-pixel start and several colored single-pixel stations, with station colors all distinct.
    - A contiguous strip of colored single pixels lies on exactly one grid edge; reading it left-to-right or top-to-bottom gives the itinerary.
    - Each strip color matches exactly one station color, and the strip is spatially separate from the stations.

    Rule:
    - Detect the unique border strip and read its colors in edge order.
    - Starting from the grey pixel, visit the matching station of each strip color in sequence.
    - For every hop, move first horizontally to the target column and then vertically to the target row; paint only background cells on that route with the destination station's color.
    - Keep the start, the station pixels, and the strip unchanged.

    Key insight:
    - The edge strip is the route legend: station layout does not determine visit order, and the path's color changes replay the strip sequence exactly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = _most_common_color(g)
    assert bg == BACKGROUND, "expected black background"
    start = _single_position(g, START_COLOR, "grey start")

    _edge_name, strip_coords, strip_colors = _find_strip(g, bg)
    strip_set = set(strip_coords)

    stations: list[tuple[int, int]] = []
    for color in strip_colors:
        coords = [tuple(map(int, pt)) for pt in np.argwhere(g == color)]
        assert len(coords) == 2, "each station color must appear once in the strip and once as a station"
        station_choices = [pt for pt in coords if pt not in strip_set]
        assert len(station_choices) == 1, "each strip color must map to one unique station"
        stations.append(station_choices[0])

    out = g.copy()
    current = start
    for color, station in zip(strip_colors, stations):
        _paint_leg(out, current, station, bg=bg, color=color)
        current = station
    return out


def _allowed_points(height: int, width: int, side: str) -> list[tuple[int, int]]:
    row_start, row_stop = 1, height - 1
    col_start, col_stop = 1, width - 1

    if side == "top":
        row_start = 2
    elif side == "bottom":
        row_stop = height - 2
    elif side == "left":
        col_start = 2
    elif side == "right":
        col_stop = width - 2

    return [(r, c) for r in range(row_start, row_stop) for c in range(col_start, col_stop)]


def _sample_separated_points(
    height: int, width: int, side: str, count: int
) -> list[tuple[int, int]] | None:
    allowed = _allowed_points(height, width, side)
    for _ in range(POINT_RETRIES):
        random.shuffle(allowed)
        chosen: list[tuple[int, int]] = []
        for pt in allowed:
            if all(max(abs(pt[0] - qr), abs(pt[1] - qc)) >= 2 for qr, qc in chosen):
                chosen.append(pt)
                if len(chosen) == count:
                    return chosen
    return None


def _nearest_neighbor_order(
    start: tuple[int, int], stations: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    current = start
    remaining = list(stations)
    order: list[tuple[int, int]] = []
    while remaining:
        remaining.sort(key=lambda pt: (abs(current[0] - pt[0]) + abs(current[1] - pt[1]), pt[0], pt[1]))
        nxt = remaining.pop(0)
        order.append(nxt)
        current = nxt
    return order


def _has_route_overlap(
    start: tuple[int, int], order: list[tuple[int, int]]
) -> bool:
    protected = {start, *order}
    seen: set[tuple[int, int]] = set()
    current = start
    for station in order:
        for cell in _leg_cells(current, station):
            if cell in protected:
                continue
            if cell in seen:
                return True
            seen.add(cell)
        current = station
    return False


def _choose_station_order(
    start: tuple[int, int],
    stations: list[tuple[int, int]],
    *,
    want_overlap: bool,
) -> list[tuple[int, int]] | None:
    row_sorted = sorted(stations)
    col_sorted = sorted(stations, key=lambda pt: (pt[1], pt[0]))
    nearest_sorted = _nearest_neighbor_order(start, stations)

    perms = list(itertools.permutations(stations))
    random.shuffle(perms)
    for perm in perms[:ORDER_RETRIES]:
        order = list(perm)
        if order in (row_sorted, col_sorted, nearest_sorted):
            continue
        good = True
        current = start
        for station in order:
            if current[0] == station[0] or current[1] == station[1]:
                good = False
                break
            current = station
        if not good:
            continue
        if _has_route_overlap(start, order) != want_overlap:
            continue
        return order
    return None


def _strip_coords(height: int, width: int, side: str, length: int) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        row = 0 if side == "top" else height - 1
        start_col = random.randint(0, width - length)
        return [(row, start_col + i) for i in range(length)]
    col = 0 if side == "left" else width - 1
    start_row = random.randint(0, height - length)
    return [(start_row + i, col) for i in range(length)]


def _render_example(
    height: int,
    width: int,
    side: str,
    start: tuple[int, int],
    station_order: list[tuple[int, int]],
    strip_colors: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[start] = START_COLOR
    for coord, color in zip(_strip_coords(height, width, side, len(strip_colors)), strip_colors):
        grid[coord] = color
    for station, color in zip(station_order, strip_colors):
        grid[station] = color
    return grid


def _expected_output(
    grid: np.ndarray,
    start: tuple[int, int],
    station_order: list[tuple[int, int]],
    strip_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    current = start
    for color, station in zip(strip_colors, station_order):
        _paint_leg(out, current, station, bg=BACKGROUND, color=color)
        current = station
    return out


def _validate_generated(
    grid: np.ndarray,
    start: tuple[int, int],
    station_order: list[tuple[int, int]],
    strip_colors: list[int],
    *,
    want_overlap: bool,
) -> bool:
    if _nearest_neighbor_order(start, station_order) == station_order:
        return False
    if sorted(station_order) == station_order:
        return False
    if sorted(station_order, key=lambda pt: (pt[1], pt[0])) == station_order:
        return False
    if _has_route_overlap(start, station_order) != want_overlap:
        return False

    expected = _expected_output(grid, start, station_order, strip_colors)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    unique_colors = {int(color) for color in np.unique(grid)}
    if unique_colors != {BACKGROUND, START_COLOR, *strip_colors}:
        return False
    if any(int(np.count_nonzero(grid == color)) != 2 for color in strip_colors):
        return False

    if np.array_equal(grid, actual):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(12, 20)
    width = random.randint(12, 20)
    station_count = random.randint(3, 5)
    side = random.choice(["top", "bottom", "left", "right"])
    want_overlap = bool(random.getrandbits(1))

    points = _sample_separated_points(height, width, side, station_count + 1)
    if points is None:
        return None
    start = points[0]
    stations = points[1:]

    station_order = _choose_station_order(start, stations, want_overlap=want_overlap)
    if station_order is None:
        return None

    strip_colors = random.sample(STATION_COLORS, station_count)
    grid = _render_example(height, width, side, start, station_order, strip_colors)
    if not _validate_generated(grid, start, station_order, strip_colors, want_overlap=want_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid itinerary-strip puzzle")

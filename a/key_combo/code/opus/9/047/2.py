import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
SOURCE_COLORS = (1, 4, 6, 7)
GEN_ATTEMPTS = 600


def _marker_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == GREEN)
    assert coords.shape == (4, 2)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (2, 2)
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREEN)
    return int(r0), int(c0), int(r1), int(c1)


def _reflect_points(seed: tuple[int, int], marker_top: int, marker_left: int) -> list[tuple[int, int]]:
    r, c = seed
    rr = 2 * marker_top + 1 - r
    cc = 2 * marker_left + 1 - c
    return [(r, c), (r, cc), (rr, c), (rr, cc)]


def _segments_for_color(points: list[tuple[int, int]], color: int) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    rows: dict[int, list[int]] = defaultdict(list)
    cols: dict[int, list[int]] = defaultdict(list)
    for r, c in points:
        rows[int(r)].append(int(c))
        cols[int(c)].append(int(r))

    assert len(rows) == 2 and all(len(v) == 2 for v in rows.values())
    assert len(cols) == 2 and all(len(v) == 2 for v in cols.values())

    horizontal = []
    for row, row_cols in rows.items():
        c0, c1 = sorted(row_cols)
        horizontal.append((row, c0, c1, color))

    vertical = []
    for col, col_rows in cols.items():
        r0, r1 = sorted(col_rows)
        vertical.append((col, r0, r1, color))

    return horizontal, vertical


def _iter_segment_cells(segment: tuple[int, int, int, int], orientation: str):
    fixed, start, end, _color = segment
    if orientation == "h":
        for c in range(start, end + 1):
            yield fixed, c
    else:
        for r in range(start, end + 1):
            yield r, fixed


def _extract_layout(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    marker_top, marker_left, _marker_bottom, _marker_right = _marker_box(g)

    sources: dict[int, tuple[int, int]] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (BG, GREEN, GREY):
            continue
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2)
        sources[color] = tuple(map(int, coords[0]))

    horizontal_segments = []
    vertical_segments = []
    reflected_points = {}
    for color, seed in sources.items():
        points = _reflect_points(seed, marker_top, marker_left)
        reflected_points[color] = points
        h_segments, v_segments = _segments_for_color(points, color)
        horizontal_segments.extend(h_segments)
        vertical_segments.extend(v_segments)

    obstacles = {tuple(map(int, coord)) for coord in np.argwhere(g == GREY)}
    protected = {
        (marker_top + dr, marker_left + dc)
        for dr in range(2)
        for dc in range(2)
    }
    return g, sources, reflected_points, horizontal_segments, vertical_segments, obstacles, protected


def _transform(grid: np.ndarray, *, horizontal_first: bool, obstacle_red: bool) -> np.ndarray:
    g, _sources, _points, horizontal_segments, vertical_segments, obstacles, protected = _extract_layout(grid)
    out = g.copy()
    locked_red: set[tuple[int, int]] = set()

    horizontal_segments = sorted(horizontal_segments, key=lambda seg: seg[0])
    vertical_segments = sorted(vertical_segments, key=lambda seg: seg[0])
    draw_plan = [("h", horizontal_segments), ("v", vertical_segments)]
    if not horizontal_first:
        draw_plan.reverse()

    for orientation, segments in draw_plan:
        for segment in segments:
            color = int(segment[3])
            for r, c in _iter_segment_cells(segment, orientation):
                if (r, c) in protected or (r, c) in locked_red:
                    continue
                if obstacle_red and (r, c) in obstacles:
                    out[r, c] = RED
                    locked_red.add((r, c))
                    continue
                out[r, c] = color

    return out


def _occupancy(horizontal_segments, vertical_segments, protected):
    h_map: dict[tuple[int, int], set[int]] = defaultdict(set)
    v_map: dict[tuple[int, int], set[int]] = defaultdict(set)
    for segment in horizontal_segments:
        for cell in _iter_segment_cells(segment, "h"):
            if cell not in protected:
                h_map[cell].add(int(segment[3]))
    for segment in vertical_segments:
        for cell in _iter_segment_cells(segment, "v"):
            if cell not in protected:
                v_map[cell].add(int(segment[3]))
    return h_map, v_map


def _non_endpoint_path_cells(horizontal_segments, vertical_segments, protected):
    path_cells = set()
    endpoints = set()
    for segment in horizontal_segments:
        row, c0, c1, _color = segment
        endpoints.add((row, c0))
        endpoints.add((row, c1))
        for cell in _iter_segment_cells(segment, "h"):
            if cell not in protected:
                path_cells.add(cell)
    for segment in vertical_segments:
        col, r0, r1, _color = segment
        endpoints.add((r0, col))
        endpoints.add((r1, col))
        for cell in _iter_segment_cells(segment, "v"):
            if cell not in protected:
                path_cells.add(cell)
    return path_cells - endpoints, endpoints


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each source marker across the green 2x2 center, connect every aligned reflected pair, and turn traversed grey cells red.

    Input structure:
    - The grid has black background, one solid green 2x2 marker square, several grey obstacle pixels, and 3-4 distinct colored single-pixel source markers.
    - Each source marker lies in the top-left quadrant relative to the green square's geometric center.
    - Source colors are unique, while grey cells only mark places that line segments may pass through.

    Rule:
    - Use the center of the green 2x2 square as both mirror axes, creating the horizontal mirror, vertical mirror, and 180-degree copy of every source marker.
    - For each color's four reflected pixels, draw the two horizontal same-row segments first from top row to bottom row, then draw the two vertical same-column segments from left column to right column.
    - If a segment passes through a grey obstacle cell, paint that cell red and never overwrite that red later; never paint over the green 2x2 square.

    Key insight:
    - The original colored pixels are only seeds in one quadrant: mirroring creates the actual line endpoints, and the final appearance depends on both the horizontal-then-vertical draw order and the obstacle-to-red override.
    """
    return _transform(grid, horizontal_first=True, obstacle_red=True)


def _candidate_source_indices(size: int, marker_top: int) -> list[int]:
    low = max(1, 2 * marker_top + 2 - size)
    high = marker_top - 2
    return list(range(low, high + 1))


def _make_seed_layout(height: int, width: int, count: int):
    valid_marker_rows = [mr for mr in range(3, height - 2) if len(_candidate_source_indices(height, mr)) >= count]
    valid_marker_cols = [mc for mc in range(3, width - 2) if len(_candidate_source_indices(width, mc)) >= count]
    if not valid_marker_rows or not valid_marker_cols:
        return None

    marker_top = random.choice(valid_marker_rows)
    marker_left = random.choice(valid_marker_cols)
    rows = sorted(random.sample(_candidate_source_indices(height, marker_top), count))
    cols_sorted = sorted(random.sample(_candidate_source_indices(width, marker_left), count))

    swap_i = random.randint(0, count - 2)
    cols = cols_sorted[:]
    cols[swap_i], cols[swap_i + 1] = cols[swap_i + 1], cols[swap_i]
    inversions = sum(cols[i] > cols[j] for i in range(count) for j in range(i + 1, count))
    assert inversions == 1
    return marker_top, marker_left, rows, cols


def _build_input_from_seeds(height: int, width: int, marker_top: int, marker_left: int, seeds: dict[int, tuple[int, int]], obstacles: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[marker_top : marker_top + 2, marker_left : marker_left + 2] = GREEN
    for color, (r, c) in seeds.items():
        grid[r, c] = color
    for r, c in obstacles:
        assert grid[r, c] == BG
        grid[r, c] = GREY
    return grid


def _crossing_cells(horizontal_segments, vertical_segments, protected):
    h_map, v_map = _occupancy(horizontal_segments, vertical_segments, protected)
    crossings = set()
    for cell in set(h_map) & set(v_map):
        if any(h_color != v_color for h_color in h_map[cell] for v_color in v_map[cell]):
            crossings.add(cell)
    return crossings, h_map, v_map


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 18)
    width = random.randint(13, 18)
    count = random.choice((3, 4))
    layout = _make_seed_layout(height, width, count)
    if layout is None:
        return None

    marker_top, marker_left, rows, cols = layout
    colors = random.sample(SOURCE_COLORS, count)
    seeds = {color: (r, c) for color, r, c in zip(colors, rows, cols)}
    grid0 = _build_input_from_seeds(height, width, marker_top, marker_left, seeds, set())

    (
        _g,
        _sources,
        _points,
        horizontal_segments,
        vertical_segments,
        _obstacles,
        protected,
    ) = _extract_layout(grid0)
    crossings, h_map, v_map = _crossing_cells(horizontal_segments, vertical_segments, protected)
    if len(crossings) != 4:
        return None

    path_cells, endpoints = _non_endpoint_path_cells(horizontal_segments, vertical_segments, protected)
    extra_candidates = sorted(path_cells - crossings)
    extra_count = random.choice((0, 1, 2))
    if len(extra_candidates) < extra_count:
        return None

    crossing_list = sorted(crossings)
    random.shuffle(crossing_list)
    obstacle_crossings = set(crossing_list[:3])
    extras = set(random.sample(extra_candidates, extra_count))
    obstacles = obstacle_crossings | extras
    if len(obstacles) < 3 or len(obstacles) > 5:
        return None
    if obstacles & endpoints:
        return None

    inp = _build_input_from_seeds(height, width, marker_top, marker_left, seeds, obstacles)
    out = main(inp)

    wrong_order = _transform(inp, horizontal_first=False, obstacle_red=True)
    wrong_obstacles = _transform(inp, horizontal_first=True, obstacle_red=False)
    if np.array_equal(out, wrong_order):
        return None
    if np.array_equal(out, wrong_obstacles):
        return None

    uncovered_crossings = crossings - obstacles
    if len(uncovered_crossings) != 1:
        return None
    visible_crossing = next(iter(uncovered_crossings))
    h_colors = h_map[visible_crossing]
    v_colors = v_map[visible_crossing]
    if len(h_colors) != 1 or len(v_colors) != 1:
        return None
    h_color = next(iter(h_colors))
    v_color = next(iter(v_colors))
    if h_color == v_color:
        return None
    if int(out[visible_crossing]) != v_color:
        return None
    if any(int(out[cell]) != RED for cell in obstacles):
        return None

    for color, (r, c) in seeds.items():
        if not (r < marker_top and c < marker_left):
            return None
        if r >= marker_top - 1 or c >= marker_left - 1:
            return None
        if int(inp[r, c]) != color:
            return None

    if len({r for r, _c in seeds.values()}) != count or len({_r for _r, c in seeds.values()}) != count:
        return None
    if not np.array_equal(out, main(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mirrored-endpoint wiring input")

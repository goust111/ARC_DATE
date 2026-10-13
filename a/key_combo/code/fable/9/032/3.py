import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BLACK = 0
INNER = 3
STRIDE = INNER + 1
LOCAL_CENTER = (1, 1)
LOCAL_TO_DIR = {
    (0, 1): (-1, 0),
    (2, 1): (1, 0),
    (1, 0): (0, -1),
    (1, 2): (0, 1),
}
DIR_TO_LOCAL = {direction: local for local, direction in LOCAL_TO_DIR.items()}
DIRS = tuple(DIR_TO_LOCAL)
NONZERO_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400


@dataclass(frozen=True)
class Marker:
    tile: tuple[int, int]
    direction: tuple[int, int]
    color: int


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return 1 + STRIDE * tile_r, 1 + STRIDE * tile_c


def _bounds_from_grid(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    assert (h - 1) % STRIDE == 0 and (w - 1) % STRIDE == 0, "grid must be a regular 3x3-tile lattice"
    return (h - 1) // STRIDE, (w - 1) // STRIDE


def _extract_doors(grid: np.ndarray, wall: int) -> tuple[np.ndarray, np.ndarray]:
    tile_rows, tile_cols = _bounds_from_grid(grid)
    h_doors = np.zeros((tile_rows - 1, tile_cols), dtype=bool)
    v_doors = np.zeros((tile_rows, tile_cols - 1), dtype=bool)

    for tile_r in range(tile_rows - 1):
        wall_r = STRIDE * (tile_r + 1)
        for tile_c in range(tile_cols):
            base_c = 1 + STRIDE * tile_c
            segment = grid[wall_r, base_c : base_c + INNER]
            center = int(segment[1])
            assert int(segment[0]) == wall and int(segment[2]) == wall, "door walls must keep their shoulders"
            assert center in (BLACK, wall), "door center must be black or wall color"
            h_doors[tile_r, tile_c] = center == BLACK

    for tile_r in range(tile_rows):
        base_r = 1 + STRIDE * tile_r
        for tile_c in range(tile_cols - 1):
            wall_c = STRIDE * (tile_c + 1)
            segment = grid[base_r : base_r + INNER, wall_c]
            center = int(segment[1])
            assert int(segment[0]) == wall and int(segment[2]) == wall, "door walls must keep their shoulders"
            assert center in (BLACK, wall), "door center must be black or wall color"
            v_doors[tile_r, tile_c] = center == BLACK

    return h_doors, v_doors


def _parse_markers(grid: np.ndarray, wall: int) -> list[Marker]:
    tile_rows, tile_cols = _bounds_from_grid(grid)
    markers: list[Marker] = []
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0, c0 = _tile_origin(tile_r, tile_c)
            interior = grid[r0 : r0 + INNER, c0 : c0 + INNER]
            locs = np.argwhere(interior != BLACK)
            assert len(locs) <= 1, "each tile may contain at most one marker"
            if len(locs) == 0:
                continue
            local = tuple(map(int, locs[0]))
            assert local in LOCAL_TO_DIR, "marker must sit on a 3x3 edge midpoint"
            color = int(interior[local])
            assert color != wall, "marker color must differ from wall color"
            markers.append(Marker((tile_r, tile_c), LOCAL_TO_DIR[local], color))
    assert markers, "expected at least one marker"
    return markers


def _door_open(
    tile: tuple[int, int],
    direction: tuple[int, int],
    tile_rows: int,
    tile_cols: int,
    h_doors: np.ndarray,
    v_doors: np.ndarray,
) -> bool:
    tile_r, tile_c = tile
    dr, dc = direction
    if dr == -1:
        return tile_r > 0 and bool(h_doors[tile_r - 1, tile_c])
    if dr == 1:
        return tile_r < tile_rows - 1 and bool(h_doors[tile_r, tile_c])
    if dc == -1:
        return tile_c > 0 and bool(v_doors[tile_r, tile_c - 1])
    assert dc == 1
    return tile_c < tile_cols - 1 and bool(v_doors[tile_r, tile_c])


def _trace_route(
    start: tuple[int, int],
    direction: tuple[int, int],
    tile_rows: int,
    tile_cols: int,
    h_doors: np.ndarray,
    v_doors: np.ndarray,
) -> tuple[list[tuple[int, int]], str]:
    tile = start
    path = [tile]
    while _door_open(tile, direction, tile_rows, tile_cols, h_doors, v_doors):
        tile = (tile[0] + direction[0], tile[1] + direction[1])
        path.append(tile)

    next_tile = (tile[0] + direction[0], tile[1] + direction[1])
    if 0 <= next_tile[0] < tile_rows and 0 <= next_tile[1] < tile_cols:
        return path, "door"
    return path, "boundary"


def _analyze(grid: np.ndarray) -> tuple[int, list[Marker], dict[int, list[tuple[int, int]]], dict[int, str], Counter]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    wall = int(g[0, 0])
    assert wall != BLACK, "outer border must be the wall color"
    assert np.all(g[0, :] == wall) and np.all(g[-1, :] == wall), "top and bottom borders must be solid walls"
    assert np.all(g[:, 0] == wall) and np.all(g[:, -1] == wall), "left and right borders must be solid walls"

    tile_rows, tile_cols = _bounds_from_grid(g)
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0, c0 = _tile_origin(tile_r, tile_c)
            assert np.all(g[r0 : r0 + INNER, c0 : c0 + INNER] != wall), "walls cannot intrude into tile interiors"

    h_doors, v_doors = _extract_doors(g, wall)
    markers = _parse_markers(g, wall)
    routes: dict[int, list[tuple[int, int]]] = {}
    stop_reasons: dict[int, str] = {}
    visits: Counter = Counter()
    for idx, marker in enumerate(markers):
        path, stop_reason = _trace_route(marker.tile, marker.direction, tile_rows, tile_cols, h_doors, v_doors)
        routes[idx] = path
        stop_reasons[idx] = stop_reason
        visits.update(path)
    return wall, markers, routes, stop_reasons, visits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each edge-midpoint marker tile by tile through wall gaps, leaving crossing tiles blank.

    Input structure:
    - A single wall color forms a regular lattice whose tile interiors are 3x3 black regions.
    - Adjacent tiles are separated by 1-cell-thick walls; some internal wall segments have a 1-cell central gap, while the outer border has no gaps.
    - A few tiles contain one colored marker pixel on an interior edge midpoint, one step away from the tile center.

    Rule:
    - The marker's offset from the tile center encodes its travel direction: top, bottom, left, or right.
    - Starting from its own tile, the marker advances one tile at a time in that coarse-grid direction, and it may cross into the next tile only when the shared wall segment has a central gap.
    - Every uniquely visited non-terminal tile gets the same color at the same interior offset, the uniquely visited stopping tile gets that color at the interior center, and any tile visited by multiple routes stays fully black inside.

    Key insight:
    - Motion and collisions are resolved on the coarse tile lattice, while the marker's 3x3 subcell offset simultaneously encodes both the movement direction and the in-tile drawing position.
    """
    g = np.asarray(grid, dtype=int)
    wall, markers, routes, _stop_reasons, visits = _analyze(g)

    out = np.where(g == wall, wall, BLACK)
    for idx, marker in enumerate(markers):
        path = routes[idx]
        local_r, local_c = DIR_TO_LOCAL[marker.direction]
        for step_idx, (tile_r, tile_c) in enumerate(path):
            if visits[(tile_r, tile_c)] > 1:
                continue
            r0, c0 = _tile_origin(tile_r, tile_c)
            if step_idx == len(path) - 1:
                out[r0 + LOCAL_CENTER[0], c0 + LOCAL_CENTER[1]] = marker.color
            else:
                out[r0 + local_r, c0 + local_c] = marker.color
    return out


def _blank_structure(tile_rows: int, tile_cols: int, wall: int, h_doors: np.ndarray, v_doors: np.ndarray) -> np.ndarray:
    grid = np.full((tile_rows * STRIDE + 1, tile_cols * STRIDE + 1), BLACK, dtype=int)
    grid[::STRIDE, :] = wall
    grid[:, ::STRIDE] = wall

    for tile_r in range(tile_rows - 1):
        wall_r = STRIDE * (tile_r + 1)
        for tile_c in range(tile_cols):
            if h_doors[tile_r, tile_c]:
                grid[wall_r, 1 + STRIDE * tile_c + 1] = BLACK

    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols - 1):
            if v_doors[tile_r, tile_c]:
                grid[1 + STRIDE * tile_r + 1, STRIDE * (tile_c + 1)] = BLACK

    return grid


def _route_tiles(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [(start[0] + step * direction[0], start[1] + step * direction[1]) for step in range(length)]


def _set_route_doors(
    path: list[tuple[int, int]],
    h_doors: np.ndarray,
    v_doors: np.ndarray,
) -> None:
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        if r0 == r1:
            v_doors[r0, min(c0, c1)] = True
        else:
            h_doors[min(r0, r1), c0] = True


def _line_end_choices(line_len: int, cross_idx: int, step_sign: int, boundary_stop: bool) -> tuple[int, int] | None:
    if step_sign == 1:
        start_lo, start_hi = 0, cross_idx - 1
        end_lo = cross_idx + 1
        end_hi = line_len - 1 if boundary_stop else line_len - 2
    else:
        start_lo, start_hi = cross_idx + 1, line_len - 1
        end_lo = 0 if boundary_stop else 1
        end_hi = cross_idx - 1
    if start_lo > start_hi or end_lo > end_hi:
        return None
    start = random.randint(start_lo, start_hi)
    end = random.randint(end_lo, end_hi)
    return start, end


def _cross_pair_specs(tile_rows: int, tile_cols: int, colors: list[int]) -> tuple[list[tuple[Marker, list[tuple[int, int]], str]], bool] | None:
    cross_r = random.randint(1, tile_rows - 2)
    cross_c = random.randint(1, tile_cols - 2)
    dir_h = random.choice([(0, 1), (0, -1)])
    dir_v = random.choice([(1, 0), (-1, 0)])
    boundary_horizontal = bool(random.getrandbits(1))

    h_choice = _line_end_choices(tile_cols, cross_c, dir_h[1], boundary_horizontal)
    v_choice = _line_end_choices(tile_rows, cross_r, dir_v[0], not boundary_horizontal)
    if h_choice is None or v_choice is None:
        return None

    start_h, end_h = h_choice
    start_v, end_v = v_choice
    path_h = _route_tiles((cross_r, start_h), dir_h, abs(end_h - start_h) + 1)
    path_v = _route_tiles((start_v, cross_c), dir_v, abs(end_v - start_v) + 1)
    assert path_h[-1] == (cross_r, end_h) and path_v[-1] == (end_v, cross_c)
    assert (cross_r, cross_c) in path_h and (cross_r, cross_c) in path_v
    if max(len(path_h), len(path_v)) < 4:
        return None

    specs = [
        (Marker(path_h[0], dir_h, colors[0]), path_h, "boundary" if boundary_horizontal else "door"),
        (Marker(path_v[0], dir_v, colors[1]), path_v, "door" if boundary_horizontal else "boundary"),
    ]
    return specs, not boundary_horizontal


def _path_stop_reason(path: list[tuple[int, int]], direction: tuple[int, int], tile_rows: int, tile_cols: int) -> str:
    tile_r, tile_c = path[-1]
    next_tile = (tile_r + direction[0], tile_c + direction[1])
    if 0 <= next_tile[0] < tile_rows and 0 <= next_tile[1] < tile_cols:
        return "door"
    return "boundary"


def _search_extra_spec(
    tile_rows: int,
    tile_cols: int,
    used_tiles: set[tuple[int, int]],
    used_starts: set[tuple[int, int]],
    used_colors: set[int],
    available_colors: list[int],
    prefer_short: bool,
) -> tuple[Marker, list[tuple[int, int]], str] | None:
    for _ in range(200):
        direction = random.choice(DIRS)
        color = random.choice([c for c in available_colors if c not in used_colors])
        start = (random.randrange(tile_rows), random.randrange(tile_cols))
        if start in used_starts:
            continue

        if direction == (0, 1):
            max_len = tile_cols - start[1]
        elif direction == (0, -1):
            max_len = start[1] + 1
        elif direction == (1, 0):
            max_len = tile_rows - start[0]
        else:
            max_len = start[0] + 1

        if max_len <= 0:
            continue

        if prefer_short:
            length_options = [1]
            if max_len >= 2:
                length_options.extend([2, min(3, max_len)])
        else:
            lo = 2 if max_len >= 2 else 1
            hi = min(max_len, 4)
            length_options = list(range(lo, hi + 1))
            if max_len > hi and random.random() < 0.35:
                length_options.append(max_len)

        path = None
        random.shuffle(length_options)
        for length in length_options:
            candidate = _route_tiles(start, direction, length)
            if any(tile in used_tiles for tile in candidate):
                continue
            path = candidate
            break
        if path is None:
            continue

        stop_reason = _path_stop_reason(path, direction, tile_rows, tile_cols)
        if prefer_short and len(path) == 1 and stop_reason == "boundary":
            continue
        return Marker(start, direction, color), path, stop_reason
    return None


def _place_markers(
    grid: np.ndarray,
    specs: list[tuple[Marker, list[tuple[int, int]], str]],
) -> None:
    for marker, _path, _reason in specs:
        r0, c0 = _tile_origin(*marker.tile)
        lr, lc = DIR_TO_LOCAL[marker.direction]
        grid[r0 + lr, c0 + lc] = marker.color


def _validate_generated(grid: np.ndarray) -> None:
    wall, markers, routes, stop_reasons, visits = _analyze(grid)
    out = main(grid)

    assert isinstance(out, np.ndarray) and out.shape == grid.shape
    assert wall == int(grid[0, 0])
    assert 3 <= len(markers) <= 4, "generator must create 3 or 4 markers"
    assert len({marker.color for marker in markers}) == len(markers), "markers should use distinct colors for clarity"
    assert len({marker.direction for marker in markers}) >= 2, "need directional variety"
    assert any(len(path) >= 4 for path in routes.values()), "need a nontrivial long route"
    assert any(reason == "door" for reason in stop_reasons.values()), "need an internal closed-door stop"
    assert any(reason == "boundary" for reason in stop_reasons.values()), "need a boundary stop"
    assert any(count > 1 for count in visits.values()), "need at least one crossing tile"

    collision_tiles = {tile for tile, count in visits.items() if count > 1}
    unique_stop_tiles = {routes[idx][-1] for idx in routes if visits[routes[idx][-1]] == 1}
    nonterminal_unique_tiles = {
        tile
        for idx, path in routes.items()
        for tile in path[:-1]
        if visits[tile] == 1
    }
    assert collision_tiles, "expected a blank crossing tile"
    assert unique_stop_tiles, "expected at least one centered stop tile"
    assert nonterminal_unique_tiles, "expected at least one non-terminal stamped tile"

    wall_mask = grid == wall
    assert np.array_equal(out[wall_mask], grid[wall_mask]), "walls must stay unchanged"
    for tile in collision_tiles:
        r0, c0 = _tile_origin(*tile)
        assert np.all(out[r0 : r0 + INNER, c0 : c0 + INNER] == BLACK), "crossing tile must stay blank"

    center_colors = 0
    for tile in unique_stop_tiles:
        r0, c0 = _tile_origin(*tile)
        center_colors += int(out[r0 + 1, c0 + 1] != BLACK)
    assert center_colors == len(unique_stop_tiles), "every unique stop tile must color its center"

    offset_colors = 0
    for idx, marker in enumerate(markers):
        lr, lc = DIR_TO_LOCAL[marker.direction]
        for tile in routes[idx][:-1]:
            if visits[tile] != 1:
                continue
            r0, c0 = _tile_origin(*tile)
            assert int(out[r0 + lr, c0 + lc]) == marker.color, "unique traversed tiles must preserve the marker offset"
            offset_colors += 1
    assert offset_colors >= 2, "need several offset stamps"

    base = np.where(grid == wall, wall, BLACK)
    assert np.count_nonzero(out != base) >= len(markers) + 3, "output must add visible information"
    assert _get_bg(grid) == BLACK and _get_bg(out) == BLACK


def _gen_attempt() -> np.ndarray | None:
    try:
        tile_rows = random.randint(4, 6)
        tile_cols = random.randint(4, 6)

        wall = random.choice(NONZERO_COLORS)
        marker_colors = random.sample([c for c in NONZERO_COLORS if c != wall], 4)

        cross = _cross_pair_specs(tile_rows, tile_cols, marker_colors)
        if cross is None:
            return None
        specs, _has_internal_cross_stop = cross

        used_tiles = set()
        used_starts = set()
        used_colors = set()
        for marker, path, _reason in specs:
            used_tiles.update(path)
            used_starts.add(marker.tile)
            used_colors.add(marker.color)

        extra_count = random.choice([1, 2])
        for extra_idx in range(extra_count):
            spec = _search_extra_spec(
                tile_rows,
                tile_cols,
                used_tiles,
                used_starts,
                used_colors,
                marker_colors,
                prefer_short=extra_idx == 0,
            )
            if spec is None:
                return None
            marker, path, _reason = spec
            specs.append(spec)
            used_tiles.update(path)
            used_starts.add(marker.tile)
            used_colors.add(marker.color)

        h_doors = np.zeros((tile_rows - 1, tile_cols), dtype=bool)
        v_doors = np.zeros((tile_rows, tile_cols - 1), dtype=bool)
        for _marker, path, _reason in specs:
            _set_route_doors(path, h_doors, v_doors)

        grid = _blank_structure(tile_rows, tile_cols, wall, h_doors, v_doors)
        _place_markers(grid, specs)
        _validate_generated(grid)
        return grid
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-door lattice input")

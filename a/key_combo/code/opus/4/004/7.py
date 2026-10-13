import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

CYCLE = (RED, BLUE, GREEN)
TILE_GRID_SIZE = 4
TILE_SIZE = 3
SEPARATOR_WIDTH = 1
LOCAL_OFFSETS = tuple((r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 300

_RNG = np.random.default_rng()
_PATH_SPECS = None


def _runs_between_stops(size: int, stops: list[int]) -> list[list[int]]:
    runs = []
    start = 0
    for stop in sorted(int(x) for x in stops):
        if start < stop:
            runs.append(list(range(start, stop)))
        start = stop + 1
    if start < size:
        runs.append(list(range(start, size)))
    return runs


def _parse_lattice(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    candidates = []
    for color in np.unique(g):
        color = int(color)
        sep_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue
        sep_mask = np.zeros(g.shape, dtype=bool)
        sep_mask[sep_rows, :] = True
        sep_mask[:, sep_cols] = True
        if np.array_equal(g == color, sep_mask):
            candidates.append((color, sep_rows, sep_cols, sep_mask))
    assert len(candidates) == 1

    sep_color, sep_rows, sep_cols, sep_mask = candidates[0]
    row_tiles = _runs_between_stops(h, sep_rows)
    col_tiles = _runs_between_stops(w, sep_cols)
    assert len(row_tiles) == TILE_GRID_SIZE
    assert len(col_tiles) == TILE_GRID_SIZE
    assert len({len(rows) for rows in row_tiles}) == 1
    assert len({len(cols) for cols in col_tiles}) == 1

    row_lookup = {
        r: (tile_r, local_r)
        for tile_r, rows in enumerate(row_tiles)
        for local_r, r in enumerate(rows)
    }
    col_lookup = {
        c: (tile_c, local_c)
        for tile_c, cols in enumerate(col_tiles)
        for local_c, c in enumerate(cols)
    }

    assert np.all(np.isin(g[~sep_mask], [BLACK, BLUE, RED, GREEN]))
    marker_coords = np.argwhere((g != sep_color) & (g != BLACK))
    assert marker_coords.size > 0

    offsets = {
        (row_lookup[int(r)][1], col_lookup[int(c)][1])
        for r, c in marker_coords
    }
    assert len(offsets) == 1
    offset = next(iter(offsets))

    markers = {color: [] for color in CYCLE}
    used_tiles = set()
    for rr, cc in marker_coords:
        r, c = int(rr), int(cc)
        tile = (row_lookup[r][0], col_lookup[c][0])
        color = int(g[r, c])
        assert color in markers
        assert tile not in used_tiles
        used_tiles.add(tile)
        markers[color].append(tile)

    for color in CYCLE:
        assert len(markers[color]) == 2

    return {
        "grid": g,
        "sep_color": int(sep_color),
        "sep_rows": sep_rows,
        "sep_cols": sep_cols,
        "row_tiles": row_tiles,
        "col_tiles": col_tiles,
        "offset": (int(offset[0]), int(offset[1])),
        "markers": {color: tuple(markers[color]) for color in CYCLE},
    }


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _segment_tiles(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = map(int, a)
    r2, c2 = map(int, b)
    assert (r1, c1) != (r2, c2)
    if r1 == r2:
        step = 1 if c2 > c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]
    assert c1 == c2
    step = 1 if r2 > r1 else -1
    return [(r, c1) for r in range(r1, r2 + step, step)]


def _orient_pair_from_neighbor(
    prev_tile: tuple[int, int],
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = pair
    a_adj = _adjacent(prev_tile, a)
    b_adj = _adjacent(prev_tile, b)
    assert a_adj != b_adj
    return (a, b) if a_adj else (b, a)


def _is_simple_path(path: list[tuple[int, int]]) -> bool:
    if len(path) < 2 or len(set(path)) != len(path):
        return False
    if any(not _adjacent(a, b) for a, b in zip(path, path[1:])):
        return False

    occupied = set(path)
    degrees = {}
    for tile in occupied:
        r, c = tile
        degrees[tile] = sum((r + dr, c + dc) in occupied for dr, dc in DIR4)

    if sum(degrees.values()) != 2 * (len(path) - 1):
        return False
    if degrees[path[0]] != 1 or degrees[path[-1]] != 1:
        return False
    return all(degrees[tile] == 2 for tile in path[1:-1])


def _ordered_path(markers: dict[int, tuple[tuple[int, int], tuple[int, int]]]) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    red_start, red_end = sorted(markers[RED])
    red_segment = _segment_tiles(red_start, red_end)

    blue_start, blue_end = _orient_pair_from_neighbor(red_segment[-1], markers[BLUE])
    blue_segment = _segment_tiles(blue_start, blue_end)

    green_start, green_end = _orient_pair_from_neighbor(blue_segment[-1], markers[GREEN])
    green_segment = _segment_tiles(green_start, green_end)

    path = red_segment + blue_segment + green_segment
    assert _is_simple_path(path)
    return path, [red_segment, blue_segment, green_segment]


def _tile_pixel(
    row_tiles: list[list[int]],
    col_tiles: list[list[int]],
    tile: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[int, int]:
    tile_r, tile_c = map(int, tile)
    local_r, local_c = map(int, offset)
    return int(row_tiles[tile_r][local_r]), int(col_tiles[tile_c][local_c])


def _blank_output(shape: tuple[int, int], sep_rows: list[int], sep_cols: list[int], sep_color: int) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    out[sep_rows, :] = sep_color
    out[:, sep_cols] = sep_color
    return out


def _cycle_grid(
    shape: tuple[int, int],
    sep_rows: list[int],
    sep_cols: list[int],
    sep_color: int,
    row_tiles: list[list[int]],
    col_tiles: list[list[int]],
    offset: tuple[int, int],
    path: list[tuple[int, int]],
) -> np.ndarray:
    out = _blank_output(shape, sep_rows, sep_cols, sep_color)
    for idx, tile in enumerate(path):
        r, c = _tile_pixel(row_tiles, col_tiles, tile, offset)
        out[r, c] = CYCLE[idx % len(CYCLE)]
    return out


def _segment_color_grid(
    shape: tuple[int, int],
    sep_rows: list[int],
    sep_cols: list[int],
    sep_color: int,
    row_tiles: list[list[int]],
    col_tiles: list[list[int]],
    offset: tuple[int, int],
    segments: list[list[tuple[int, int]]],
) -> np.ndarray:
    out = _blank_output(shape, sep_rows, sep_cols, sep_color)
    for color, segment in zip(CYCLE, segments):
        for tile in segment:
            r, c = _tile_pixel(row_tiles, col_tiles, tile, offset)
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Connect the colored marker tiles into one ordered chain and repaint it with a red-blue-green cycle.

    Input structure:
    - A black grid is partitioned by one-pixel grey separator lines into a 4x4 lattice of equal tiles.
    - Exactly six marker pixels appear at one shared local offset inside their tiles.
    - The marker colors are red, blue, and green, with two tiles per color, and each same-color pair lies in one tile row or one tile column.

    Rule:
    - Detect the separator lattice and convert each marker pixel into its tile coordinate plus shared local offset.
    - Connect each same-color pair along its tile row or tile column, preserving that local offset, and orient the three segments as red then blue then green.
    - The red segment starts at the upper-left of the two red endpoints; the blue and green starts are the unique endpoints adjacent to the previous segment's end.
    - Traverse the resulting tile chain from first to last and repaint the shared local offset in every visited tile with the cycle red -> blue -> green -> red -> ...
    - Keep separator lines grey and leave every unvisited tile black.

    Key insight:
    - The same-color connections are not the final output colors; they define a unique path order, and the final colors come from cycling along that recovered order.
    """
    parsed = _parse_lattice(np.asarray(grid, dtype=int))
    path, _segments = _ordered_path(parsed["markers"])
    return _cycle_grid(
        parsed["grid"].shape,
        parsed["sep_rows"],
        parsed["sep_cols"],
        parsed["sep_color"],
        parsed["row_tiles"],
        parsed["col_tiles"],
        parsed["offset"],
        path,
    )


def _build_regular_lattice() -> tuple[list[list[int]], list[int], int]:
    tile_runs = []
    sep_lines = []
    cursor = 0
    for idx in range(TILE_GRID_SIZE):
        tile_runs.append(list(range(cursor, cursor + TILE_SIZE)))
        cursor += TILE_SIZE
        if idx != TILE_GRID_SIZE - 1:
            sep_lines.append(cursor)
            cursor += SEPARATOR_WIDTH
    return tile_runs, sep_lines, cursor


def _cycle_assignment(path: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {tile: CYCLE[idx % len(CYCLE)] for idx, tile in enumerate(path)}


def _build_path_specs() -> list[dict]:
    specs = []
    primary_dirs = ((0, 1), (1, 0))
    for start_r in range(TILE_GRID_SIZE):
        for start_c in range(TILE_GRID_SIZE):
            for d1 in primary_dirs:
                for d2 in DIR4:
                    if d2 == d1 or d2 == (-d1[0], -d1[1]):
                        continue
                    for d3 in DIR4:
                        if d3 == d2 or d3 == (-d2[0], -d2[1]):
                            continue
                        for len1 in range(2, TILE_GRID_SIZE + 1):
                            seg1 = []
                            ok = True
                            for step in range(len1):
                                tile = (start_r + step * d1[0], start_c + step * d1[1])
                                if not (0 <= tile[0] < TILE_GRID_SIZE and 0 <= tile[1] < TILE_GRID_SIZE):
                                    ok = False
                                    break
                                seg1.append(tile)
                            if not ok:
                                continue

                            for len2 in range(2, TILE_GRID_SIZE + 1):
                                seg2 = []
                                pivot_r, pivot_c = seg1[-1]
                                for step in range(1, len2 + 1):
                                    tile = (pivot_r + step * d2[0], pivot_c + step * d2[1])
                                    if not (0 <= tile[0] < TILE_GRID_SIZE and 0 <= tile[1] < TILE_GRID_SIZE):
                                        ok = False
                                        break
                                    seg2.append(tile)
                                if not ok:
                                    ok = True
                                    continue

                                for len3 in range(2, TILE_GRID_SIZE + 1):
                                    seg3 = []
                                    pivot_r, pivot_c = seg2[-1]
                                    for step in range(1, len3 + 1):
                                        tile = (pivot_r + step * d3[0], pivot_c + step * d3[1])
                                        if not (0 <= tile[0] < TILE_GRID_SIZE and 0 <= tile[1] < TILE_GRID_SIZE):
                                            ok = False
                                            break
                                        seg3.append(tile)
                                    if not ok:
                                        ok = True
                                        continue

                                    path = seg1 + seg2 + seg3
                                    if not _is_simple_path(path):
                                        continue

                                    actual = _cycle_assignment(path)
                                    if actual == _cycle_assignment(sorted(path)):
                                        continue
                                    if actual == _cycle_assignment(sorted(path, key=lambda tile: (tile[1], tile[0]))):
                                        continue

                                    specs.append(
                                        {
                                            "path": tuple(path),
                                            "segments": (tuple(seg1), tuple(seg2), tuple(seg3)),
                                        }
                                    )
    assert specs
    return specs


def _get_path_specs() -> list[dict]:
    global _PATH_SPECS
    if _PATH_SPECS is None:
        _PATH_SPECS = _build_path_specs()
    return _PATH_SPECS


def _validate_generated_case(grid: np.ndarray) -> bool:
    parsed = _parse_lattice(grid)
    path, segments = _ordered_path(parsed["markers"])
    out = main(grid)

    if not np.array_equal(
        out,
        _cycle_grid(
            parsed["grid"].shape,
            parsed["sep_rows"],
            parsed["sep_cols"],
            parsed["sep_color"],
            parsed["row_tiles"],
            parsed["col_tiles"],
            parsed["offset"],
            path,
        ),
    ):
        return False

    if np.array_equal(
        out,
        _segment_color_grid(
            parsed["grid"].shape,
            parsed["sep_rows"],
            parsed["sep_cols"],
            parsed["sep_color"],
            parsed["row_tiles"],
            parsed["col_tiles"],
            parsed["offset"],
            segments,
        ),
    ):
        return False

    row_major = sorted(path)
    col_major = sorted(path, key=lambda tile: (tile[1], tile[0]))
    if np.array_equal(
        out,
        _cycle_grid(
            parsed["grid"].shape,
            parsed["sep_rows"],
            parsed["sep_cols"],
            parsed["sep_color"],
            parsed["row_tiles"],
            parsed["col_tiles"],
            parsed["offset"],
            row_major,
        ),
    ):
        return False
    if np.array_equal(
        out,
        _cycle_grid(
            parsed["grid"].shape,
            parsed["sep_rows"],
            parsed["sep_cols"],
            parsed["sep_color"],
            parsed["row_tiles"],
            parsed["col_tiles"],
            parsed["offset"],
            col_major,
        ),
    ):
        return False

    marker_pixels = np.argwhere((grid != GREY) & (grid != BLACK))
    if marker_pixels.shape[0] != 6:
        return False

    return True


def generate_input() -> np.ndarray:
    row_tiles, sep_lines, size = _build_regular_lattice()
    col_tiles = [cols[:] for cols in row_tiles]
    path_specs = _get_path_specs()

    for _ in range(GEN_ATTEMPTS):
        spec = path_specs[int(_RNG.integers(len(path_specs)))]
        offset = LOCAL_OFFSETS[int(_RNG.integers(len(LOCAL_OFFSETS)))]

        grid = np.full((size, size), BLACK, dtype=int)
        grid[sep_lines, :] = GREY
        grid[:, sep_lines] = GREY

        for color, segment in zip(CYCLE, spec["segments"]):
            for tile in (segment[0], segment[-1]):
                r, c = _tile_pixel(row_tiles, col_tiles, tile, offset)
                grid[r, c] = color

        if _validate_generated_case(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")

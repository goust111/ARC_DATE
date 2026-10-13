from collections import Counter, defaultdict

import numpy as np


def _runs_between_stops(n: int, stops: list[int]) -> list[tuple[int, ...]]:
    stop_set = set(stops)
    runs = []
    start = None
    for i in range(n):
        if i in stop_set:
            if start is not None:
                runs.append(tuple(range(start, i)))
                start = None
        elif start is None:
            start = i
    if start is not None:
        runs.append(tuple(range(start, n)))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """Complete straight same-color marker segments across the tiled lattice.

    Input structure
    The grid is a 3x3 array of equal 2x2 tiles separated by full-width and
    full-height separator lines.  Each tile is mostly the same base color, with
    a few non-base marker pixels at fixed local offsets inside their tiles.

    Rule
    For each marker color and local tile offset, connect any pair of markers
    that lie in the same tile row or same tile column by recoloring the base
    cells between them to that marker color at the same local offset.

    Key insight
    The colored pixels are endpoints in tile coordinates, not free cells in the
    raw grid; the missing pixels are found by completing axis-aligned segments
    in the 3x3 tile lattice while preserving the endpoint's local position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    separator_candidates = []
    for color in set(map(int, g.ravel())):
        sep_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if sep_rows and sep_cols:
            separator_candidates.append((color, sep_rows, sep_cols))
    assert len(separator_candidates) == 1

    sep_color, sep_rows, sep_cols = separator_candidates[0]
    sep_mask = np.zeros(g.shape, dtype=bool)
    sep_mask[sep_rows, :] = True
    sep_mask[:, sep_cols] = True
    assert np.array_equal(g == sep_color, sep_mask)

    row_tiles = _runs_between_stops(h, sep_rows)
    col_tiles = _runs_between_stops(w, sep_cols)
    assert len(row_tiles) == 3
    assert len(col_tiles) == 3
    assert {len(rows) for rows in row_tiles} == {2}
    assert {len(cols) for cols in col_tiles} == {2}

    content = g[~sep_mask]
    counts = Counter(map(int, content))
    base_color, base_count = counts.most_common(1)[0]
    marker_count = sum(n for color, n in counts.items() if color != base_color)
    assert base_color != sep_color
    assert base_count > marker_count

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

    markers = defaultdict(set)
    for r, c in np.argwhere((g != sep_color) & (g != base_color)):
        tile_r, local_r = row_lookup[int(r)]
        tile_c, local_c = col_lookup[int(c)]
        markers[(int(g[r, c]), local_r, local_c)].add((tile_r, tile_c))
    assert markers

    out = g.copy()
    for (color, local_r, local_c), tile_points in markers.items():
        assert len(tile_points) >= 2
        ordered_points = sorted(tile_points)
        assert any(
            r1 == r2 or c1 == c2
            for i, (r1, c1) in enumerate(ordered_points)
            for r2, c2 in ordered_points[i + 1 :]
        )
        for i, (r1, c1) in enumerate(ordered_points):
            for r2, c2 in ordered_points[i + 1 :]:
                if r1 == r2:
                    for tile_c in range(min(c1, c2), max(c1, c2) + 1):
                        r = row_tiles[r1][local_r]
                        c = col_tiles[tile_c][local_c]
                        assert out[r, c] in (base_color, color)
                        out[r, c] = color
                elif c1 == c2:
                    for tile_r in range(min(r1, r2), max(r1, r2) + 1):
                        r = row_tiles[tile_r][local_r]
                        c = col_tiles[c1][local_c]
                        assert out[r, c] in (base_color, color)
                        out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

# Geometry: fixed 8x8 grid with a 3x3 lattice of 2x2 content tiles.
GRID_SIZE = 8
TILE_GRID_SIZE = 3
LAST_TILE = TILE_GRID_SIZE - 1
TILE_CELLS = ((0, 1), (3, 4), (6, 7))
SEPARATOR_LINES = [2, 5]

# Sampling controls: colors, group count, offsets, and shape mix.
N_COLORS = 10
GROUP_COUNT_RANGE = (1, 3)
LOCAL_OFFSETS = [(0, 0), (0, 1), (1, 0), (1, 1)]
STRAIGHT_SHAPE_PROB = 0.5
N_LINE_AXES = 2
EXTREME_TILES = [0, LAST_TILE]
GENERATION_RETRIES = 300


def _gen_shape_tiles():
    """Return (output tiles, input endpoint tiles) for one marker group."""
    if np.random.random() < STRAIGHT_SHAPE_PROB:
        axis = np.random.randint(N_LINE_AXES)
        fixed = np.random.randint(TILE_GRID_SIZE)
        if axis == 0:
            return (
                {(fixed, c) for c in range(TILE_GRID_SIZE)},
                {(fixed, 0), (fixed, LAST_TILE)},
            )
        return (
            {(r, fixed) for r in range(TILE_GRID_SIZE)},
            {(0, fixed), (LAST_TILE, fixed)},
        )
    corner_r = int(np.random.choice(EXTREME_TILES))
    corner_c = int(np.random.choice(EXTREME_TILES))
    out_tiles = {(corner_r, c) for c in range(TILE_GRID_SIZE)}
    out_tiles.update((r, corner_c) for r in range(TILE_GRID_SIZE))
    return (
        out_tiles,
        {
            (corner_r, corner_c),
            (corner_r, LAST_TILE - corner_c),
            (LAST_TILE - corner_r, corner_c),
        },
    )


def _choose_palette_and_offsets():
    colors = list(np.random.permutation(N_COLORS))
    separator_color = int(colors[0])
    base_color = int(colors[1])
    n_groups = int(np.random.randint(*GROUP_COUNT_RANGE))
    marker_colors = [int(c) for c in colors[2 : 2 + n_groups]]
    offsets = [(int(r), int(c)) for r, c in np.random.permutation(LOCAL_OFFSETS)[:n_groups]]
    return separator_color, base_color, marker_colors, offsets


def _make_groups(marker_colors, offsets):
    groups = []
    for color, offset in zip(marker_colors, offsets):
        out_tiles, in_tiles = _gen_shape_tiles()
        groups.append({"color": color, "offset": offset, "out_tiles": out_tiles, "in_tiles": in_tiles})
    return groups


def _render_groups(groups, tile_key, *, separator_color, base_color):
    grid = np.full((GRID_SIZE, GRID_SIZE), base_color, dtype=int)
    grid[SEPARATOR_LINES, :] = separator_color
    grid[:, SEPARATOR_LINES] = separator_color

    occupied = set()
    for group in groups:
        local_r, local_c = group["offset"]
        for tile_r, tile_c in group[tile_key]:
            r = TILE_CELLS[tile_r][local_r]
            c = TILE_CELLS[tile_c][local_c]
            if (r, c) in occupied:
                return None
            occupied.add((r, c))
            grid[r, c] = group["color"]
    return grid


def _gen_attempt():
    separator_color, base_color, marker_colors, offsets = _choose_palette_and_offsets()
    groups = _make_groups(marker_colors, offsets)

    inp = _render_groups(groups, "in_tiles", separator_color=separator_color, base_color=base_color)
    if inp is None:
        return None
    expected = _render_groups(groups, "out_tiles", separator_color=separator_color, base_color=base_color)
    if expected is None or not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid cbded52d input")

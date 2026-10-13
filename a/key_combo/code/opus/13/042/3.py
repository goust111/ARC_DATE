from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
MAROON = 9

CORE_COLORS = (BLUE, RED, YELLOW)
TABLE_SHAPES = ((3, 4), (4, 3), (4, 4))
BAND_VALUES = (6, 8)
MAX_GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _regions_between(lines: list[int], limit: int) -> list[tuple[int, int]]:
    lines = sorted(int(x) for x in lines)
    assert lines, "expected at least one separator line"
    regions: list[tuple[int, int]] = []
    prev = -1
    for line in lines:
        assert 0 < line < limit - 1, "separator lines may not touch the outer border"
        assert line - prev > 1, "separator lines may not be adjacent"
        regions.append((prev + 1, line - 1))
        prev = line
    assert prev < limit - 1
    regions.append((prev + 1, limit - 1))
    return regions


def _position_type(row_i: int, col_i: int, n_rows: int, n_cols: int) -> str:
    on_outer_row = row_i in (0, n_rows - 1)
    on_outer_col = col_i in (0, n_cols - 1)
    if on_outer_row and on_outer_col:
        return "corner"
    if on_outer_row or on_outer_col:
        return "edge"
    return "interior"


def _fill_box(arr: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    h, w = arr.shape
    r0 = max(0, int(top))
    c0 = max(0, int(left))
    r1 = min(h, int(top + height))
    c1 = min(w, int(left + width))
    if r0 < r1 and c0 < c1:
        arr[r0:r1, c0:c1] = int(color)


def _square_patch(core_color: int) -> np.ndarray:
    patch = np.full((4, 4), GREY, dtype=int)
    patch[1:3, 1:3] = int(core_color)
    return patch


def _detect_square(cell: np.ndarray) -> dict[str, int] | None:
    coords = np.argwhere(cell != BLACK)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (4, 4), "non-empty cell must contain one 4x4 square"
    assert int(coords.shape[0]) == 16, "square must occupy its full 4x4 bounding box"

    patch = cell[r0 : r1 + 1, c0 : c1 + 1]
    ring_mask = np.ones((4, 4), dtype=bool)
    ring_mask[1:3, 1:3] = False
    assert np.all(patch[ring_mask] == GREY), "square border must be gray"

    core = patch[1:3, 1:3]
    core_color = int(core[0, 0])
    assert core_color in CORE_COLORS, "core color must be blue, red, or yellow"
    assert np.all(core == core_color), "core must be a solid 2x2 block"

    return {"core_color": core_color, "r0": r0, "c0": c0}


def _corner_growth(cell_shape: tuple[int, int], square: dict[str, int]) -> np.ndarray:
    out = np.zeros(cell_shape, dtype=int)
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    core_color = int(square["core_color"])

    _fill_box(out, r0 - 2, c0 - 2, 8, 8, MAROON)
    _fill_box(out, r0 - 1, c0 - 1, 6, 6, GREY)
    _fill_box(out, r0 + 1, c0 + 1, 2, 2, core_color)
    return out


def _transform_cell(cell: np.ndarray, pos_type: str, square: dict[str, int] | None) -> np.ndarray:
    if square is None:
        if pos_type == "corner":
            return np.full(cell.shape, RED, dtype=int)
        if pos_type == "interior":
            return np.full(cell.shape, ORANGE, dtype=int)
        return np.zeros_like(cell)

    core_color = int(square["core_color"])
    if pos_type == "corner":
        return _corner_growth(cell.shape, square)
    if pos_type == "interior":
        return np.full(cell.shape, core_color, dtype=int)
    return cell.copy()


def _table_structure(grid: np.ndarray) -> tuple[list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _get_bg(g) == BLACK, "background must be black"

    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREEN)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREEN)]
    assert len(full_rows) >= 2 and len(full_cols) >= 2, "table needs at least 3x3 cells"

    green_mask = g == GREEN
    expected = np.zeros_like(green_mask)
    expected[full_rows, :] = True
    expected[:, full_cols] = True
    assert np.array_equal(green_mask, expected), "green pixels must lie exactly on full separator rows and columns"

    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    assert len(row_regions) >= 3 and len(col_regions) >= 3, "table must have at least 3 rows and 3 columns of cells"
    return full_rows, full_cols, row_regions, col_regions


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _full_rows, _full_cols, row_regions, col_regions = _table_structure(g)

    out = g.copy()
    n_rows = len(row_regions)
    n_cols = len(col_regions)
    for row_i, (r0, r1) in enumerate(row_regions):
        for col_i, (c0, c1) in enumerate(col_regions):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            square = _detect_square(cell)
            pos_type = _position_type(row_i, col_i, n_rows, n_cols)
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_cell(cell, pos_type, square)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Differentiate green-table cells by table position and square presence, then grow, flood-fill, or recolor them.

    Input structure:
    - The grid has a black background with complete green separator rows and columns that partition it into a rectangular table of cells.
    - Each non-empty cell contains exactly one centered 4x4 motif: a gray 1-pixel border around a solid 2x2 blue, red, or yellow core.
    - Other cells are completely black, and the separator lines themselves are unchanged structural scaffolding.

    Rule:
    - First classify every cell by its table position: corner, non-corner edge, or strict interior; then detect whether that cell contains the bordered square.
    - Corner cells with a square turn into a clipped 8x8 nested growth centered on the square: maroon outer border, gray grown layer, original 2x2 core preserved. Empty corner cells become solid red.
    - Interior cells with a square are flood-filled entirely with the square's core color, empty interior cells become solid orange, and edge cells stay unchanged whether they contain a square or are empty.

    Key insight:
    - The green separators do more than split space: they assign each cell a role, and that role must be combined with local content to choose among six distinct transformations.
    """
    return _solve(np.asarray(grid, dtype=int))


def _band_regions(bands: tuple[int, ...]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for i, width in enumerate(bands):
        width = int(width)
        regions.append((start, start + width - 1))
        start += width
        if i < len(bands) - 1:
            start += 1
    return regions


def _band_options(count: int) -> list[tuple[int, ...]]:
    options = []
    for bands in itertools.product(BAND_VALUES, repeat=count):
        if sum(bands) + count - 1 <= 30:
            options.append(tuple(int(x) for x in bands))
    return options


BAND_OPTIONS = {count: _band_options(count) for count in (3, 4)}


def _sample_bands(count: int, rng: np.random.Generator) -> tuple[int, ...]:
    options = list(BAND_OPTIONS[count])
    assert options
    return options[int(rng.integers(0, len(options)))]


def _corner_keys(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, n_cols - 1), (n_rows - 1, 0), (n_rows - 1, n_cols - 1)]


def _edge_keys(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    keys = []
    for row_i in range(n_rows):
        for col_i in range(n_cols):
            if _position_type(row_i, col_i, n_rows, n_cols) == "edge":
                keys.append((row_i, col_i))
    return keys


def _interior_keys(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    return [(row_i, col_i) for row_i in range(1, n_rows - 1) for col_i in range(1, n_cols - 1)]


def _choose_nontrivial_subset(keys: list[tuple[int, int]], rng: np.random.Generator) -> set[tuple[int, int]]:
    count = int(rng.integers(1, len(keys)))
    perm = list(keys)
    rng.shuffle(perm)
    return set(perm[:count])


def _build_empty_table(row_bands: tuple[int, ...], col_bands: tuple[int, ...]) -> np.ndarray:
    height = sum(row_bands) + len(row_bands) - 1
    width = sum(col_bands) + len(col_bands) - 1
    grid = np.zeros((height, width), dtype=int)

    row_lines = []
    pos = row_bands[0]
    for width_band in row_bands[1:]:
        row_lines.append(pos)
        pos += 1 + width_band

    col_lines = []
    pos = col_bands[0]
    for width_band in col_bands[1:]:
        col_lines.append(pos)
        pos += 1 + width_band

    grid[row_lines, :] = GREEN
    grid[:, col_lines] = GREEN
    return grid


def _place_square(grid: np.ndarray, row_region: tuple[int, int], col_region: tuple[int, int], core_color: int) -> None:
    r0, r1 = row_region
    c0, c1 = col_region
    cell_h = r1 - r0 + 1
    cell_w = c1 - c0 + 1
    top = r0 + (cell_h - 4) // 2
    left = c0 + (cell_w - 4) // 2
    grid[top : top + 4, left : left + 4] = _square_patch(core_color)


def _same_patch(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _validate_generated_input(grid: np.ndarray) -> None:
    _full_rows, _full_cols, row_regions, col_regions = _table_structure(grid)
    n_rows = len(row_regions)
    n_cols = len(col_regions)
    assert (n_rows, n_cols) in TABLE_SHAPES

    branch_counts = {
        ("corner", True): 0,
        ("corner", False): 0,
        ("edge", True): 0,
        ("edge", False): 0,
        ("interior", True): 0,
        ("interior", False): 0,
    }
    pos_colors = {"corner": set(), "edge": set(), "interior": set()}
    all_colors: list[int] = []
    representative_out: dict[tuple[str, bool], np.ndarray] = {}
    representative_color_out: dict[tuple[str, int], np.ndarray] = {}

    out = main(grid)
    assert np.array_equal(out[_full_rows, :], grid[_full_rows, :])
    assert np.array_equal(out[:, _full_cols], grid[:, _full_cols])

    corner_filled_full = 0
    corner_filled_clipped = 0

    for row_i, row_region in enumerate(row_regions):
        for col_i, col_region in enumerate(col_regions):
            r0, r1 = row_region
            c0, c1 = col_region
            cell_in = grid[r0 : r1 + 1, c0 : c1 + 1]
            cell_out = out[r0 : r1 + 1, c0 : c1 + 1]
            square = _detect_square(cell_in)
            pos_type = _position_type(row_i, col_i, n_rows, n_cols)
            filled = square is not None
            branch_counts[(pos_type, filled)] += 1
            representative_out.setdefault((pos_type, filled), cell_out.copy())
            assert np.array_equal(cell_out, _transform_cell(cell_in, pos_type, square))

            if square is not None:
                core_color = int(square["core_color"])
                pos_colors[pos_type].add(core_color)
                all_colors.append(core_color)
                representative_color_out.setdefault((pos_type, core_color), cell_out.copy())
                if pos_type == "corner":
                    cell_h = r1 - r0 + 1
                    cell_w = c1 - c0 + 1
                    if cell_h >= 8 and cell_w >= 8:
                        corner_filled_full += 1
                    else:
                        corner_filled_clipped += 1

    assert branch_counts[("corner", True)] >= 2
    assert branch_counts[("corner", False)] >= 1
    assert branch_counts[("edge", True)] >= 1
    assert branch_counts[("edge", False)] >= 1
    assert branch_counts[("interior", True)] >= 1
    assert branch_counts[("interior", False)] >= 1
    assert len(set(all_colors)) >= 2, "filled squares should use multiple core colors"
    shared = pos_colors["corner"] & pos_colors["edge"] & pos_colors["interior"]
    assert shared, "the same core color should appear in filled cells of all three position types"
    assert corner_filled_full >= 1 and corner_filled_clipped >= 1

    shared_color = min(shared)
    assert not _same_patch(representative_color_out[("corner", shared_color)], representative_color_out[("edge", shared_color)])
    assert not _same_patch(representative_color_out[("corner", shared_color)], representative_color_out[("interior", shared_color)])
    assert not _same_patch(representative_color_out[("edge", shared_color)], representative_color_out[("interior", shared_color)])

    for pos_type in ("corner", "edge", "interior"):
        assert not _same_patch(representative_out[(pos_type, True)], representative_out[(pos_type, False)])


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_rows, n_cols = TABLE_SHAPES[int(rng.integers(0, len(TABLE_SHAPES)))]

    for _ in range(200):
        row_bands = _sample_bands(n_rows, rng)
        col_bands = _sample_bands(n_cols, rng)

        corners = _corner_keys(n_rows, n_cols)
        full_corners = [
            key
            for key in corners
            if row_bands[key[0]] >= 8 and col_bands[key[1]] >= 8
        ]
        clipped_corners = [key for key in corners if key not in full_corners]
        if not full_corners or not clipped_corners:
            continue

        corner_fill_count = int(rng.integers(2, 4))
        corner_filled = {
            full_corners[int(rng.integers(0, len(full_corners)))],
            clipped_corners[int(rng.integers(0, len(clipped_corners)))],
        }
        remaining_corners = [key for key in corners if key not in corner_filled]
        rng.shuffle(remaining_corners)
        while len(corner_filled) < corner_fill_count:
            corner_filled.add(remaining_corners.pop())

        edge_keys = _edge_keys(n_rows, n_cols)
        interior_keys = _interior_keys(n_rows, n_cols)
        edge_filled = _choose_nontrivial_subset(edge_keys, rng)
        interior_filled = _choose_nontrivial_subset(interior_keys, rng)

        filled = set(corner_filled) | set(edge_filled) | set(interior_filled)
        color_map: dict[tuple[int, int], int] = {}
        shared_color = int(rng.choice(CORE_COLORS))
        anchor_keys = [
            next(iter(corner_filled)),
            next(iter(edge_filled)),
            next(iter(interior_filled)),
        ]
        for key in anchor_keys:
            color_map[key] = shared_color

        other_filled = [key for key in filled if key not in color_map]
        for key in other_filled:
            color_map[key] = int(rng.choice(CORE_COLORS))
        if len({color_map[key] for key in filled}) == 1:
            key = other_filled[0]
            color_map[key] = next(color for color in CORE_COLORS if color != shared_color)

        row_regions = _band_regions(row_bands)
        col_regions = _band_regions(col_bands)
        grid = _build_empty_table(row_bands, col_bands)
        for row_i in range(n_rows):
            for col_i in range(n_cols):
                key = (row_i, col_i)
                if key in filled:
                    _place_square(grid, row_regions[row_i], col_regions[col_i], color_map[key])

        try:
            _validate_generated_input(grid)
        except Exception:
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")

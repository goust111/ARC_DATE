import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

GRID_SIZE = 15
TILE_SIZE = 3
TILE_COUNT = 4
SEPARATOR_INDEXES = (3, 7, 11)
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class PathInfo:
    color: int
    orientation: str
    local_offset: tuple[int, int]
    endpoint_tiles: tuple[tuple[int, int], tuple[int, int]]
    visited_tiles: tuple[tuple[int, int], ...]
    visited_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ParsedInput:
    grid: np.ndarray
    row_tiles: tuple[tuple[int, ...], ...]
    col_tiles: tuple[tuple[int, ...], ...]
    separator_color: int
    base_color: int
    paths: dict[int, PathInfo]
    intersection_tiles: frozenset[tuple[int, int]]


def _runs_between_stops(length: int, stops: list[int]) -> tuple[tuple[int, ...], ...]:
    runs = []
    prev = -1
    for stop in [*sorted(stops), length]:
        if stop - prev > 1:
            runs.append(tuple(range(prev + 1, stop)))
        prev = stop
    return tuple(runs)


def _detect_lattice(grid: np.ndarray) -> tuple[int, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    h, w = grid.shape
    candidates = []
    for color in sorted(set(map(int, grid.ravel()))):
        sep_rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue
        sep_mask = np.zeros_like(grid, dtype=bool)
        sep_mask[sep_rows, :] = True
        sep_mask[:, sep_cols] = True
        if np.array_equal(grid == color, sep_mask):
            row_tiles = _runs_between_stops(h, sep_rows)
            col_tiles = _runs_between_stops(w, sep_cols)
            candidates.append((color, row_tiles, col_tiles))
    assert len(candidates) == 1, "expected exactly one separator lattice"
    separator_color, row_tiles, col_tiles = candidates[0]
    assert len(row_tiles) == TILE_COUNT and len(col_tiles) == TILE_COUNT, "expected a 4x4 tile lattice"
    assert {len(run) for run in row_tiles} == {TILE_SIZE}, "expected 3x3 tiles"
    assert {len(run) for run in col_tiles} == {TILE_SIZE}, "expected 3x3 tiles"
    return int(separator_color), row_tiles, col_tiles


def _build_lookup(tile_runs: tuple[tuple[int, ...], ...]) -> dict[int, tuple[int, int]]:
    return {
        idx: (tile_index, local_index)
        for tile_index, run in enumerate(tile_runs)
        for local_index, idx in enumerate(run)
    }


def _tile_cell(row_tiles: tuple[tuple[int, ...], ...], col_tiles: tuple[tuple[int, ...], ...], tile_r: int, tile_c: int, offset: tuple[int, int]) -> tuple[int, int]:
    local_r, local_c = offset
    return row_tiles[tile_r][local_r], col_tiles[tile_c][local_c]


def _extract_paths(
    grid: np.ndarray,
    row_tiles: tuple[tuple[int, ...], ...],
    col_tiles: tuple[tuple[int, ...], ...],
    separator_color: int,
    base_color: int,
) -> dict[int, PathInfo]:
    row_lookup = _build_lookup(row_tiles)
    col_lookup = _build_lookup(col_tiles)

    marker_colors = [int(color) for color in sorted(set(map(int, grid.ravel()))) if color not in {separator_color, base_color}]
    assert marker_colors == [BLUE, RED], "expected exactly blue and red marker colors"

    paths: dict[int, PathInfo] = {}
    for color in marker_colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), f"expected exactly two markers for color {color}"

        marker_tiles = []
        local_offsets = []
        for r, c in coords:
            tile_r, local_r = row_lookup[int(r)]
            tile_c, local_c = col_lookup[int(c)]
            marker_tiles.append((tile_r, tile_c))
            local_offsets.append((local_r, local_c))

        assert len(set(marker_tiles)) == 2, "markers of one color must occupy distinct tiles"
        assert len(set(local_offsets)) == 1, "markers of one color must share the same local offset"
        local_offset = local_offsets[0]

        (r1, c1), (r2, c2) = sorted(marker_tiles)
        if r1 == r2:
            orientation = "horizontal"
            tile_range = tuple((r1, tile_c) for tile_c in range(min(c1, c2), max(c1, c2) + 1))
        else:
            assert c1 == c2, "markers of one color must align in the same tile row or tile column"
            orientation = "vertical"
            tile_range = tuple((tile_r, c1) for tile_r in range(min(r1, r2), max(r1, r2) + 1))

        visited_cells = tuple(_tile_cell(row_tiles, col_tiles, tile_r, tile_c, local_offset) for tile_r, tile_c in tile_range)
        paths[color] = PathInfo(
            color=color,
            orientation=orientation,
            local_offset=local_offset,
            endpoint_tiles=((r1, c1), (r2, c2)),
            visited_tiles=tile_range,
            visited_cells=visited_cells,
        )

    return paths


def _parse_input(grid: np.ndarray) -> ParsedInput:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 15x15 grid"

    separator_color, row_tiles, col_tiles = _detect_lattice(g)
    assert separator_color == GREY, "expected grey separator lines"

    sep_mask = np.zeros_like(g, dtype=bool)
    sep_mask[list(SEPARATOR_INDEXES), :] = True
    sep_mask[:, list(SEPARATOR_INDEXES)] = True
    content = g[~sep_mask]
    counts = Counter(map(int, content))
    base_color, _ = counts.most_common(1)[0]
    assert base_color == YELLOW, "expected yellow tile backgrounds"

    paths = _extract_paths(g, row_tiles, col_tiles, separator_color, base_color)
    assert paths[BLUE].local_offset != paths[RED].local_offset, "blue and red paths must use different local offsets"

    tile_to_marker_count = Counter(path.endpoint_tiles[0] for path in paths.values())
    tile_to_marker_count.update(path.endpoint_tiles[1] for path in paths.values())
    assert all(count == 1 for count in tile_to_marker_count.values()), "each marker tile must contain exactly one marker"

    intersection_tiles = frozenset(paths[BLUE].visited_tiles) & frozenset(paths[RED].visited_tiles)
    assert intersection_tiles, "the two color paths must cross in at least one tile"
    return ParsedInput(
        grid=g,
        row_tiles=row_tiles,
        col_tiles=col_tiles,
        separator_color=separator_color,
        base_color=base_color,
        paths=paths,
        intersection_tiles=intersection_tiles,
    )


def _swap_colors_in_tile(tile: np.ndarray) -> None:
    blue_mask = tile == BLUE
    red_mask = tile == RED
    tile[blue_mask] = RED
    tile[red_mask] = BLUE


def _draw_paths_without_swap(parsed: ParsedInput) -> np.ndarray:
    out = parsed.grid.copy()
    for color in (BLUE, RED):
        for r, c in parsed.paths[color].visited_cells:
            assert out[r, c] in (parsed.base_color, color), "path cells may only cover base color or their own marker"
            out[r, c] = color
    return out


def _draw_raw_segments(parsed: ParsedInput) -> np.ndarray:
    out = parsed.grid.copy()
    row_lookup = _build_lookup(parsed.row_tiles)
    col_lookup = _build_lookup(parsed.col_tiles)
    for color in (BLUE, RED):
        endpoints = np.argwhere(parsed.grid == color)
        assert endpoints.shape == (2, 2)
        (r1, c1), (r2, c2) = sorted((tuple(map(int, rc)) for rc in endpoints))
        tile_r1, _ = row_lookup[r1]
        tile_r2, _ = row_lookup[r2]
        tile_c1, _ = col_lookup[c1]
        tile_c2, _ = col_lookup[c2]
        if tile_r1 == tile_r2:
            for c in range(min(c1, c2), max(c1, c2) + 1):
                if out[r1, c] != parsed.separator_color:
                    out[r1, c] = color
        else:
            assert tile_c1 == tile_c2
            for r in range(min(r1, r2), max(r1, r2) + 1):
                if out[r, c1] != parsed.separator_color:
                    out[r, c1] = color
    return out


def _validate_semantics(grid: np.ndarray) -> None:
    parsed = _parse_input(grid)
    output = main(grid)
    assert np.any(output == BLUE) and np.any(output == RED), "output must preserve both colors somewhere"

    no_swap = _draw_paths_without_swap(parsed)
    assert not np.array_equal(output, no_swap), "input degenerates to a no-swap shortcut"

    raw_segment = _draw_raw_segments(parsed)
    assert not np.array_equal(output, raw_segment), "input degenerates to raw-pixel segment drawing"

    for tile_r, tile_c in parsed.intersection_tiles:
        rows = parsed.row_tiles[tile_r]
        cols = parsed.col_tiles[tile_c]
        tile = output[np.ix_(rows, cols)]
        assert np.any(tile == BLUE) and np.any(tile == RED), "crossing tiles must contain both swapped colors"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect matching markers through the tile lattice and swap blue and red inside crossed tiles.

    Input structure:
    - The grid is a 15x15 board split by full grey separator rows and columns into a 4x4 lattice of 3x3 yellow tiles.
    - Exactly two blue marker pixels and two red marker pixels appear inside tiles, never on separators.
    - The two markers of each color share one local tile offset and lie in the same tile row or the same tile column.

    Rule:
    - Parse the separator lattice to recover tile coordinates and local tile offsets.
    - For each color independently, traverse every tile between its two endpoint tiles along the shared tile row or tile column and color the same local offset inside each visited tile with that marker color.
    - After both paths are drawn, every tile visited by both colors is a crossing tile; inside each such tile swap blue pixels to red and red pixels to blue, while leaving separators and all non-crossing tiles unchanged.

    Key insight:
    - The markers define paths in tile coordinates, not in raw pixel coordinates, and crossings are decided per visited tile even when the two path pixels occupy different cells inside that tile.
    """
    parsed = _parse_input(grid)
    out = _draw_paths_without_swap(parsed)

    for tile_r, tile_c in parsed.intersection_tiles:
        rows = parsed.row_tiles[tile_r]
        cols = parsed.col_tiles[tile_c]
        tile = out[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        _swap_colors_in_tile(tile)

    return out


def _empty_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), YELLOW, dtype=int)
    grid[list(SEPARATOR_INDEXES), :] = GREY
    grid[:, list(SEPARATOR_INDEXES)] = GREY
    return grid


def _all_local_offsets() -> list[tuple[int, int]]:
    return [(r, c) for r in range(TILE_SIZE) for c in range(TILE_SIZE)]


def _choose_distinct_offsets() -> tuple[tuple[int, int], tuple[int, int]]:
    offsets = _all_local_offsets()
    blue_offset = random.choice(offsets)
    candidates = [offset for offset in offsets if offset != blue_offset]
    red_offset = random.choice(candidates)
    return blue_offset, red_offset


def _path_cells_from_spec(
    row_tiles: tuple[tuple[int, ...], ...],
    col_tiles: tuple[tuple[int, ...], ...],
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    offset: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    if orientation == "horizontal":
        tile_positions = ((fixed, tile_c) for tile_c in range(start, end + 1))
    else:
        tile_positions = ((tile_r, fixed) for tile_r in range(start, end + 1))
    return tuple(_tile_cell(row_tiles, col_tiles, tile_r, tile_c, offset) for tile_r, tile_c in tile_positions)


def _perpendicular_specs() -> tuple[dict[str, int | str], dict[str, int | str]]:
    shared_row = random.choice([1, 2])
    shared_col = random.choice([1, 2])

    horizontal_intervals = {1: [(0, 2), (0, 3)], 2: [(1, 3), (0, 3)]}
    vertical_intervals = {1: [(0, 2), (0, 3)], 2: [(1, 3), (0, 3)]}

    blue_horizontal = random.choice([True, False])
    if blue_horizontal:
        blue_start, blue_end = random.choice(horizontal_intervals[shared_col])
        red_start, red_end = random.choice(vertical_intervals[shared_row])
        return (
            {"orientation": "horizontal", "fixed": shared_row, "start": blue_start, "end": blue_end},
            {"orientation": "vertical", "fixed": shared_col, "start": red_start, "end": red_end},
        )

    blue_start, blue_end = random.choice(vertical_intervals[shared_row])
    red_start, red_end = random.choice(horizontal_intervals[shared_col])
    return (
        {"orientation": "vertical", "fixed": shared_col, "start": blue_start, "end": blue_end},
        {"orientation": "horizontal", "fixed": shared_row, "start": red_start, "end": red_end},
    )


def _parallel_specs(orientation: str) -> tuple[dict[str, int | str], dict[str, int | str]]:
    interval_pairs = [((0, 2), (1, 3)), ((1, 3), (0, 2)), ((0, 3), (1, 2)), ((1, 2), (0, 3))]
    first, second = random.choice(interval_pairs)
    fixed = random.randrange(TILE_COUNT)
    return (
        {"orientation": orientation, "fixed": fixed, "start": first[0], "end": first[1]},
        {"orientation": orientation, "fixed": fixed, "start": second[0], "end": second[1]},
    )


def _paint_markers(grid: np.ndarray, spec: dict[str, int | str], color: int, offset: tuple[int, int]) -> None:
    orientation = str(spec["orientation"])
    fixed = int(spec["fixed"])
    start = int(spec["start"])
    end = int(spec["end"])
    if orientation == "horizontal":
        endpoints = [(fixed, start), (fixed, end)]
    else:
        endpoints = [(start, fixed), (end, fixed)]
    for tile_r, tile_c in endpoints:
        r = tile_r * (TILE_SIZE + 1) + offset[0]
        c = tile_c * (TILE_SIZE + 1) + offset[1]
        assert grid[r, c] == YELLOW
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    grid = _empty_grid()
    row_tiles = _runs_between_stops(GRID_SIZE, list(SEPARATOR_INDEXES))
    col_tiles = _runs_between_stops(GRID_SIZE, list(SEPARATOR_INDEXES))
    blue_offset, red_offset = _choose_distinct_offsets()

    branch = random.choice(["perpendicular", "horizontal_overlap", "vertical_overlap"])
    if branch == "perpendicular":
        blue_spec, red_spec = _perpendicular_specs()
    elif branch == "horizontal_overlap":
        blue_spec, red_spec = _parallel_specs("horizontal")
    else:
        blue_spec, red_spec = _parallel_specs("vertical")

    blue_cells = _path_cells_from_spec(row_tiles, col_tiles, str(blue_spec["orientation"]), int(blue_spec["fixed"]), int(blue_spec["start"]), int(blue_spec["end"]), blue_offset)
    red_cells = _path_cells_from_spec(row_tiles, col_tiles, str(red_spec["orientation"]), int(red_spec["fixed"]), int(red_spec["start"]), int(red_spec["end"]), red_offset)
    if set(blue_cells) & set(red_cells):
        return None

    _paint_markers(grid, blue_spec, BLUE, blue_offset)
    _paint_markers(grid, red_spec, RED, red_offset)

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")

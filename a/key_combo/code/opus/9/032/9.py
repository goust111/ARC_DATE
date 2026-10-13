from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

ENDPOINT_COLORS = (BLUE, RED, ORANGE)
TILE_ROWS = 4
TILE_COLS = 4
TILE_SIZE = 3
STRIDE = 4
GRID_SHAPE = (STRIDE * TILE_ROWS + 1, STRIDE * TILE_COLS + 1)
SUBCELLS = tuple((dr, dc) for dr in range(TILE_SIZE) for dc in range(TILE_SIZE))
MAX_GENERATION_ATTEMPTS = 500


def _runs(active: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _tile_origin(tile_r: int, tile_c: int) -> tuple[int, int]:
    return 1 + STRIDE * tile_r, 1 + STRIDE * tile_c


def _pixel_from_tile(tile_r: int, tile_c: int, dr: int, dc: int) -> tuple[int, int]:
    r0, c0 = _tile_origin(tile_r, tile_c)
    return r0 + dr, c0 + dc


def _parse_lattice(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"
    assert np.all(g[0, :] == GREY) and np.all(g[-1, :] == GREY), "expected grey border rows"
    assert np.all(g[:, 0] == GREY) and np.all(g[:, -1] == GREY), "expected grey border columns"

    row_runs = _runs([not np.all(g[r, :] == GREY) for r in range(g.shape[0])])
    col_runs = _runs([not np.all(g[:, c] == GREY) for c in range(g.shape[1])])
    assert row_runs and col_runs, "expected a tile lattice"
    assert all(end - start == TILE_SIZE for start, end in row_runs), "expected 3-cell tile interiors"
    assert all(end - start == TILE_SIZE for start, end in col_runs), "expected 3-cell tile interiors"
    return row_runs, col_runs


def _extract_endpoint_groups(
    grid: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
) -> dict[tuple[int, int, int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    groups: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for tile_r, (r0, r1) in enumerate(row_runs):
        for tile_c, (c0, c1) in enumerate(col_runs):
            tile = g[r0:r1, c0:c1]
            assert tile.shape == (TILE_SIZE, TILE_SIZE)
            for dr in range(TILE_SIZE):
                for dc in range(TILE_SIZE):
                    color = int(tile[dr, dc])
                    if color in ENDPOINT_COLORS:
                        groups[(color, dr, dc)].append((tile_r, tile_c))
    return groups


def _full_group_tiles(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    covered = set(points)
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for tile_r, tile_c in points:
        by_row[tile_r].append(tile_c)
        by_col[tile_c].append(tile_r)

    for tile_r, cols in by_row.items():
        if len(cols) >= 2:
            for tile_c in range(min(cols), max(cols) + 1):
                covered.add((tile_r, tile_c))
    for tile_c, rows in by_col.items():
        if len(rows) >= 2:
            for tile_r in range(min(rows), max(rows) + 1):
                covered.add((tile_r, tile_c))
    return covered


def _connect_groups(
    grid: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    groups = _extract_endpoint_groups(out, row_runs, col_runs)
    assert groups, "expected at least one endpoint group"

    for (color, dr, dc), points in groups.items():
        assert len(points) in (2, 3), "each endpoint group must have 2 or 3 markers"
        covered = _full_group_tiles(points)
        for tile_r, tile_c in covered:
            r = row_runs[tile_r][0] + dr
            c = col_runs[tile_c][0] + dc
            out[r, c] = color
    return out


def _compress_columns(grid: np.ndarray) -> np.ndarray:
    src = np.asarray(grid, dtype=int)
    out = src.copy()
    height, width = src.shape
    for c in range(width):
        channel_rows = [r for r in range(height) if int(src[r, c]) != GREY]
        if not channel_rows:
            continue
        colors = [int(src[r, c]) for r in channel_rows if int(src[r, c]) not in (BLACK, GREY)]
        out[channel_rows, c] = BLACK
        if not colors:
            continue
        dst_rows = channel_rows[-len(colors) :]
        for r, color in zip(dst_rows, colors):
            out[r, c] = color
    return out


def _solve_without_recolor(grid: np.ndarray) -> np.ndarray:
    row_runs, col_runs = _parse_lattice(grid)
    connected = _connect_groups(grid, row_runs, col_runs)
    return _compress_columns(connected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color tile-lattice endpoints, compress each pixel column to the bottom of its channel, then recolor compressed yellow pixels green.

    Input structure:
    - A grey separator lattice partitions the grid into 3x3 black tile interiors on a 4-cell period.
    - Blue, red, and orange endpoint markers appear inside tile subcells; each color occurs 2 or 3 times at one shared relative subcell position.
    - Some tile interiors also contain yellow pixels, and grey separator rows and columns remain intact.

    Rule:
    - Group endpoints by both color and relative 3x3 subcell position, then connect every same-row or same-column pair across the tile lattice by filling the intervening tiles at that same subcell.
    - For each pixel column, read all non-black, non-grey colors from top to bottom while ignoring grey separator cells, clear the non-grey channel cells to black, and restack the collected colors at the bottom of that channel in the same top-to-bottom order.
    - After compression, recolor every yellow pixel to green.

    Key insight:
    - The tile-lattice connection step changes per-column color counts before compression, so horizontal linking only becomes visible after the later vertical stacking.
    """
    compressed = _solve_without_recolor(grid)
    out = compressed.copy()
    out[out == YELLOW] = GREEN
    return out


def _blank_grid() -> np.ndarray:
    grid = np.full(GRID_SHAPE, GREY, dtype=int)
    for tile_r in range(TILE_ROWS):
        for tile_c in range(TILE_COLS):
            r0, c0 = _tile_origin(tile_r, tile_c)
            grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = BLACK
    return grid


def _sample_span_pairs(limit: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(limit) for b in range(a + 2, limit)]


def _sample_group_points(kind: str) -> list[tuple[int, int]]:
    if kind == "H":
        row = random.randrange(TILE_ROWS)
        c0, c1 = random.choice(_sample_span_pairs(TILE_COLS))
        return [(row, c0), (row, c1)]
    if kind == "V":
        col = random.randrange(TILE_COLS)
        r0, r1 = random.choice(_sample_span_pairs(TILE_ROWS))
        return [(r0, col), (r1, col)]
    assert kind == "L"
    corner_r = random.randrange(TILE_ROWS)
    corner_c = random.randrange(TILE_COLS)
    horiz = [c for c in range(TILE_COLS) if abs(c - corner_c) >= 2]
    vert = [r for r in range(TILE_ROWS) if abs(r - corner_r) >= 2]
    end_c = random.choice(horiz)
    end_r = random.choice(vert)
    return [(corner_r, corner_c), (corner_r, end_c), (end_r, corner_c)]


def _build_groups() -> list[dict[str, object]] | None:
    colors = list(ENDPOINT_COLORS)
    random.shuffle(colors)
    subcells = random.sample(SUBCELLS, len(colors))
    kinds = ["H", "V", "L"]
    random.shuffle(kinds)

    groups: list[dict[str, object]] = []
    used_endpoint_tiles: set[tuple[int, int]] = set()
    for color, subcell, kind in zip(colors, subcells, kinds):
        points = None
        for _ in range(100):
            candidate = _sample_group_points(kind)
            if used_endpoint_tiles & set(candidate):
                continue
            points = candidate
            break
        if points is None:
            return None
        groups.append(
            {
                "color": color,
                "subcell": subcell,
                "points": points,
                "kind": kind,
            }
        )
        used_endpoint_tiles.update(points)
    return groups


def _group_reserved_pixels(groups: list[dict[str, object]]) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for group in groups:
        dr, dc = group["subcell"]  # type: ignore[misc]
        points = group["points"]  # type: ignore[misc]
        for tile_r, tile_c in _full_group_tiles(points):
            reserved.add(_pixel_from_tile(tile_r, tile_c, dr, dc))
    return reserved


def _place_endpoints(grid: np.ndarray, groups: list[dict[str, object]]) -> set[tuple[int, int]]:
    endpoint_pixels: set[tuple[int, int]] = set()
    for group in groups:
        color = int(group["color"])  # type: ignore[arg-type]
        dr, dc = group["subcell"]  # type: ignore[misc]
        points = group["points"]  # type: ignore[misc]
        for tile_r, tile_c in points:
            r, c = _pixel_from_tile(tile_r, tile_c, dr, dc)
            grid[r, c] = color
            endpoint_pixels.add((r, c))
    return endpoint_pixels


def _available_tile_cells(
    tile_r: int,
    tile_c: int,
    forbidden: set[tuple[int, int]],
    grid: np.ndarray,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r0, c0 = _tile_origin(tile_r, tile_c)
    for dr in range(TILE_SIZE):
        for dc in range(TILE_SIZE):
            r, c = r0 + dr, c0 + dc
            if (r, c) in forbidden or int(grid[r, c]) != BLACK:
                continue
            cells.append((r, c))
    return cells


def _place_yellows(
    grid: np.ndarray,
    groups: list[dict[str, object]],
    reserved_pixels: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    input_tile_cols = {
        tile_c
        for group in groups
        for (_tile_r, tile_c) in group["points"]  # type: ignore[misc]
    }
    missing_cols = [tile_c for tile_c in range(TILE_COLS) if tile_c not in input_tile_cols]

    tile_load: dict[tuple[int, int], int] = defaultdict(int)
    for group in groups:
        for tile in group["points"]:  # type: ignore[misc]
            tile_load[tile] += 1

    yellow_positions: list[tuple[int, int]] = []
    yellow_tiles: set[tuple[int, int]] = set()

    def add_yellow_tile(tile_r: int, tile_c: int, want_two: bool) -> bool:
        tile = (tile_r, tile_c)
        cells = _available_tile_cells(tile_r, tile_c, reserved_pixels, grid)
        if not cells:
            return False
        count = 2 if want_two and len(cells) >= 2 else 1
        picks = random.sample(cells, count)
        for r, c in picks:
            grid[r, c] = YELLOW
            yellow_positions.append((r, c))
        tile_load[tile] += count
        yellow_tiles.add(tile)
        return True

    for tile_c in missing_cols:
        tile_rows = list(range(TILE_ROWS))
        random.shuffle(tile_rows)
        placed = False
        for tile_r in tile_rows:
            if tile_load[(tile_r, tile_c)] >= 2:
                continue
            if add_yellow_tile(tile_r, tile_c, want_two=False):
                placed = True
                break
        if not placed:
            return []

    target_tiles = random.randint(max(3, len(yellow_tiles)), 6)
    candidates = [(tile_r, tile_c) for tile_r in range(TILE_ROWS) for tile_c in range(TILE_COLS)]
    random.shuffle(candidates)
    for tile_r, tile_c in candidates:
        if len(yellow_tiles) >= target_tiles:
            break
        if (tile_r, tile_c) in yellow_tiles or tile_load[(tile_r, tile_c)] >= 2:
            continue
        want_two = random.random() < 0.35
        add_yellow_tile(tile_r, tile_c, want_two=want_two)

    if not yellow_positions:
        return []
    return yellow_positions


def _compress_without_connect(grid: np.ndarray) -> np.ndarray:
    compressed = _compress_columns(grid)
    out = compressed.copy()
    out[out == YELLOW] = GREEN
    return out


def _validate_groups(
    grid: np.ndarray,
    groups: dict[tuple[int, int, int], list[tuple[int, int]]],
) -> bool:
    if {color for color, _dr, _dc in groups} != set(ENDPOINT_COLORS):
        return False

    by_color: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for key in groups:
        by_color[key[0]].append(key)
    if any(len(keys) != 1 for keys in by_color.values()):
        return False

    for (_color, _dr, _dc), points in groups.items():
        if len(points) not in (2, 3):
            return False
        shared = False
        for i, (r0, c0) in enumerate(points):
            for r1, c1 in points[i + 1 :]:
                if r0 == r1 or c0 == c1:
                    shared = True
        if not shared:
            return False

    input_tile_cols = set()
    for points in groups.values():
        for _tile_r, tile_c in points:
            input_tile_cols.add(tile_c)
    if input_tile_cols != set(range(TILE_COLS)):
        yellow_cols = {
            (c - 1) // STRIDE
            for r, c in np.argwhere(grid == YELLOW)
            if 1 <= c < grid.shape[1] - 1 and c % STRIDE != 0 and r % STRIDE != 0
        }
        if input_tile_cols | yellow_cols != set(range(TILE_COLS)):
            return False
    return True


def _build_attempt() -> np.ndarray | None:
    groups = _build_groups()
    if groups is None:
        return None

    grid = _blank_grid()
    reserved_pixels = _group_reserved_pixels(groups)
    _place_endpoints(grid, groups)
    yellow_positions = _place_yellows(grid, groups, reserved_pixels)
    if not yellow_positions:
        return None

    row_runs, col_runs = _parse_lattice(grid)
    extracted = _extract_endpoint_groups(grid, row_runs, col_runs)
    if not _validate_groups(grid, extracted):
        return None

    connected = _connect_groups(grid, row_runs, col_runs)
    added = (connected != grid) & (grid != GREY)
    if int(np.count_nonzero(added)) < 3:
        return None

    if np.array_equal(connected, grid):
        return None

    compressed = _compress_columns(connected)
    if np.array_equal(compressed, connected):
        return None

    if not np.any(compressed == YELLOW):
        return None

    out = compressed.copy()
    out[out == YELLOW] = GREEN
    if np.any(out == YELLOW) or not np.any(out == GREEN):
        return None

    if np.array_equal(out, _compress_without_connect(grid)):
        return None
    if np.array_equal(out, compressed):
        return None
    if np.array_equal(out, grid):
        return None

    if not np.array_equal(main(grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-combo ARC input")

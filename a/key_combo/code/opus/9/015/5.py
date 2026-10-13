import random
from typing import Iterable

import numpy as np


# Coordinate convention follows arc_utils: grids are indexed as (r, c).
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

N_CELLS = 5
CELL_SIZE = 3
STRIDE = CELL_SIZE + 1
GRID_SIZE = N_CELLS * CELL_SIZE + (N_CELLS + 1)
CYCLE_COLORS = (BLUE, GREEN, YELLOW)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L
SIDE_CENTERS = {
    0: (0, 1),  # up
    1: (1, 2),  # right
    2: (2, 1),  # down
    3: (1, 0),  # left
}
GEN_CATEGORY_ORDER = ("outer_ring", "inner_left", "outer_zone")
GEN_CALL_COUNT = 0


def _turn_left(dir_idx: int) -> int:
    return (dir_idx - 1) % 4


def _turn_right(dir_idx: int) -> int:
    return (dir_idx + 1) % 4


def _line_positions() -> list[int]:
    return [i * STRIDE for i in range(N_CELLS + 1)]


def _make_bands(lines: Iterable[int]) -> list[tuple[int, int]]:
    lines = list(lines)
    bands = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            bands.append((a + 1, b - 1))
    return bands


def _band_index(value: int, bands: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= value <= hi:
            return idx
    raise AssertionError(f"value {value} does not belong to any band")


def _in_bounds(cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < N_CELLS and 0 <= c < N_CELLS


def _step_cell(cell: tuple[int, int], dir_idx: int) -> tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return cell[0] + dr, cell[1] + dc


def _paint_cell(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    cell: tuple[int, int],
    color: int,
) -> None:
    r0, r1 = row_bands[cell[0]]
    c0, c1 = col_bands[cell[1]]
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _is_inner(cell: tuple[int, int], inner_cells: set[tuple[int, int]]) -> bool:
    return cell in inner_cells


def _crosses_zone(
    cell: tuple[int, int],
    nxt: tuple[int, int],
    inner_cells: set[tuple[int, int]],
) -> bool:
    return _is_inner(cell, inner_cells) != _is_inner(nxt, inner_cells)


def _simulate_path(
    start_cell: tuple[int, int],
    start_dir: int,
    inner_cells: set[tuple[int, int]],
) -> dict:
    visited = [start_cell]
    seen = {start_cell}
    cell = start_cell
    dir_idx = int(start_dir)
    collisions: list[dict[str, object]] = []

    while True:
        nxt = _step_cell(cell, dir_idx)
        if nxt in seen:
            return {"visited": visited, "collisions": collisions, "stop": "painted"}

        if not _in_bounds(nxt) or _crosses_zone(cell, nxt, inner_cells):
            collision_kind = "wall" if not _in_bounds(nxt) else "zone"
            collisions.append(
                {
                    "cell": cell,
                    "dir": dir_idx,
                    "kind": collision_kind,
                    "inner": _is_inner(cell, inner_cells),
                }
            )
            dir_idx = _turn_left(dir_idx) if _is_inner(cell, inner_cells) else _turn_right(dir_idx)
            turned = _step_cell(cell, dir_idx)
            if turned in seen:
                return {"visited": visited, "collisions": collisions, "stop": "painted-after-turn"}
            if not _in_bounds(turned):
                return {"visited": visited, "collisions": collisions, "stop": "wall-after-turn"}
            if _crosses_zone(cell, turned, inner_cells):
                return {"visited": visited, "collisions": collisions, "stop": "zone-after-turn"}
            cell = turned
            visited.append(cell)
            seen.add(cell)
            continue

        cell = nxt
        visited.append(cell)
        seen.add(cell)


def _base_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BLACK, dtype=int)
    lines = _line_positions()
    grid[lines, :] = GREY
    grid[:, lines] = GREY
    top, bottom = lines[1], lines[4]
    left, right = lines[1], lines[4]
    grid[top, left : right + 1] = ORANGE
    grid[bottom, left : right + 1] = ORANGE
    grid[top : bottom + 1, left] = ORANGE
    grid[top : bottom + 1, right] = ORANGE
    row_bands = _make_bands(lines)
    col_bands = _make_bands(lines)
    cycle_row = row_bands[0][0] + 1
    left_col = col_bands[0][0]
    for offset, color in enumerate(CYCLE_COLORS):
        grid[cycle_row, left_col + offset] = int(color)
    return grid


def _render_input(start_cell: tuple[int, int], start_dir: int) -> np.ndarray:
    grid = _base_grid()
    lines = _line_positions()
    row_bands = _make_bands(lines)
    col_bands = _make_bands(lines)
    r0, _r1 = row_bands[start_cell[0]]
    c0, _c1 = col_bands[start_cell[1]]
    local_r, local_c = SIDE_CENTERS[start_dir]
    grid[r0 + local_r, c0 + local_c] = RED
    return grid


def _render_output_from_path(grid: np.ndarray, visited: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    lines = _line_positions()
    row_bands = _make_bands(lines)
    col_bands = _make_bands(lines)
    for idx, cell in enumerate(visited):
        _paint_cell(out, row_bands, col_bands, cell, CYCLE_COLORS[idx % len(CYCLE_COLORS)])
    return out


def _find_grid_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row_lines = [r for r in range(grid.shape[0]) if set(map(int, grid[r, :])) <= {GREY, ORANGE}]
    col_lines = [c for c in range(grid.shape[1]) if set(map(int, grid[:, c])) <= {GREY, ORANGE}]
    assert len(row_lines) == N_CELLS + 1, f"expected {N_CELLS + 1} horizontal grid lines"
    assert len(col_lines) == N_CELLS + 1, f"expected {N_CELLS + 1} vertical grid lines"
    return row_lines, col_lines


def _parse_cycle(grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]) -> tuple[int, ...]:
    r0, r1 = row_bands[0]
    c0, c1 = col_bands[0]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    coords = np.argwhere(np.isin(patch, CYCLE_COLORS))
    assert len(coords) == 3, "top-left cell must contain exactly three cycle markers"
    rows = {int(r) for r, _ in coords}
    cols = sorted(int(c) for _, c in coords)
    assert len(rows) == 1 and cols == [0, 1, 2], "cycle markers must form one left-to-right row"
    ordered = [int(patch[int(r), int(c)]) for r, c in sorted(coords, key=lambda rc: int(rc[1]))]
    assert set(ordered) == set(CYCLE_COLORS), "cycle markers must be blue, green, and yellow"
    return tuple(ordered)


def _parse_inner_cells(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    row_scores = [(int(np.count_nonzero(grid[r, :] == ORANGE)), r) for r in row_lines]
    col_scores = [(int(np.count_nonzero(grid[:, c] == ORANGE)), c) for c in col_lines]
    orange_rows = [r for count, r in sorted(row_scores, reverse=True)[:2] if count > 0]
    orange_cols = [c for count, c in sorted(col_scores, reverse=True)[:2] if count > 0]
    assert len(orange_rows) == 2 and len(orange_cols) == 2, "orange boundary must provide two rows and two columns"
    top, bottom = min(orange_rows), max(orange_rows)
    left, right = min(orange_cols), max(orange_cols)
    inner_rows = [i for i, (r0, r1) in enumerate(row_bands) if top < r0 and r1 < bottom]
    inner_cols = [j for j, (c0, c1) in enumerate(col_bands) if left < c0 and c1 < right]
    assert len(inner_rows) == 3 and len(inner_cols) == 3, "orange boundary must enclose a 3x3 inner zone"
    return {(r, c) for r in inner_rows for c in inner_cols}


def _parse_start(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> tuple[tuple[int, int], int]:
    starts = np.argwhere(grid == RED)
    assert len(starts) == 1, "expected exactly one red start marker"
    r, c = map(int, starts[0])
    cell = (_band_index(r, row_bands), _band_index(c, col_bands))
    r0, _r1 = row_bands[cell[0]]
    c0, _c1 = col_bands[cell[1]]
    local = (r - r0, c - c0)
    for dir_idx, marker in SIDE_CENTERS.items():
        if local == marker:
            return cell, dir_idx
    raise AssertionError("start marker must sit at the center of one cell edge")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate a color-cycling particle that turns differently in the inner and outer lattice zones.

    Input structure:
    - A 5x5 lattice of 3x3 black cells is separated by one-pixel grey grid lines.
    - An orange square is overlaid on the grid lines around the central 3x3 block of cells, marking the inner zone.
    - The top-left cell contains three one-pixel markers, blue/green/yellow from left to right, defining the paint cycle.
    - One non-template cell contains a single red pixel on the center of one cell edge, defining the start cell and initial heading.

    Rule:
    - Start in the red marker's cell and paint each visited cell's full 3x3 interior with the next cycle color; the start cell uses the first color.
    - Moving one step means moving to an orthogonally adjacent cell.
    - If the next step would leave the current zone (cross the orange boundary) or leave the 5x5 lattice, turn instead: inner-zone cells turn left, outer-zone cells turn right.
    - After such a turn, stop if the turned step is still invalid or lands on an already painted cell; also stop immediately if the forward step itself would land on an already painted cell. Grid lines stay unchanged.

    Key insight:
    - The orange partition is a live steering wall, not decoration: you must track both the particle's heading and whether its current cell is in the inner 3x3 block or the outer ring at every step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), f"expected a {GRID_SIZE}x{GRID_SIZE} grid"

    row_lines, col_lines = _find_grid_lines(g)
    row_bands = _make_bands(row_lines)
    col_bands = _make_bands(col_lines)
    assert len(row_bands) == N_CELLS and all(r1 - r0 + 1 == CELL_SIZE for r0, r1 in row_bands)
    assert len(col_bands) == N_CELLS and all(c1 - c0 + 1 == CELL_SIZE for c0, c1 in col_bands)

    cycle = _parse_cycle(g, row_bands, col_bands)
    inner_cells = _parse_inner_cells(g, row_lines, col_lines, row_bands, col_bands)
    start_cell, start_dir = _parse_start(g, row_bands, col_bands)

    sim = _simulate_path(start_cell, start_dir, inner_cells)
    out = np.array(g, copy=True)
    for idx, cell in enumerate(sim["visited"]):
        _paint_cell(out, row_bands, col_bands, cell, cycle[idx % len(cycle)])
    return out


def _build_candidates() -> dict[str, list[dict]]:
    inner_cells = {(r, c) for r in range(1, 4) for c in range(1, 4)}
    buckets: dict[str, dict[tuple[tuple[int, int], ...], dict]] = {
        "outer_ring": {},
        "inner_left": {},
        "outer_zone": {},
    }
    for cell_r in range(N_CELLS):
        for cell_c in range(N_CELLS):
            cell = (cell_r, cell_c)
            if cell == (0, 0):
                continue
            for dir_idx in range(4):
                sim = _simulate_path(cell, dir_idx, inner_cells)
                visited = tuple(sim["visited"])
                collisions = sim["collisions"]
                is_inner = cell in inner_cells
                if is_inner and len(visited) >= 6 and len(collisions) >= 3:
                    buckets["inner_left"].setdefault(
                        visited,
                        {"start_cell": cell, "start_dir": dir_idx, "sim": sim},
                    )
                if (not is_inner) and len(visited) >= 12 and len(collisions) >= 3 and not any(c["kind"] == "zone" for c in collisions):
                    buckets["outer_ring"].setdefault(
                        visited,
                        {"start_cell": cell, "start_dir": dir_idx, "sim": sim},
                    )
                if (not is_inner) and collisions and collisions[0]["kind"] == "zone" and len(visited) >= 4:
                    buckets["outer_zone"].setdefault(
                        visited,
                        {"start_cell": cell, "start_dir": dir_idx, "sim": sim},
                    )

    out = {name: list(items.values()) for name, items in buckets.items()}
    assert all(out[name] for name in GEN_CATEGORY_ORDER), "generator lost one required branch"
    return out


CANDIDATES = _build_candidates()


def generate_input() -> np.ndarray:
    global GEN_CALL_COUNT
    category = GEN_CATEGORY_ORDER[GEN_CALL_COUNT % len(GEN_CATEGORY_ORDER)]
    GEN_CALL_COUNT += 1

    candidate = random.choice(CANDIDATES[category])
    inp = _render_input(candidate["start_cell"], candidate["start_dir"])
    expected = _render_output_from_path(inp, candidate["sim"]["visited"])

    actual = main(inp)
    assert np.array_equal(actual, expected), f"oracle mismatch for category {category}"
    return inp

from __future__ import annotations

import numpy as np


BG = 0
RED = 2
GREY = 5
PALETTE = (1, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Return an orthogonal segment from start to end, excluding start and including end."""
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    if (r0, c0) == (r1, c1):
        return []
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0 + step, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(r, c0) for r in range(r0 + step, r1 + step, step)]
    raise AssertionError("segments must be horizontal or vertical")


def _cycle_from_band(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    for color in grid[0]:
        color = int(color)
        if color == BG:
            break
        colors.append(color)
    assert 2 <= len(colors) <= 4, "expected a 2-4 pixel color band in the top-left corner"
    assert len(set(colors)) == len(colors), "cycle colors must be distinct"
    return colors


def _markers(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(grid == RED)
    assert coords.shape[0] == 2, "expected exactly two red markers"
    markers = [tuple(map(int, coord)) for coord in coords]
    top = [pos for pos in markers if pos[0] == 0]
    bottom = [pos for pos in markers if pos[0] == grid.shape[0] - 1]
    assert len(top) == 1 and len(bottom) == 1, "markers must sit on the top and bottom rows"
    return top[0], bottom[0]


def _wall_rows_and_centers(grid: np.ndarray) -> tuple[list[int], list[int]]:
    wall_rows: list[int] = []
    centers: list[int] = []
    width = grid.shape[1]
    for r in range(grid.shape[0]):
        row = grid[r]
        if GREY not in row:
            continue
        zero_cols = [c for c, value in enumerate(row) if int(value) == BG]
        grey_cols = [c for c, value in enumerate(row) if int(value) == GREY]
        assert len(zero_cols) == 3, "each wall must contain one 3-cell doorway"
        assert len(grey_cols) == width - 3, "wall row must otherwise be solid grey"
        assert zero_cols == list(range(zero_cols[0], zero_cols[0] + 3)), "doorway must be contiguous"
        wall_rows.append(r)
        centers.append(zero_cols[1])
    assert 3 <= len(wall_rows) <= 5, "expected 3-5 horizontal walls"
    assert wall_rows == sorted(wall_rows), "wall rows must be ordered top-to-bottom"
    assert len(set(centers)) == len(centers), "door centers must be in distinct columns"
    return wall_rows, centers


def _trace_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    wall_rows: list[int],
    door_cols: list[int],
) -> list[tuple[int, int]]:
    height, _width = map(int, shape)
    assert start[0] == 0 and end[0] == height - 1
    assert len(wall_rows) == len(door_cols)
    assert wall_rows[0] >= 2
    assert wall_rows[-1] <= height - 3
    assert all(b - a >= 2 for a, b in zip(wall_rows, wall_rows[1:]))

    path = [start]
    cur = start

    first_corridor_row = 1
    path.extend(_line_cells(cur, (first_corridor_row, cur[1])))
    cur = (first_corridor_row, cur[1])
    path.extend(_line_cells(cur, (cur[0], door_cols[0])))
    cur = (cur[0], door_cols[0])
    path.extend(_line_cells(cur, (wall_rows[0], cur[1])))
    cur = (wall_rows[0], cur[1])

    for next_wall_row, next_door_col in zip(wall_rows[1:], door_cols[1:]):
        corridor_row = cur[0] + 1
        path.extend(_line_cells(cur, (corridor_row, cur[1])))
        cur = (corridor_row, cur[1])
        path.extend(_line_cells(cur, (cur[0], next_door_col)))
        cur = (cur[0], next_door_col)
        path.extend(_line_cells(cur, (next_wall_row, cur[1])))
        cur = (next_wall_row, cur[1])

    final_corridor_row = cur[0] + 1
    path.extend(_line_cells(cur, (final_corridor_row, cur[1])))
    cur = (final_corridor_row, cur[1])
    path.extend(_line_cells(cur, (cur[0], end[1])))
    cur = (cur[0], end[1])
    path.extend(_line_cells(cur, end))

    assert path[-1] == end
    assert len(path) == len(set(path)), "constructed path should not self-intersect"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the doorway-threading path so each wall crossing advances one step in a visible color cycle.

    Input structure:
    - A black grid contains 3-5 full-width horizontal grey walls, each with one
      3-cell doorway cut out as black background.
    - One red marker sits on the top row and one red marker sits on the bottom
      row; a short left-to-right color band in the top-left corner defines the
      cycle colors.

    Rule:
    - Read the top-left color band from left to right as a repeating cycle.
    - Draw the orthogonal route from the top marker to the bottom marker by
      entering the first corridor, aligning horizontally to each doorway center,
      and crossing the doorway centers in top-to-bottom wall order.
    - Color route cells by the number of walls already crossed: before the first
      doorway use cycle color 0, each doorway center starts the next segment,
      and the red endpoint markers remain unchanged.

    Key insight:
    - The geometry comes from the ordered doorway centers, while the visible
      color changes are driven by the cumulative wall-crossing count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cycle = _cycle_from_band(g)
    start, end = _markers(g)
    wall_rows, door_cols = _wall_rows_and_centers(g)
    path = _trace_path(g.shape, start, end, wall_rows, door_cols)
    door_centers = {(r, c) for r, c in zip(wall_rows, door_cols)}

    out = g.copy()
    crossed = 0
    for cell in path[1:]:
        if cell in door_centers:
            crossed += 1
        if cell == end:
            continue
        out[cell] = cycle[crossed % len(cycle)]

    out[start] = RED
    out[end] = RED
    return out


def _sample_spaced_positions(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    if not values:
        return None
    compressed = len(values) - (count - 1) * (min_gap - 1)
    if compressed < count:
        return None
    bases = sorted(np.random.choice(np.arange(compressed), size=count, replace=False).tolist())
    return [lo + base + i * (min_gap - 1) for i, base in enumerate(bases)]


def _generate_attempt() -> np.ndarray | None:
    height = int(np.random.randint(15, 25))
    width = int(np.random.randint(11, 19))

    cycle_len = int(np.random.randint(2, 5))
    cycle = np.random.choice(np.array(PALETTE, dtype=int), size=cycle_len, replace=False).tolist()

    wall_capacity = ((height - 4) - 3) // 2 + 1
    max_walls = min(5, wall_capacity)
    if max_walls < 3:
        return None
    wall_count = int(np.random.randint(3, max_walls + 1))
    wall_rows = _sample_spaced_positions(3, height - 4, wall_count, min_gap=2)
    if wall_rows is None:
        return None

    if width - 2 < wall_count:
        return None
    door_cols = sorted(np.random.choice(np.arange(1, width - 1), size=wall_count, replace=False).tolist())
    np.random.shuffle(door_cols)

    start_choices = [c for c in range(cycle_len + 2, width) if c not in door_cols]
    if not start_choices:
        return None
    start_col = int(np.random.choice(start_choices))

    end_choices = [c for c in range(width) if c != start_col and c not in door_cols]
    if not end_choices:
        return None
    end_col = int(np.random.choice(end_choices))

    grid = np.zeros((height, width), dtype=int)
    grid[0, :cycle_len] = np.array(cycle, dtype=int)
    grid[0, start_col] = RED
    grid[height - 1, end_col] = RED

    for r, center in zip(wall_rows, door_cols):
        grid[r, :] = GREY
        grid[r, center - 1 : center + 2] = BG

    path = _trace_path(grid.shape, (0, start_col), (height - 1, end_col), wall_rows, door_cols)
    if any(grid[r, c] not in (BG, RED) for r, c in path if (r, c) not in {(0, start_col), (height - 1, end_col)}):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    door_centers = [(r, c) for r, c in zip(wall_rows, door_cols)]
    colors_at_doors = [int(out[r, c]) for r, c in door_centers]
    expected_at_doors = [cycle[(i + 1) % cycle_len] for i in range(wall_count)]
    if colors_at_doors != expected_at_doors:
        return None

    if out[0, start_col] != RED or out[height - 1, end_col] != RED:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid segmented-wall-path input")

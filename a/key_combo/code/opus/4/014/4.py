import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7
TEAL = 8

LEGEND_COLORS = (BLUE, YELLOW, PINK, ORANGE)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

MIN_GRID_SIZE = 14
MAX_GRID_SIZE = 18
MIN_SEGMENT_LEN = 2
MAX_HORIZONTAL_LEN = 5
MAX_VERTICAL_LEN = 6
GENERATION_RETRIES = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _turn_options(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(-dc, dr), (dc, -dr)]


def _single_position(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return tuple(map(int, coords[0]))


def _read_legend(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in np.flatnonzero(grid[:, 0] != BLACK)]
    assert 2 <= len(rows) <= 4, "expected a 2-4 pixel legend in the leftmost column"
    assert rows == list(range(rows[0], rows[0] + len(rows))), "legend cells must be contiguous"

    legend = [int(grid[r, 0]) for r in rows]
    assert len(set(legend)) == len(legend), "legend colors must be distinct"
    assert set(legend) <= set(LEGEND_COLORS), "legend must use the allowed palette"
    return legend, rows


def _trace_from_direction(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict | None:
    h, w = grid.shape
    pos = start
    path: list[tuple[int, int]] = []
    segment_ids: list[int] = []
    segment_index = 0
    steps_in_segment = 0
    seen: set[tuple[int, int, int, int]] = set()

    for _ in range(h * w * 4):
        state = (pos[0], pos[1], direction[0], direction[1])
        if state in seen:
            return None
        seen.add(state)

        nr = pos[0] + direction[0]
        nc = pos[1] + direction[1]
        blocked = not in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == TEAL
        if blocked:
            if steps_in_segment == 0:
                return None
            options = []
            for nd in _turn_options(direction):
                tr = pos[0] + nd[0]
                tc = pos[1] + nd[1]
                if not in_bounds(grid.shape, tr, tc):
                    continue
                cell = int(grid[tr, tc])
                if cell in (BLACK, RED):
                    options.append(nd)
            if len(options) != 1:
                return None
            direction = options[0]
            segment_index += 1
            steps_in_segment = 0
            continue

        cell = int(grid[nr, nc])
        if cell == RED:
            if steps_in_segment == 0:
                return None
            return {
                "path": path,
                "segment_ids": segment_ids,
                "segment_count": segment_index + 1,
                "initial_dir": direction if not path else None,
            }
        if cell != BLACK:
            return None

        pos = (nr, nc)
        path.append(pos)
        segment_ids.append(segment_index)
        steps_in_segment += 1

    return None


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend, legend_rows = _read_legend(g)
    start = _single_position(g, GREEN)
    end = _single_position(g, RED)

    candidates = []
    for direction in DIRS:
        traced = _trace_from_direction(g, start, direction)
        if traced is None:
            continue
        traced["initial_dir"] = direction
        candidates.append(traced)

    assert len(candidates) == 1, "expected exactly one valid initial direction"
    info = candidates[0]
    info.update({"legend": legend, "legend_rows": legend_rows, "start": start, "end": end})
    assert info["segment_count"] >= len(legend) + 1, "path must cycle through the legend"
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the forced turning path and recolor each straight segment by cycling through the left-edge legend.

    Input structure:
    - A black background grid contains a single green start pixel, a single red end pixel, and several teal obstacle pixels.
    - The leftmost column holds a contiguous 2-4 cell legend made from distinct colors chosen from blue, yellow, pink, and orange.
    - All non-legend path cells are black in the input, and the path never enters the legend column.

    Rule:
    - Read the legend from top to bottom as an ordered color cycle.
    - Starting from the green pixel, follow the unique valid initial direction and move straight through black cells.
    - Whenever the next step is blocked by a teal pixel or the grid boundary, turn 90 degrees in the only legal direction, and stop when the next step would enter the red end pixel.
    - Color every traversed black path pixel by its straight-segment index modulo the legend length, while keeping the legend, start, end, and obstacles unchanged.

    Key insight:
    - A path pixel's color is determined by the cumulative number of turns taken before reaching it, so the solution must recover the whole forced path before any recoloring can be assigned.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    legend = info["legend"]

    out = g.copy()
    for (r, c), segment_id in zip(info["path"], info["segment_ids"]):
        out[r, c] = legend[segment_id % len(legend)]
    return out


def _direction_sequence(start_at_top: bool, horizontal_first: bool, n_segments: int) -> list[tuple[int, int]]:
    inward = DOWN if start_at_top else UP
    outward = UP if start_at_top else DOWN
    dirs: list[tuple[int, int]] = []
    vertical_index = 0

    for idx in range(n_segments):
        wants_horizontal = (idx % 2 == 0) if horizontal_first else (idx % 2 == 1)
        if wants_horizontal:
            dirs.append(LEFT)
        else:
            dirs.append(inward if vertical_index % 2 == 0 else outward)
            vertical_index += 1
    return dirs


def _choose_lengths(
    h: int,
    w: int,
    start: tuple[int, int],
    dirs: list[tuple[int, int]],
) -> list[int] | None:
    r, c = start
    lengths: list[int] = []
    final_dir = dirs[-1]

    for idx, direction in enumerate(dirs):
        if direction == LEFT:
            future_left = sum(1 for d in dirs[idx + 1 :] if d == LEFT)
            terminal_min_col = 3 if final_dir == LEFT else 2
            max_len = c - (2 * future_left + terminal_min_col)
            max_len = min(max_len, MAX_HORIZONTAL_LEN)
        elif direction == DOWN:
            boundary_margin = 1 if idx == len(dirs) - 1 else 0
            max_len = (h - 1 - boundary_margin) - r
            max_len = min(max_len, MAX_VERTICAL_LEN)
        else:
            boundary_margin = 1 if idx == len(dirs) - 1 else 0
            max_len = r - boundary_margin
            max_len = min(max_len, MAX_VERTICAL_LEN)

        if max_len < MIN_SEGMENT_LEN:
            return None

        length = random.randint(MIN_SEGMENT_LEN, max_len)
        lengths.append(length)
        r += direction[0] * length
        c += direction[1] * length

    red_r = r + final_dir[0]
    red_c = c + final_dir[1]
    if not in_bounds((h, w), red_r, red_c) or red_c < 2:
        return None
    return lengths


def _other_turn(
    direction: tuple[int, int],
    chosen: tuple[int, int],
) -> tuple[int, int]:
    options = _turn_options(direction)
    return options[1] if options[0] == chosen else options[0]


def _build_blueprint(
    h: int,
    w: int,
    legend: list[int],
    legend_top: int,
    start_at_top: bool,
    horizontal_first: bool,
    n_segments: int,
) -> dict | None:
    start = (0, w - 1) if start_at_top else (h - 1, w - 1)
    dirs = _direction_sequence(start_at_top, horizontal_first, n_segments)
    lengths = _choose_lengths(h, w, start, dirs)
    if lengths is None:
        return None

    path: list[tuple[int, int]] = []
    segment_ids: list[int] = []
    turnpoints: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    current = start

    for segment_id, (direction, length) in enumerate(zip(dirs, lengths)):
        for _ in range(length):
            current = (current[0] + direction[0], current[1] + direction[1])
            path.append(current)
            segment_ids.append(segment_id)
        if segment_id + 1 < len(dirs):
            turnpoints.append((current, direction, dirs[segment_id + 1]))

    end = (current[0] + dirs[-1][0], current[1] + dirs[-1][1])

    occupied = {start, end, *path}
    legend_cells = {(legend_top + i, 0) for i in range(len(legend))}
    if occupied & legend_cells:
        return None

    if len(path) != len(set(path)):
        return None

    obstacles: set[tuple[int, int]] = set()

    for candidate in ((start[0] + d[0], start[1] + d[1]) for d in DIRS if d != dirs[0]):
        if not in_bounds((h, w), candidate[0], candidate[1]):
            continue
        if candidate in occupied or candidate in legend_cells:
            return None
        obstacles.add(candidate)
        break

    for pos, direction, next_direction in turnpoints:
        forward = (pos[0] + direction[0], pos[1] + direction[1])
        if in_bounds((h, w), forward[0], forward[1]):
            obstacles.add(forward)

        blocked_side = _other_turn(direction, next_direction)
        blocked_cell = (pos[0] + blocked_side[0], pos[1] + blocked_side[1])
        if in_bounds((h, w), blocked_cell[0], blocked_cell[1]):
            obstacles.add(blocked_cell)

    if obstacles & occupied or obstacles & legend_cells:
        return None

    grid = np.zeros((h, w), dtype=int)
    for idx, color in enumerate(legend):
        grid[legend_top + idx, 0] = color
    for r, c in obstacles:
        grid[r, c] = TEAL
    grid[start] = GREEN
    grid[end] = RED

    blueprint = {
        "grid": grid,
        "legend": legend,
        "start": start,
        "end": end,
        "dirs": dirs,
        "lengths": lengths,
        "path": path,
        "segment_ids": segment_ids,
        "segment_count": n_segments,
        "initial_dir": dirs[0],
    }
    return blueprint


def _validate_blueprint(blueprint: dict) -> bool:
    grid = blueprint["grid"]
    analyzed = _analyze_grid(grid)
    if analyzed["legend"] != blueprint["legend"]:
        return False
    if analyzed["start"] != blueprint["start"] or analyzed["end"] != blueprint["end"]:
        return False
    if analyzed["initial_dir"] != blueprint["initial_dir"]:
        return False
    if analyzed["path"] != blueprint["path"]:
        return False
    if analyzed["segment_ids"] != blueprint["segment_ids"]:
        return False
    if analyzed["segment_count"] != blueprint["segment_count"]:
        return False

    legend = blueprint["legend"]
    expected = grid.copy()
    for (r, c), segment_id in zip(blueprint["path"], blueprint["segment_ids"]):
        expected[r, c] = legend[segment_id % len(legend)]

    if not np.array_equal(main(grid), expected):
        return False
    if blueprint["segment_count"] <= len(legend):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)

    legend_len = random.randint(2, 4)
    legend = random.sample(list(LEGEND_COLORS), k=legend_len)
    legend_top = random.randint(0, h - legend_len)

    start_at_top = random.choice([True, False])
    horizontal_first = random.choice([True, False])
    min_segments = max(legend_len + 1, 4)
    n_segments = random.randint(min_segments, min_segments + 3)

    blueprint = _build_blueprint(
        h=h,
        w=w,
        legend=legend,
        legend_top=legend_top,
        start_at_top=start_at_top,
        horizontal_first=horizontal_first,
        n_segments=n_segments,
    )
    if blueprint is None:
        return None
    if not _validate_blueprint(blueprint):
        return None
    return blueprint["grid"]


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid turning-path legend sample")

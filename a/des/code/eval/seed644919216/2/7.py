import random
from typing import Optional

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TERMINAL_COLORS = (6, 7, 8, 9)

NODE_COLORS = {RED, BLUE, GREEN}
RESERVED_COLORS = {BLACK, BLUE, RED, GREEN, YELLOW, GREY}
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_INDEX = {direction: idx for idx, direction in enumerate(DIRS)}

MIN_PATHS = 3
MAX_PATHS = 4
MIN_WIDTH = 14
MAX_WIDTH = 24
BAND_PITCH = 4
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _on_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _step(pos: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return pos[0] + direction[0], pos[1] + direction[1]


def _turn(direction: tuple[int, int], node_color: int) -> tuple[int, int]:
    idx = DIR_TO_INDEX[direction]
    if node_color == RED:
        return DIRS[(idx + 1) % 4]
    if node_color == BLUE:
        return DIRS[(idx - 1) % 4]
    if node_color == GREEN:
        return direction
    raise ValueError(f"invalid node color: {node_color}")


def _turn_color(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> int:
    delta = (DIR_TO_INDEX[out_dir] - DIR_TO_INDEX[in_dir]) % 4
    if delta == 0:
        return GREEN
    if delta == 1:
        return RED
    if delta == 3:
        return BLUE
    raise ValueError(f"unsupported turn: {in_dir} -> {out_dir}")


def _is_terminal_color(color: int) -> bool:
    return color not in RESERVED_COLORS


def _nonblack_neighbor_dirs(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) != BLACK:
            result.append((dr, dc))
    return result


def _trace_from_source(
    grid: np.ndarray, source: tuple[int, int]
) -> tuple[bool, list[tuple[int, int]], Optional[int]]:
    sr, sc = source
    start_dirs = _nonblack_neighbor_dirs(grid, sr, sc)
    if len(start_dirs) != 1:
        return False, [], None

    direction = start_dirs[0]
    pos = _step(source, direction)
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    path: list[tuple[int, int]] = []

    while True:
        if not _in_bounds(grid.shape, pos[0], pos[1]):
            return False, [], None

        state = (pos, direction)
        if state in visited:
            return False, [], None
        visited.add(state)

        color = int(grid[pos])
        if color == BLACK or color == YELLOW:
            return False, [], None

        if color == GREY:
            path.append(pos)
            nxt = _step(pos, direction)
            if not _in_bounds(grid.shape, nxt[0], nxt[1]) or int(grid[nxt]) == BLACK:
                return False, [], None
            pos = nxt
            continue

        if color in NODE_COLORS:
            path.append(pos)
            direction = _turn(direction, color)
            nxt = _step(pos, direction)
            if not _in_bounds(grid.shape, nxt[0], nxt[1]) or int(grid[nxt]) == BLACK:
                return False, [], None
            pos = nxt
            continue

        if _is_terminal_color(color) and _on_edge(grid.shape, pos[0], pos[1]):
            return True, path, color

        return False, [], None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    sources = sorted(tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW))
    for source in sources:
        success, path, terminal_color = _trace_from_source(g, source)
        if success and terminal_color is not None:
            for r, c in path:
                out[r, c] = terminal_color
    return out


def _band_rows(index: int) -> list[int]:
    base = index * BAND_PITCH
    return [base + 1, base + 2, base + 3]


def _choose_change_columns(endpoint_x: int, change_count: int) -> Optional[list[int]]:
    cols: list[int] = []
    min_x = 3
    for idx in range(change_count):
        remaining = change_count - idx - 1
        max_x = endpoint_x - 2 - 3 * remaining
        if min_x > max_x:
            return None
        x = random.randint(min_x, max_x)
        cols.append(x)
        min_x = x + 3
    return cols


def _straight_candidates(cells: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    support = set(cells)
    for coord, color in cells.items():
        if color != GREY:
            continue
        dirs = []
        neighbor_colors = []
        for dr, dc in DIRS:
            nxt = (coord[0] + dr, coord[1] + dc)
            if nxt in support:
                dirs.append((dr, dc))
                neighbor_colors.append(cells[nxt])
        if len(dirs) != 2:
            continue
        if dirs[0][0] != -dirs[1][0] or dirs[0][1] != -dirs[1][1]:
            continue
        if any(color == YELLOW or _is_terminal_color(color) for color in neighbor_colors):
            continue
        candidates.append(coord)
    return candidates


def _select_non_adjacent(coords: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    random.shuffle(coords)
    chosen: list[tuple[int, int]] = []
    for coord in coords:
        if any(abs(coord[0] - other[0]) + abs(coord[1] - other[1]) <= 1 for other in chosen):
            continue
        chosen.append(coord)
        if len(chosen) == count:
            break
    return chosen


def _render_path(
    width: int,
    band_index: int,
    complete: bool,
    terminal_color: Optional[int],
    force_bend: bool,
    force_green: bool,
) -> Optional[dict[tuple[int, int], int]]:
    rows = _band_rows(band_index)
    start_row = random.choice(rows)
    end_x = width - 2 if complete else random.randint(max(6, width // 2), width - 4)

    max_changes = min(2, max(0, (end_x - 5) // 3))
    min_changes = 1 if force_bend else 0
    if max_changes < min_changes:
        return None

    change_count = random.randint(min_changes, max_changes)
    change_cols = _choose_change_columns(end_x, change_count)
    if change_cols is None:
        return None

    cells: dict[tuple[int, int], int] = {}
    cells[(start_row, 0)] = YELLOW

    current_row = start_row
    cursor = 1

    for x in change_cols:
        next_row = random.choice([row for row in rows if row != current_row])
        for c in range(cursor, x):
            cells[(current_row, c)] = GREY

        vertical_step = 1 if next_row > current_row else -1
        cells[(current_row, x)] = _turn_color((0, 1), (vertical_step, 0))
        for r in range(current_row + vertical_step, next_row, vertical_step):
            cells[(r, x)] = GREY
        cells[(next_row, x)] = _turn_color((vertical_step, 0), (0, 1))

        current_row = next_row
        cursor = x + 1

    for c in range(cursor, end_x + 1):
        cells[(current_row, c)] = GREY

    if complete:
        if terminal_color is None:
            return None
        cells[(current_row, width - 1)] = terminal_color

    candidates = _straight_candidates(cells)
    max_green = min(2, len(candidates))
    if max_green == 0:
        green_count = 0
    elif force_green:
        green_count = random.randint(1, max_green)
    else:
        green_count = random.choice([0, 1, 1, min(2, max_green)])
        green_count = min(green_count, max_green)

    if force_green and green_count == 0:
        return None

    for coord in _select_non_adjacent(candidates, green_count):
        cells[coord] = GREEN

    return cells


def _apply_cells(grid: np.ndarray, cells: dict[tuple[int, int], int]) -> None:
    for (r, c), color in cells.items():
        if int(grid[r, c]) != BLACK:
            raise ValueError("paths should not overlap")
        grid[r, c] = color


def _validate_generated_grid(grid: np.ndarray) -> bool:
    sources = [tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)]
    if not (MIN_PATHS <= len(sources) <= MAX_PATHS):
        return False

    node_colors = {int(color) for color in np.unique(grid) if int(color) in NODE_COLORS}
    if node_colors != NODE_COLORS:
        return False

    terminals = np.argwhere(np.isin(grid, TERMINAL_COLORS))
    if len(terminals) == 0:
        return False
    for r, c in terminals:
        rr, cc = int(r), int(c)
        if not _on_edge(grid.shape, rr, cc):
            return False
        if len(_nonblack_neighbor_dirs(grid, rr, cc)) != 1:
            return False

    results = [_trace_from_source(grid, source)[0] for source in sources]
    if not any(results) or all(results):
        return False

    output = main(grid)
    if np.array_equal(output, grid):
        return False
    return True


def _gen_attempt() -> Optional[np.ndarray]:
    path_count = random.randint(MIN_PATHS, MAX_PATHS)
    height = path_count * BAND_PITCH + 1
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    grid = np.full((height, width), BLACK, dtype=int)

    complete_flags = [random.random() < 0.5 for _ in range(path_count)]
    if not any(complete_flags):
        complete_flags[random.randrange(path_count)] = True
    if all(complete_flags):
        complete_flags[random.randrange(path_count)] = False

    bend_index = random.randrange(path_count)
    green_index = random.randrange(path_count)

    complete_count = sum(complete_flags)
    terminal_colors = list(random.sample(TERMINAL_COLORS, k=complete_count))
    terminal_iter = iter(terminal_colors)

    try:
        for idx in range(path_count):
            terminal_color = next(terminal_iter) if complete_flags[idx] else None
            cells = _render_path(
                width=width,
                band_index=idx,
                complete=complete_flags[idx],
                terminal_color=terminal_color,
                force_bend=idx == bend_index,
                force_green=idx == green_index,
            )
            if cells is None:
                return None
            _apply_cells(grid, cells)
    except ValueError:
        return None

    rotation = random.randrange(4)
    grid = np.rot90(grid, rotation).copy()

    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid circuit-routing ARC input")

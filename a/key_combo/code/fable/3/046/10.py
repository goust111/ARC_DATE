from __future__ import annotations

from collections import Counter
import random

import numpy as np


# Direction order: north, east, south, west.
_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_DIRS = range(4)

# For ordinary tiles the wall-midpoint pixel is a relative instruction glyph:
# top = straight, left = turn left, right = turn right.
_TURN_BY_SYMBOL_SIDE = {0: 0, 3: -1, 1: 1}
_SYMBOL_SIDE_BY_TURN = {0: 0, -1: 3, 1: 1}


def _in_bounds(rows: int, cols: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(rows) and 0 <= int(c) < int(cols)


def _regular_line_indices(indices: list[int], limit: int) -> int | None:
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != int(limit) - 1:
        return None
    step = indices[1] - indices[0]
    if step < 4:
        return None
    if any(b - a != step for a, b in zip(indices, indices[1:])):
        return None
    interior = step - 1
    if interior < 3 or interior % 2 == 0:
        return None
    return interior


def _infer_lattice(grid: np.ndarray) -> tuple[int, list[int], list[int], int, int]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, list[int], list[int], int, int]] = []

    for color in sorted(set(map(int, grid.flatten()))):
        row_lines = [r for r in range(h) if bool(np.all(grid[r, :] == int(color)))]
        col_lines = [c for c in range(w) if bool(np.all(grid[:, c] == int(color)))]
        tile_h = _regular_line_indices(row_lines, h)
        tile_w = _regular_line_indices(col_lines, w)
        if tile_h is None or tile_w is None:
            continue
        if len(row_lines) < 3 or len(col_lines) < 3:
            continue
        candidates.append((int(color), row_lines, col_lines, tile_h, tile_w))

    assert len(candidates) == 1, "expected one regular full-grid tile frame color"
    return candidates[0]


def _tile_slice(lines: list[int], index: int) -> slice:
    return slice(lines[index] + 1, lines[index + 1])


def _pointer_positions(tile_h: int, tile_w: int) -> dict[tuple[int, int], int]:
    return {
        (0, tile_w // 2): 0,
        (tile_h // 2, tile_w - 1): 1,
        (tile_h - 1, tile_w // 2): 2,
        (tile_h // 2, 0): 3,
    }


def _parse_tiles(
    grid: np.ndarray,
) -> tuple[list[int], list[int], int, int, int, int, tuple[int, int], dict[tuple[int, int], int]]:
    _frame_color, row_lines, col_lines, tile_h, tile_w = _infer_lattice(grid)
    tile_rows = len(row_lines) - 1
    tile_cols = len(col_lines) - 1
    assert tile_rows * tile_cols >= 4, "expected enough tiles to distinguish the start color"

    interior_values: list[int] = []
    for tr in range(tile_rows):
        rs = _tile_slice(row_lines, tr)
        for tc in range(tile_cols):
            cs = _tile_slice(col_lines, tc)
            interior_values.extend(map(int, grid[rs, cs].flatten()))

    ranked = Counter(interior_values).most_common()
    assert len(ranked) >= 3, "expected background, ordinary pointer, and start colors"
    assert len(ranked) == 3, "tile interiors should contain exactly three colors"
    bg = int(ranked[0][0])
    assert ranked[0][1] > ranked[1][1], "expected a unique tile-interior background color"

    pointer_sides: dict[tuple[int, int], int] = {}
    pointer_colors: dict[tuple[int, int], int] = {}
    pos_to_side = _pointer_positions(tile_h, tile_w)

    for tr in range(tile_rows):
        rs = _tile_slice(row_lines, tr)
        for tc in range(tile_cols):
            cs = _tile_slice(col_lines, tc)
            patch = grid[rs, cs]
            coords = np.argwhere(patch != bg)
            assert coords.shape[0] == 1, "each tile interior must contain exactly one pointer pixel"
            pr, pc = map(int, coords[0])
            assert (pr, pc) in pos_to_side, "pointer must lie at a wall midpoint"
            pointer_sides[(tr, tc)] = pos_to_side[(pr, pc)]
            pointer_colors[(tr, tc)] = int(patch[pr, pc])

    color_counts = Counter(pointer_colors.values())
    assert len(color_counts) == 2, "expected one ordinary pointer color and one start/path color"
    start_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(start_colors) == 1, "the start/path color must appear on exactly one pointer"
    start_color = int(start_colors[0])
    ordinary_colors = [color for color in color_counts if color != start_color]
    assert color_counts[ordinary_colors[0]] == tile_rows * tile_cols - 1

    starts = [pos for pos, color in pointer_colors.items() if color == start_color]
    assert len(starts) == 1
    return row_lines, col_lines, tile_h, tile_w, bg, start_color, starts[0], pointer_sides


def _trace_relative_path(
    tile_rows: int,
    tile_cols: int,
    start: tuple[int, int],
    pointer_sides: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    r, c = start
    direction = pointer_sides[start]
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    while _in_bounds(tile_rows, tile_cols, r, c):
        tile = (int(r), int(c))
        assert tile not in seen, "the route must not revisit a tile"
        seen.add(tile)
        path.append(tile)

        if tile != start:
            side = pointer_sides[tile]
            assert side in _TURN_BY_SYMBOL_SIDE, "ordinary pointers must be straight/left/right symbols"
            direction = (direction + _TURN_BY_SYMBOL_SIDE[side]) % 4

        dr, dc = _DELTAS[direction]
        r += dr
        c += dc

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint the lattice-tile route commanded by one absolute start pointer and later relative turn pointers.

    Input structure:
    - A one-pixel frame color divides the grid into a regular rectangular array of equal odd-sized tile interiors.
    - Each tile interior is otherwise background and contains exactly one pointer pixel at a wall midpoint.
    - Exactly one pointer has the path/start color; all other pointers share one ordinary instruction color.

    Rule:
    - The special start pointer is read as an absolute initial direction and the route first steps that way.
    - In every later tile, the ordinary pointer is a relative turn symbol: front/left/right means
      straight/turn-left/turn-right from the current heading, then the route steps into that neighbor.
    - Fill every visited tile interior completely with the start/path color; leave unvisited tiles and all frame lines unchanged.

    Key insight:
    - Only the start pointer is compass-like.  The repeated ordinary pointers are relative turn commands,
      so interpreting all pointers as absolute directions sends the route to the wrong tiles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_lines, col_lines, tile_h, tile_w, _bg, path_color, start, pointer_sides = _parse_tiles(g)
    tile_rows = len(row_lines) - 1
    tile_cols = len(col_lines) - 1
    path = _trace_relative_path(tile_rows, tile_cols, start, pointer_sides)

    out = g.copy()
    for tr, tc in path:
        rs = _tile_slice(row_lines, tr)
        cs = _tile_slice(col_lines, tc)
        out[rs, cs] = int(path_color)
    return out


# --------------- generator ---------------

_GEN_COLORS = tuple(range(10))
_GEN_INTERIOR_SIZES = (3, 5)
_GEN_TILE_COUNT_RANGE = (3, 6)
_GEN_RETRIES = 1000
_GEN_MIN_PATH = 7


def _gen_lattice_shape() -> tuple[int, int, int, int]:
    options: list[tuple[int, int, int, int]] = []
    for tile_rows in range(_GEN_TILE_COUNT_RANGE[0], _GEN_TILE_COUNT_RANGE[1] + 1):
        for tile_cols in range(_GEN_TILE_COUNT_RANGE[0], _GEN_TILE_COUNT_RANGE[1] + 1):
            for tile_h in _GEN_INTERIOR_SIZES:
                for tile_w in _GEN_INTERIOR_SIZES:
                    if tile_rows * (tile_h + 1) + 1 <= 30 and tile_cols * (tile_w + 1) + 1 <= 30:
                        if tile_rows * tile_cols >= _GEN_MIN_PATH + 3:
                            options.append((tile_rows, tile_cols, tile_h, tile_w))
    return random.choice(options)


def _gen_walk(tile_rows: int, tile_cols: int) -> tuple[tuple[int, int], int, dict[tuple[int, int], int], list[tuple[int, int]]] | None:
    total = tile_rows * tile_cols
    max_path = max(_GEN_MIN_PATH, total - 3)

    start = (random.randrange(tile_rows), random.randrange(tile_cols))
    initial_dirs = list(_DIRS)
    random.shuffle(initial_dirs)

    for initial_dir in initial_dirs:
        sr, sc = start
        dr, dc = _DELTAS[initial_dir]
        first = (sr + dr, sc + dc)
        if not _in_bounds(tile_rows, tile_cols, *first):
            continue

        direction = initial_dir
        path = [start]
        seen = {start}
        current = first
        turns: dict[tuple[int, int], int] = {}

        for _ in range(total + 3):
            if not _in_bounds(tile_rows, tile_cols, *current):
                break
            if current in seen:
                return None

            path.append(current)
            seen.add(current)
            choices: list[tuple[int, int, tuple[int, int], bool]] = []
            for turn in (-1, 0, 1):
                ndir = (direction + turn) % 4
                rr = current[0] + _DELTAS[ndir][0]
                cc = current[1] + _DELTAS[ndir][1]
                outside = not _in_bounds(tile_rows, tile_cols, rr, cc)
                if outside:
                    if len(path) >= _GEN_MIN_PATH:
                        choices.append((turn, ndir, (rr, cc), True))
                elif (rr, cc) not in seen and len(path) < max_path:
                    choices.append((turn, ndir, (rr, cc), False))

            if not choices:
                return None
            if len(path) >= max_path:
                exits = [choice for choice in choices if choice[3]]
                if not exits:
                    return None
                choices = exits
            elif len(path) >= _GEN_MIN_PATH and random.random() < 0.28:
                exits = [choice for choice in choices if choice[3]]
                if exits:
                    choices = exits

            turn, direction, nxt, outside = random.choice(choices)
            turns[current] = turn
            current = nxt
            if outside:
                break
        else:
            return None

        turn_values = set(turns.values())
        if len(path) < _GEN_MIN_PATH:
            return None
        if total - len(path) < 3:
            return None
        if not {-1, 0, 1} <= turn_values:
            return None
        return start, initial_dir, turns, path

    return None


def _gen_tile_slices(row_lines: list[int], col_lines: list[int], tr: int, tc: int) -> tuple[slice, slice]:
    return _tile_slice(row_lines, tr), _tile_slice(col_lines, tc)


def _render_scene(
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    start: tuple[int, int],
    initial_dir: int,
    turns: dict[tuple[int, int], int],
    path: list[tuple[int, int]],
    bg: int,
    frame: int,
    ordinary: int,
    special: int,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    height = tile_rows * (tile_h + 1) + 1
    width = tile_cols * (tile_w + 1) + 1
    grid = np.full((height, width), int(bg), dtype=int)
    row_lines = [i * (tile_h + 1) for i in range(tile_rows + 1)]
    col_lines = [i * (tile_w + 1) for i in range(tile_cols + 1)]

    for r in row_lines:
        grid[r, :] = int(frame)
    for c in col_lines:
        grid[:, c] = int(frame)

    pointer_sides: dict[tuple[int, int], int] = {}
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            pointer_sides[(tr, tc)] = random.choice(tuple(_TURN_BY_SYMBOL_SIDE))

    pointer_sides[start] = int(initial_dir)
    for tile, turn in turns.items():
        pointer_sides[tile] = _SYMBOL_SIDE_BY_TURN[int(turn)]

    local_pos = {side: pos for pos, side in _pointer_positions(tile_h, tile_w).items()}
    for tr in range(tile_rows):
        rs = _tile_slice(row_lines, tr)
        for tc in range(tile_cols):
            color = special if (tr, tc) == start else ordinary
            lr, lc = local_pos[pointer_sides[(tr, tc)]]
            grid[rs.start + lr, _tile_slice(col_lines, tc).start + lc] = int(color)

    expected = grid.copy()
    for tr, tc in path:
        rs, cs = _gen_tile_slices(row_lines, col_lines, tr, tc)
        expected[rs, cs] = int(special)
    return grid, expected, pointer_sides


def _absolute_wall_path(
    tile_rows: int,
    tile_cols: int,
    start: tuple[int, int],
    pointer_sides: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    r, c = start
    while _in_bounds(tile_rows, tile_cols, r, c):
        tile = (int(r), int(c))
        if tile in seen:
            break
        seen.add(tile)
        path.append(tile)
        direction = pointer_sides[tile]
        dr, dc = _DELTAS[direction]
        r += dr
        c += dc
    return path


def _painted_from_path(
    grid: np.ndarray,
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    path: list[tuple[int, int]],
    color: int,
) -> np.ndarray:
    out = grid.copy()
    row_lines = [i * (tile_h + 1) for i in range(tile_rows + 1)]
    col_lines = [i * (tile_w + 1) for i in range(tile_cols + 1)]
    for tr, tc in path:
        out[_tile_slice(row_lines, tr), _tile_slice(col_lines, tc)] = int(color)
    return out


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    start: tuple[int, int],
    pointer_sides: dict[tuple[int, int], int],
    special: int,
) -> bool:
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(set(map(int, grid.flatten()))) != 4:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    relative_path = _trace_relative_path(tile_rows, tile_cols, start, pointer_sides)
    absolute_path = _absolute_wall_path(tile_rows, tile_cols, start, pointer_sides)
    if set(relative_path) == set(absolute_path):
        return False
    absolute_out = _painted_from_path(grid, tile_rows, tile_cols, tile_h, tile_w, absolute_path, special)
    if np.array_equal(absolute_out, expected):
        return False
    if len(relative_path) < _GEN_MIN_PATH or tile_rows * tile_cols - len(relative_path) < 3:
        return False

    ordinary_turns = []
    for tile in relative_path[1:]:
        side = pointer_sides[tile]
        if side not in _TURN_BY_SYMBOL_SIDE:
            return False
        ordinary_turns.append(_TURN_BY_SYMBOL_SIDE[side])
    return {-1, 0, 1} <= set(ordinary_turns)


def _gen_attempt() -> np.ndarray | None:
    tile_rows, tile_cols, tile_h, tile_w = _gen_lattice_shape()
    walk = _gen_walk(tile_rows, tile_cols)
    if walk is None:
        return None
    start, initial_dir, turns, path = walk

    bg, frame, ordinary, special = random.sample(_GEN_COLORS, 4)
    grid, expected, pointer_sides = _render_scene(
        tile_rows,
        tile_cols,
        tile_h,
        tile_w,
        start,
        initial_dir,
        turns,
        path,
        bg,
        frame,
        ordinary,
        special,
    )

    if not _semantic_ok(
        grid,
        expected,
        tile_rows,
        tile_cols,
        tile_h,
        tile_w,
        start,
        pointer_sides,
        special,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid turning-cruise input")

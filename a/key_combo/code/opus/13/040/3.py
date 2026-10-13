from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREEN = 3
GREY = 5
PINK = 6
TEAL = 8

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_CONTROLS = {PINK, TEAL}

TOTAL_ROWS_RANGE = (14, 19)
TOP_ROWS_RANGE = (9, 13)
BOTTOM_ROWS_RANGE = (3, 5)
WIDTH_RANGE = (16, 24)
TURN_COUNT_RANGE = (3, 6)
SEGMENT_LEN_RANGE = (1, 5)
GEN_ATTEMPTS = 400


def _turn(direction: tuple[int, int], control_color: int) -> tuple[int, int]:
    idx = DIRS.index(direction)
    if control_color == PINK:
        return DIRS[(idx + 1) % 4]
    if control_color == TEAL:
        return DIRS[(idx - 1) % 4]
    raise AssertionError("unexpected control color")


def _add(a: tuple[int, int], b: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * scale, a[1] + b[1] * scale


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _in_top(top_rows: int, width: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < top_rows and 0 <= c < width


def _distance_to_exit(pos: tuple[int, int], direction: tuple[int, int], top_rows: int, width: int) -> int:
    r, c = pos
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return top_rows - 1 - r
    if dc < 0:
        return c
    return width - 1 - c


def _ray(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [_add(start, direction, step) for step in range(1, length + 1)]


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected one full-width separator row"
    return int(rows[0])


def _visible_control_directions(grid: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    sr, sc = start
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w:
            value = int(grid[r, c])
            if value != BG:
                if value in TURN_CONTROLS:
                    candidates.append((dr, dc))
                break
            r += dr
            c += dc
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the top-region ray and convert its per-column green-pixel density into a bottom histogram.

    Input structure:
    - A black grid contains one full-width gray separator row that splits an upper path area from a lower empty area.
    - The upper area contains exactly one green start pixel plus single-pixel pink and teal turn controls on black background.
    - Exactly one cardinal sightline from the start first meets a turn control; the lower area starts black.

    Rule:
    - Move from the green start toward that unique visible control, painting visited black cells green.
    - If the next cell is pink, turn right around that control; if it is teal, turn left around that control. Do not overwrite the control, and continue from the adjacent cell on the turned side of the control.
    - Stop when the next move would leave the upper area by crossing the separator or the outer grid boundary.
    - Count green pixels in each column of the upper area, then fill the lower area downward from just below the separator with a green bar of that column height, clipped by the available lower rows.

    Key insight:
    - The lower output is not a copy of the ray. The ray's geometry only matters through the columnwise counts of its green path pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    top = g[:sep]
    bottom_rows = g.shape[0] - sep - 1
    assert bottom_rows >= 1, "expected rows below the separator"

    starts = np.argwhere(top == GREEN)
    assert len(starts) == 1, "expected exactly one green start pixel in the upper area"
    start = tuple(map(int, starts[0]))

    allowed = {BG, GREEN, GREY, PINK, TEAL}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors present"
    assert np.all(g[sep + 1 :] != GREY), "separator color must be confined to the separator row"

    candidates = _visible_control_directions(g, start)
    assert len(candidates) == 1, "expected one visible control direction from the start"

    out = g.copy()
    pos = start
    direction = candidates[0]
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (pos[0], pos[1], direction[0], direction[1])
        assert state not in seen_states, "path must terminate instead of cycling"
        seen_states.add(state)

        nxt = _add(pos, direction)
        if not _in_top(sep, g.shape[1], nxt):
            break

        value = int(out[nxt])
        if value == BG:
            out[nxt] = GREEN
            pos = nxt
            continue

        if value in TURN_CONTROLS:
            direction = _turn(direction, value)
            turned = _add(nxt, direction)
            if not _in_top(sep, g.shape[1], turned):
                break
            assert int(out[turned]) == BG, "turned-to cell must be empty background"
            out[turned] = GREEN
            pos = turned
            continue

        raise AssertionError("path encountered an unexpected non-background cell")

    column_heights = np.count_nonzero(out[:sep] == GREEN, axis=0)
    for c, height in enumerate(column_heights):
        fill = min(int(height), bottom_rows)
        if fill > 0:
            out[sep + 1 : sep + 1 + fill, c] = GREEN
    return out


def _render_input(
    top_rows: int,
    width: int,
    bottom_rows: int,
    start: tuple[int, int],
    controls: Iterable[tuple[tuple[int, int], int]],
) -> np.ndarray:
    grid = np.zeros((top_rows + 1 + bottom_rows, width), dtype=int)
    grid[top_rows, :] = GREY
    grid[start] = GREEN
    for (r, c), color in controls:
        grid[r, c] = color
    return grid


def _render_expected(
    inp: np.ndarray,
    top_rows: int,
    path_cells: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in path_cells:
        out[r, c] = GREEN
    counts = np.zeros(inp.shape[1], dtype=int)
    for _r, c in path_cells:
        counts[c] += 1
    bottom_rows = inp.shape[0] - top_rows - 1
    for c, height in enumerate(counts):
        fill = min(int(height), bottom_rows)
        if fill > 0:
            out[top_rows + 1 : top_rows + 1 + fill, c] = GREEN
    return out


def _sample_turn_colors(turn_count: int) -> list[int]:
    if random.random() < 0.7:
        colors = [PINK, TEAL]
        while len(colors) < turn_count:
            colors.append(random.choice((PINK, TEAL)))
        random.shuffle(colors)
        return colors
    return [random.choice((PINK, TEAL)) for _ in range(turn_count)]


def _build_path_plan(
    top_rows: int,
    width: int,
    start: tuple[int, int],
    initial_direction: tuple[int, int],
    control_colors: list[int],
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], int]], list[tuple[int, int]]] | None:
    occupied_path: set[tuple[int, int]] = {start}
    occupied_controls: set[tuple[int, int]] = set()

    def search(
        idx: int,
        pos: tuple[int, int],
        direction: tuple[int, int],
        path_cells: set[tuple[int, int]],
        controls: list[tuple[tuple[int, int], int]],
        directions_used: list[tuple[int, int]],
    ) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], int]], list[tuple[int, int]]] | None:
        occupied = path_cells | {cell for cell, _ in controls}
        if idx == len(control_colors):
            final_len = _distance_to_exit(pos, direction, top_rows, width)
            if final_len < 1:
                return None
            final_cells = _ray(pos, direction, final_len)
            if any(cell in occupied for cell in final_cells):
                return None
            return path_cells | set(final_cells), controls, directions_used + [direction]

        max_straight = min(SEGMENT_LEN_RANGE[1], _distance_to_exit(pos, direction, top_rows, width) - 1)
        if max_straight < SEGMENT_LEN_RANGE[0]:
            return None

        lengths = list(range(SEGMENT_LEN_RANGE[0], max_straight + 1))
        random.shuffle(lengths)
        control_color = control_colors[idx]

        for seg_len in lengths:
            straight = _ray(pos, direction, seg_len)
            control = _add(pos, direction, seg_len + 1)
            new_direction = _turn(direction, control_color)
            corner = _add(control, new_direction)

            if not _in_top(top_rows, width, control) or not _in_top(top_rows, width, corner):
                continue
            if any(cell in occupied for cell in straight):
                continue
            if control in occupied or corner in occupied:
                continue

            result = search(
                idx + 1,
                corner,
                new_direction,
                path_cells | set(straight) | {corner},
                controls + [(control, control_color)],
                directions_used + [direction],
            )
            if result is not None:
                return result
        return None

    return search(0, start, initial_direction, occupied_path, [], [])


def _column_heights(path_cells: Iterable[tuple[int, int]], width: int) -> np.ndarray:
    counts = np.zeros(width, dtype=int)
    for _r, c in path_cells:
        counts[c] += 1
    return counts


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    top_rows: int,
    start: tuple[int, int],
    controls: list[tuple[tuple[int, int], int]],
    path_cells: set[tuple[int, int]],
    directions_used: list[tuple[int, int]],
    want_clipped: bool,
) -> bool:
    if len(_visible_control_directions(inp, start)) != 1:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(solved, expected):
        return False

    if len(controls) < 3:
        return False

    if len(set(directions_used)) < 2:
        return False

    if not any(color == PINK for _cell, color in controls):
        return False
    if not any(color == TEAL for _cell, color in controls):
        return False

    counts = _column_heights(path_cells, inp.shape[1])
    touched = np.flatnonzero(counts > 0)
    if len(touched) == 0:
        return False
    if int(touched[-1] - touched[0] + 1) < (inp.shape[1] + 1) // 2:
        return False
    if int(counts.max()) < 3:
        return False
    if len(set(map(int, counts[counts > 0]))) < 2:
        return False
    if np.count_nonzero(counts == 0) < 2:
        return False

    bottom_rows = inp.shape[0] - top_rows - 1
    clipped = counts > bottom_rows
    if bool(np.any(clipped)) != want_clipped:
        return False

    bottom = expected[top_rows + 1 :]
    for c, height in enumerate(counts):
        fill = min(int(height), bottom_rows)
        if fill > 0:
            if not np.all(bottom[:fill, c] == GREEN):
                return False
        if fill < bottom_rows:
            if not np.all(bottom[fill:, c] == BG):
                return False

    if np.count_nonzero(expected[:top_rows] == GREEN) != len(path_cells):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    top_rows = random.randint(*TOP_ROWS_RANGE)
    bottom_rows = random.randint(*BOTTOM_ROWS_RANGE)
    total_rows = top_rows + 1 + bottom_rows
    if not (TOTAL_ROWS_RANGE[0] <= total_rows <= TOTAL_ROWS_RANGE[1]):
        return None

    want_clipped = random.random() < 0.5
    turn_count = random.randint(*TURN_COUNT_RANGE)
    control_colors = _sample_turn_colors(turn_count)
    initial_direction = random.choice(DIRS)

    row_candidates = list(range(1, top_rows - 1))
    col_candidates = list(range(1, width - 1))
    random.shuffle(row_candidates)
    random.shuffle(col_candidates)

    starts = [(r, c) for r in row_candidates for c in col_candidates]
    random.shuffle(starts)
    for start in starts[:80]:
        built = _build_path_plan(top_rows, width, start, initial_direction, control_colors)
        if built is None:
            continue
        path_cells, controls, directions_used = built
        inp = _render_input(top_rows, width, bottom_rows, start, controls)
        expected = _render_expected(inp, top_rows, path_cells)
        if _validate_semantics(
            inp,
            expected,
            top_rows,
            start,
            controls,
            path_cells,
            directions_used,
            want_clipped,
        ):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-density histogram input")

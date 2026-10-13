import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

COLOR_ORDER = [RED, BLUE, GREEN, YELLOW, ORANGE, MAGENTA, AZURE, MAROON]
PRIORITY = {color: index for index, color in enumerate(COLOR_ORDER)}
PATH_COLORS = tuple(COLOR_ORDER)


def _is_path(value: int) -> bool:
    return int(value) not in (BLACK, GREY)


def _priority(color: int) -> int:
    return PRIORITY[int(color)]


def _pick_color(rng: random.Random, avoid: set[int] | None = None) -> int:
    blocked = avoid or set()
    choices = [color for color in PATH_COLORS if color not in blocked]
    return int(rng.choice(choices or list(PATH_COLORS)))


def _paint_segment(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = start
    r1, c1 = end
    color = int(color)

    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            existing = int(grid[r0, c])
            if existing == GREY:
                grid[r0, c] = color
            else:
                assert existing != BLACK, "source markers must not be overwritten by a path"
                if _priority(color) < _priority(existing):
                    grid[r0, c] = color
        return

    assert c0 == c1, "segments must be orthogonal"
    step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step, step):
        existing = int(grid[r, c0])
        if existing == GREY:
            grid[r, c0] = color
        else:
            assert existing != BLACK, "source markers must not be overwritten by a path"
            if _priority(color) < _priority(existing):
                grid[r, c0] = color


def _sources(grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]]:
    h, w = grid.shape
    result: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]] = []

    for r in range(h):
        if int(grid[r, 0]) == BLACK and w > 1 and _is_path(int(grid[r, 1])):
            result.append(((r, 0), (r, 1), (0, 1), int(grid[r, 1])))

    for c in range(w):
        if int(grid[0, c]) == BLACK and h > 1 and _is_path(int(grid[1, c])):
            result.append(((0, c), (1, c), (1, 0), int(grid[1, c])))

    assert result, "expected at least one edge source"
    return result


def _trace_signal(grid: np.ndarray, start_pos: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    r, c = start_pos
    dr, dc = direction
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "signal path must not loop"
        seen.add(state)

        if dr == 0:
            candidates = [(-1, 0), (1, 0)]
        else:
            candidates = [(0, -1), (0, 1)]

        turns: list[tuple[int, int]] = []
        for ndr, ndc in candidates:
            nr = r + ndr
            nc = c + ndc
            if 0 <= nr < h and 0 <= nc < w and _is_path(int(grid[nr, nc])):
                turns.append((ndr, ndc))

        if turns:
            if len(turns) == 1:
                dr, dc = turns[0]
            else:
                dr, dc = min(
                    turns,
                    key=lambda d: (
                        _priority(int(grid[r + d[0], c + d[1]])),
                        int(grid[r + d[0], c + d[1]]),
                        d,
                    ),
                )
            r += dr
            c += dc
            continue

        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and _is_path(int(grid[nr, nc])):
            r, c = nr, nc
            continue

        assert 0 <= nr < h and 0 <= nc < w, "signal must terminate inside the output canvas"
        assert int(grid[nr, nc]) == GREY, "terminal exterior cell must be grey"
        assert nr == h - 1 or nc == w - 1, "terminals must exit on the right or bottom edge"
        return (nr, nc)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for _, start_pos, direction, source_color in _sources(g):
        tr, tc = _trace_signal(g, start_pos, direction)
        out[tr, tc] = int(source_color)
    return out


def _build_left_mode(rng: random.Random) -> np.ndarray:
    gadget_count = rng.choice([1, 2])
    band_height = 11
    min_h = gadget_count * band_height + (gadget_count - 1)
    h = rng.randint(min_h, min(30, min_h + 4))
    w = rng.randint(18, 29)
    grid = np.full((h, w), GREY, dtype=int)

    extra_rows = h - min_h
    paddings = [0] * (gadget_count + 1)
    for _ in range(extra_rows):
        paddings[rng.randrange(gadget_count + 1)] += 1

    band_starts: list[int] = []
    cursor = paddings[0]
    for index in range(gadget_count):
        band_starts.append(cursor)
        cursor += band_height
        if index < gadget_count - 1:
            cursor += 1 + paddings[index + 1]

    for base in band_starts:
        rows = [base + offset for offset in (1, 3, 5, 7, 9)]
        source_row = rows[2]
        first_col = rng.randint(4, min(8, w - 10))
        use_second_choice = w >= 22 and rng.random() < 0.65
        second_col = rng.randint(first_col + 4, min(first_col + 8, w - 6)) if use_second_choice else None

        source_color = _pick_color(rng)
        grid[source_row, 0] = BLACK
        _paint_segment(grid, (source_row, 1), (source_row, first_col), source_color)

        first_double = rng.random() < 0.55
        if first_double:
            up_color = _pick_color(rng, {source_color})
            down_color = _pick_color(rng, {source_color, up_color})
            _paint_segment(grid, (rows[1], first_col), (source_row, first_col), up_color)
            _paint_segment(grid, (source_row, first_col), (rows[3], first_col), down_color)
            chosen_upper = _priority(up_color) < _priority(down_color)
            route_row = rows[1] if chosen_upper else rows[3]
            decoy_row = rows[3] if chosen_upper else rows[1]
            decoy_color = _pick_color(rng, {up_color, down_color})
            _paint_segment(grid, (decoy_row, first_col), (decoy_row, w - 2), decoy_color)
        else:
            route_row = rows[1] if rng.random() < 0.5 else rows[3]
            turn_color = _pick_color(rng, {source_color})
            _paint_segment(grid, (min(source_row, route_row), first_col), (max(source_row, route_row), first_col), turn_color)

        if use_second_choice:
            bridge_color = _pick_color(rng)
            assert second_col is not None
            _paint_segment(grid, (route_row, first_col), (route_row, second_col), bridge_color)

            if route_row == rows[1]:
                candidate_rows = (rows[0], rows[2])
            else:
                candidate_rows = (rows[2], rows[4])

            second_double = rng.random() < 0.55
            a_row, b_row = candidate_rows
            if second_double:
                a_color = _pick_color(rng, {bridge_color})
                b_color = _pick_color(rng, {bridge_color, a_color})
                _paint_segment(grid, (min(route_row, a_row), second_col), (max(route_row, a_row), second_col), a_color)
                _paint_segment(grid, (min(route_row, b_row), second_col), (max(route_row, b_row), second_col), b_color)
                choose_a = _priority(a_color) < _priority(b_color)
                final_row = a_row if choose_a else b_row
                decoy_row = b_row if choose_a else a_row
                decoy_color = _pick_color(rng, {a_color, b_color})
                _paint_segment(grid, (decoy_row, second_col), (decoy_row, w - 2), decoy_color)
            else:
                final_row = a_row if rng.random() < 0.5 else b_row
                turn_color = _pick_color(rng, {bridge_color})
                _paint_segment(grid, (min(route_row, final_row), second_col), (max(route_row, final_row), second_col), turn_color)

            final_color = _pick_color(rng)
            _paint_segment(grid, (final_row, second_col), (final_row, w - 2), final_color)
        else:
            final_color = _pick_color(rng)
            _paint_segment(grid, (route_row, first_col), (route_row, w - 2), final_color)

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each edge-launched signal through the colored path network and stamp the source color at the reached terminal edge cell.

    Input structure:
    - The background is grey.
    - Black pixels on the left edge or top edge are signal sources, each touching the first pixel of a colored 1-pixel-wide orthogonal path.
    - Colored path segments may meet at shared intersection pixels, and the visible color at a shared pixel is the earlier one in the fixed order red, blue, green, yellow, orange, magenta, azure, maroon.
    - True terminals are grey cells on the right edge or bottom edge that sit immediately beyond the last path pixel.

    Rule:
    - For each source, enter the adjacent path pixel and keep moving in the current direction while no perpendicular branch is available.
    - At any intersection, the signal must turn onto a perpendicular branch and may not continue straight.
    - If both perpendicular turn directions are available, compare the colors of the two entered branches with the fixed visible color order and take the earlier one.
    - When the signal reaches a path endpoint whose next cell is a grey right-edge or bottom-edge cell, paint that edge cell with the starting path color and leave all paths and sources unchanged.

    Key insight:
    - The task is solved by tracing signal motion on local path topology: intersections force a turn, and only the branch-color priority breaks left-vs-right (or up-vs-down) ambiguity.
    """
    return _solve(grid)


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        grid = _build_left_mode(rng)

        if rng.random() < 0.5:
            grid = np.flipud(grid).copy()
        if rng.random() < 0.5:
            grid = grid.T.copy()

        try:
            _ = _solve(grid)
        except AssertionError:
            continue

        sources = _sources(grid)
        assert 1 <= len(sources) <= 2
        return grid

    raise RuntimeError("failed to generate a valid signal-routing input")

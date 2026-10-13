from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

TURN_RIGHT = PINK
TURN_LEFT = TEAL
CONTROL_COLORS = (TURN_RIGHT, TURN_LEFT)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
RANK_PALETTE = [BLUE, RED, GREEN, YELLOW, GREY, ORANGE, MAROON]

GEN_GRID_MIN = 15
GEN_GRID_MAX = 20
GEN_TURN_COUNTS = [3, 4, 5, 6]
GEN_TURN_WEIGHTS = [1, 2, 3, 3]
GEN_TURN_COLOR_MIX_PROB = 0.75
GEN_NONFINAL_STEP_MAX = 8
GEN_MAX_ATTEMPTS = 800


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _turn(direction: tuple[int, int], control_color: int) -> tuple[int, int]:
    dr, dc = direction
    if control_color == TURN_RIGHT:
        return dc, -dr
    if control_color == TURN_LEFT:
        return -dc, dr
    raise AssertionError("unexpected turn control color")


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    starts = np.argwhere(grid == GREEN)
    assert starts.shape[0] == 1, "expected exactly one green start pixel"
    return tuple(map(int, starts[0]))


def _initial_direction(grid: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    h, w = grid.shape
    sr, sc = start
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        r, c = sr + dr, sc + dc
        while 0 <= r < h and 0 <= c < w:
            value = int(grid[r, c])
            if value != BG:
                if value in CONTROL_COLORS:
                    candidates.append((dr, dc))
                break
            r += dr
            c += dc
    assert len(candidates) == 1, "start must have exactly one visible control direction"
    return candidates[0]


def _ray_cells(pos: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, steps + 1)]


def _edge_distance(shape: tuple[int, int], pos: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return h - 1 - r
    if dc < 0:
        return c
    return w - 1 - c


def _trace_segments(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BG, GREEN, PINK, TEAL}, "unexpected colors in the input scene"

    start = _find_start(g)
    direction = _initial_direction(g, start)
    h, w = g.shape

    segments: list[list[tuple[int, int]]] = [[start]]
    turn_events: list[tuple[int, int, int]] = []
    visited_path = {start}
    seen_states: set[tuple[int, int, int, int]] = set()
    seen_controls: set[tuple[int, int]] = set()

    r, c = start
    dr, dc = direction
    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path must terminate instead of cycling"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            break

        value = int(g[nr, nc])
        if value == BG:
            r, c = nr, nc
            assert (r, c) not in visited_path, "path must not self-cross"
            visited_path.add((r, c))
            segments[-1].append((r, c))
            continue

        if value in CONTROL_COLORS:
            assert segments[-1], "segment must contain at least one path pixel"
            assert (nr, nc) not in seen_controls, "each control should be encountered once"
            seen_controls.add((nr, nc))
            turn_events.append((nr, nc, value))
            dr, dc = _turn((dr, dc), value)
            segments.append([(r, c)])
            continue

        raise AssertionError("path may only move through background and turn before controls")

    assert len(segments) >= 1
    return segments, turn_events


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the turning path from the green start, rank its straight segments by length, and recolor them by rank.

    Input structure:
    - A black-background grid contains exactly one green start pixel.
    - The only other foreground pixels are one-pixel turn controls: pink means turn right and teal means turn left.
    - The green start has exactly one cardinal sightline whose first non-background pixel is a turn control.

    Rule:
    - Start at the green pixel and move toward its uniquely visible control, filling only background cells along the way.
    - Whenever the next cell is a pink or teal control, do not overwrite the control; rotate 90 degrees right or left and begin a new straight segment from the bend cell.
    - Stop when the next step would leave the grid, then measure each traced segment by its pixel count, sort the segments by decreasing length, and recolor them with the rank palette blue, red, green, yellow, grey, orange, maroon.
    - Control pixels keep their original colors, and bend cells belong to both adjacent segments for length counting but take the later segment's final recolor because segments are repainted in traversal order.

    Key insight:
    - The output colors depend on a full path simulation first: control colors determine the turn sequence, that sequence determines the segment lengths, and only then can the segments be ranked and recolored.
    """
    g = np.asarray(grid, dtype=int)
    segments, turn_events = _trace_segments(g)

    assert 4 <= len(segments) <= 7, "expected 4 to 7 path segments"
    lengths = [len(segment) for segment in segments]
    assert len(set(lengths)) == len(lengths), "segment lengths must be pairwise distinct"
    assert len(segments) <= len(RANK_PALETTE), "rank palette is too short for the segment count"

    order = sorted(range(len(segments)), key=lambda idx: lengths[idx], reverse=True)
    segment_color = {idx: RANK_PALETTE[rank] for rank, idx in enumerate(order)}

    out = g.copy()
    for idx, segment in enumerate(segments):
        color = segment_color[idx]
        for r, c in segment:
            out[r, c] = color

    for r, c, color in turn_events:
        out[r, c] = color
    return out


def _choose_turn_colors(turn_count: int) -> list[int]:
    if turn_count >= 2 and random.random() < GEN_TURN_COLOR_MIX_PROB:
        colors = [random.choice(CONTROL_COLORS) for _ in range(turn_count)]
        if TURN_RIGHT not in colors:
            colors[random.randrange(turn_count)] = TURN_RIGHT
        if TURN_LEFT not in colors:
            colors[random.randrange(turn_count)] = TURN_LEFT
        return colors
    return [random.choice(CONTROL_COLORS) for _ in range(turn_count)]


def _search_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    turn_colors: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[int]] | None:
    h, w = shape
    start_set = {start}

    def rec(
        turn_idx: int,
        pos: tuple[int, int],
        cur_dir: tuple[int, int],
        used_steps: set[int],
        path_cells: set[tuple[int, int]],
        controls: list[tuple[int, int, int]],
        segment_steps: list[int],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[int]] | None:
        occupied = path_cells | {cell[:2] for cell in controls}
        if turn_idx == len(turn_colors):
            final_steps = _edge_distance((h, w), pos, cur_dir)
            if final_steps < 1 or final_steps in used_steps:
                return None
            final_cells = _ray_cells(pos, cur_dir, final_steps)
            if any(cell in occupied for cell in final_cells):
                return None
            return list(path_cells | set(final_cells)), controls, segment_steps + [final_steps]

        max_forward = _edge_distance((h, w), pos, cur_dir) - 1
        if max_forward < 1:
            return None

        candidates = list(range(1, min(GEN_NONFINAL_STEP_MAX, max_forward) + 1))
        random.shuffle(candidates)
        for steps in candidates:
            if steps in used_steps:
                continue

            cells = _ray_cells(pos, cur_dir, steps)
            control_r = pos[0] + cur_dir[0] * (steps + 1)
            control_c = pos[1] + cur_dir[1] * (steps + 1)
            control_cell = (control_r, control_c)
            if not _in_bounds((h, w), control_r, control_c):
                continue
            if any(cell in occupied for cell in cells):
                continue
            if control_cell in occupied or control_cell in start_set:
                continue

            next_pos = cells[-1]
            next_dir = _turn(cur_dir, turn_colors[turn_idx])
            min_future = 1 if turn_idx + 1 == len(turn_colors) else 2
            if _edge_distance((h, w), next_pos, next_dir) < min_future:
                continue

            result = rec(
                turn_idx + 1,
                next_pos,
                next_dir,
                used_steps | {steps},
                path_cells | set(cells),
                controls + [(control_r, control_c, turn_colors[turn_idx])],
                segment_steps + [steps],
            )
            if result is not None:
                return result
        return None

    return rec(0, start, direction, set(), {start}, [], [])


def _lengths_from_segments(grid: np.ndarray) -> list[int]:
    segments, _turn_events = _trace_segments(grid)
    return [len(segment) for segment in segments]


def _is_non_monotonic(lengths: list[int]) -> bool:
    return lengths != sorted(lengths) and lengths != sorted(lengths, reverse=True)


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, GREEN, PINK, TEAL}:
        return False

    starts = np.argwhere(grid == GREEN)
    controls = np.argwhere(np.isin(grid, CONTROL_COLORS))
    if starts.shape[0] != 1 or not (3 <= controls.shape[0] <= 6):
        return False

    try:
        start = _find_start(grid)
        _initial_direction(grid, start)
        segments, turn_events = _trace_segments(grid)
        out = main(grid)
    except AssertionError:
        return False

    if len(turn_events) != controls.shape[0]:
        return False
    if len(segments) != len(turn_events) + 1:
        return False

    lengths = [len(segment) for segment in segments]
    if len(set(lengths)) != len(lengths):
        return False
    if not (4 <= len(lengths) <= 7):
        return False
    if not _is_non_monotonic(lengths):
        return False

    control_colors = {color for _r, _c, color in turn_events}
    if control_colors == {TURN_RIGHT} or control_colors == {TURN_LEFT}:
        if len(turn_events) >= 5:
            return False

    output_colors = set(map(int, np.unique(out)))
    if PINK in output_colors - set(map(int, grid[grid == PINK])):
        return False
    if TEAL in output_colors - set(map(int, grid[grid == TEAL])):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    turn_count = random.choices(GEN_TURN_COUNTS, weights=GEN_TURN_WEIGHTS, k=1)[0]
    turn_colors = _choose_turn_colors(turn_count)

    directions = DIRS[:]
    random.shuffle(directions)
    rows = list(range(2, h - 2))
    cols = list(range(2, w - 2))
    random.shuffle(rows)
    random.shuffle(cols)

    starts = [(r, c) for r in rows for c in cols]
    random.shuffle(starts)
    for start in starts:
        for direction in directions:
            built = _search_path((h, w), start, direction, turn_colors)
            if built is None:
                continue

            _path_cells, controls, _segment_steps = built
            grid = np.zeros((h, w), dtype=int)
            grid[start] = GREEN
            for r, c, color in controls:
                grid[r, c] = color

            if _validate_generated_input(grid):
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid ranked-turn-path input")

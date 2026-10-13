from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
START = 3
SEPARATOR = 5
PINK = 6
TEAL = 8

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_CONTROLS = {PINK, TEAL}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _find_separator(grid: np.ndarray) -> int:
    H, W = map(int, grid.shape)
    cols = [c for c in range(W) if int(grid[0, c]) == SEPARATOR and np.all(grid[:, c] == SEPARATOR)]
    assert len(cols) == 1, "expected exactly one full-height gray separator"
    sep = int(cols[0])
    assert 2 * sep + 1 == W, "separator must split equal-width panels"
    assert not np.any(grid[:, :sep] == SEPARATOR), "separator color must stay out of the left panel"
    assert not np.any(grid[:, sep + 1 :] == SEPARATOR), "separator color must stay out of the right panel"
    assert H > 0 and sep > 0, "panels must be non-empty"
    return sep


def _fold_controls(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert _get_bg(g) == BG, "background must be black"

    sep = _find_separator(g)
    left = g[:, :sep]
    right = np.fliplr(g[:, sep + 1 :])
    assert left.shape == right.shape

    out = right.copy()
    out[left != BG] = left[left != BG]
    colors = set(map(int, np.unique(out)))
    assert colors <= {BG, START, PINK, TEAL}, "folded scene may only contain background, start, and controls"
    return out


def _visible_control_directions(scene: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = map(int, scene.shape)
    sr, sc = start
    candidates: list[tuple[int, int]] = []

    for dr, dc in DIRS:
        r, c = sr + dr, sc + dc
        while 0 <= r < H and 0 <= c < W:
            value = int(scene[r, c])
            if value != BG:
                if value in TURN_CONTROLS:
                    candidates.append((dr, dc))
                break
            r += dr
            c += dc
    return candidates


def _turn(direction: tuple[int, int], control_color: int) -> tuple[int, int]:
    dr, dc = direction
    if control_color == PINK:
        return dc, -dr
    assert control_color == TEAL, "expected a magenta or teal turn control"
    return -dc, dr


def _trace_path(scene: np.ndarray) -> np.ndarray:
    starts = np.argwhere(scene == START)
    assert len(starts) == 1, "expected exactly one green start"
    start = tuple(map(int, starts[0]))

    candidates = _visible_control_directions(scene, start)
    assert len(candidates) == 1, "expected exactly one visible control direction after folding"

    out = scene.copy()
    r, c = start
    dr, dc = candidates[0]
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path must terminate instead of looping"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        if not (0 <= nr < scene.shape[0] and 0 <= nc < scene.shape[1]):
            return out

        value = int(scene[nr, nc])
        if value == BG:
            out[nr, nc] = START
            r, c = nr, nc
        elif value in TURN_CONTROLS:
            dr, dc = _turn((dr, dc), value)
        else:
            raise AssertionError("the path may only move through background and turn before controls")


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the separator-split panels, merge their turn controls, then trace a green path directed by the folded layout.

    Input structure:
    - A black-background grid is split into equal left and right panels by one full-height gray separator column.
    - The left panel contains exactly one green start pixel and several magenta turn controls.
    - The right panel contains several teal turn controls and no start pixel.

    Rule:
    - Remove the gray separator, mirror the right panel horizontally, and overlay it onto the left-panel coordinate system with left-panel foreground taking priority on overlaps.
    - In the folded result, find the unique cardinal direction from the green start whose first visible non-background pixel is a turn control.
    - Trace a green path through background cells, turning right before magenta controls and left before teal controls, and stop when the next step would leave the grid.

    Key insight:
    - The path is determined by the control-point layout that only appears after folding the right panel into the left-panel coordinates.
    """
    scene = _fold_controls(grid)
    return _trace_path(scene)


# --------------- generator ---------------

HEIGHT_RANGE = (8, 14)
HALF_WIDTH_RANGE = (5, 9)
LEFT_CTRL_COUNT_RANGE = (2, 3)
RIGHT_CTRL_COUNT_RANGE = (2, 3)
SEGMENT_LENGTHS = (1, 2, 3, 4)
TOTAL_PATH_RANGE = (6, 18)
MAX_ATTEMPTS = 500


def _ray_cells(pos: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(r + dr * step, c + dc * step) for step in range(1, length + 1)]


def _in_bounds(H: int, W: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W


def _edge_distance(H: int, W: int, pos: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pos
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return H - 1 - r
    if dc < 0:
        return c
    return W - 1 - c


def _build_path(
    H: int,
    W: int,
    start: tuple[int, int],
    initial_direction: tuple[int, int],
    turns: tuple[int, ...],
) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], int]]] | None:
    def search(
        i: int,
        pos: tuple[int, int],
        direction: tuple[int, int],
        path_cells: set[tuple[int, int]],
        controls: list[tuple[tuple[int, int], int]],
        path_len: int,
    ) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], int]]] | None:
        occupied = path_cells | {start} | {cell for cell, _ in controls}

        if i == len(turns):
            final_len = _edge_distance(H, W, pos, direction)
            if not (1 <= final_len <= max(H, W)):
                return None
            final_cells = _ray_cells(pos, direction, final_len)
            if any(cell in occupied for cell in final_cells):
                return None
            total_len = path_len + final_len
            if not (TOTAL_PATH_RANGE[0] <= total_len <= TOTAL_PATH_RANGE[1]):
                return None
            return path_cells | set(final_cells), controls

        lengths = list(SEGMENT_LENGTHS)
        random.shuffle(lengths)
        control_color = turns[i]
        for seg_len in lengths:
            cells = _ray_cells(pos, direction, seg_len)
            control_cell = _ray_cells(pos, direction, seg_len + 1)[-1]
            if not all(_in_bounds(H, W, cell) for cell in cells):
                continue
            if not _in_bounds(H, W, control_cell):
                continue
            if any(cell in occupied for cell in cells):
                continue
            if control_cell in occupied:
                continue

            next_len = path_len + seg_len
            min_remaining = (len(turns) - i - 1) + 1
            if next_len + min_remaining > TOTAL_PATH_RANGE[1]:
                continue

            result = search(
                i + 1,
                cells[-1],
                _turn(direction, control_color),
                path_cells | set(cells),
                controls + [(control_cell, control_color)],
                next_len,
            )
            if result is not None:
                return result
        return None

    return search(0, start, initial_direction, set(), [], 0)


def _render_folded_scene(
    H: int,
    W: int,
    start: tuple[int, int],
    controls: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    scene = np.zeros((H, W), dtype=int)
    scene[start] = START
    for (r, c), color in controls:
        scene[r, c] = int(color)
    return scene


def _control_sequence(left_count: int, right_count: int) -> tuple[int, ...]:
    turns = [TEAL]
    turns.extend([PINK] * left_count)
    turns.extend([TEAL] * (right_count - 1))
    tail = turns[1:]
    random.shuffle(tail)
    return tuple([turns[0], *tail])


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*HEIGHT_RANGE)
    W = random.randint(*HALF_WIDTH_RANGE)
    left_count = random.randint(*LEFT_CTRL_COUNT_RANGE)
    right_count = random.randint(*RIGHT_CTRL_COUNT_RANGE)
    turns = _control_sequence(left_count, right_count)

    directions = DIRS[:]
    random.shuffle(directions)
    starts = [(r, c) for r in range(H) for c in range(W)]
    random.shuffle(starts)

    for initial_direction in directions:
        for start in starts:
            built = _build_path(H, W, start, initial_direction, turns)
            if built is None:
                continue

            path_cells, controls = built
            folded = _render_folded_scene(H, W, start, controls)
            if _visible_control_directions(folded, start) != [initial_direction]:
                continue

            left_only = np.zeros((H, W), dtype=int)
            left_only[start] = START
            right_folded = np.zeros((H, W), dtype=int)
            for cell, color in controls:
                if color == PINK:
                    left_only[cell] = PINK
                else:
                    right_folded[cell] = TEAL

            if np.count_nonzero(left_only == PINK) != left_count:
                continue
            if np.count_nonzero(right_folded == TEAL) != right_count:
                continue
            if _visible_control_directions(left_only, start):
                continue

            inp = np.zeros((H, 2 * W + 1), dtype=int)
            inp[:, :W] = left_only
            inp[:, W] = SEPARATOR
            inp[:, W + 1 :] = np.fliplr(right_folded)

            out = main(inp)
            expected = _trace_path(folded)
            if not np.array_equal(out, expected):
                continue
            if len(path_cells) < TOTAL_PATH_RANGE[0]:
                continue
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folded-turn-path input")

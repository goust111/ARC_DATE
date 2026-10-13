from __future__ import annotations

import numpy as np

from arc_utils import Color, coords_where, get_bg, in_bounds
import random


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
TURN_CONTROLS = {Color.PINK, Color.TEAL}


def _turn(direction: tuple[int, int], control_color: int) -> tuple[int, int]:
    dr, dc = direction
    if control_color == Color.PINK:
        return dc, -dr

    assert control_color == Color.TEAL, "expected a pink or teal turn control"
    return -dc, dr


def _initial_direction(g: np.ndarray, bg: int, start: tuple[int, int]) -> tuple[int, int]:
    H, W = g.shape
    sr, sc = start
    candidates: list[tuple[int, int]] = []

    for dr, dc in DIRS:
        r = sr + dr
        c = sc + dc
        while 0 <= r < H and 0 <= c < W:
            value = int(g[r, c])
            if value != bg:
                if value in TURN_CONTROLS:
                    candidates.append((dr, dc))
                break
            r += dr
            c += dc

    assert len(candidates) == 1, "expected one visible turn control from the green start"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a green path from the start, turning right at pink controls and left at teal controls until it exits the grid.

    Input structure
    - A black-background grid contains exactly one green start pixel.
    - The remaining foreground pixels are pink and teal one-pixel turn controls.
    - The green start has exactly one cardinal sightline whose first non-background pixel is a turn control.

    Rule
    - Begin moving from the green start toward that visible control.
    - Paint green through background cells in a straight line.
    - When the next cell is a pink control, turn right without overwriting it; when it is a teal control, turn left without overwriting it.
    - Stop when the next step would leave the grid.

    Key insight
    - The colored control pixels are not targets to fill; they are turn instructions for a single ray that leaves a continuous green trail.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = set(map(int, np.unique(g)))
    assert colors <= {bg, Color.GREEN, Color.PINK, Color.TEAL}, "unexpected colors in path scene"
    assert bg not in {Color.GREEN, Color.PINK, Color.TEAL}, "background must be distinct from path colors"

    starts = coords_where(g == Color.GREEN)
    assert len(starts) == 1, "expected exactly one green start pixel"
    start = tuple(map(int, starts[0]))
    assert np.any(np.isin(g, list(TURN_CONTROLS))), "expected at least one turn control"

    out = g.copy()
    r, c = start
    dr, dc = _initial_direction(g, bg, start)
    seen_states: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen_states, "path must terminate instead of cycling"
        seen_states.add(state)

        nr = r + dr
        nc = c + dc
        if not in_bounds(g.shape, nr, nc):
            return out

        value = int(g[nr, nc])
        if value == bg:
            out[nr, nc] = Color.GREEN
            r, c = nr, nc
        elif value in TURN_CONTROLS:
            dr, dc = _turn((dr, dc), value)
        else:
            raise AssertionError("path may only enter background cells or turn before controls")

# --------------- generator (o5x) ---------------

# Grid/canvas bounds.
GEN_GRID_H_RANGE = (6, 10)
GEN_GRID_W_RANGE = (6, 12)

# Path/control structure.
GEN_TURN_COUNT_RANGE = (1, 6)
GEN_TURN_COLORS = (Color.PINK, Color.TEAL)
GEN_SEGMENT_LENGTHS = (1, 2, 3, 4)
GEN_PATH_LENGTH_RANGE = (5, 17)
GEN_START_DIRECTION = (0, 1)
GEN_MIN_CONTROL_COL = 1

# Generation retry limit.
GEN_MAX_ATTEMPTS = 300


def _gen_turn_dir(direction: tuple[int, int], control_color: int) -> tuple[int, int]:
    dr, dc = direction
    if control_color == Color.PINK:
        return dc, -dr
    return -dc, dr


def _gen_in_bounds(H: int, W: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W


def _gen_ray_cells(
    pos: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    r, c = pos
    dr, dc = direction
    return [(r + dr * k, c + dc * k) for k in range(1, length + 1)]


def _gen_edge_distance(
    H: int, W: int, pos: tuple[int, int], direction: tuple[int, int]
) -> int:
    r, c = pos
    dr, dc = direction
    if dr < 0:
        return r
    if dr > 0:
        return H - 1 - r
    if dc < 0:
        return c
    return W - 1 - c


def _gen_build_path(
    H: int,
    W: int,
    start: tuple[int, int],
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
            final_len = _gen_edge_distance(H, W, pos, direction)
            if final_len not in GEN_SEGMENT_LENGTHS:
                return None
            final_cells = _gen_ray_cells(pos, direction, final_len)
            if any(cell in occupied for cell in final_cells):
                return None
            total_len = path_len + final_len
            if not GEN_PATH_LENGTH_RANGE[0] <= total_len <= GEN_PATH_LENGTH_RANGE[1]:
                return None
            return path_cells | set(final_cells), controls

        lengths = list(GEN_SEGMENT_LENGTHS)
        random.shuffle(lengths)
        control_color = turns[i]
        for seg_len in lengths:
            cells = _gen_ray_cells(pos, direction, seg_len)
            control_cell = _gen_ray_cells(pos, direction, seg_len + 1)[-1]
            if not all(_gen_in_bounds(H, W, cell) for cell in cells):
                continue
            if not _gen_in_bounds(H, W, control_cell):
                continue
            if control_cell[1] < GEN_MIN_CONTROL_COL:
                continue
            if any(cell in occupied for cell in cells):
                continue
            if control_cell in occupied:
                continue

            remaining_min = (len(turns) - i - 1) + 1
            next_path_len = path_len + seg_len
            if next_path_len + remaining_min > GEN_PATH_LENGTH_RANGE[1]:
                continue

            next_pos = cells[-1]
            next_direction = _gen_turn_dir(direction, control_color)
            result = search(
                i + 1,
                next_pos,
                next_direction,
                path_cells | set(cells),
                controls + [(control_cell, control_color)],
                next_path_len,
            )
            if result is not None:
                return result
        return None

    return search(0, start, GEN_START_DIRECTION, set(), [], 0)


def _gen_render_pair(H, W, start, path_cells, controls):
    inp = np.zeros((H, W), dtype=int)
    out = inp.copy()
    inp[start] = Color.GREEN
    out[start] = Color.GREEN
    for cell, color in controls:
        inp[cell] = color
        out[cell] = color
    for cell in path_cells:
        out[cell] = Color.GREEN
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_GRID_H_RANGE)
    W = random.randint(*GEN_GRID_W_RANGE)
    turn_count = random.randint(*GEN_TURN_COUNT_RANGE)
    turns = tuple(random.choice(GEN_TURN_COLORS) for _ in range(turn_count))

    start_rows = list(range(H))
    random.shuffle(start_rows)
    for sr in start_rows:
        start = (sr, 0)
        built = _gen_build_path(H, W, start, turns)
        if built is None:
            continue

        path_cells, controls = built
        inp, out = _gen_render_pair(H, W, start, path_cells, controls)

        if np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid e5790162 input")

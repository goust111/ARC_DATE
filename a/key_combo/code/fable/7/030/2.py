from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds


BG = Color.BLACK
TURN_RIGHT = Color.BLUE
TURN_LEFT = Color.RED
PATH = Color.GREEN
SEPARATOR = Color.YELLOW
TAIL = Color.GREY

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
LEFT_OF = {
    (-1, 0): (0, -1),
    (0, -1): (1, 0),
    (1, 0): (0, 1),
    (0, 1): (-1, 0),
}
RIGHT_OF = {
    (-1, 0): (0, 1),
    (0, 1): (1, 0),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
}

WIDTH_RANGE = (6, 12)
TERRAIN_H_RANGE = (2, 6)
TOP_H_MIN = 7
TOP_H_MAX = 18
GEN_ATTEMPTS = 600


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates = [
        r
        for r in range(grid.shape[0])
        if 0 < r < grid.shape[0] - 1 and int(grid[r, 0]) != bg and np.all(grid[r] == grid[r, 0])
    ]
    assert len(candidates) == 1, "expected exactly one solid separator row"
    return int(candidates[0])


def _find_arrow(canvas: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    allowed = {bg, PATH, TAIL}
    assert set(map(int, np.unique(canvas))) <= allowed, "canvas should contain only background and the arrow"

    heads = np.argwhere(canvas == PATH)
    tails = np.argwhere(canvas == TAIL)
    assert heads.shape == (1, 2), "expected exactly one green head"
    assert tails.shape == (1, 2), "expected exactly one grey tail"

    hr, hc = map(int, heads[0])
    tr, tc = map(int, tails[0])
    assert abs(hr - tr) + abs(hc - tc) == 1, "arrow cells must be orthogonally adjacent"
    direction = (hr - tr, hc - tc)
    return (hr, hc), (tr, tc), direction


def _decode_terrain(panel: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    allowed = {bg, TURN_LEFT, TURN_RIGHT}
    assert set(map(int, np.unique(panel))) <= allowed, "terrain panel should use only black, red, and blue"
    assert panel.shape[0] >= 1 and panel.shape[1] >= 1

    steps: list[int] = []
    turns: list[int] = []
    for c in range(panel.shape[1]):
        nz = np.flatnonzero(panel[:, c] != bg)
        assert nz.size > 0, "every terrain column must encode one instruction"
        assert int(nz[-1]) == panel.shape[0] - 1, "terrain must touch the panel bottom"
        assert np.array_equal(nz, np.arange(nz[0], panel.shape[0])), "terrain must be vertically contiguous"
        color = int(panel[nz[0], c])
        assert color in (TURN_LEFT, TURN_RIGHT)
        assert np.all(panel[nz[0] :, c] == color), "each terrain column must be monochrome"
        steps.append(int(nz.size))
        turns.append(color)

    assert TURN_LEFT in turns and TURN_RIGHT in turns, "generator expects both turn colors to appear"
    return steps, turns


def _turn(direction: tuple[int, int], turn_color: int) -> tuple[int, int]:
    return LEFT_OF[direction] if turn_color == TURN_LEFT else RIGHT_OF[direction]


def _trace_program(
    start: tuple[int, int],
    direction: tuple[int, int],
    steps: list[int],
    turns: list[int],
    canvas_shape: tuple[int, int],
    blocked: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    blocked = set() if blocked is None else set(blocked)
    occupied = set(blocked)
    occupied.add(start)
    path: list[tuple[int, int]] = []
    r, c = start

    for step_len, turn_color in zip(steps, turns):
        for _ in range(int(step_len)):
            r += direction[0]
            c += direction[1]
            assert in_bounds(canvas_shape, r, c), "decoded polyline left the canvas"
            assert (r, c) not in occupied, "decoded polyline must visit only new cells"
            occupied.add((r, c))
            path.append((r, c))
        direction = _turn(direction, int(turn_color))
    return path


def _parse_input(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = get_bg(g)
    sep = _find_separator_row(g, bg)
    canvas = g[:sep, :]
    panel = g[sep + 1 :, :]
    assert canvas.shape[0] >= 2 and panel.shape[0] >= 1
    assert int(g[sep, 0]) == SEPARATOR, "separator color must be yellow"
    head, tail, direction = _find_arrow(canvas, bg)
    steps, turns = _decode_terrain(panel, bg)
    return bg, canvas, panel, head, tail, direction, steps, turns


def _render_input(
    top_h: int,
    terrain_h: int,
    head: tuple[int, int],
    tail: tuple[int, int],
    steps: list[int],
    turns: list[int],
) -> np.ndarray:
    width = len(steps)
    grid = np.full((top_h + 1 + terrain_h, width), BG, dtype=int)
    grid[top_h, :] = SEPARATOR
    grid[head] = PATH
    grid[tail] = TAIL

    base_r = top_h + 1 + terrain_h
    for c, (step_len, turn_color) in enumerate(zip(steps, turns)):
        grid[base_r - step_len : base_r, c] = int(turn_color)
    return grid


def _output_from_parts(
    grid: np.ndarray,
    canvas_shape: tuple[int, int],
    head: tuple[int, int],
    tail: tuple[int, int],
    direction: tuple[int, int],
    steps: list[int],
    turns: list[int],
) -> np.ndarray:
    path = _trace_program(head, direction, steps, turns, canvas_shape, blocked={tail})
    out = np.asarray(grid, dtype=int).copy()
    out[head] = PATH
    for r, c in path:
        out[r, c] = PATH
    out[tail] = TAIL
    return out


def _path_bbox(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _modified_path_changes(
    head: tuple[int, int],
    tail: tuple[int, int],
    direction: tuple[int, int],
    steps: list[int],
    turns: list[int],
    canvas_shape: tuple[int, int],
    terrain_h: int,
) -> bool:
    base = tuple(_trace_program(head, direction, steps, turns, canvas_shape, blocked={tail}))

    for idx in range(len(steps)):
        alt_turns = list(turns)
        alt_turns[idx] = TURN_LEFT if turns[idx] == TURN_RIGHT else TURN_RIGHT
        try:
            if tuple(_trace_program(head, direction, steps, alt_turns, canvas_shape, blocked={tail})) != base:
                return True
        except AssertionError:
            return True

    for idx, step_len in enumerate(steps):
        alternatives = [h for h in {1, terrain_h, step_len - 1, step_len + 1} if 1 <= h <= terrain_h and h != step_len]
        for alt_h in alternatives:
            alt_steps = list(steps)
            alt_steps[idx] = alt_h
            try:
                if tuple(_trace_program(head, direction, alt_steps, turns, canvas_shape, blocked={tail})) != base:
                    return True
            except AssertionError:
                return True
    return False


def _validate_generated_instance(
    grid: np.ndarray,
    expected_head: tuple[int, int],
    expected_tail: tuple[int, int],
    expected_direction: tuple[int, int],
    expected_steps: list[int],
    expected_turns: list[int],
) -> None:
    bg, canvas, panel, head, tail, direction, steps, turns = _parse_input(grid)
    assert bg == BG
    assert head == expected_head
    assert tail == expected_tail
    assert direction == expected_direction
    assert steps == expected_steps
    assert turns == expected_turns

    path = _trace_program(head, direction, steps, turns, canvas.shape, blocked={tail})
    traced = {head, *path}
    assert len(traced) == 1 + sum(steps)
    path_h, path_w = _path_bbox([head, *path])
    assert path_h >= 3 and path_w >= 3, "path should occupy both axes clearly"
    assert _modified_path_changes(head, tail, direction, steps, turns, canvas.shape, panel.shape[0])

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out[canvas.shape[0] :, :], grid[canvas.shape[0] :, :]), "separator and terrain must stay unchanged"

    expected = _output_from_parts(grid, canvas.shape, head, tail, direction, steps, turns)
    assert np.array_equal(out, expected)

    canvas_out = out[: canvas.shape[0], :]
    green_cells = {tuple(cell) for cell in np.argwhere(canvas_out == PATH)}
    assert green_cells == traced
    assert int(out[tail]) == TAIL


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the bottom terrain columns into a turn program and trace its polyline on the upper canvas.

    Input structure:
    - A solid yellow separator row splits the grid into an upper canvas and a lower terrain panel.
    - The upper canvas contains exactly one two-cell arrow: a green head and an orthogonally adjacent grey tail.
    - In the lower panel, every column has one bottom-aligned monochrome terrain bar in red or blue, with height read inside that panel.

    Rule:
    - Read the terrain columns from left to right. A column's height is the step length for that instruction.
    - Starting from the green head and facing from the grey tail toward the green head, paint that many new green cells, then turn left after a red column or right after a blue column.
    - Return the full grid with the separator row, the terrain panel, and the grey tail unchanged, plus the complete traced green polyline on the canvas.

    Key insight:
    - The lower terrain is not scenery to copy; each column is an instruction token whose height encodes distance and whose color encodes the next quarter-turn.
    """
    g = np.asarray(grid, dtype=int)
    _bg, canvas, _panel, head, tail, direction, steps, turns = _parse_input(g)
    return _output_from_parts(g, canvas.shape, head, tail, direction, steps, turns)


def _sample_steps(width: int, terrain_h: int) -> list[int]:
    steps = [random.randint(1, terrain_h) for _ in range(width)]
    forced = {0: 1, width - 1: terrain_h}
    mid = random.randrange(width)
    forced[mid] = random.randint(1, terrain_h)
    for idx, value in forced.items():
        steps[idx] = value
    if len(set(steps)) == 1:
        j = random.randrange(width)
        steps[j] = terrain_h if steps[j] == 1 else 1
    return steps


def _sample_turns(width: int) -> list[int]:
    turns = [random.choice([TURN_LEFT, TURN_RIGHT]) for _ in range(width)]
    turns[0] = TURN_LEFT
    turns[-1] = TURN_RIGHT
    if TURN_LEFT not in turns:
        turns[random.randrange(width)] = TURN_LEFT
    if TURN_RIGHT not in turns:
        turns[random.randrange(width)] = TURN_RIGHT
    return turns


def _head_candidates(top_h: int, width: int, direction: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    dr, dc = direction
    candidates = []
    for hr in range(top_h):
        for hc in range(width):
            tr, tc = hr - dr, hc - dc
            if in_bounds((top_h, width), tr, tc):
                candidates.append(((hr, hc), (tr, tc)))
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    terrain_h = random.randint(*TERRAIN_H_RANGE)
    top_h_max = min(TOP_H_MAX, 29 - terrain_h)
    if top_h_max < TOP_H_MIN:
        return None
    top_h = random.randint(TOP_H_MIN, top_h_max)

    steps = _sample_steps(width, terrain_h)
    turns = _sample_turns(width)
    direction = random.choice(DIRS)

    for head, tail in _head_candidates(top_h, width, direction):
        try:
            path = _trace_program(head, direction, steps, turns, (top_h, width), blocked={tail})
        except AssertionError:
            continue
        if len(path) < width + 2:
            continue
        span_h, span_w = _path_bbox([head, *path])
        if span_h < 3 or span_w < 3:
            continue

        grid = _render_input(top_h, terrain_h, head, tail, steps, turns)
        try:
            _validate_generated_instance(grid, head, tail, direction, steps, turns)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid terrain-program ARC input")

from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
TURN_LEFT = "L"
TURN_RIGHT = "R"
UP = (-1, 0)

GEN_ATTEMPTS = 500
GEN_PROGRAM_ATTEMPTS = 300
GEN_BAR_COUNT_CHOICES = (3, 4, 4, 5, 5, 6)
GEN_STEP_MIN = 2
GEN_STEP_MAX = 6
GEN_MAX_PATH_H = 17
GEN_MAX_PATH_W = 17
GEN_MAX_GRID = 30
GEN_GAP_RANGE = (2, 4)
GEN_EXTRA_W_RANGE = (1, 5)
GEN_EXTRA_H_RANGE = (2, 6)
GEN_COLORS = tuple(range(1, 10))
GEN_SIDES = ("left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _turn(direction: Coord, kind: str) -> Coord:
    dr, dc = direction
    if kind == TURN_LEFT:
        return -dc, dr
    if kind == TURN_RIGHT:
        return dc, -dr
    raise ValueError(f"unknown turn {kind}")


def _trace_program(turns: list[str], steps: list[int]) -> list[Coord]:
    assert len(turns) == len(steps)
    direction = UP
    pos = (0, 0)
    path = [pos]
    for kind, length in zip(turns, steps):
        direction = _turn(direction, kind)
        for _ in range(int(length)):
            pos = (pos[0] + direction[0], pos[1] + direction[1])
            path.append(pos)
    return path


def _path_is_clear(path: list[Coord]) -> bool:
    visited: set[Coord] = set()
    for i, cell in enumerate(path):
        if cell in visited:
            return False
        prev = path[i - 1] if i > 0 else None
        for nbr in _neighbors4(*cell):
            if nbr in visited and nbr != prev:
                return False
        visited.add(cell)
    return True


def _path_bbox(path: list[Coord]) -> BBox:
    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    return min(rows), min(cols), max(rows), max(cols)


def _decode_program(grid: np.ndarray) -> tuple[int, Coord, list[tuple[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    components = _find_components(g != bg)
    assert components, "expected bars plus one seed"

    seed_cells = [coords for coords in components if int(coords.shape[0]) == 1]
    assert len(seed_cells) == 1, "expected exactly one singleton seed"
    seed = tuple(map(int, seed_cells[0][0]))
    body = int(g[seed])
    assert body != bg, "seed must be foreground"

    instructions: list[tuple[int, str, int]] = []
    for coords in components:
        if int(coords.shape[0]) == 1:
            continue

        r0, c0, r1, c1 = _bbox_coords(coords)
        assert r0 == r1, "instruction bars must be horizontal"
        cols = sorted(int(c) for _r, c in coords)
        assert cols == list(range(c0, c1 + 1)), "instruction bar must be contiguous"

        row_vals = [int(g[r0, c]) for c in range(c0, c1 + 1)]
        body_count = sum(val == body for val in row_vals)
        assert body_count == len(row_vals) - 1, "bar must have exactly one head cell"
        assert body_count >= 1, "bar body must contain at least one cell"

        head_left = row_vals[0] != body
        head_right = row_vals[-1] != body
        assert head_left ^ head_right, "head must occupy exactly one endpoint"
        if head_left:
            assert all(val == body for val in row_vals[1:])
            instructions.append((r0, TURN_LEFT, body_count))
        else:
            assert all(val == body for val in row_vals[:-1])
            instructions.append((r0, TURN_RIGHT, body_count))

    assert instructions, "expected at least one instruction bar"
    instructions.sort(key=lambda item: item[0])
    rows = [row for row, _turn_kind, _count in instructions]
    assert len(set(rows)) == len(rows), "instruction rows must be unique"
    assert all(b - a >= 2 for a, b in zip(rows, rows[1:])), "bars must be separated by blank rows"
    return body, seed, [(turn_kind, count) for _row, turn_kind, count in instructions]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the sidebar's horizontal bars as turtle instructions and draw the resulting polyline.

    Input structure:
    - The background is black.
    - One side of the grid holds a vertical list of horizontal bars with one blank row between neighbors.
    - Each bar consists of body-colored cells plus exactly one differently colored endpoint head, and one isolated seed pixel elsewhere uses the same body color.

    Rule:
    - Start the turtle at the seed, initially facing upward.
    - Read the bars from top to bottom: a left-end head means turn left by ninety degrees, and a right-end head means turn right by ninety degrees.
    - After each turn, advance in the new direction for as many steps as there are body-colored cells in that bar, painting every visited cell with the body color while leaving the instruction bars unchanged.

    Key insight:
    - The sidebar is a program, not a picture: each bar encodes one turn and one segment length, and the seed is only the starting point of the hidden polyline.
    """
    g = np.asarray(grid, dtype=int)
    body, pos, instructions = _decode_program(g)
    out = g.copy()
    direction = UP
    bg = BG

    for turn_kind, steps in instructions:
        direction = _turn(direction, turn_kind)
        for _ in range(steps):
            nr = pos[0] + direction[0]
            nc = pos[1] + direction[1]
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "path must stay inside the grid"
            assert int(g[nr, nc]) in {bg, body}, "generated path may not overwrite instruction heads"
            out[nr, nc] = body
            pos = (nr, nc)

    return out


def _sample_program() -> tuple[list[str], list[int], list[Coord]] | None:
    for _ in range(GEN_PROGRAM_ATTEMPTS):
        n_bars = random.choice(GEN_BAR_COUNT_CHOICES)
        turns = [random.choice((TURN_LEFT, TURN_RIGHT)) for _ in range(n_bars)]
        if len(set(turns)) < 2:
            continue

        steps = [random.randint(GEN_STEP_MIN, GEN_STEP_MAX) for _ in range(n_bars)]
        if len(set(steps)) < 2:
            continue

        path = _trace_program(turns, steps)
        if not _path_is_clear(path):
            continue

        r0, c0, r1, c1 = _path_bbox(path)
        if r1 - r0 + 1 > GEN_MAX_PATH_H or c1 - c0 + 1 > GEN_MAX_PATH_W:
            continue
        return turns, steps, path
    return None


def _render_program(turns: list[str], steps: list[int], path: list[Coord]) -> tuple[np.ndarray, np.ndarray] | None:
    body, head = random.sample(GEN_COLORS, 2)
    side = random.choice(GEN_SIDES)

    bar_h = 2 * len(turns) - 1
    bar_w = max(steps) + 1
    gap = random.randint(*GEN_GAP_RANGE)
    extra_w = random.randint(*GEN_EXTRA_W_RANGE)
    extra_h = random.randint(*GEN_EXTRA_H_RANGE)

    path_r0, path_c0, path_r1, path_c1 = _path_bbox(path)
    path_h = path_r1 - path_r0 + 1
    path_w = path_c1 - path_c0 + 1

    width = bar_w + gap + path_w + extra_w
    height = max(bar_h, path_h) + extra_h
    if height > GEN_MAX_GRID or width > GEN_MAX_GRID:
        return None

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    bar_top = random.randint(0, height - bar_h)
    path_top = random.randint(0, height - path_h)

    if side == "left":
        path_left = random.randint(bar_w + gap, width - path_w)
        bar_right = None
    else:
        path_left = random.randint(0, width - (bar_w + gap + path_w))
        bar_right = width - 1

    shift_r = path_top - path_r0
    shift_c = path_left - path_c0
    seed = (shift_r, shift_c)
    grid[seed] = body
    expected[seed] = body

    for r, c in path[1:]:
        expected[r + shift_r, c + shift_c] = body

    for i, (turn_kind, length) in enumerate(zip(turns, steps)):
        row = bar_top + 2 * i
        total_len = length + 1
        if side == "left":
            c0 = 0
            c1 = total_len - 1
        else:
            assert bar_right is not None
            c1 = bar_right
            c0 = c1 - total_len + 1

        grid[row, c0 : c1 + 1] = body
        expected[row, c0 : c1 + 1] = body
        head_col = c0 if turn_kind == TURN_LEFT else c1
        grid[row, head_col] = head
        expected[row, head_col] = head

    changed = int(np.count_nonzero(expected != grid))
    assert changed == sum(steps), "independent renderer must add exactly one cell per turtle step"
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    program = _sample_program()
    if program is None:
        return None
    turns, steps, path = program

    rendered = _render_program(turns, steps, path)
    if rendered is None:
        return None
    grid, expected = rendered

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid turtle-program grid after {GEN_ATTEMPTS} attempts")

from __future__ import annotations

import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

DIR_BY_COLOR = {
    RED: (0, 1),
    BLUE: (1, 0),
    GREEN: (0, -1),
    YELLOW: (-1, 0),
}

BG_CHOICES = (0, 6, 8, 9)
CANVAS_MIN_COL = 2
GRID_HEIGHT_RANGE = (7, 18)
GRID_WIDTH_RANGE = (7, 18)
INSTRUCTION_LEN_RANGE = (4, 10)
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _instruction_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    instruction_colors = tuple(DIR_BY_COLOR)
    for c in range(w):
        for r in range(h):
            if int(grid[r, c]) in instruction_colors:
                return c
    raise ValueError("no instruction column found")


def _read_instructions(grid: np.ndarray, col: int) -> list[int]:
    return [int(grid[r, col]) for r in range(grid.shape[0]) if int(grid[r, col]) in DIR_BY_COLOR]


def _trace_move(
    start: tuple[int, int],
    delta: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], list[tuple[int, int]], bool]:
    h, w = shape
    r, c = start
    dr, dc = delta
    traversed: list[tuple[int, int]] = []
    clipped = False
    for _ in range(2):
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            clipped = True
            break
        r, c = nr, nc
        traversed.append((r, c))
    return (r, c), traversed, clipped


def _canvas_legal_exact_moves(pos: tuple[int, int], shape: tuple[int, int]) -> list[int]:
    h, w = shape
    r, c = pos
    out: list[int] = []
    if c + 2 < w:
        out.append(RED)
    if r + 2 < h:
        out.append(BLUE)
    if c - 2 >= CANVAS_MIN_COL:
        out.append(GREEN)
    if r - 2 >= 0:
        out.append(YELLOW)
    return out


def _advance_exact(pos: tuple[int, int], color: int) -> tuple[int, int]:
    dr, dc = DIR_BY_COLOR[color]
    return pos[0] + 2 * dr, pos[1] + 2 * dc


def _build_no_clip_walk(
    rng: random.Random,
    shape: tuple[int, int],
    length: int,
    start: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], list[int]] | None:
    h, w = shape
    if start is None:
        start = (rng.randrange(h), rng.randrange(CANVAS_MIN_COL, w))
    pos = start
    instructions: list[int] = []
    previous: int | None = None

    for _ in range(length):
        choices = _canvas_legal_exact_moves(pos, shape)
        if not choices:
            return None
        if previous in choices and len(choices) > 1 and rng.random() < 0.7:
            choices = [color for color in choices if color != previous]
        color = int(rng.choice(choices))
        instructions.append(color)
        pos = _advance_exact(pos, color)
        previous = color

    return start, instructions


def _build_clip_walk(
    rng: random.Random,
    shape: tuple[int, int],
    length: int,
) -> tuple[tuple[int, int], list[int]] | None:
    h, w = shape
    first = int(rng.choice((YELLOW, BLUE, RED)))
    if first == YELLOW:
        start = (int(rng.choice((0, 1))), rng.randrange(CANVAS_MIN_COL, w))
    elif first == BLUE:
        start = (int(rng.choice((h - 2, h - 1))), rng.randrange(CANVAS_MIN_COL, w))
    else:
        start = (rng.randrange(h), int(rng.choice((w - 2, w - 1))))

    pos, _cells, _clipped = _trace_move(start, DIR_BY_COLOR[first], shape)
    remainder = _build_no_clip_walk(rng, shape, length - 1, start=pos)
    if remainder is None:
        return None
    _, tail = remainder
    return start, [first] + tail


def _walk_summary(
    start: tuple[int, int],
    instructions: list[int],
    shape: tuple[int, int],
) -> tuple[int, bool]:
    clip_count = 0
    touches_left_panel = False
    pos = start
    for color in instructions:
        pos, cells, clipped = _trace_move(pos, DIR_BY_COLOR[color], shape)
        clip_count += int(clipped)
        if any(c < CANVAS_MIN_COL for _, c in cells) or pos[1] < CANVAS_MIN_COL:
            touches_left_panel = True
    return clip_count, touches_left_panel


def main(grid: np.ndarray) -> np.ndarray:
    """
    Execute the left-column color program and draw the orange brush trail.

    Input structure:
    - The dominant color is the background.
    - A vertical instruction sequence sits in the left column.
    - A single orange pixel marks the brush start.
    - A background separator column lies between the instructions and the canvas.

    Rule:
    - Read the instruction colors from top to bottom.
    - red/right, blue/down, green/left, yellow/up; every instruction moves by 2 cells.
    - After each move, paint the new brush position orange.
    - Any traversed background cell becomes grey.
    - If a move would leave the grid, stop on the boundary.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    instruction_col = _instruction_column(g)
    instructions = _read_instructions(g, instruction_col)
    if not instructions:
        return g.copy()

    starts = [(int(r), int(c)) for r, c in np.argwhere(g == ORANGE) if int(c) != instruction_col]
    if len(starts) != 1:
        raise ValueError("expected exactly one orange brush start")

    out = g.copy()
    pos = starts[0]
    for color in instructions:
        pos, traversed, _clipped = _trace_move(pos, DIR_BY_COLOR[int(color)], (h, w))
        if not traversed:
            out[pos] = ORANGE
            continue
        for rr, cc in traversed[:-1]:
            if int(out[rr, cc]) == bg:
                out[rr, cc] = GREY
        out[traversed[-1]] = ORANGE
    return out


# --------------- generator ---------------


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    max_len = min(INSTRUCTION_LEN_RANGE[1], h)
    if max_len < INSTRUCTION_LEN_RANGE[0]:
        return None
    instruction_len = rng.randint(INSTRUCTION_LEN_RANGE[0], max_len)
    bg = int(rng.choice(BG_CHOICES))

    clip_mode = rng.random() < 0.5
    if clip_mode:
        built = _build_clip_walk(rng, (h, w), instruction_len)
    else:
        built = _build_no_clip_walk(rng, (h, w), instruction_len)
    if built is None:
        return None

    start, instructions = built
    if len(set(instructions)) < 2:
        return None

    clip_count, touches_left_panel = _walk_summary(start, instructions, (h, w))
    if clip_mode and clip_count == 0:
        return None
    if not clip_mode and clip_count != 0:
        return None
    if touches_left_panel:
        return None

    seq_top = rng.randrange(0, h - instruction_len + 1)
    grid = np.full((h, w), bg, dtype=int)
    for i, color in enumerate(instructions):
        grid[seq_top + i, 0] = int(color)
    grid[start] = ORANGE

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out[:, :CANVAS_MIN_COL] != grid[:, :CANVAS_MIN_COL]):
        return None
    if int(np.count_nonzero(out == ORANGE)) < 2:
        return None
    if int(np.count_nonzero(out == GREY)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

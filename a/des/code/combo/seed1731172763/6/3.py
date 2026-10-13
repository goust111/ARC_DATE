import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

DIRS = {
    RED: (0, 1),
    BLUE: (1, 0),
    GREEN: (0, -1),
    YELLOW: (-1, 0),
}

OPPOSITE = {
    RED: GREEN,
    GREEN: RED,
    BLUE: YELLOW,
    YELLOW: BLUE,
}

GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_canvas_side(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    max_side = min(h, w - 1)
    for side in range(max_side, 0, -1):
        patch = grid[:side, w - side :]
        if patch.shape == (side, side) and np.all(patch == bg):
            return side
    return max_side


def _run_instructions(instructions: list[int], side: int, bg: int) -> tuple[np.ndarray, int, int, set[tuple[int, int]], tuple[int, int]]:
    out = np.full((side, side), bg, dtype=int)
    r = 0
    c = 0
    moved = 0
    skipped = 0
    visited = {(0, 0)}
    out[0, 0] = GREY

    for color in instructions:
        step = DIRS.get(int(color))
        if step is None:
            continue
        dr, dc = step
        nr = r + dr
        nc = c + dc
        if 0 <= nr < side and 0 <= nc < side:
            r = nr
            c = nc
            out[r, c] = int(color)
            moved += 1
            visited.add((r, c))
        else:
            skipped += 1

    out[0, 0] = GREY
    return out, moved, skipped, visited, (r, c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a left-column color instruction list into a path on the right-side canvas.

    Red = right, Blue = down, Green = left, Yellow = up.
    The output is the square canvas alone, with the origin marked in Grey.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    side = _find_canvas_side(g, bg)
    instructions = [int(color) for color in g[:, 0] if int(color) in DIRS]
    out, _moved, _skipped, _visited, _pos = _run_instructions(instructions, side, bg)
    return out


def _prefix_sequence(side: int, rng: random.Random) -> list[int]:
    mode = rng.choice(("skip_first", "return_skip", "top_edge_skip", "left_edge_skip"))
    if mode == "skip_first":
        return [rng.choice((GREEN, YELLOW)), rng.choice((RED, BLUE))]
    if mode == "return_skip":
        outward = rng.choice((RED, BLUE))
        return [outward, OPPOSITE[outward], rng.choice((GREEN, YELLOW))]
    if mode == "top_edge_skip":
        run = rng.randint(1, min(4, side - 2))
        return [RED] * run + [YELLOW]
    run = rng.randint(1, min(4, side - 2))
    return [BLUE] * run + [GREEN]


def _extend_sequence(prefix: list[int], side: int, rng: random.Random) -> list[int]:
    instructions = list(prefix)
    _out, _moved, _skipped, visited, (r, c) = _run_instructions(instructions, side, BLACK)

    while len(instructions) < side:
        valid: list[tuple[int, int, int]] = []
        invalid: list[int] = []
        for color, (dr, dc) in DIRS.items():
            nr = r + dr
            nc = c + dc
            if 0 <= nr < side and 0 <= nc < side:
                valid.append((color, nr, nc))
            else:
                invalid.append(color)

        if invalid and rng.random() < 0.18:
            instructions.append(rng.choice(invalid))
            continue

        frontier = [(color, nr, nc) for color, nr, nc in valid if (nr, nc) not in visited]
        if frontier and rng.random() < 0.85:
            color, nr, nc = rng.choice(frontier)
        else:
            color, nr, nc = rng.choice(valid)

        instructions.append(color)
        r = nr
        c = nc
        visited.add((r, c))

    return instructions


def _good_sequence(instructions: list[int], side: int) -> bool:
    out, moved, skipped, visited, _pos = _run_instructions(instructions, side, BLACK)
    if moved < 2 or skipped < 1:
        return False
    if len(set(instructions)) < 3:
        return False
    if len(visited) < min(4, side):
        return False
    if np.count_nonzero(out != BLACK) < 4:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.randint(7, 15)
    gap = rng.randint(1, 4)

    prefix = _prefix_sequence(side, rng)
    instructions = _extend_sequence(prefix, side, rng)
    if not _good_sequence(instructions, side):
        return None

    width = 1 + gap + side
    grid = np.full((side, width), BLACK, dtype=int)
    grid[:, 0] = np.array(instructions, dtype=int)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid instruction/canvas grid")

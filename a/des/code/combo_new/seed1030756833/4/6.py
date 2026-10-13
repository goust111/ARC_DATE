import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
ORANGE = 7

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TURN_DELTA = {
    GREEN: 0,
    RED: 1,
    YELLOW: 2,
    BLUE: -1,
}
INSTRUCTION_COLORS = tuple(TURN_DELTA)


def _turn(direction: tuple[int, int], instruction: int) -> tuple[int, int]:
    assert instruction in TURN_DELTA, "unknown top-edge instruction color"
    idx = DIRS.index(direction)
    return DIRS[(idx + TURN_DELTA[int(instruction)]) % len(DIRS)]


def _instruction_sequence(g: np.ndarray) -> list[int]:
    seq = []
    for value in g[0, :]:
        color = int(value)
        if color == BLACK:
            continue
        assert color in TURN_DELTA, "top edge may contain only instruction colors and black gaps"
        seq.append(color)
    assert seq, "expected at least one instruction on the top edge"
    return seq


def _start_and_direction(g: np.ndarray) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    orange = np.argwhere(g[1:, :] == ORANGE)
    gray = np.argwhere(g[1:, :] == GRAY)
    assert len(orange) == 1, "expected one non-top-row orange start pixel"
    assert len(gray) == 1, "expected one non-top-row gray direction marker"

    sr = int(orange[0, 0] + 1)
    sc = int(orange[0, 1])
    gr = int(gray[0, 0] + 1)
    gc = int(gray[0, 1])
    direction = (gr - sr, gc - sc)
    assert direction in DIRS, "gray marker must be orthogonally adjacent to the start"
    return sr, sc, direction, (gr, gc)


def _trace(grid: np.ndarray, collect: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert 1 <= h <= 30 and 1 <= w <= 30, "ARC grids must fit within 30x30"

    seq = _instruction_sequence(g)
    r, c, direction, marker = _start_and_direction(g)
    out = g.copy()
    out[marker] = BLACK

    moved = 0
    skipped = 0
    visited = [(int(r), int(c))]
    for instruction in seq:
        direction = _turn(direction, int(instruction))
        nr = r + direction[0]
        nc = c + direction[1]
        if 0 <= nr < h and 0 <= nc < w:
            r, c = nr, nc
            out[r, c] = ORANGE
            moved += 1
            visited.append((int(r), int(c)))
        else:
            skipped += 1

    out[0, :] = g[0, :]
    if collect:
        return out, {"moved": moved, "skipped": skipped, "visited": visited}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Decode top-edge turn instructions and draw the resulting orange polyline path.

    Input structure:
    - The background is black.
    - The top row contains a left-to-right sequence of red, blue, green, and
      yellow instruction pixels, with optional black gaps between them.
    - Below the top row there is one orange start pixel and one gray direction
      marker orthogonally adjacent to it.

    Rule:
    - The gray marker gives the initial facing direction from the orange start
      toward the gray cell.
    - Read the non-black top-edge pixels from left to right: red turns right,
      blue turns left, green keeps the direction, and yellow turns around.
    - After each turn, step one cell in the current direction and paint that
      cell orange; if that step would leave the grid, do not move for that
      instruction.
    - Erase the gray marker to black and leave the top-edge instruction row
      unchanged in the output.

    Key insight:
    - The edge colors are a compact turn program, while the gray neighbor is
      only an initial heading marker for simulating the path.
    """
    return _trace(grid)


# --------------- generator ---------------

GRID_ROWS_RANGE = (8, 16)
GRID_COLS_RANGE = (14, 30)
GEN_RETRIES = 500
MIN_TOTAL_INSTRUCTIONS = 10
MAX_TOTAL_INSTRUCTIONS = 26
SKIP_TARGETS = ("left", "right", "bottom")


def _turn_color(current: tuple[int, int], target: tuple[int, int]) -> int:
    delta = (DIRS.index(target) - DIRS.index(current)) % len(DIRS)
    if delta == 0:
        return GREEN
    if delta == 1:
        return RED
    if delta == 2:
        return YELLOW
    assert delta == 3
    return BLUE


def _near_edge_start(h: int, w: int, target: str) -> tuple[int, int, tuple[int, int], int]:
    if target == "left":
        sr = random.randint(2, h - 3)
        sc = random.randint(2, min(6, w - 3))
        direction = (0, -1)
        distance = sc
    elif target == "right":
        sr = random.randint(2, h - 3)
        sc = random.randint(max(2, w - 7), w - 3)
        direction = (0, 1)
        distance = w - 1 - sc
    else:
        assert target == "bottom"
        sr = random.randint(max(2, h - 7), h - 3)
        sc = random.randint(2, w - 3)
        direction = (1, 0)
        distance = h - 1 - sr
    return sr, sc, direction, distance


def _place_marker_for_direction(
    grid: np.ndarray, sr: int, sc: int, initial_direction: tuple[int, int]
) -> bool:
    gr = sr + initial_direction[0]
    gc = sc + initial_direction[1]
    h, w = grid.shape
    if not (1 <= gr < h and 0 <= gc < w):
        return False
    grid[sr, sc] = ORANGE
    grid[gr, gc] = GRAY
    return True


def _instruction_row(width: int, sequence: list[int]) -> np.ndarray:
    top = np.zeros(width, dtype=int)
    cols = sorted(random.sample(range(width), len(sequence)))
    for c, color in zip(cols, sequence):
        top[c] = int(color)
    return top


def _make_sequence(width: int, initial_direction: tuple[int, int], target_direction: tuple[int, int], distance: int) -> list[int] | None:
    prefix = [_turn_color(initial_direction, target_direction)] + [GREEN] * distance
    minimum = max(MIN_TOTAL_INSTRUCTIONS, len(prefix) + 4)
    maximum = min(width - 2, MAX_TOTAL_INSTRUCTIONS)
    if minimum > maximum:
        return None

    total = random.randint(minimum, maximum)
    suffix_len = total - len(prefix)
    suffix = [color for color in INSTRUCTION_COLORS if color not in prefix]
    if len(suffix) > suffix_len:
        return None
    while len(suffix) < suffix_len:
        suffix.append(random.choice(INSTRUCTION_COLORS))
    random.shuffle(suffix)
    return prefix + suffix


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    target = random.choice(SKIP_TARGETS)
    sr, sc, target_direction, distance = _near_edge_start(h, w, target)
    initial_direction = random.choice(DIRS)

    sequence = _make_sequence(w, initial_direction, target_direction, distance)
    if sequence is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = _instruction_row(w, sequence)
    if not _place_marker_for_direction(grid, sr, sc, initial_direction):
        return None

    try:
        out, stats = _trace(grid, collect=True)
    except Exception:
        return None

    visited = stats["visited"]
    if stats["skipped"] < 1 or stats["moved"] < 5:
        return None
    if any(r == 0 for r, _c in visited):
        return None
    if len(set(visited)) < 5:
        return None
    if not np.array_equal(out[0, :], grid[0, :]):
        return None
    if not all(color in sequence for color in INSTRUCTION_COLORS):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")

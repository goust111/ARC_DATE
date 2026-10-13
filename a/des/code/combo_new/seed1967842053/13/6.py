import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

TOWER_COLORS = (BLUE, RED, YELLOW)
MIX_COLORS = {
    frozenset((BLUE, RED)): MAGENTA,
    frozenset((BLUE, YELLOW)): GREEN,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, RED, YELLOW)): GREY,
}


def _mixed_color(colors):
    colors = frozenset(colors)
    if not colors:
        return 0
    if len(colors) == 1:
        return next(iter(colors))
    return MIX_COLORS.get(colors, GREY)


def main(grid: np.ndarray) -> np.ndarray:
    """Broadcast odd Chebyshev-distance square ripples from colored signal towers.

    Input structure:
    The input is a black background containing isolated one-cell signal towers.
    Each tower is blue, red, or yellow, and all non-black cells are treated as
    tower centers.

    Rule:
    Every tower emits square rings centered on itself.  A cell is covered by a
    tower when its Chebyshev distance from that tower is positive and odd; even
    distances remain black unless covered by another odd ring.  Single-color
    coverage keeps that tower color.  Mixed coverage maps blue+red to magenta,
    blue+yellow to green, red+yellow to orange, and all three colors to grey.
    Tower cells themselves always keep their original input color.

    Key insight:
    The visible waves are not continuous fills but odd-distance square rings,
    and overlapping rings must be interpreted as color-set mixtures.
    """
    arr = np.asarray(grid)
    out = np.zeros(arr.shape, dtype=arr.dtype)
    towers = []
    for r, c in np.argwhere(arr != 0):
        color = int(arr[r, c])
        towers.append((int(r), int(c), color))
        out[r, c] = arr[r, c]

    tower_positions = {(r, c) for r, c, _ in towers}
    rows, cols = arr.shape
    for r in range(rows):
        for c in range(cols):
            if (r, c) in tower_positions:
                continue
            colors = set()
            for tr, tc, color in towers:
                if color not in TOWER_COLORS:
                    continue
                distance = max(abs(r - tr), abs(c - tc))
                if distance % 2 == 1:
                    colors.add(color)
            out[r, c] = _mixed_color(colors)
    return out


def _choose_positions(rows, cols, count, min_gap):
    positions = []
    candidates = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if all(max(abs(r - pr), abs(c - pc)) >= min_gap for pr, pc in positions):
            positions.append((r, c))
            if len(positions) == count:
                return positions
    return positions


def generate_input() -> np.ndarray:
    rows = random.randint(12, 30)
    cols = random.randint(12, 30)
    use_three_colors = random.random() < 0.5
    palette = list(TOWER_COLORS if use_three_colors else random.sample(TOWER_COLORS, 2))
    count = random.randint(len(palette), min(6, len(palette) + 3))

    min_gap = random.choice((4, 5, 6))
    positions = _choose_positions(rows, cols, count, min_gap)
    while len(positions) < count:
        rows = random.randint(14, 30)
        cols = random.randint(14, 30)
        min_gap = random.choice((3, 4, 5))
        positions = _choose_positions(rows, cols, count, min_gap)

    colors = palette[:]
    while len(colors) < count:
        colors.append(random.choice(palette))
    random.shuffle(colors)

    grid = np.zeros((rows, cols), dtype=int)
    for (r, c), color in zip(positions, colors):
        grid[r, c] = color
    return grid

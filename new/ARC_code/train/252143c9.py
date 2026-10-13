from __future__ import annotations

import numpy as np

from arc_utils import draw_line, get_bg
import math


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the center color and redraw it as a diagonal ray to its indicated corner.

    Input structure:
    - The grid is an odd square with a unique dominant background color.
    - The center pixel is foreground.
    - Every off-center pixel with the center color lies in the same diagonal
      quadrant relative to the center.

    Rule:
    - Fill the whole output with the background color.
    - Use the quadrant of the center-colored pixels to choose one diagonal
      direction from the center.
    - Draw a continuous line of the center color from the center to the grid
      edge in that direction.

    Key insight:
    The color at the exact center identifies the surviving color, while the
    other pixels of that color identify which corner the center should connect to.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1, "grid must be an odd square"

    bg = get_bg(g)
    vals, counts = np.unique(g, return_counts=True)
    assert int(np.sum(counts == counts.max())) == 1, "background must be unique"
    assert int(counts[vals == bg][0]) == int(counts.max()), "background must be dominant"

    center = (h // 2, w // 2)
    color = int(g[center])
    assert color != bg, "center pixel must be foreground"

    directions: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == color):
        r = int(r)
        c = int(c)
        if (r, c) == center:
            continue
        dr = _sign(r - center[0])
        dc = _sign(c - center[1])
        assert dr != 0 and dc != 0, "center-colored pixels must be diagonal markers"
        directions.add((dr, dc))

    assert len(directions) == 1, "center color must indicate one diagonal direction"
    direction = next(iter(directions))

    out = np.full_like(g, bg)
    draw_line(out, center[0], center[1], direction=direction, color=color)
    return out

# --------------- generator (o5x) ---------------

# Canvas sizes: side = 2 * n + 1, giving 7, 9, 11, or 13.
N_MIN, N_MAX = 3, 6
# Palette: one background color plus one color per quadrant.
COLOR_COUNT, PALETTE_SIZE = 10, 5
# Density targets observed in the examples and enforced jointly.
QUADRANT_FILL_MIN, QUADRANT_FILL_MAX = 0.33, 0.56
BACKGROUND_FILL_MIN, BACKGROUND_FILL_MAX = 0.63, 0.69
CENTER_PIXEL_COUNT = 1
# Generation retry limit.
MAX_GENERATION_ATTEMPTS = 300
QUADRANT_DIRECTIONS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _quadrant_cells(n: int, dr: int, dc: int) -> list[tuple[int, int]]:
    rows = range(0, n) if dr < 0 else range(n + 1, 2 * n + 1)
    cols = range(0, n) if dc < 0 else range(n + 1, 2 * n + 1)
    return [(r, c) for r in rows for c in cols]


def _choose_fill_counts(rng: np.random.Generator, n: int) -> list[int]:
    q_area = n * n
    min_k = math.ceil(QUADRANT_FILL_MIN * q_area)
    max_k = math.floor(QUADRANT_FILL_MAX * q_area)
    side = 2 * n + 1
    cells = side * side
    bg_min = math.ceil(BACKGROUND_FILL_MIN * cells)
    bg_max = math.floor(BACKGROUND_FILL_MAX * cells)
    quadrant_count = len(QUADRANT_DIRECTIONS)
    total_min = max(quadrant_count * min_k, cells - CENTER_PIXEL_COUNT - bg_max)
    total_max = min(quadrant_count * max_k, cells - CENTER_PIXEL_COUNT - bg_min)
    target = int(rng.integers(total_min, total_max + 1))

    counts: list[int] = []
    remaining = target
    for slots in range(quadrant_count, 0, -1):
        lo = max(min_k, remaining - (slots - 1) * max_k)
        hi = min(max_k, remaining - (slots - 1) * min_k)
        k = int(rng.integers(lo, hi + 1))
        counts.append(k)
        remaining -= k
    rng.shuffle(counts)
    return counts


def _ordered_directions(rng: np.random.Generator, answer_direction: tuple[int, int]) -> list[tuple[int, int]]:
    """Return directions with the center-color quadrant first and distractors shuffled."""
    directions = list(QUADRANT_DIRECTIONS)
    rng.shuffle(directions)
    directions.remove(answer_direction)
    directions.insert(0, answer_direction)
    return directions


def _render_grid(rng: np.random.Generator, n: int, bg: int, quadrant_colors: list[int], directions: list[tuple[int, int]]) -> np.ndarray:
    side = 2 * n + 1
    grid = np.full((side, side), bg, dtype=int)
    center_color = quadrant_colors[0]
    grid[n, n] = center_color

    for (dr, dc), color, count in zip(directions, quadrant_colors, _choose_fill_counts(rng, n)):
        cells = _quadrant_cells(n, dr, dc)
        chosen = rng.choice(len(cells), size=count, replace=False)
        for idx in chosen:
            r, c = cells[int(idx)]
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(N_MIN, N_MAX + 1))
    colors = rng.choice(np.arange(COLOR_COUNT), size=PALETTE_SIZE, replace=False).astype(int).tolist()
    bg = colors[0]
    quadrant_colors = colors[1:]
    direction = QUADRANT_DIRECTIONS[int(rng.integers(0, len(QUADRANT_DIRECTIONS)))]
    directions = _ordered_directions(rng, direction)
    grid = _render_grid(rng, n, bg, quadrant_colors, directions)
    expected = np.full_like(grid, bg)
    draw_line(expected, n, n, direction=direction, color=quadrant_colors[0])
    try:
        return grid if np.array_equal(main(grid), expected) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 252143c9 input")

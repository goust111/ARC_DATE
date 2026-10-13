from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg
import random


COLOR_OFFSETS = {
    Color.ORANGE: (-2, 0),
    Color.RED: (0, -2),
    Color.MAROON: (2, 0),
    Color.BLUE: (0, 1),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every foreground pixel by the fixed offset indicated by its color.

    Input structure:
    - The grid has a unique dominant background color.
    - The foreground is a sparse set of colored pixels.
    - Each foreground color is one of the trained motion colors: orange, red,
      maroon, or blue.

    Rule:
    - Clear the canvas to the background color.
    - Translate orange pixels up 2 cells, red pixels left 2 cells, maroon
      pixels down 2 cells, and blue pixels right 1 cell.
    - Preserve each moved pixel's original color.

    Key insight:
    The foreground color is the motion instruction; position only determines
    where that color-specific displacement lands.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    vals, counts = np.unique(g, return_counts=True)
    assert int(np.sum(counts == counts.max())) == 1, "background must be unique"
    assert int(counts[vals == bg][0]) == int(counts.max()), "background must be dominant"

    out = np.full_like(g, bg)
    occupied: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g != bg):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        assert color in COLOR_OFFSETS, "foreground color must have a known motion"
        dr, dc = COLOR_OFFSETS[color]
        nr = r + dr
        nc = c + dc
        assert 0 <= nr < h and 0 <= nc < w, "translated pixel must stay in bounds"
        assert (nr, nc) not in occupied, "translated foreground pixels must not collide"
        out[nr, nc] = color
        occupied.add((nr, nc))

    return out

# --------------- generator (att2, o5x) ---------------
# Canvas and foreground size bounds.
BACKGROUND_COLOR = 8
GRID_SIZE_RANGE = (10, 16)
FOREGROUND_COUNT_RANGE = (1, 10)

# Color movement options.
COLOR_OFFSETS = {7: (-2, 0), 2: (0, -2), 9: (2, 0), 1: (0, 1)}
MAX_SELECTED_COLORS = 4

# Placement constraints and retry budget.
ISOLATION_RADIUS = 1
GENERATION_RETRIES = 300


def _positive_counts(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _sample_canvas_and_colors() -> tuple[int, list[int]]:
    n = random.randint(*GRID_SIZE_RANGE)
    total = random.randint(*FOREGROUND_COUNT_RANGE)
    num_colors = random.randint(1, min(MAX_SELECTED_COLORS, total))
    colors = random.sample(list(COLOR_OFFSETS), num_colors)
    counts = _positive_counts(total, num_colors)

    color_sequence = []
    for color, count in zip(colors, counts):
        color_sequence.extend([color] * count)
    random.shuffle(color_sequence)
    return n, color_sequence


def _is_isolated(pos: tuple[int, int], taken: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr in range(r - ISOLATION_RADIUS, r + ISOLATION_RADIUS + 1):
        for cc in range(c - ISOLATION_RADIUS, c + ISOLATION_RADIUS + 1):
            if (rr, cc) in taken:
                return False
    return True


def _render_sequence(n: int, color_sequence: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    """Render sampled movements, rejecting plans that run out of legal positions."""
    grid = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    out = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    sources, destinations, occupied = set(), set(), set()
    for color in color_sequence:
        dr, dc = COLOR_OFFSETS[color]
        candidates = []
        for r in range(max(0, -dr), min(n, n - dr)):
            for c in range(max(0, -dc), min(n, n - dc)):
                src = (r, c)
                dst = (r + dr, c + dc)
                if (
                    src not in occupied
                    and dst not in occupied
                    and _is_isolated(src, sources)
                    and _is_isolated(dst, destinations)
                ):
                    candidates.append((src, dst))

        if not candidates:
            return None

        (r, c), (nr, nc) = random.choice(candidates)
        grid[r, c] = color
        out[nr, nc] = color
        sources.add((r, c))
        destinations.add((nr, nc))
        occupied.add((r, c))
        occupied.add((nr, nc))

    return grid, out


def _gen_attempt() -> np.ndarray | None:
    n, color_sequence = _sample_canvas_and_colors()
    rendered = _render_sequence(n, color_sequence)
    if rendered is None:
        return None
    grid, out = rendered
    return grid if np.array_equal(main(grid), out) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 342dd610 input")

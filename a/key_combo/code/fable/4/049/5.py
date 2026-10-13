from __future__ import annotations

from collections import Counter
import random

import numpy as np


ARC_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400
GEN_BRANCHES = ("short", "long")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _parse_instance(grid: np.ndarray) -> tuple[int, int, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2, "expected a 2D grid with a palette row and lower area"

    bg = _get_bg(g)
    top = g[0]
    gap_cols = np.flatnonzero(top == bg)
    assert gap_cols.size == 1, "expected exactly one background gap in the top palette row"
    gap_col = int(gap_cols[0])

    palette_colors = [int(v) for i, v in enumerate(top) if i != gap_col]
    assert len(set(palette_colors)) == len(palette_colors), "expected distinct non-gap palette colors"
    assert all(color != bg for color in palette_colors), "only the gap may use the background color in the palette row"

    lower_fg = np.argwhere(g[1:] != bg)
    assert lower_fg.shape[0] == 1, "expected exactly one isolated control pixel below the palette row"
    control = (1 + int(lower_fg[0, 0]), int(lower_fg[0, 1]))
    return bg, gap_col, control


def _ring_color(top: np.ndarray, gap_col: int, distance: int, bg: int) -> int:
    color = int(top[(gap_col + int(distance)) % top.shape[0]])
    return bg if color == bg else color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top palette row as a cyclic color wheel to paint concentric square rings around the lone control pixel.

    Input structure
    - The top row is a palette row with exactly one background gap and otherwise pairwise-distinct colors.
    - Every row below is background except for one isolated control pixel.

    Rule
    - Keep the palette row unchanged.
    - For each pixel below the palette row, compute its Chebyshev distance from the control pixel.
    - Starting at the gap cell and reading the palette row cyclically to the right, use the color at offset distance d; if that offset lands on the gap, keep background instead.

    Key insight
    - The top row is a one-dimensional cyclic palette, and Chebyshev-distance rings unwrap it into square annual rings whose zero ring erases the control pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg, gap_col, control = _parse_instance(g)
    out = g.copy()
    top = g[0]
    cr, cc = control

    for r in range(1, g.shape[0]):
        for c in range(g.shape[1]):
            distance = max(abs(r - cr), abs(c - cc))
            out[r, c] = _ring_color(top, gap_col, distance, bg)

    out[0] = top
    return out


def _max_lower_distance(height: int, width: int, control: tuple[int, int]) -> int:
    cr, cc = control
    corners = ((1, 0), (1, width - 1), (height - 1, 0), (height - 1, width - 1))
    return max(max(abs(r - cr), abs(c - cc)) for r, c in corners)


def _build_palette_row(width: int, bg: int) -> tuple[np.ndarray, int]:
    gap_col = random.randrange(width)
    colors = random.sample([color for color in ARC_COLORS if color != bg], width - 1)
    row = np.empty(width, dtype=int)
    idx = 0
    for c in range(width):
        if c == gap_col:
            row[c] = bg
        else:
            row[c] = colors[idx]
            idx += 1
    return row, gap_col


def _choose_control(height: int, width: int, want_long_branch: bool) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for r in range(1, height):
        for c in range(width):
            max_distance = _max_lower_distance(height, width, (r, c))
            if want_long_branch and max_distance >= width:
                candidates.append((r, c))
            if not want_long_branch and max_distance < width:
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_input(branch: str) -> np.ndarray | None:
    if branch == "long":
        width = random.randint(4, 8)
        height = random.randint(width + 2, min(22, width + 12))
        if height <= width:
            return None
    else:
        width = random.randint(5, 10)
        upper_height = min(14, width)
        if upper_height < 5:
            return None
        height = random.randint(5, upper_height)

    bg = random.choice(ARC_COLORS)
    top, gap_col = _build_palette_row(width, bg)
    control = _choose_control(height, width, want_long_branch=(branch == "long"))
    if control is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[0] = top

    # Fix the irrelevant control-pixel color to the first visible ring color to avoid hidden ambiguity.
    control_color = int(top[(gap_col + 1) % width])
    grid[control] = control_color

    if branch == "long" and _max_lower_distance(height, width, control) < width:
        return None
    if branch == "short" and _max_lower_distance(height, width, control) >= width:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or not np.array_equal(out[0], top):
        return None

    hollow_count = int(np.count_nonzero(out[1:] == bg))
    if branch == "long" and hollow_count <= 1:
        return None
    if branch == "short" and hollow_count != 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        branch = random.choice(GEN_BRANCHES)
        grid = _build_input(branch)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid palette-rings input")

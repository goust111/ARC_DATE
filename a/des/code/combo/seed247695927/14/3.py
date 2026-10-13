from __future__ import annotations

from collections import Counter
import numpy as np


RED = 2
YELLOW = 4
CANVAS_SIZE = 16
MAX_RETRIES = 300
RNG = np.random.default_rng()


def _background_color(grid: np.ndarray) -> int:
    corners = [
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    ]
    return Counter(corners).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace scattered markers by a bottom-left aligned red rectangle.

    The rectangle height is the smaller count of the two foreground colors and
    the width is the larger count. Two rising yellow diagonals start from the
    rectangle's bottom-left and bottom-right corners.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    counts = Counter(map(int, g.reshape(-1)))
    fg_counts = {color: count for color, count in counts.items() if color != bg}
    assert len(fg_counts) == 2, "expected exactly two foreground colors"

    sizes = sorted(int(count) for count in fg_counts.values())
    height, width = sizes[0], sizes[1]
    assert 1 <= height < width <= CANVAS_SIZE

    out = np.full((CANVAS_SIZE, CANVAS_SIZE), bg, dtype=int)
    top = CANVAS_SIZE - height
    out[top:CANVAS_SIZE, 0:width] = RED

    for k in range(height):
        row = CANVAS_SIZE - 1 - k
        out[row, k] = YELLOW
        out[row, width - 1 - k] = YELLOW

    return out


def _sample_positions(
    height: int,
    width: int,
    total: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    RNG.shuffle(cells)

    chosen: list[tuple[int, int]] = []
    blocked = set(occupied)
    for r, c in cells:
        if (r, c) in blocked:
            continue
        chosen.append((r, c))
        if len(chosen) == total:
            return chosen
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((nr, nc))

    remaining = [cell for cell in cells if cell not in occupied and cell not in chosen]
    need = total - len(chosen)
    if need <= len(remaining):
        chosen.extend(remaining[:need])
        return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = int(RNG.integers(0, 10))
    marker_colors = [c for c in range(10) if c not in (bg, RED, YELLOW)]
    color_a, color_b = RNG.choice(marker_colors, size=2, replace=False)

    small = int(RNG.integers(1, CANVAS_SIZE))
    large = int(RNG.integers(small + 1, CANVAS_SIZE + 1))

    min_h = max(8, small + 2)
    min_w = max(8, large + 2)
    height = int(RNG.integers(min_h, 21))
    width = int(RNG.integers(min_w, 21))

    total = small + large
    corners = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
    positions = _sample_positions(height, width, total, corners)
    if positions is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    for r, c in positions[:small]:
        grid[r, c] = int(color_a)
    for r, c in positions[small:]:
        grid[r, c] = int(color_b)

    fg = [int(x) for x in np.unique(grid) if int(x) != bg]
    if len(fg) != 2:
        return None

    counts = sorted(int(np.count_nonzero(grid == color)) for color in fg)
    if counts != [small, large]:
        return None
    try:
        main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

import numpy as np


BG = 0
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MIN_CENTER = 2
MIN_GAP = 4
MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _axis_markers(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    row_markers = [(r, int(grid[r, 0])) for r in range(1, h) if int(grid[r, 0]) != bg]
    col_markers = [(c, int(grid[0, c])) for c in range(1, w) if int(grid[0, c]) != bg]
    return row_markers, col_markers


def _paint_symbol(out: np.ndarray, center_r: int, center_c: int, row_color: int, col_color: int) -> None:
    h, w = out.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = center_r + dr
            cc = center_c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if rr == 0 or cc == 0:
                continue
            if row_color == col_color:
                out[rr, cc] = row_color
            elif dr == 0 and dc == 0:
                out[rr, cc] = col_color
            else:
                out[rr, cc] = row_color


def main(grid: np.ndarray) -> np.ndarray:
    """Draw 3x3 markers at row/column-axis intersections."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    row_markers, col_markers = _axis_markers(g, bg)

    out = g.copy()
    for r, row_color in row_markers:
        for c, col_color in col_markers:
            _paint_symbol(out, int(r), int(c), int(row_color), int(col_color))
    return out


def _sample_spaced_positions(rng: np.random.Generator, size: int, count: int) -> list[int]:
    max_center = size - 2
    baseline_last = MIN_CENTER + MIN_GAP * (count - 1)
    slack = max_center - baseline_last
    if slack < 0:
        raise ValueError("grid too small for requested marker count")
    extras = np.sort(rng.integers(0, slack + 1, size=count))
    return [MIN_CENTER + MIN_GAP * i + int(extra) for i, extra in enumerate(extras)]


def _has_both_branches(row_colors: list[int], col_colors: list[int]) -> bool:
    same = False
    different = False
    for row_color in row_colors:
        for col_color in col_colors:
            if row_color == col_color:
                same = True
            else:
                different = True
    return same and different


def _forced_colors(rng: np.random.Generator, row_count: int, col_count: int) -> tuple[list[int], list[int]]:
    palette_size = int(rng.integers(2, 5))
    palette = [int(x) for x in rng.choice(PALETTE, size=palette_size, replace=False)]
    shared_color = int(rng.choice(palette))
    different_choices = [color for color in palette if color != shared_color]
    different_color = int(rng.choice(different_choices))

    row_colors = [shared_color, different_color]
    if row_count > 2:
        row_colors.extend(int(x) for x in rng.choice(palette, size=row_count - 2, replace=True))

    col_colors = [shared_color]
    if col_count > 1:
        col_colors.extend(int(x) for x in rng.choice(palette, size=col_count - 1, replace=True))

    if rng.random() < 0.5 and col_count > 1:
        col_colors[1] = different_color

    rng.shuffle(row_colors)
    rng.shuffle(col_colors)
    return row_colors, col_colors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_count = int(rng.integers(2, 5))
    col_count = int(rng.integers(2, 5))

    min_height = 5 + MIN_GAP * (row_count - 1)
    min_width = 5 + MIN_GAP * (col_count - 1)
    height = int(rng.integers(min_height, min_height + 9))
    width = int(rng.integers(min_width, min_width + 9))

    row_positions = _sample_spaced_positions(rng, height, row_count)
    col_positions = _sample_spaced_positions(rng, width, col_count)
    row_colors, col_colors = _forced_colors(rng, row_count, col_count)

    if not _has_both_branches(row_colors, col_colors):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for r, color in zip(row_positions, row_colors):
        grid[r, 0] = int(color)
    for c, color in zip(col_positions, col_colors):
        grid[0, c] = int(color)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

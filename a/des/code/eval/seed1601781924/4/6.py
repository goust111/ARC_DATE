import random

import numpy as np


BG = 8
COLOR_POOL = (0, 1, 2, 3, 4, 5, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 300


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if indices.size == 0:
        return runs
    start = int(indices[0])
    prev = start
    for value in map(int, indices[1:]):
        if value != prev + 1:
            runs.append((start, prev))
            start = value
        prev = value
    runs.append((start, prev))
    return runs


def _read_legend_and_structure(grid: np.ndarray) -> tuple[list[int], tuple[int, int, int, int]]:
    occupied_cols = np.where((grid != BG).any(axis=0))[0]
    runs = _contiguous_runs(occupied_cols)
    assert len(runs) == 2, "expected one legend column and one main structure block"

    (legend_c0, legend_c1), (main_c0, main_c1) = runs
    assert legend_c0 == legend_c1, "legend must occupy exactly one column"

    legend_rows = np.where(grid[:, legend_c0] != BG)[0]
    assert legend_rows.size > 0, "legend must contain at least one color"
    if legend_rows.size > 1:
        gaps = np.diff(legend_rows)
        assert np.all(gaps == 2), "legend pixels must be separated by one background row"

    legend_colors = [int(grid[int(row), legend_c0]) for row in legend_rows]
    assert len(set(legend_colors)) == len(legend_colors), "legend colors must be distinct"

    main_mask = grid[:, main_c0 : main_c1 + 1] != BG
    main_rows = np.where(main_mask.any(axis=1))[0]
    assert main_rows.size > 0, "main structure must be present"
    r0 = int(main_rows[0])
    r1 = int(main_rows[-1])
    return legend_colors, (r0, main_c0, r1, main_c1)


def _shell_color(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, solid: bool) -> int:
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    if solid:
        colors = np.unique(window)
    else:
        top = window[0, :]
        bottom = window[-1, :]
        left = window[1:-1, 0]
        right = window[1:-1, -1]
        colors = np.unique(np.concatenate((top, bottom, left, right)))
    assert colors.size == 1, "each layer must have a single color"
    color = int(colors[0])
    assert color != BG, "layer color must differ from background"
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split a concentric rectangle stack into solid rectangles ordered by the legend.

    The input uses a single legend column on the left and one concentric structure
    on the right. Each legend pixel defines the output order, while each layer in
    the concentric structure contributes one solid rectangle whose size equals that
    layer's outer bounding box.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_colors, (r0, c0, r1, c1) = _read_legend_and_structure(g)
    layer_count = len(legend_colors)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    assert outer_h - 2 * (layer_count - 1) > 0
    assert outer_w - 2 * (layer_count - 1) > 0

    dims_by_color: dict[int, tuple[int, int]] = {}
    for depth in range(layer_count):
        rr0 = r0 + depth
        cc0 = c0 + depth
        rr1 = r1 - depth
        cc1 = c1 - depth
        color = _shell_color(g, rr0, cc0, rr1, cc1, solid=depth == layer_count - 1)
        dims_by_color[color] = (rr1 - rr0 + 1, cc1 - cc0 + 1)

    assert set(dims_by_color) == set(legend_colors), "legend colors must match layer colors"

    out_h = max(height for height, _ in dims_by_color.values())
    out_w = sum(width for _, width in (dims_by_color[color] for color in legend_colors)) + layer_count - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for color in legend_colors:
        height, width = dims_by_color[color]
        out[:height, col : col + width] = color
        col += width + 1

    return out


def _draw_structure(
    grid: np.ndarray,
    top: int,
    left: int,
    outer_h: int,
    outer_w: int,
    colors: list[int],
) -> None:
    last = len(colors) - 1
    for depth, color in enumerate(colors):
        r0 = top + depth
        c0 = left + depth
        r1 = top + outer_h - depth - 1
        c1 = left + outer_w - depth - 1
        if depth == last:
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            continue
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    layer_count = rng.randint(2, 6)
    inner_h = rng.randint(1, 6)
    inner_w = rng.randint(1, 6)
    outer_h = inner_h + 2 * (layer_count - 1)
    outer_w = inner_w + 2 * (layer_count - 1)
    legend_h = 2 * layer_count - 1

    nested_colors = rng.sample(COLOR_POOL, layer_count)
    if rng.random() < 0.5:
        legend_colors = list(nested_colors)
    else:
        legend_colors = list(nested_colors)
        while legend_colors == list(nested_colors):
            rng.shuffle(legend_colors)

    top_padding = rng.randint(0, 3)
    bottom_padding = rng.randint(0, 3)
    grid_h = max(outer_h, legend_h) + top_padding + bottom_padding
    legend_top = rng.randint(0, grid_h - legend_h)
    main_top = rng.randint(0, grid_h - outer_h)

    left_padding = rng.randint(0, 3)
    right_padding = rng.randint(0, 3)
    grid_w = left_padding + 1 + 1 + outer_w + right_padding
    legend_col = left_padding
    main_col = left_padding + 2

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for idx, color in enumerate(legend_colors):
        grid[legend_top + 2 * idx, legend_col] = color
    _draw_structure(grid, main_top, main_col, outer_h, outer_w, list(nested_colors))

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape[0] != outer_h:
        return None
    if len(np.unique(out[out != BG])) != layer_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

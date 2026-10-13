from collections import Counter

import numpy as np
import random


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty region"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the noisy four-region board split by a solid cross into a 3x3 color map.

    Input structure:
    - A single full row and a single full column have the same solid separator color.
    - The separator cross divides the input into four nonempty rectangular regions.
    - Each region has a unique dominant color, with other colors acting as noise.

    Rule:
    - Find the unique solid same-color row/column pair.
    - Fill the middle row and middle column of the 3x3 output with the separator color.
    - Put the unique dominant color of each quadrant into the corresponding corner.

    Key insight:
    - The input grid is a noisy enlarged version of the desired 3x3 layout; the cross marks the coarse cell boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    solid_rows = [
        (row, int(g[row, 0]))
        for row in range(height)
        if np.all(g[row, :] == int(g[row, 0]))
    ]
    solid_cols = [
        (col, int(g[0, col]))
        for col in range(width)
        if np.all(g[:, col] == int(g[0, col]))
    ]
    crosses = [
        (row, col, row_color)
        for row, row_color in solid_rows
        for col, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(crosses) == 1, "expected exactly one solid same-color separator cross"

    cross_row, cross_col, separator = crosses[0]
    assert 0 < cross_row < height - 1, "separator row must split nonempty top and bottom regions"
    assert 0 < cross_col < width - 1, "separator column must split nonempty left and right regions"

    out = np.full((3, 3), int(separator), dtype=int)
    regions = [
        ((slice(0, cross_row), slice(0, cross_col)), (0, 0)),
        ((slice(0, cross_row), slice(cross_col + 1, width)), (0, 2)),
        ((slice(cross_row + 1, height), slice(0, cross_col)), (2, 0)),
        ((slice(cross_row + 1, height), slice(cross_col + 1, width)), (2, 2)),
    ]
    for src, dst in regions:
        out[dst] = _unique_mode(g[src])

    return out

# --------------- generator (att2) ---------------

GRID_SIZE = 19
COLOR_COUNT = 10

# Cross placement keeps every quadrant at least 6 cells wide/tall.
CROSS_ROW_MIN = 6
CROSS_ROW_MAX = 11
CROSS_COL_MIN = 6
CROSS_COL_MAX = 10

# Quadrant borders are mostly non-dominant while preserving the mode.
MAX_MODE_RATIO = 0.73
MAX_DOMINANT_BORDER_RATIO = 0.30
NOISE_DISTINCT_MIN = 6
NOISE_DISTINCT_MAX = 8

GEN_MAX_ATTEMPTS = 300


def _gen_border_coords(r0: int, r1: int, c0: int, c1: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(c0, c1):
        coords.append((r0, c))
        coords.append((r1 - 1, c))
    for r in range(r0 + 1, r1 - 1):
        coords.append((r, c0))
        coords.append((r, c1 - 1))
    return coords


def _gen_fill_quadrant(
    grid: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    dominant: int,
) -> None:
    grid[r0:r1, c0:c1] = dominant
    border = _gen_border_coords(r0, r1, c0, c1)
    interior_area = max(0, r1 - r0 - 2) * max(0, c1 - c0 - 2)
    total_area = (r1 - r0) * (c1 - c0)
    min_dominant_on_border = max(0, total_area // 2 + 1 - interior_area)
    max_by_mode = int(np.floor(MAX_MODE_RATIO * total_area)) - interior_area
    max_by_noise = int(np.floor(MAX_DOMINANT_BORDER_RATIO * len(border)))
    max_dominant_on_border = max(min_dominant_on_border, min(max_by_mode, max_by_noise))
    dominant_on_border = random.randint(min_dominant_on_border, max_dominant_on_border)

    random.shuffle(border)
    dominant_cells = set(border[:dominant_on_border])
    noise_cells = border[dominant_on_border:]
    noise_palette = [color for color in range(COLOR_COUNT) if color != dominant]
    distinct_noise = random.randint(
        NOISE_DISTINCT_MIN,
        min(NOISE_DISTINCT_MAX, len(noise_palette), len(noise_cells)),
    )
    used_noise = random.sample(noise_palette, distinct_noise)
    noise_values = used_noise[:]
    noise_values.extend(random.choice(used_noise) for _ in range(len(noise_cells) - len(noise_values)))
    random.shuffle(noise_values)

    for r, c in dominant_cells:
        grid[r, c] = dominant
    for (r, c), color in zip(noise_cells, noise_values):
        grid[r, c] = color


def _gen_non_separator_noise(separator: int, dominant: int) -> int:
    return random.choice([color for color in range(COLOR_COUNT) if color not in (separator, dominant)])


def _gen_prevent_extra_solid_crosses(
    grid: np.ndarray,
    cross_r: int,
    cross_c: int,
    separator: int,
    dominants: list[int],
) -> None:
    for r in range(grid.shape[0]):
        if r != cross_r and np.all(grid[r, :] == separator):
            dominant = dominants[0] if r < cross_r else dominants[2]
            grid[r, 0] = _gen_non_separator_noise(separator, dominant)
    for c in range(grid.shape[1]):
        if c != cross_c and np.all(grid[:, c] == separator):
            dominant = dominants[0] if c < cross_c else dominants[1]
            grid[0, c] = _gen_non_separator_noise(separator, dominant)


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(COLOR_COUNT), 5)
    separator = colors[0]
    dominants = colors[1:]
    expected = np.array(
        [
            [dominants[0], separator, dominants[1]],
            [separator, separator, separator],
            [dominants[2], separator, dominants[3]],
        ],
        dtype=int,
    )

    height = width = GRID_SIZE
    cross_r = random.randint(CROSS_ROW_MIN, CROSS_ROW_MAX)
    cross_c = random.randint(CROSS_COL_MIN, CROSS_COL_MAX)
    grid = np.full((height, width), separator, dtype=int)

    quadrants = [
        (0, cross_r, 0, cross_c, dominants[0]),
        (0, cross_r, cross_c + 1, width, dominants[1]),
        (cross_r + 1, height, 0, cross_c, dominants[2]),
        (cross_r + 1, height, cross_c + 1, width, dominants[3]),
    ]
    for r0, r1, c0, c1, dominant in quadrants:
        _gen_fill_quadrant(grid, r0, r1, c0, c1, dominant)

    grid[cross_r, :] = separator
    grid[:, cross_c] = separator
    _gen_prevent_extra_solid_crosses(grid, cross_r, cross_c, separator, dominants)

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_MAX_ATTEMPTS} attempts")

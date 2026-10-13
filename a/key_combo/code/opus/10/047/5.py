from __future__ import annotations

import random

import numpy as np


TRANSFORM_BY_LEGEND = {
    1: lambda tile: tile.copy(),
    2: np.fliplr,
    3: np.flipud,
    4: lambda tile: np.rot90(tile, 2),
}

TILE_SIZE = 3
OUTPUT_TILE_REPEATS = 3
GEN_ATTEMPTS = 500
TILE_COLOR_POOL = tuple(range(1, 10))
LEGEND_COLORS = (1, 2, 3, 4)


def _extract_structure(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    nonzero = [tuple(map(int, rc)) for rc in np.argwhere(g != 0)]
    assert len(nonzero) == 12, "expected a 3x3 tile plus 3 legend pixels"
    nonzero_set = set(nonzero)

    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []
    for r0 in range(h - TILE_SIZE + 1):
        for c0 in range(w - TILE_SIZE + 1):
            tile = g[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]
            if tile.shape != (TILE_SIZE, TILE_SIZE) or np.any(tile == 0):
                continue

            tile_cells = {
                (r, c)
                for r in range(r0, r0 + TILE_SIZE)
                for c in range(c0, c0 + TILE_SIZE)
            }
            if not tile_cells.issubset(nonzero_set):
                continue

            legend = sorted(nonzero_set - tile_cells)
            if len(legend) != 3:
                continue
            if len({c for _r, c in legend}) != 1:
                continue
            if len({r for r, _c in legend}) != 3:
                continue

            legend_col = legend[0][1]
            if legend_col <= c0 + TILE_SIZE + 1:
                continue

            legend_colors = [int(g[r, c]) for r, c in legend]
            if any(color not in TRANSFORM_BY_LEGEND for color in legend_colors):
                continue

            candidates.append((r0, c0, legend))

    assert len(candidates) == 1, "expected one unambiguous tile-plus-legend parse"
    r0, c0, legend = candidates[0]
    tile = g[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE].copy()
    legend_colors = [int(g[r, c]) for r, c in legend]
    return tile, legend_colors


def _render_output(tile: np.ndarray, legend_colors: list[int]) -> np.ndarray:
    row_bands = []
    for color in legend_colors:
        transformed = TRANSFORM_BY_LEGEND[color](tile)
        row_bands.append(np.hstack([transformed] * OUTPUT_TILE_REPEATS))
    return np.vstack(row_bands)


def main(grid: np.ndarray) -> np.ndarray:
    """Tile a transformed 3x3 source sprite into a 3x3 array using a color legend of row-wise transforms.

    Input structure:
    - Black background contains exactly one fully colored 3x3 tile on the left and one right-side legend column made of 3 colored pixels.
    - The three legend pixels share one column, may have gaps between rows, and are separated from the tile by at least two black columns.
    - Legend colors use blue(1), red(2), green(3), or yellow(4) as operation codes rather than paint colors.

    Rule:
    - Read the 3x3 tile and sort the three legend pixels from top to bottom.
    - Decode colors as transforms: 1 = original, 2 = horizontal mirror, 3 = vertical mirror, 4 = 180-degree rotation.
    - Build a 9x9 output by placing three copies of the transformed tile across each output row band, using the corresponding legend transform for each of the three tile rows.

    Key insight:
    - The legend does not say which color to repaint with; it encodes which geometric transform to apply to the same tile before tiling that whole output row.
    """
    tile, legend_colors = _extract_structure(grid)
    return _render_output(tile, legend_colors)


def _sample_tile() -> np.ndarray:
    for _ in range(200):
        color_count = random.randint(2, 4)
        palette = random.sample(TILE_COLOR_POOL, color_count)
        values = palette + [random.choice(palette) for _ in range(TILE_SIZE * TILE_SIZE - color_count)]
        random.shuffle(values)
        tile = np.array(values, dtype=int).reshape(TILE_SIZE, TILE_SIZE)
        transforms = (
            tile,
            np.fliplr(tile),
            np.flipud(tile),
            np.rot90(tile, 2),
        )
        if len({arr.tobytes() for arr in transforms}) != 4:
            continue
        return tile
    raise RuntimeError("failed to sample a non-degenerate tile")


def _sample_legend_colors() -> list[int]:
    mode = random.choices(
        ["all_distinct", "two_same", "all_same"],
        weights=[4, 4, 1],
        k=1,
    )[0]
    if mode == "all_distinct":
        return random.sample(list(LEGEND_COLORS), 3)
    if mode == "all_same":
        color = random.choice(LEGEND_COLORS)
        return [color, color, color]

    repeated, single = random.sample(list(LEGEND_COLORS), 2)
    colors = [repeated, repeated, single]
    random.shuffle(colors)
    return colors


def _sample_legend_rows(height: int) -> list[int]:
    mode = random.choice(["adjacent", "gapped", "random"])
    if mode == "adjacent":
        start = random.randint(0, height - 3)
        return [start, start + 1, start + 2]
    if mode == "gapped":
        step1 = random.randint(1, 3)
        step2 = random.randint(1, 3)
        if step1 == 1 and step2 == 1:
            step2 = 2
        start = random.randint(0, height - 1 - step1 - step2)
        return [start, start + step1, start + step1 + step2]
    return sorted(random.sample(range(height), 3))


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(7, 15)
    tile = _sample_tile()
    legend_colors = _sample_legend_colors()
    legend_rows = _sample_legend_rows(height)

    tile_row = random.randint(0, height - TILE_SIZE)
    tile_col = random.randint(0, 4)
    gap_cols = random.randint(2, 6)
    legend_col = tile_col + TILE_SIZE + gap_cols
    right_margin = random.randint(0, 4)
    width = legend_col + 1 + right_margin

    grid = np.zeros((height, width), dtype=int)
    grid[tile_row : tile_row + TILE_SIZE, tile_col : tile_col + TILE_SIZE] = tile
    for row, color in zip(legend_rows, legend_colors):
        grid[row, legend_col] = color

    expected = _render_output(tile, legend_colors)
    return grid, expected


def _validate_generated_case(grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    try:
        tile, legend_colors = _extract_structure(grid)
    except AssertionError:
        return False

    transforms = (
        tile,
        np.fliplr(tile),
        np.flipud(tile),
        np.rot90(tile, 2),
    )
    if len({arr.tobytes() for arr in transforms}) != 4:
        return False
    if len(set(map(int, tile.flat))) < 2:
        return False
    if any(color not in LEGEND_COLORS for color in legend_colors):
        return False

    actual = main(grid)
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid, expected = _build_case()
        if _validate_generated_case(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")

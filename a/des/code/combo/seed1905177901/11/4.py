from __future__ import annotations

import random

import numpy as np


# ARC palette ids for the six visible colors used by the puzzle.
BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7

COLOR_CYCLE = (RED, ORANGE, YELLOW, GREEN, BLUE, MAGENTA)
COLOR_RANK = {color: rank for rank, color in enumerate(COLOR_CYCLE)}


def _band_center(start: int, thickness: int) -> float:
    return start + (thickness - 1) / 2.0


def _extract_bands(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a non-empty grid"

    bands: list[dict[str, int]] = []
    row = 0
    while row < height:
        color = int(g[row, 0])
        assert color in COLOR_RANK, "stripe color must belong to the visible color cycle"
        assert np.all(g[row] == color), "each stripe row must be monochromatic"

        row_end = row + 1
        while row_end < height:
            next_color = int(g[row_end, 0])
            assert np.all(g[row_end] == next_color), "each stripe row must be monochromatic"
            if next_color != color:
                break
            row_end += 1

        bands.append({"color": color, "start": row, "thickness": row_end - row})
        row = row_end

    assert len(bands) >= 2, "expected multiple horizontal stripes"
    colors = [band["color"] for band in bands]
    assert len(set(colors)) == len(colors), "stripe colors must be distinct"
    return bands


def _adjust_thickness(start: int, thickness: int, total_height: int) -> int:
    # Classify a stripe by the position of its center relative to the grid midpoint.
    if _band_center(start, thickness) < total_height / 2.0:
        return thickness + 1
    return max(1, thickness - 1)


def _render_bands(bands: list[tuple[int, int]], width: int) -> np.ndarray:
    out_h = sum(thickness for _color, thickness in bands)
    out = np.full((out_h, width), BLACK, dtype=int)
    row = 0
    for color, thickness in bands:
        out[row : row + thickness, :] = int(color)
        row += thickness
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder horizontal color stripes by the visible color-wheel order and adjust
    each stripe thickness according to its original vertical position.
    """
    g = np.asarray(grid, dtype=int)
    bands = _extract_bands(g)
    height, width = map(int, g.shape)

    reordered: list[tuple[int, int]] = []
    for band in sorted(bands, key=lambda band: COLOR_RANK[band["color"]]):
        new_thickness = _adjust_thickness(band["start"], band["thickness"], height)
        reordered.append((int(band["color"]), int(new_thickness)))

    return _render_bands(reordered, width)


# --------------- generator ---------------

MIN_BANDS = 3
MAX_BANDS = len(COLOR_CYCLE)
WIDTH_RANGE = (5, 18)
THICKNESS_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 300


def _sorted_colors(colors: list[int]) -> list[int]:
    return sorted(colors, key=lambda color: COLOR_RANK[color])


def _sample_color_order(colors: list[int]) -> list[int]:
    ordered = _sorted_colors(colors)
    reversed_order = list(reversed(ordered))
    mode = random.choice(("sorted", "reversed", "random"))
    if mode == "sorted":
        return ordered
    if mode == "reversed":
        return reversed_order

    while True:
        perm = random.sample(colors, len(colors))
        if perm != ordered and perm != reversed_order:
            return perm


def _layout_stats(thicknesses: list[int]) -> tuple[list[int], list[int], list[int]]:
    starts: list[int] = []
    top_indices: list[int] = []
    bottom_indices: list[int] = []
    row = 0
    total_height = sum(thicknesses)
    for idx, thickness in enumerate(thicknesses):
        starts.append(row)
        if _band_center(row, thickness) < total_height / 2.0:
            top_indices.append(idx)
        else:
            bottom_indices.append(idx)
        row += thickness
    return starts, top_indices, bottom_indices


def _sample_thicknesses(require_bottom_one: bool) -> list[int] | None:
    for _ in range(200):
        count = random.randint(MIN_BANDS, MAX_BANDS)
        thicknesses = [random.randint(*THICKNESS_RANGE) for _ in range(count)]
        starts, top_indices, bottom_indices = _layout_stats(thicknesses)
        total_height = sum(thicknesses)
        if any(_band_center(start, thickness) == total_height / 2.0 for start, thickness in zip(starts, thicknesses)):
            continue
        if not top_indices or not bottom_indices:
            continue

        bottom_values = [thicknesses[idx] for idx in bottom_indices]
        has_bottom_one = any(value == 1 for value in bottom_values)
        has_bottom_shrink = any(value >= 2 for value in bottom_values)

        if require_bottom_one and not has_bottom_one:
            continue
        if not require_bottom_one and not has_bottom_shrink:
            continue
        return thicknesses
    return None


def _build_input(order: list[int], thicknesses: list[int], width: int) -> np.ndarray:
    height = sum(thicknesses)
    grid = np.full((height, width), BLACK, dtype=int)
    row = 0
    for color, thickness in zip(order, thicknesses):
        grid[row : row + thickness, :] = int(color)
        row += thickness
    return grid


def _expected_from_metadata(order: list[int], thicknesses: list[int], width: int) -> np.ndarray:
    starts, _top_indices, _bottom_indices = _layout_stats(thicknesses)
    transformed = []
    for color, start, thickness in zip(order, starts, thicknesses):
        transformed.append((int(color), _adjust_thickness(start, thickness, sum(thicknesses))))
    transformed.sort(key=lambda item: COLOR_RANK[item[0]])
    return _render_bands(transformed, width)


def _gen_attempt() -> np.ndarray | None:
    require_bottom_one = random.random() < 0.5
    thicknesses = _sample_thicknesses(require_bottom_one)
    if thicknesses is None:
        return None

    band_count = len(thicknesses)
    colors = random.sample(list(COLOR_CYCLE), band_count)
    order = _sample_color_order(colors)
    width = random.randint(*WIDTH_RANGE)

    grid = _build_input(order, thicknesses, width)
    expected = _expected_from_metadata(order, thicknesses, width)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PATTERN_COLORS = (GREEN, YELLOW)
SWAP = {GREEN: YELLOW, YELLOW: GREEN}

GEN_RETRY_LIMIT = 400
BAND_RETRY_LIMIT = 120


def _swap_array(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    green_mask = arr == GREEN
    yellow_mask = arr == YELLOW
    out[green_mask] = YELLOW
    out[yellow_mask] = GREEN
    return out


def _marker_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape[0] == 1, f"expected exactly one marker of color {color}"
    r, c = coords[0]
    return int(r), int(c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw two horizontal marker lines, split the canvas into three bands, and apply a different transform in each band.

    Input structure:
    - The grid is a black background with one red marker pixel, one blue marker pixel, and a sparse green/yellow pattern.
    - The red marker lies above the blue marker, and their rows contain no green/yellow pattern pixels.
    - Green and yellow pixels may appear in the top band, middle band, and bottom band formed by the two marker rows.

    Rule:
    - Replace the red marker row with a full red horizontal line and the blue marker row with a full blue horizontal line.
    - In rows above the red line, mirror every green/yellow pixel left-to-right, keeping its color.
    - In rows between the two lines, mirror every green/yellow pixel left-to-right and swap green with yellow.
    - In rows below the blue line, keep positions fixed and swap green with yellow; background remains black.

    Key insight:
    - The two isolated markers are not output decorations: their rows define three horizontal regions, and each region has its own fixed transformation recipe.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(np.unique(g)).issubset({BG, BLUE, RED, GREEN, YELLOW})

    red_r, _red_c = _marker_coord(g, RED)
    blue_r, _blue_c = _marker_coord(g, BLUE)
    assert red_r < blue_r

    pattern_mask = (g == GREEN) | (g == YELLOW)
    assert not np.any(pattern_mask[red_r, :])
    assert not np.any(pattern_mask[blue_r, :])

    h, w = g.shape
    out = np.zeros_like(g)
    out[red_r, :] = RED
    out[blue_r, :] = BLUE

    for r, c in np.argwhere(pattern_mask):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if r < red_r:
            out[r, w - 1 - c] = color
        elif red_r < r < blue_r:
            out[r, w - 1 - c] = SWAP[color]
        elif r > blue_r:
            out[r, c] = SWAP[color]

    return out


def _shape_templates(band_h: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    templates = [
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ]
    valid = []
    for cells in templates:
        max_r = max(r for r, _ in cells)
        max_c = max(c for _, c in cells)
        if max_r < band_h and max_c < width:
            valid.append(cells)
    return valid


def _can_place(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if occupied[r, c]:
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _band_variant_ok(local: np.ndarray, mode: str) -> bool:
    colors = set(np.unique(local)) - {BG}
    if colors != set(PATTERN_COLORS):
        return False
    if np.count_nonzero(local) < 4:
        return False

    mirrored = np.fliplr(local)
    swapped = _swap_array(local)
    combined = _swap_array(mirrored)

    if mode == "mirror":
        return not np.array_equal(local, mirrored)
    if mode == "swap":
        return not np.array_equal(local, swapped)
    assert mode == "mirror_swap"
    return (
        not np.array_equal(local, combined)
        and not np.array_equal(mirrored, combined)
        and not np.array_equal(swapped, combined)
    )


def _generate_band(band_h: int, width: int, mode: str) -> np.ndarray | None:
    templates = _shape_templates(band_h, width)
    if not templates:
        return None

    for _ in range(BAND_RETRY_LIMIT):
        local = np.zeros((band_h, width), dtype=int)
        occupied = np.zeros((band_h, width), dtype=bool)
        max_shapes = 4 if band_h * width >= 28 else 3
        shape_count = random.randint(2, max_shapes)
        shapes: list[list[tuple[int, int]]] = []

        for _shape_index in range(shape_count):
            placements: list[list[tuple[int, int]]] = []
            random.shuffle(templates)
            for template in templates:
                max_r = max(r for r, _ in template)
                max_c = max(c for _, c in template)
                for r0 in range(band_h - max_r):
                    for c0 in range(width - max_c):
                        cells = [(r0 + r, c0 + c) for r, c in template]
                        if _can_place(occupied, cells):
                            placements.append(cells)
            if not placements:
                break
            cells = random.choice(placements)
            shapes.append(cells)
            for r, c in cells:
                occupied[r, c] = True

        if len(shapes) != shape_count:
            continue

        shape_colors = [GREEN, YELLOW]
        shape_colors.extend(random.choice(PATTERN_COLORS) for _ in range(shape_count - 2))
        random.shuffle(shape_colors)
        for color, cells in zip(shape_colors, shapes):
            for r, c in cells:
                local[r, c] = color

        if _band_variant_ok(local, mode):
            return local

    return None


def _build_candidate() -> np.ndarray | None:
    top_h = random.randint(2, 5)
    mid_h = random.randint(3, 6)
    bottom_h = random.randint(2, 5)
    height = top_h + mid_h + bottom_h + 2
    width = random.randint(10, 18)

    red_row = top_h
    blue_row = top_h + mid_h + 1

    top_band = _generate_band(top_h, width, "mirror")
    mid_band = _generate_band(mid_h, width, "mirror_swap")
    bottom_band = _generate_band(bottom_h, width, "swap")
    if top_band is None or mid_band is None or bottom_band is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[:top_h, :] = top_band
    grid[red_row + 1 : blue_row, :] = mid_band
    grid[blue_row + 1 :, :] = bottom_band

    red_col = random.randrange(width)
    blue_col = random.randrange(width)
    grid[red_row, red_col] = RED
    grid[blue_row, blue_col] = BLUE

    if np.any(grid[red_row, :] == GREEN) or np.any(grid[red_row, :] == YELLOW):
        return None
    if np.any(grid[blue_row, :] == GREEN) or np.any(grid[blue_row, :] == YELLOW):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _build_candidate()
        if grid is None:
            continue
        out = main(grid)
        red_r, _ = _marker_coord(grid, RED)
        blue_r, _ = _marker_coord(grid, BLUE)
        assert red_r < blue_r
        if np.array_equal(grid, out):
            continue
        if grid.shape[0] > 30 or grid.shape[1] > 30:
            continue
        if out.shape[0] > 30 or out.shape[1] > 30:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")

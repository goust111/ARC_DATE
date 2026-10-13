from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    arc_utils_dir = str(_ARC_UTILS_DIR)
    if arc_utils_dir not in sys.path:
        sys.path.append(arc_utils_dir)

try:
    from arc_utils import Color, bbox, in_bounds
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        height, width = map(int, shape)
        return 0 <= int(r) < height and 0 <= int(c) < width

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


BG = Color.BLACK
SEP = Color.GREY
PALETTE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

HEIGHT_RANGE = (13, 19)
WIDTH_RANGE = (13, 22)
PALETTE_LENGTH_RANGE = (3, 5)
TRANSPARENT_PROBABILITY = 0.4
CLIP_PROBABILITY = 0.5
MAX_GENERATION_ATTEMPTS = 400


def _read_palette(row: np.ndarray) -> list[int]:
    nonzero = np.flatnonzero(row != BG)
    assert nonzero.size > 0, "palette row must contain at least one non-background cell"
    assert int(nonzero[0]) == 0, "palette must start at the left edge"
    last = int(nonzero[-1])
    palette = [int(v) for v in row[: last + 1]]
    assert PALETTE_LENGTH_RANGE[0] <= len(palette) <= PALETTE_LENGTH_RANGE[1], "palette length must be 3..5"
    visible = sum(color != BG for color in palette)
    assert 3 <= visible <= 5, "palette must expose 3..5 visible colors"
    assert palette[0] != BG and palette[-1] != BG, "palette endpoints must be visible colors"
    assert np.all(row[last + 1 :] == BG), "palette tail must be background"
    return palette


def _find_seed(grid: np.ndarray, seed_color: int) -> tuple[int, int]:
    coords = np.argwhere(grid[2:] != BG)
    assert coords.shape[0] == 1, "expected exactly one seed pixel below the separator"
    seed_r = int(coords[0, 0]) + 2
    seed_c = int(coords[0, 1])
    assert int(grid[seed_r, seed_c]) == int(seed_color), "seed color must match the first palette color"
    return seed_r, seed_c


def _paint_square_ring(canvas: np.ndarray, center: tuple[int, int], radius: int, color: int) -> None:
    if color == BG:
        return
    seed_r, seed_c = center
    r0 = seed_r - radius
    c0 = seed_c - radius
    r1 = seed_r + radius
    c1 = seed_c + radius
    assert in_bounds(canvas.shape, r0, c0) and in_bounds(canvas.shape, r1, c1), "ring must fit before shearing"
    canvas[r0, c0 : c1 + 1] = color
    canvas[r1, c0 : c1 + 1] = color
    canvas[r0 : r1 + 1, c0] = color
    canvas[r0 : r1 + 1, c1] = color


def _build_rings(shape: tuple[int, int], center: tuple[int, int], palette: list[int]) -> np.ndarray:
    rings = np.zeros(shape, dtype=int)
    for radius, color in enumerate(palette):
        _paint_square_ring(rings, center, radius, int(color))
    return rings


def _shear_right(figure: np.ndarray, bottom: int) -> np.ndarray:
    out = np.zeros_like(figure)
    for r_raw, c_raw in np.argwhere(figure != BG):
        r = int(r_raw)
        c = int(c_raw)
        new_c = c + (int(bottom) - r)
        if in_bounds(figure.shape, r, new_c):
            out[r, new_c] = int(figure[r, c])
    return out


def _compose_output(
    shape: tuple[int, int],
    palette: list[int],
    center: tuple[int, int],
    preserved_rows: tuple[np.ndarray, np.ndarray],
    *,
    shear: bool,
) -> np.ndarray:
    figure = _build_rings(shape, center, palette)
    bottom = int(center[0]) + len(palette) - 1
    shifted = _shear_right(figure, bottom) if shear else figure
    out = np.zeros(shape, dtype=int)
    out[0, :] = preserved_rows[0]
    out[1, :] = preserved_rows[1]
    mask = shifted != BG
    out[mask] = shifted[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build concentric Chebyshev square rings from the palette row, then shear the whole stack rightward while keeping the legend rows fixed.

    Input structure:
    - Black background fills the grid except for a left-anchored palette prefix on row 0 and an all-grey separator on row 1.
    - The palette prefix has length 3..5, starts at column 0, may contain internal black cells as transparent layers, and ends before a black suffix.
    - Exactly one seed pixel appears below the separator, and its color matches the first palette entry.

    Rule:
    - Read the palette from row 0 as the prefix from column 0 through the last non-black cell; keep any internal black entries as transparent layers.
    - For each Chebyshev radius d from 0 to len(palette)-1, draw the square ring centered on the seed with palette[d]; skip drawing when palette[d] is black.
    - Let bottom be the bottom row of the outermost radius and move every drawn pixel (r, c) to (r, c + bottom - r), clipping anything that exits the right edge.
    - Return a black canvas with the sheared figure plus the original palette row and grey separator row unchanged.

    Key insight:
    - The palette index controls both ring radius and ring color, while the outer bottom row anchors a second distance-based shear, so the solver must track two coupled but different notions of distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert 3 <= height <= 30 and 3 <= width <= 30, "grid must satisfy ARC size limits"
    assert np.all(g[1, :] == SEP), "row 1 must be the grey separator"

    palette = _read_palette(g[0])
    seed = _find_seed(g, palette[0])
    radius = len(palette) - 1
    seed_r, seed_c = seed
    assert seed_r - radius >= 2, "rings must stay below the legend rows before shearing"
    assert seed_r + radius < height, "rings must fit vertically before shearing"
    assert seed_c - radius >= 0 and seed_c + radius < width, "rings must fit horizontally before shearing"

    return _compose_output(g.shape, palette, seed, (g[0].copy(), g[1].copy()), shear=True)


def _sample_palette() -> tuple[list[int], bool]:
    use_transparent = random.random() < TRANSPARENT_PROBABILITY
    if use_transparent:
        length = random.randint(4, PALETTE_LENGTH_RANGE[1])
        visible_colors = random.sample(PALETTE_COLORS, length - 1)
        hole = random.randint(1, length - 2)
        palette: list[int] = []
        visible_index = 0
        for idx in range(length):
            if idx == hole:
                palette.append(BG)
            else:
                palette.append(int(visible_colors[visible_index]))
                visible_index += 1
        return palette, True

    length = random.randint(*PALETTE_LENGTH_RANGE)
    return [int(color) for color in random.sample(PALETTE_COLORS, length)], False


def _sample_dimensions(radius: int, clip: bool) -> tuple[int, int]:
    min_height = max(HEIGHT_RANGE[0], 2 * radius + 5)
    height = random.randint(min_height, HEIGHT_RANGE[1])

    if clip:
        min_width = max(WIDTH_RANGE[0], 2 * radius + 5)
    else:
        min_width = max(WIDTH_RANGE[0], 4 * radius + 1)
    width = random.randint(min_width, WIDTH_RANGE[1])
    return height, width


def _sample_seed(height: int, width: int, radius: int, clip: bool) -> tuple[int, int]:
    seed_r = random.randint(2 + radius, height - 1 - radius)
    if clip:
        left = max(radius, width - 3 * radius)
        right = width - 1 - radius
    else:
        left = radius
        right = width - 1 - 3 * radius
    assert left <= right, "seed range must be non-empty"
    seed_c = random.randint(left, right)
    return seed_r, seed_c


def _count_clipped_pixels(shape: tuple[int, int], figure: np.ndarray, bottom: int) -> int:
    width = shape[1]
    clipped = 0
    for r_raw, c_raw in np.argwhere(figure != BG):
        r = int(r_raw)
        c = int(c_raw)
        if c + (int(bottom) - r) >= width:
            clipped += 1
    return clipped


def _semantic_checks(grid: np.ndarray, *, expect_clip: bool, expect_transparent: bool) -> bool:
    try:
        palette = _read_palette(grid[0])
        seed = _find_seed(grid, palette[0])
        radius = len(palette) - 1
        figure = _build_rings(grid.shape, seed, palette)
        figure_box = bbox(figure != BG)
        assert figure_box == (seed[0] - radius, seed[1] - radius, seed[0] + radius, seed[1] + radius)

        bottom = seed[0] + radius
        clipped = _count_clipped_pixels(grid.shape, figure, bottom)
        assert (clipped > 0) == expect_clip

        actual = main(grid)
        expected = _compose_output(grid.shape, palette, seed, (grid[0].copy(), grid[1].copy()), shear=True)
        assert np.array_equal(actual, expected)
        assert np.array_equal(actual[0], grid[0]) and np.array_equal(actual[1], grid[1])

        unsheared = _compose_output(grid.shape, palette, seed, (grid[0].copy(), grid[1].copy()), shear=False)
        assert not np.array_equal(actual, unsheared), "shear must matter"

        sheared_visible = set(map(int, np.unique(actual[2:])))
        sheared_visible.discard(BG)
        assert sheared_visible == {int(color) for color in palette if color != BG}, "every visible layer must survive in output"

        moved = False
        for r_raw, c_raw in np.argwhere(figure != BG):
            r = int(r_raw)
            c = int(c_raw)
            new_c = c + (bottom - r)
            if 0 <= new_c < grid.shape[1] and new_c != c:
                moved = True
                break
        assert moved, "at least one visible pixel must move under the shear"

        has_internal_transparent = any(color == BG for color in palette[1:-1])
        assert has_internal_transparent == expect_transparent
        if has_internal_transparent:
            compressed = [int(color) for color in palette if color != BG]
            alt = _compose_output(grid.shape, compressed, seed, (grid[0].copy(), grid[1].copy()), shear=True)
            assert not np.array_equal(actual, alt), "transparent layers must affect the final geometry"

        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    palette, use_transparent = _sample_palette()
    clip = random.random() < CLIP_PROBABILITY
    radius = len(palette) - 1

    height, width = _sample_dimensions(radius, clip)
    seed_r, seed_c = _sample_seed(height, width, radius, clip)

    grid = np.zeros((height, width), dtype=int)
    grid[0, : len(palette)] = np.array(palette, dtype=int)
    grid[1, :] = SEP
    grid[seed_r, seed_c] = int(palette[0])

    if not _semantic_checks(grid, expect_clip=clip, expect_transparent=use_transparent):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

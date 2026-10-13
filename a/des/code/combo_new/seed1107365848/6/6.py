import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
LAYER_COUNT_RANGE = (2, 7)
INNER_RECT_MAX_SIDE = 6
GRID_MAX_SIDE = 30


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_border(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_nested_layers(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    colors: list[int],
) -> None:
    r0, c0 = top, left
    r1, c1 = top + height - 1, left + width - 1
    for color in colors[:-1]:
        _paint_border(grid, r0, c0, r1, c1, color)
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = colors[-1]


def _extract_palette(grid: np.ndarray) -> tuple[int, int, int, list[int]]:
    non_bg_rows = np.flatnonzero(np.any(grid != BACKGROUND, axis=1))
    assert len(non_bg_rows) >= 2, "expected a palette row and a lower pattern"

    palette_row = int(non_bg_rows[0])
    cols = np.flatnonzero(grid[palette_row] != BACKGROUND)
    assert len(cols) > 0, "palette row must contain foreground colors"
    assert np.all(np.diff(cols) == 1), "palette colors must form one contiguous strip"

    c0, c1 = int(cols[0]), int(cols[-1])
    palette = [int(v) for v in grid[palette_row, c0 : c1 + 1]]
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return palette_row, c0, c1, palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor concentric rectangular layers according to the left-to-right order of the palette strip.

    Input structure:
    - The background is black.
    - The top region contains one horizontal palette strip made of adjacent
      single-pixel colors in a specific left-to-right order.
    - After at least one blank background row, the lower region contains one
      filled rectangular bbox whose color bands are concentric nested
      one-pixel-thick rectangular layers; the innermost layer may be a solid
      rectangle or a single pixel.

    Rule:
    - Read the palette strip from left to right to obtain colors C1..Cn.
    - Locate the lower nested-rectangle structure and interpret its layers from
      the outer border inward as layer 1..n.
    - Recolor layer k to Ck, ignoring the original layer colors.
    - Leave the palette strip and all background cells unchanged.

    Key insight:
    - The palette does not label object identities; it encodes the target color
      sequence for successive geometric depths of the nested rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_row, palette_c0, palette_c1, palette = _extract_palette(g)

    mask = g != BACKGROUND
    mask[palette_row, palette_c0 : palette_c1 + 1] = False
    assert not mask[palette_row].any(), "palette row should not contain other foreground pixels"

    lower_rows = np.flatnonzero(np.any(mask, axis=1))
    assert len(lower_rows) > 0, "expected a nested lower structure"
    assert int(lower_rows[0]) >= palette_row + 2, "lower structure must sit below a blank separator row"

    r0, c0, r1, c1 = _bbox_from_mask(mask)
    expected_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(mask.sum()) == expected_area, "nested layers must fill one solid rectangular bbox"

    layer_count = len(palette)
    assert r1 - r0 + 1 >= 2 * layer_count - 1, "bbox height is too small for the palette depth"
    assert c1 - c0 + 1 >= 2 * layer_count - 1, "bbox width is too small for the palette depth"

    out = g.copy()
    _paint_nested_layers(out, r0, c0, r1 - r0 + 1, c1 - c0 + 1, palette)
    return out


def _random_inner_dims() -> tuple[int, int]:
    if random.random() < 0.5:
        return 1, 1

    inner_h = random.randint(1, INNER_RECT_MAX_SIDE)
    inner_w = random.randint(1, INNER_RECT_MAX_SIDE)
    if inner_h == 1 and inner_w == 1:
        if random.random() < 0.5:
            inner_h = random.randint(2, INNER_RECT_MAX_SIDE)
        else:
            inner_w = random.randint(2, INNER_RECT_MAX_SIDE)
    return inner_h, inner_w


def _random_ring_colors(layer_count: int, palette: list[int]) -> list[int]:
    sample_pool = random.sample(COLOR_POOL, random.randint(2, min(5, len(COLOR_POOL))))
    colors = [random.choice(sample_pool) for _ in range(layer_count)]
    if all(colors[i] == palette[i] for i in range(layer_count)):
        idx = random.randrange(layer_count)
        alternatives = [c for c in COLOR_POOL if c != palette[idx]]
        colors[idx] = random.choice(alternatives)
    return colors


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(*LAYER_COUNT_RANGE)
    palette = random.sample(list(COLOR_POOL), layer_count)

    inner_h, inner_w = _random_inner_dims()
    shape_h = inner_h + 2 * (layer_count - 1)
    shape_w = inner_w + 2 * (layer_count - 1)

    min_h = shape_h + 2
    min_w = max(shape_w, layer_count)
    if min_h > GRID_MAX_SIDE or min_w > GRID_MAX_SIDE:
        return None

    grid_h = random.randint(min_h, min(GRID_MAX_SIDE, min_h + 8))
    grid_w = random.randint(min_w, min(GRID_MAX_SIDE, min_w + 10))

    palette_row = random.randint(0, grid_h - shape_h - 2)
    shape_top = palette_row + 2
    palette_left = random.randint(0, grid_w - layer_count)
    shape_left = random.randint(0, grid_w - shape_w)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[palette_row, palette_left : palette_left + layer_count] = palette

    ring_colors = _random_ring_colors(layer_count, palette)
    _paint_nested_layers(grid, shape_top, shape_left, shape_h, shape_w, ring_colors)

    out = main(grid)
    assert np.array_equal(out[palette_row], grid[palette_row]), "palette row must stay unchanged"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-ring palette task")

from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected at least one marked cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _perimeter_values(rect: np.ndarray) -> np.ndarray:
    perimeter = np.zeros(rect.shape, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    return rect[perimeter]


def _template_row_and_cols(g: np.ndarray) -> tuple[int, list[int]]:
    grey_rows = np.flatnonzero(np.any(g == GREY, axis=1))
    assert len(grey_rows) == 1, "expected exactly one grey template row"
    row = int(grey_rows[0])
    assert set(map(int, g[row, :])) <= {BLACK, GREY}, "template row must contain only black and grey"
    cols = [int(c) for c in np.flatnonzero(g[row, :] == GREY)]
    assert cols, "expected at least one grey placeholder"
    return row, cols


def _nested_colors(g: np.ndarray, template_row: int) -> list[int]:
    work = np.array(g, dtype=int, copy=True)
    work[template_row, :] = BLACK

    colors: list[int] = []
    search = (0, 0, work.shape[0] - 1, work.shape[1] - 1)

    while True:
        sr0, sc0, sr1, sc1 = search
        assert sr0 <= sr1 and sc0 <= sc1, "nested search region became empty"

        region = work[sr0 : sr1 + 1, sc0 : sc1 + 1]
        br0, bc0, br1, bc1 = _bbox_from_mask(region != BLACK)
        r0, c0 = sr0 + br0, sc0 + bc0
        r1, c1 = sr0 + br1, sc0 + bc1

        rect = work[r0 : r1 + 1, c0 : c1 + 1]
        perimeter_colors = set(map(int, _perimeter_values(rect)))
        assert len(perimeter_colors) == 1, "each visible layer must have one perimeter color"
        color = next(iter(perimeter_colors))
        assert color not in (BLACK, GREY), "layer colors must be distinct from markers"
        colors.append(color)

        if np.all(rect == color):
            break

        assert rect.shape[0] >= 3 and rect.shape[1] >= 3, "a non-solid frame needs an interior"
        interior = work[r0 + 1 : r1, c0 + 1 : c1]
        assert np.any(interior != BLACK), "every frame must contain the next nested layer"
        search = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        assert len(colors) <= len(COLOR_POOL), "too many nested layers"

    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract nested rectangle colors and write them into the bottom template row.

    Input structure:
    - The background is black.
    - One nested rectangular structure appears away from a grey template row:
      one-pixel rectangular borders surround smaller borders, ending in a solid
      inner rectangle.
    - The template row contains grey placeholder pixels whose count equals the
      number of nested layers including the solid center.

    Rule:
    - Read the visible layer colors from the outermost rectangle to the innermost
      solid rectangle.
    - Clear the nested structure to black.
    - Replace the grey placeholders from left to right with the extracted color
      sequence, preserving the grid size.

    Key insight:
    The color order is supplied by containment depth, not by color identity; the
    grey row is only a slot template for the outer-to-inner sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template_row, placeholder_cols = _template_row_and_cols(g)
    colors = _nested_colors(g, template_row)
    assert len(colors) == len(placeholder_cols), "template size must match nested layer count"

    out = np.zeros_like(g)
    for col, color in zip(placeholder_cols, colors):
        out[template_row, col] = color
    return out


def _draw_nested(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    colors: list[int],
    gaps: list[int],
) -> None:
    r0, c0 = int(top), int(left)
    r1, c1 = r0 + int(height) - 1, c0 + int(width) - 1

    for color, gap in zip(colors[:-1], gaps):
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)

        inset = int(gap) + 1
        r0 += inset
        c0 += inset
        r1 -= inset
        c1 -= inset

    assert r0 <= r1 and c0 <= c1, "solid center must have positive size"
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(colors[-1])


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(2, 5)
    border_count = layer_count - 1

    inner_h = random.randint(1, 4)
    inner_w = random.randint(1, 5)
    gaps = [random.randint(1, 2) for _ in range(border_count)]

    outer_h = inner_h + sum(2 * (gap + 1) for gap in gaps)
    outer_w = inner_w + sum(2 * (gap + 1) for gap in gaps)

    min_h = outer_h + 2
    min_w = max(outer_w, layer_count + 2)
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(max(8, min_h), 30)
    width = random.randint(max(8, min_w), 30)

    max_top = height - outer_h - 2
    max_left = width - outer_w
    if max_top < 0 or max_left < 0:
        return None

    top = random.randint(0, max_top)
    left = random.randint(0, max_left)
    colors = random.sample(list(COLOR_POOL), layer_count)

    grid = np.zeros((height, width), dtype=int)
    _draw_nested(grid, top, left, outer_h, outer_w, colors, gaps)

    slot_start = random.randint(0, width - layer_count)
    grid[-1, slot_start : slot_start + layer_count] = GREY

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.array_equal(out[-1, slot_start : slot_start + layer_count], np.array(colors, dtype=int)):
        return None
    if int(np.count_nonzero(out)) != layer_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle template input")

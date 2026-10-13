from __future__ import annotations

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack the bottom-grounded foreground rectangles into an overlapping diagonal chain starting at the upper-left.

    Input structure
    The background is the most common color. Every foreground object is a solid
    monochrome rectangle touching the bottom row, and the rectangles have unique
    left edges that define their left-to-right order.

    Rule
    Clear the canvas to the background. Copy the rectangles in left-to-right
    order, placing the first at the upper-left corner. After placing a rectangle
    of height h and width w, place the next rectangle h-1 rows lower and w-1
    columns to the right, so later rectangles overwrite the shared corner cell.

    Key insight
    The bottom arrangement is a horizontal list of rectangles; the output keeps
    the list order and rectangle shapes while converting the list into a compact
    diagonal chain of overlapping bounding-box corners.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = int(get_bg(g))

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert objects, "expected at least one foreground rectangle"

    left_edges = [int(obj["bbox"][1]) for obj in objects]
    assert len(set(left_edges)) == len(left_edges), "expected unique left-to-right object order"

    out = np.full_like(g, bg)
    row = 0
    col = 0
    for obj in sorted(objects, key=lambda item: int(item["bbox"][1])):
        r0, c0, r1, c1 = map(int, obj["bbox"])
        assert r1 == height - 1, "each object must touch the bottom row"

        patch = crop_bbox(g, (r0, c0, r1, c1))
        assert np.all(patch == int(obj["color"])), "each object must be a solid rectangle"

        h, w = map(int, patch.shape)
        assert row + h <= height and col + w <= width, "packed chain must fit on the canvas"
        paste(out, patch, row, col)
        row += h - 1
        col += w - 1

    return out

# --------------- generator (o5x) ---------------
# Canvas and palette.
GRID_SIZE, BACKGROUND_COLOR = 10, 0
FOREGROUND_COLORS = np.arange(1, 10)
# Rectangle sampling ranges, inclusive.
RECT_COUNT_MIN, RECT_COUNT_MAX = 3, 4
RECT_HEIGHT_MIN, RECT_HEIGHT_MAX = 1, 5
RECT_WIDTH_MIN, RECT_WIDTH_MAX = 1, 4
# Packing constraints and retry limit.
CHAIN_OVERLAP, INPUT_GAP, MAX_GENERATION_ATTEMPTS = 1, 1, 300
_DIMENSION_CHOICES_CACHE: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}


def _gen_dimension_choices(n: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    from itertools import product

    if n not in _DIMENSION_CHOICES_CACHE:
        choices = []
        for heights in product(range(RECT_HEIGHT_MIN, RECT_HEIGHT_MAX + 1), repeat=n):
            if sum(h - CHAIN_OVERLAP for h in heights) + CHAIN_OVERLAP > GRID_SIZE:
                continue
            for widths in product(range(RECT_WIDTH_MIN, RECT_WIDTH_MAX + 1), repeat=n):
                if sum(w - CHAIN_OVERLAP for w in widths) + CHAIN_OVERLAP > GRID_SIZE:
                    continue
                if sum(widths) + (n - 1) * INPUT_GAP > GRID_SIZE:
                    continue
                choices.append((tuple(map(int, heights)), tuple(map(int, widths))))
        _DIMENSION_CHOICES_CACHE[n] = choices
    return _DIMENSION_CHOICES_CACHE[n]


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(RECT_COUNT_MIN, RECT_COUNT_MAX + 1))
    choices = _gen_dimension_choices(n)
    heights, widths = choices[int(np.random.randint(len(choices)))]
    colors = [int(c) for c in np.random.choice(FOREGROUND_COLORS, size=n, replace=False)]

    expected = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    row = col = 0
    for h, w, color in zip(heights, widths, colors):
        patch = np.full((h, w), color, dtype=int)
        paste(expected, patch, row, col)
        row += h - CHAIN_OVERLAP
        col += w - CHAIN_OVERLAP

    inp = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    left = 0
    for h, w, color in zip(heights, widths, colors):
        patch = np.full((h, w), color, dtype=int)
        paste(inp, patch, GRID_SIZE - h, left)
        left += w + INPUT_GAP

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")

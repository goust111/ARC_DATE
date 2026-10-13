from __future__ import annotations

import random

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))
GRID_LIMIT = 30
LAYER_RANGE = (2, 5)
STEP_RANGE = (2, 4)
INNER_HEIGHT_RANGE = (3, 8)
INNER_WIDTH_RANGE = (3, 8)
MARGIN_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shifted_mask(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_hollow_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    actual = _shifted_mask(coords, (r0, c0, r1, c1))
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(actual, expected))


def _is_stripe(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h == 1 and w >= 2:
        return coords.shape[0] == w
    if w == 1 and h >= 2:
        return coords.shape[0] == h
    return False


def _stripe_colors(grid: np.ndarray, coords: np.ndarray) -> list[int]:
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1:
        return [int(grid[r0, c]) for c in range(c0, c1 + 1)]
    return [int(grid[r, c0]) for r in range(r0, r1 + 1)]


def _sorted_layers(rectangles: list[np.ndarray]) -> list[np.ndarray]:
    ordered = sorted(
        rectangles,
        key=lambda coords: (
            -(_bbox(coords)[2] - _bbox(coords)[0] + 1) * (_bbox(coords)[3] - _bbox(coords)[1] + 1),
            _bbox(coords)[0],
            _bbox(coords)[1],
        ),
    )
    boxes = [_bbox(coords) for coords in ordered]
    for outer, inner in zip(boxes, boxes[1:]):
        assert outer[0] < inner[0] <= inner[2] < outer[2]
        assert outer[1] < inner[1] <= inner[3] < outer[3]
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangle outline by the depth-indexed colors listed in the separate stripe and erase that stripe.

    Input structure:
    - The background is black.
    - One region contains 2-5 nested hollow axis-aligned rectangle outlines, each one pixel thick and separated by at least one black cell.
    - A separate horizontal or vertical stripe of adjacent colored pixels sits in a corner and does not overlap the rectangles.

    Rule:
    - Split the non-black pixels into 4-connected components.
    - Identify the unique line component as the color stripe and read its colors in geometric order: left-to-right for a horizontal stripe or top-to-bottom for a vertical one.
    - Identify the remaining components as nested rectangle outlines, order them from outermost to innermost, and recolor depth k with the k-th stripe color.
    - Remove the stripe by replacing its cells with black.

    Key insight:
    - The rectangles encode an ordered depth structure spatially, while the detached stripe encodes a lookup table positionally; matching depth order to stripe position gives the recoloring map.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pieces = _components(g != BG)
    stripe_parts = [coords for coords in pieces if _is_stripe(coords)]
    rect_parts = [coords for coords in pieces if _is_hollow_rectangle(coords)]

    assert len(stripe_parts) == 1, "expected exactly one stripe component"
    assert len(rect_parts) >= 2, "expected multiple nested rectangles"
    assert len(rect_parts) + 1 == len(pieces), "unexpected extra foreground component"

    stripe = stripe_parts[0]
    colors = _stripe_colors(g, stripe)
    layers = _sorted_layers(rect_parts)
    assert len(colors) == len(layers), "stripe length must match rectangle depth count"

    out = g.copy()
    out[stripe[:, 0], stripe[:, 1]] = BG
    for color, coords in zip(colors, layers):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _paint_rectangle_outline(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _corner_margin(corner: str) -> tuple[int, int, int, int]:
    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    if "top" in corner:
        top = max(top, 2)
    else:
        bottom = max(bottom, 2)
    if "left" in corner:
        left = max(left, 2)
    else:
        right = max(right, 2)
    return top, bottom, left, right


def _layer_boxes(
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    row_steps: list[int],
    col_steps: list[int],
) -> list[tuple[int, int, int, int]]:
    pad_h = sum(row_steps)
    pad_w = sum(col_steps)
    r0 = top
    c0 = left
    r1 = top + inner_h + 2 * pad_h - 1
    c1 = left + inner_w + 2 * pad_w - 1
    boxes = [(r0, c0, r1, c1)]
    for dr, dc in zip(row_steps, col_steps):
        r0 += int(dr)
        c0 += int(dc)
        r1 -= int(dr)
        c1 -= int(dc)
        boxes.append((r0, c0, r1, c1))
    return boxes


def _place_stripe(
    grid: np.ndarray,
    colors: list[int],
    corner: str,
    orientation: str,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    out = grid.copy()
    n = len(colors)
    h, w = out.shape
    if orientation == "horizontal":
        row = 0 if "top" in corner else h - 1
        start_col = 0 if "left" in corner else w - n
        out[row, start_col : start_col + n] = np.array(colors, dtype=int)
        return out, (row, start_col, row, start_col + n - 1)

    col = 0 if "left" in corner else w - 1
    start_row = 0 if "top" in corner else h - n
    out[start_row : start_row + n, col] = np.array(colors, dtype=int)
    return out, (start_row, col, start_row + n - 1, col)


def _expected_output(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    stripe_box: tuple[int, int, int, int],
    stripe_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    sr0, sc0, sr1, sc1 = stripe_box
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = BG
    for color, box in zip(stripe_colors, boxes):
        _paint_rectangle_outline(out, box, int(color))
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_layers = random.randint(*LAYER_RANGE)
    row_steps = [random.randint(*STEP_RANGE) for _ in range(n_layers - 1)]
    col_steps = [random.randint(*STEP_RANGE) for _ in range(n_layers - 1)]
    inner_h = random.randint(*INNER_HEIGHT_RANGE)
    inner_w = random.randint(*INNER_WIDTH_RANGE)
    outer_h = inner_h + 2 * sum(row_steps)
    outer_w = inner_w + 2 * sum(col_steps)

    corner = random.choice(("top-left", "top-right", "bottom-left", "bottom-right"))
    orientation = random.choice(("horizontal", "vertical"))
    top, bottom, left, right = _corner_margin(corner)
    height = outer_h + top + bottom
    width = outer_w + left + right
    if max(height, width) > GRID_LIMIT:
        return None

    palette = random.sample(list(NON_BG_COLORS), n_layers)
    stripe_colors = list(palette)
    current_colors = list(palette)
    while current_colors == stripe_colors:
        random.shuffle(current_colors)

    grid = np.full((height, width), BG, dtype=int)
    boxes = _layer_boxes(top, left, inner_h, inner_w, row_steps, col_steps)
    for color, box in zip(current_colors, boxes):
        _paint_rectangle_outline(grid, box, int(color))

    grid, stripe_box = _place_stripe(grid, stripe_colors, corner, orientation)
    expected = _expected_output(grid, boxes, stripe_box, stripe_colors)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid nested-rectangle recoloring input")

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 4
RECT_COLORS = tuple(color for color in range(10) if color not in (BACKGROUND, ANCHOR_COLOR))

QUADRANTS = ("tl", "tr", "bl", "br")
MIN_RECT_SIDE = 2

GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
RECT_COUNT_RANGE = (2, 4)
MAX_AREA_FRACTION = 0.42
GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _quadrant_of(shape: tuple[int, int], anchor: tuple[int, int]) -> str:
    h, w = shape
    r, c = anchor
    top = r < h // 2
    left = c < w // 2
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _aligned_bbox(anchor: tuple[int, int], height: int, width: int, quadrant: str) -> tuple[int, int, int, int]:
    ar, ac = anchor
    if quadrant == "tl":
        r0, c0 = ar, ac
    elif quadrant == "tr":
        r0, c0 = ar, ac - width + 1
    elif quadrant == "bl":
        r0, c0 = ar - height + 1, ac
    else:
        r0, c0 = ar - height + 1, ac - width + 1
    return r0, c0, r0 + height - 1, c0 + width - 1


def _fits_in_bounds(
    shape: tuple[int, int], anchor: tuple[int, int], height: int, width: int, quadrant: str
) -> bool:
    h, w = shape
    r0, c0, r1, c1 = _aligned_bbox(anchor, height, width, quadrant)
    return 0 <= r0 and 0 <= c0 and r1 < h and c1 < w


def _paint_clipped(
    out: np.ndarray, anchor: tuple[int, int], height: int, width: int, color: int, quadrant: str
) -> None:
    rows, cols = out.shape
    r0, c0, r1, c1 = _aligned_bbox(anchor, height, width, quadrant)
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(rows - 1, r1)
    cc1 = min(cols - 1, c1)
    if rr0 <= rr1 and cc0 <= cc1:
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = color


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    rectangles = []
    colors = [int(color) for color in np.unique(grid) if color not in (BACKGROUND, ANCHOR_COLOR)]
    for color in colors:
        mask = grid == color
        box = _bbox(mask)
        assert box is not None, "every rectangle color must appear"
        r0, c0, r1, c1 = box
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        area = height * width
        assert int(mask.sum()) == area, "each foreground color must form one solid rectangle"
        assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), "rectangle bbox must be fully filled"
        rectangles.append(
            {
                "color": int(color),
                "bbox": box,
                "height": int(height),
                "width": int(width),
                "area": int(area),
            }
        )
    rectangles.sort(
        key=lambda rect: (
            int(rect["area"]),
            int(rect["height"]),
            int(rect["width"]),
            int(rect["bbox"][0]),
            int(rect["bbox"][1]),
        )
    )
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather scattered solid rectangles and redraw them as an area-ordered stack anchored at the yellow marker.

    Input structure:
    - The background is black.
    - Several non-black, non-yellow colors each form one solid axis-aligned rectangle, with no overlaps.
    - Exactly one yellow pixel marks the anchor and does not belong to any rectangle.

    Rule:
    - Find every colored rectangle and sort them by area from smallest to largest.
    - Determine which quadrant contains the anchor, using the grid midpoint as the horizontal and vertical split.
    - Redraw the rectangles at the anchor by matching the anchor to the quadrant's corner: top-left, top-right, bottom-left, or bottom-right.
    - Paint from largest area to smallest area so smaller rectangles overwrite larger ones, and clip any out-of-bounds parts.
    - Remove all original rectangles; the output keeps only the reconstructed stack on a black canvas.

    Key insight:
    - Two independent cues must be combined: rectangle area decides the layer order, while the anchor quadrant decides which corner of every rectangle is pinned to the same point.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    anchor_cells = np.argwhere(g == ANCHOR_COLOR)
    assert anchor_cells.shape[0] == 1, "expected exactly one yellow anchor"
    anchor = tuple(map(int, anchor_cells[0]))

    rectangles = _extract_rectangles(g)
    out = np.full(g.shape, BACKGROUND, dtype=int)
    if not rectangles:
        out[anchor] = ANCHOR_COLOR
        return out

    quadrant = _quadrant_of(g.shape, anchor)
    for rect in reversed(rectangles):
        _paint_clipped(
            out,
            anchor,
            int(rect["height"]),
            int(rect["width"]),
            int(rect["color"]),
            quadrant,
        )
    if out[anchor] == BACKGROUND:
        out[anchor] = ANCHOR_COLOR
    return out


def _sample_monotone_sequence(largest: int, count: int) -> list[int] | None:
    values = list(range(MIN_RECT_SIDE, largest))
    if len(values) < count - 1:
        return None
    seq = random.sample(values, count - 1)
    seq.append(largest)
    seq.sort()
    return seq


def _build_rectangle_dims(height: int, width: int, count: int, clip_mode: bool) -> list[tuple[int, int]] | None:
    need = MIN_RECT_SIDE + count - 1
    if clip_mode:
        axis = random.choice(("h", "w"))
        if axis == "h":
            low_h = max(need, height // 2 + 1)
            high_h = min(height - 2, low_h + 5)
            low_w = need
            high_w = min(width - 2, max(low_w, need + 3))
        else:
            low_h = need
            high_h = min(height - 2, max(low_h, need + 3))
            low_w = max(need, width // 2 + 1)
            high_w = min(width - 2, low_w + 5)
    else:
        low_h = need
        high_h = min(height - 2, max(low_h, height // 2 + 2))
        low_w = need
        high_w = min(width - 2, max(low_w, width // 2 + 2))

    if low_h > high_h or low_w > high_w:
        return None

    largest_h = random.randint(low_h, high_h)
    largest_w = random.randint(low_w, high_w)
    h_seq = _sample_monotone_sequence(largest_h, count)
    w_seq = _sample_monotone_sequence(largest_w, count)
    if h_seq is None or w_seq is None:
        return None

    dims = list(zip(h_seq, w_seq))
    total_area = sum(h * w for h, w in dims)
    if total_area >= height * width * MAX_AREA_FRACTION:
        return None
    return dims


def _candidate_anchor_cells(
    height: int, width: int, quadrant: str, largest: tuple[int, int], clip_mode: bool
) -> list[tuple[int, int]]:
    largest_h, largest_w = largest
    candidates = []
    for r in range(height):
        if height % 2 == 1 and r == height // 2:
            continue
        for c in range(width):
            if width % 2 == 1 and c == width // 2:
                continue
            if _quadrant_of((height, width), (r, c)) != quadrant:
                continue
            fits = _fits_in_bounds((height, width), (r, c), largest_h, largest_w, quadrant)
            if fits != clip_mode:
                candidates.append((r, c))
    return candidates


def _free_positions(occupied: np.ndarray, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    rows, cols = occupied.shape
    positions = []
    for r0 in range(rows - rect_h + 1):
        for c0 in range(cols - rect_w + 1):
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(rows, r0 + rect_h + 1)
            cc1 = min(cols, c0 + rect_w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_input_rectangles(
    shape: tuple[int, int], anchor: tuple[int, int], dims: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    rows, cols = shape
    occupied = np.zeros((rows, cols), dtype=bool)
    ar, ac = anchor
    occupied[max(0, ar - 1) : min(rows, ar + 2), max(0, ac - 1) : min(cols, ac + 2)] = True

    placements: dict[tuple[int, int], tuple[int, int]] = {}
    for rect_h, rect_w in sorted(dims, key=lambda hw: hw[0] * hw[1], reverse=True):
        choices = _free_positions(occupied, rect_h, rect_w)
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        placements[(rect_h, rect_w)] = (r0, c0)
        occupied[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

    return [placements[(rect_h, rect_w)] for rect_h, rect_w in dims]


def _build_expected(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    quadrant: str,
    dims: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for (rect_h, rect_w), color in sorted(
        zip(dims, colors),
        key=lambda item: (item[0][0] * item[0][1], item[0][0], item[0][1]),
        reverse=True,
    ):
        _paint_clipped(out, anchor, rect_h, rect_w, color, quadrant)
    if out[anchor] == BACKGROUND:
        out[anchor] = ANCHOR_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    count = random.randint(*RECT_COUNT_RANGE)
    quadrant = random.choice(QUADRANTS)
    clip_mode = random.random() < 0.5

    dims = _build_rectangle_dims(height, width, count, clip_mode)
    if dims is None:
        return None

    anchor_choices = _candidate_anchor_cells(height, width, quadrant, dims[-1], clip_mode)
    if not anchor_choices:
        return None
    anchor = random.choice(anchor_choices)

    placements = _place_input_rectangles((height, width), anchor, dims)
    if placements is None:
        return None

    colors = random.sample(list(RECT_COLORS), count)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[anchor] = ANCHOR_COLOR
    for (rect_h, rect_w), color, (r0, c0) in zip(dims, colors, placements):
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color

    expected = _build_expected((height, width), anchor, quadrant, dims, colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

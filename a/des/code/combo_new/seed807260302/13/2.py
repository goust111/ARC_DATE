from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.asarray(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _filled_rect_dims(cells: set[tuple[int, int]]) -> tuple[int, int] | None:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    return (h, w) if len(cells) == h * w else None


def _horizontal_rect_dims(a: dict, b: dict) -> tuple[int, int] | None:
    r0a, c0a, r1a, c1a = a["bbox"]
    r0b, c0b, r1b, c1b = b["bbox"]
    if (r0a, r1a) != (r0b, r1b):
        return None

    a_local = {(int(r - r0a), int(c - c0a)) for r, c in a["coords"]}
    b_local = {(int(r - r0b), int(c - c0b)) for r, c in b["coords"]}
    wa = c1a - c0a + 1
    wb = c1b - c0b + 1
    a_left = (c0a + c1a) <= (c0b + c1b)
    dims: set[tuple[int, int]] = set()

    for shift in range(-(wa + wb), wa + wb + 1):
        if a_left and 2 * shift + wa - 1 > wb - 1:
            continue
        if (not a_left) and 2 * shift + wa - 1 < wb - 1:
            continue

        moved = {(r, c + shift) for r, c in a_local}
        if moved & b_local:
            continue

        rect_dims = _filled_rect_dims(moved | b_local)
        if rect_dims is not None:
            dims.add(rect_dims)

    if len(dims) != 1:
        return None
    return next(iter(dims))


def _vertical_rect_dims(a: dict, b: dict) -> tuple[int, int] | None:
    r0a, c0a, r1a, c1a = a["bbox"]
    r0b, c0b, r1b, c1b = b["bbox"]
    if (c0a, c1a) != (c0b, c1b):
        return None

    a_local = {(int(r - r0a), int(c - c0a)) for r, c in a["coords"]}
    b_local = {(int(r - r0b), int(c - c0b)) for r, c in b["coords"]}
    ha = r1a - r0a + 1
    hb = r1b - r0b + 1
    a_top = (r0a + r1a) <= (r0b + r1b)
    dims: set[tuple[int, int]] = set()

    for shift in range(-(ha + hb), ha + hb + 1):
        if a_top and 2 * shift + ha - 1 > hb - 1:
            continue
        if (not a_top) and 2 * shift + ha - 1 < hb - 1:
            continue

        moved = {(r + shift, c) for r, c in a_local}
        if moved & b_local:
            continue

        rect_dims = _filled_rect_dims(moved | b_local)
        if rect_dims is not None:
            dims.add(rect_dims)

    if len(dims) != 1:
        return None
    return next(iter(dims))


def _midpoint_center(a: dict, b: dict) -> tuple[int, int]:
    r0a, c0a, r1a, c1a = a["bbox"]
    r0b, c0b, r1b, c1b = b["bbox"]
    center_r = (r0a + r1a + r0b + r1b + 2) // 4
    center_c = (c0a + c1a + c0b + c1b + 2) // 4
    return int(center_r), int(center_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each color's two fragments, merge them into a solid rectangle, and place that rectangle midway between the originals.

    Input structure:
    - The grid has a dominant background color.
    - Each non-background color appears in exactly two 4-connected objects.
    - Same-colored objects lie in one shared row band or one shared column band, and their shapes are complementary parts of one axis-aligned rectangle.

    Rule:
    - Group objects by color and pair the two components of each color.
    - Infer whether each pair should merge horizontally or vertically from the shared bbox projection, and find the unique solid rectangle that those two shapes can form after sliding along that axis.
    - Erase the original fragments and draw the merged solid rectangle centered at the rounded midpoint of the two original bbox centers.

    Key insight:
    - Color gives the pairing, but the rectangle size comes from the two fragment shapes themselves; the original spacing only decides where the completed rectangle is re-centered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    components = _find_components(g, bg)
    assert components, "expected at least one foreground component"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    out = np.full_like(g, int(bg))
    for color, parts in by_color.items():
        assert len(parts) == 2, "each foreground color must have exactly two components"
        dims = []
        horiz = _horizontal_rect_dims(parts[0], parts[1])
        if horiz is not None:
            dims.append(horiz)
        vert = _vertical_rect_dims(parts[0], parts[1])
        if vert is not None:
            dims.append(vert)
        assert len(dims) == 1, "each color pair must determine one merged rectangle"

        rect_h, rect_w = dims[0]
        center_r, center_c = _midpoint_center(parts[0], parts[1])
        top = center_r - rect_h // 2
        left = center_c - rect_w // 2
        assert 0 <= top and top + rect_h <= h
        assert 0 <= left and left + rect_w <= w
        assert np.all(out[top : top + rect_h, left : left + rect_w] == bg), "merged rectangles must not overlap"
        out[top : top + rect_h, left : left + rect_w] = int(color)

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
PAIR_COUNT_RANGE = (1, 3)
RECT_HEIGHT_RANGE = (2, 6)
RECT_WIDTH_RANGE = (4, 8)
SEPARATION_RANGE = (1, 6)
PAIR_PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 400


def _boxes_separated(a: BBox, b: BBox, pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _mask_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([int(top), int(left)], dtype=int)


def _sample_horizontal_masks(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray]:
    widths = [random.randint(1, rect_w - 1) for _ in range(rect_h)]
    if rect_h > 1 and rect_w > 2 and len(set(widths)) == 1 and random.choice([False, True]):
        i = random.randrange(rect_h)
        options = [v for v in range(1, rect_w) if v != widths[i]]
        widths[i] = random.choice(options)
    if random.choice([False, True]):
        widths.reverse()

    left_w = max(widths)
    right_start = min(widths)
    right_w = rect_w - right_start
    left = np.zeros((rect_h, left_w), dtype=bool)
    right = np.zeros((rect_h, right_w), dtype=bool)

    for r, width in enumerate(widths):
        left[r, :width] = True
        right[r, width - right_start :] = True

    return left, right


def _sample_vertical_masks(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray]:
    heights = [random.randint(1, rect_h - 1) for _ in range(rect_w)]
    if rect_w > 1 and rect_h > 2 and len(set(heights)) == 1 and random.choice([False, True]):
        i = random.randrange(rect_w)
        options = [v for v in range(1, rect_h) if v != heights[i]]
        heights[i] = random.choice(options)
    if random.choice([False, True]):
        heights.reverse()

    top_h = max(heights)
    bottom_start = min(heights)
    bottom_h = rect_h - bottom_start
    top = np.zeros((top_h, rect_w), dtype=bool)
    bottom = np.zeros((bottom_h, rect_w), dtype=bool)

    for c, height in enumerate(heights):
        top[:height, c] = True
        bottom[height - bottom_start :, c] = True

    return top, bottom


def _place_horizontal_pair(
    grid_shape: tuple[int, int],
    zones: list[BBox],
    rect_h: int,
    rect_w: int,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, BBox, BBox] | None:
    grid_h, grid_w = grid_shape
    left_w = left_mask.shape[1]
    right_w = right_mask.shape[1]
    max_gap = min(SEPARATION_RANGE[1], grid_w - left_w - right_w - 1)
    if max_gap < SEPARATION_RANGE[0]:
        return None

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        gap = random.randint(SEPARATION_RANGE[0], max_gap)
        max_left = grid_w - left_w - gap - right_w
        if max_left < 0:
            continue

        row = random.randint(0, grid_h - rect_h)
        left_col = random.randint(0, max_left)
        right_col = left_col + left_w + gap

        center_c_num = 2 * left_col + left_w - 1 + 2 * right_col + right_w - 1
        center_c = (center_c_num + 2) // 4
        out_left = center_c - rect_w // 2
        if out_left < 0 or out_left + rect_w > grid_w:
            continue

        out_bbox = (row, out_left, row + rect_h - 1, out_left + rect_w - 1)
        zone = (
            row,
            min(left_col, out_left),
            row + rect_h - 1,
            max(right_col + right_w - 1, out_left + rect_w - 1),
        )
        if any(not _boxes_separated(zone, other) for other in zones):
            continue

        left_coords = _mask_coords(left_mask, row, left_col)
        right_coords = _mask_coords(right_mask, row, right_col)
        return left_coords, right_coords, out_bbox, zone

    return None


def _place_vertical_pair(
    grid_shape: tuple[int, int],
    zones: list[BBox],
    rect_h: int,
    rect_w: int,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, BBox, BBox] | None:
    grid_h, grid_w = grid_shape
    top_h = top_mask.shape[0]
    bottom_h = bottom_mask.shape[0]
    max_gap = min(SEPARATION_RANGE[1], grid_h - top_h - bottom_h - 1)
    if max_gap < SEPARATION_RANGE[0]:
        return None

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        gap = random.randint(SEPARATION_RANGE[0], max_gap)
        max_top = grid_h - top_h - gap - bottom_h
        if max_top < 0:
            continue

        col = random.randint(0, grid_w - rect_w)
        top_row = random.randint(0, max_top)
        bottom_row = top_row + top_h + gap

        center_r_num = 2 * top_row + top_h - 1 + 2 * bottom_row + bottom_h - 1
        center_r = (center_r_num + 2) // 4
        out_top = center_r - rect_h // 2
        if out_top < 0 or out_top + rect_h > grid_h:
            continue

        out_bbox = (out_top, col, out_top + rect_h - 1, col + rect_w - 1)
        zone = (
            min(top_row, out_top),
            col,
            max(bottom_row + bottom_h - 1, out_top + rect_h - 1),
            col + rect_w - 1,
        )
        if any(not _boxes_separated(zone, other) for other in zones):
            continue

        top_coords = _mask_coords(top_mask, top_row, col)
        bottom_coords = _mask_coords(bottom_mask, bottom_row, col)
        return top_coords, bottom_coords, out_bbox, zone

    return None


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    if pair_count > len(palette):
        return None
    colors = random.sample(palette, pair_count)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    expected = np.full_like(grid, bg)
    zones: list[BBox] = []

    for color in colors:
        horizontal = random.choice([False, True])
        rect_h = random.randint(*RECT_HEIGHT_RANGE)
        rect_w = random.randint(*RECT_WIDTH_RANGE)

        if horizontal:
            left_mask, right_mask = _sample_horizontal_masks(rect_h, rect_w)
            placed = _place_horizontal_pair(
                (grid_h, grid_w),
                zones,
                rect_h,
                rect_w,
                left_mask,
                right_mask,
            )
        else:
            top_mask, bottom_mask = _sample_vertical_masks(rect_h, rect_w)
            placed = _place_vertical_pair(
                (grid_h, grid_w),
                zones,
                rect_h,
                rect_w,
                top_mask,
                bottom_mask,
            )

        if placed is None:
            return None

        first_coords, second_coords, out_bbox, zone = placed
        zones.append(zone)
        _paint_coords(grid, first_coords, color)
        _paint_coords(grid, second_coords, color)

        r0, c0, r1, c1 = out_bbox
        expected[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sample = _sample_case()
        if sample is None:
            continue
        grid, expected = sample
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid fragment-merging input")

from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MIN_SIDE = 3
MAX_GENERATION_ATTEMPTS = 400

Rect = tuple[int, int, int, int]


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _border_cells(rect: Rect) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _draw_border(grid: np.ndarray, rect: Rect, color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _marker_pair(rect: Rect, diag: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if int(diag) == 0:
        return (r0, c0), (r1, c1)
    return (r0, c1), (r1, c0)


def _boxes_overlap(a: Rect, b: Rect, pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar0 - pad <= br1 + pad
        and br0 - pad <= ar1 + pad
        and ac0 - pad <= bc1 + pad
        and bc0 - pad <= ac1 + pad
    )


def _scan_order_colors(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == BLACK or color in seen:
                continue
            seen.add(color)
            colors.append(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct all same-color marker pairs as rectangular borders, drawing larger rectangles first so smaller overlaps stay visible on top.

    Input structure:
    - The background is black.
    - Each non-black color appears exactly twice as two isolated marker pixels.
    - Each same-color pair marks opposite corners of one axis-aligned rectangle.

    Rule:
    - Group the two pixels of each non-black color into one rectangle by taking
      their min/max row and column as the rectangle bounds.
    - Compute every rectangle's area as height times width, counting the border
      corners inside the bounding box.
    - Draw the rectangles as one-pixel-thick borders in descending area order,
      leaving each interior transparent.
    - Later, smaller rectangles overwrite earlier ones at border intersections.

    Key insight:
    - The colored pixels are not final objects; each color encodes a rectangle
      through its diagonal pair, and overlap colors are explained by a hidden
      layering rule based on rectangle area rather than by color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    coords_by_color: dict[int, list[tuple[int, int]]] = {}
    color_order = _scan_order_colors(g)
    for color in color_order:
        coords = np.argwhere(g == int(color))
        coords_by_color[int(color)] = [tuple(map(int, pt)) for pt in coords]

    assert coords_by_color, "expected at least one marker pair"

    items = []
    for color in color_order:
        points = coords_by_color[int(color)]
        assert len(points) == 2, "each non-background color must appear exactly twice"
        (r_a, c_a), (r_b, c_b) = points
        rect = (
            min(r_a, r_b),
            min(c_a, c_b),
            max(r_a, r_b),
            max(c_a, c_b),
        )
        items.append({"color": int(color), "rect": rect, "area": _rect_area(rect)})

    out = np.full(g.shape, BLACK, dtype=int)
    items.sort(key=lambda item: int(item["area"]), reverse=True)
    for item in items:
        _draw_border(out, item["rect"], int(item["color"]))
    return out


def _render_output(shape: tuple[int, int], rect_specs: list[dict[str, int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    ordered = sorted(rect_specs, key=lambda spec: _rect_area(spec["rect"]), reverse=True)
    for spec in ordered:
        _draw_border(out, spec["rect"], int(spec["color"]))
    return out


def _choose_diagonals(rects: list[Rect]) -> list[int] | None:
    borders = [_border_cells(rect) for rect in rects]
    masks = list(range(1 << len(rects)))
    random.shuffle(masks)

    for mask in masks:
        markers: list[tuple[int, int]] = []
        valid = True
        for i, rect in enumerate(rects):
            diag = (int(mask) >> i) & 1
            for marker in _marker_pair(rect, diag):
                if any(marker in borders[j] for j in range(len(rects)) if j != i):
                    valid = False
                    break
                if any(max(abs(marker[0] - other[0]), abs(marker[1] - other[1])) <= 1 for other in markers):
                    valid = False
                    break
                markers.append(marker)
            if not valid:
                break
        if valid:
            return [(int(mask) >> i) & 1 for i in range(len(rects))]
    return None


def _random_transform(shape: tuple[int, int], rects: list[Rect]) -> tuple[tuple[int, int], list[Rect]]:
    height, width = shape
    out_rects = list(rects)

    if random.random() < 0.5:
        out_rects = [(c0, r0, c1, r1) for r0, c0, r1, c1 in out_rects]
        height, width = width, height

    if random.random() < 0.5:
        out_rects = [(height - 1 - r1, c0, height - 1 - r0, c1) for r0, c0, r1, c1 in out_rects]

    if random.random() < 0.5:
        out_rects = [(r0, width - 1 - c1, r1, width - 1 - c0) for r0, c0, r1, c1 in out_rects]

    return (height, width), out_rects


def _crop_layout(shape: tuple[int, int], rects: list[Rect]) -> tuple[tuple[int, int], list[Rect]]:
    height, width = shape
    min_r = min(rect[0] for rect in rects)
    min_c = min(rect[1] for rect in rects)
    max_r = max(rect[2] for rect in rects)
    max_c = max(rect[3] for rect in rects)

    top_margin = random.randint(0, min_r)
    left_margin = random.randint(0, min_c)
    bottom_margin = random.randint(0, height - 1 - max_r)
    right_margin = random.randint(0, width - 1 - max_c)

    crop_r0 = min_r - top_margin
    crop_c0 = min_c - left_margin
    crop_r1 = max_r + bottom_margin
    crop_c1 = max_c + right_margin

    shifted = [
        (r0 - crop_r0, c0 - crop_c0, r1 - crop_r0, c1 - crop_c0)
        for r0, c0, r1, c1 in rects
    ]
    return (crop_r1 - crop_r0 + 1, crop_c1 - crop_c0 + 1), shifted


def _has_border_overlap(rects: list[Rect]) -> bool:
    borders = [_border_cells(rect) for rect in rects]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if borders[i] & borders[j]:
                return True
    return False


def _sample_extras(rects: list[Rect], shape: tuple[int, int], total_count: int) -> list[Rect] | None:
    height, width = shape
    out_rects = list(rects)
    used_areas = {_rect_area(rect) for rect in out_rects}

    while len(out_rects) < total_count:
        placed = False
        for _ in range(300):
            max_h = min(9, height - 2)
            max_w = min(9, width - 2)
            if max_h < MIN_SIDE or max_w < MIN_SIDE:
                return None

            h = random.randint(MIN_SIDE, max_h)
            w = random.randint(MIN_SIDE, max_w)
            area = h * w
            if area in used_areas:
                continue

            r0 = random.randint(1, height - h - 1)
            c0 = random.randint(1, width - w - 1)
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_overlap(rect, other, pad=1) for other in out_rects):
                continue

            out_rects.append(rect)
            used_areas.add(area)
            placed = True
            break

        if not placed:
            return None

    return out_rects


def _build_disjoint_layout() -> tuple[tuple[int, int], list[Rect]] | None:
    count = random.randint(2, 5)
    min_side = 8 + 3 * count
    height = random.randint(min_side, min(28, min_side + 4))
    width = random.randint(min_side, min(28, min_side + 4))
    rects = _sample_extras([], (height, width), count)
    if rects is None:
        return None
    return (height, width), rects


def _build_overlap_layout() -> tuple[tuple[int, int], list[Rect]] | None:
    count = random.randint(2, 5)
    height = random.randint(16 if count <= 3 else 20, 28)
    width = random.randint(18 if count <= 3 else 22, 28)

    for _ in range(300):
        h_a_max = min(12, height - 5)
        if h_a_max < 6:
            continue
        h_a = random.randint(6, h_a_max)
        w_a = random.randint(8, min(16, width - 4))
        r0_a = random.randint(2, height - h_a - 3)
        c0_a = random.randint(2, width - w_a - 2)
        r1_a = r0_a + h_a - 1
        c1_a = c0_a + w_a - 1

        top_max = min(4, r0_a - 1)
        bottom_max = min(4, height - r1_a - 2)
        max_w_b = min(6, w_a - 2)
        if top_max < 1 or bottom_max < 1 or max_w_b < MIN_SIDE:
            continue

        top = random.randint(1, top_max)
        bottom = random.randint(1, bottom_max)
        h_b = h_a + top + bottom
        w_b = random.randint(MIN_SIDE, max_w_b)
        if h_b * w_b >= h_a * w_a:
            continue

        c0_b_min = c0_a + 1
        c0_b_max = c1_a - w_b
        if c0_b_min > c0_b_max:
            continue
        c0_b = random.randint(c0_b_min, c0_b_max)

        rect_a = (r0_a, c0_a, r1_a, c1_a)
        rect_b = (r0_a - top, c0_b, r1_a + bottom, c0_b + w_b - 1)
        rects = [rect_a, rect_b]
        if not _has_border_overlap(rects):
            continue

        rects = _sample_extras(rects, (height, width), count)
        if rects is None:
            continue

        return (height, width), rects

    return None


def _build_case() -> np.ndarray | None:
    overlap_required = random.random() < 0.5
    layout = _build_overlap_layout() if overlap_required else _build_disjoint_layout()
    if layout is None:
        return None

    shape, rects = layout
    shape, rects = _random_transform(shape, rects)
    shape, rects = _crop_layout(shape, rects)

    if overlap_required != _has_border_overlap(rects):
        return None

    areas = [_rect_area(rect) for rect in rects]
    if len(areas) != len(set(areas)):
        return None

    diags = _choose_diagonals(rects)
    if diags is None:
        return None

    colors = random.sample(PALETTE, len(rects))
    specs = []
    inp = np.full(shape, BLACK, dtype=int)
    for color, rect, diag in zip(colors, rects, diags):
        (r_a, c_a), (r_b, c_b) = _marker_pair(rect, diag)
        inp[r_a, c_a] = int(color)
        inp[r_b, c_b] = int(color)
        specs.append({"color": int(color), "rect": rect})

    expected = _render_output(shape, specs)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-layering input")

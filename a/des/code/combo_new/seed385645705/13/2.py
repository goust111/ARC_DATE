from __future__ import annotations

import numpy as np

BBox = tuple[int, int, int, int]

# --------------- solver ---------------


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _non_background_colors_in_scan_order(grid: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for r, c in np.argwhere(grid != int(bg)):
        color = int(grid[r, c])
        if color not in seen:
            seen.add(color)
            order.append(color)
    return order


def _bbox_of_color(grid: np.ndarray, color: int) -> BBox:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, "expected color pixels"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _draw_border(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each color's scattered pixels with the one-cell border of that color's minimum bounding rectangle.

    Input structure:
    - The background is the dominant color.
    - Several non-background colors appear as sparse pixels or tiny clusters, with
      each color occurring a small number of times.
    - Pixels of different colors may be interleaved spatially, but each color is
      defined only by all cells of that exact color.

    Rule:
    - For each non-background color, find the minimum axis-aligned bounding box of
      all pixels of that color.
    - Draw the full one-pixel-thick border of that bounding box in the same color.
    - Clear every other original pixel of that color to the background.
    - If borders from different colors overlap, process colors by the first place
      they appear in row-major order and let later colors overwrite earlier ones.

    Key insight:
    - The scattered pixels are only markers for color-wise grouping; the real output
      object for each color is its full enclosing rectangle frame, layered by
      scan-order appearance rather than by color id.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    out = np.full_like(g, int(bg))

    for color in _non_background_colors_in_scan_order(g, bg):
        _draw_border(out, _bbox_of_color(g, color), color)

    return out


# --------------- generator ---------------

GRID_MIN_SIDE = 12
GRID_MAX_SIDE = 26
COLOR_COUNT_RANGE = (2, 5)
RECT_SIDE_RANGE = (2, 7)
LINE_LENGTH_RANGE = (2, 9)
PIXEL_COUNT_RANGE = (2, 6)
PLACEMENT_GAP = 1
BOX_RETRIES = 200
GENERATION_RETRIES = 500

RELATION_OVERLAP = "overlap"
RELATION_SEPARATE = "separate"

KIND_RECT = "rect"
KIND_HLINE = "hline"
KIND_VLINE = "vline"
LINE_KINDS = (KIND_HLINE, KIND_VLINE)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _bbox_height(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_width(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _bbox_area(bbox: BBox) -> int:
    return _bbox_height(bbox) * _bbox_width(bbox)


def _boxes_intersect(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _inflate_bbox(bbox: BBox, pad: int) -> BBox:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _boxes_too_close(a: BBox, b: BBox, pad: int) -> bool:
    return _boxes_intersect(_inflate_bbox(a, pad), _inflate_bbox(b, pad))


def _border_cells(bbox: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expected_from_specs(
    shape: tuple[int, int], bg: int, specs: list[dict[str, object]], color_order: list[int]
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    bbox_by_color = {int(spec["color"]): spec["bbox"] for spec in specs}
    for color in color_order:
        _draw_border(out, bbox_by_color[int(color)], int(color))
    return out


def _choose_grid_shape(rng: np.random.Generator, n_colors: int) -> tuple[int, int]:
    min_side = max(GRID_MIN_SIDE, 10 + 2 * n_colors)
    rows = _randint(rng, min_side, GRID_MAX_SIDE)
    cols = _randint(rng, min_side, GRID_MAX_SIDE)
    return rows, cols


def _sample_dims(rng: np.random.Generator, kind: str) -> tuple[int, int]:
    if kind == KIND_RECT:
        return _randint(rng, *RECT_SIDE_RANGE), _randint(rng, *RECT_SIDE_RANGE)
    if kind == KIND_HLINE:
        return 1, _randint(rng, *LINE_LENGTH_RANGE)
    assert kind == KIND_VLINE
    return _randint(rng, *LINE_LENGTH_RANGE), 1


def _random_bbox_of_kind(
    rng: np.random.Generator, rows: int, cols: int, kind: str
) -> BBox | None:
    height, width = _sample_dims(rng, kind)
    if height > rows or width > cols:
        return None
    r0 = _randint(rng, 0, rows - height)
    c0 = _randint(rng, 0, cols - width)
    return r0, c0, r0 + height - 1, c0 + width - 1


def _place_separate_bbox(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    kind: str,
    existing: list[BBox],
) -> BBox | None:
    for _ in range(BOX_RETRIES):
        bbox = _random_bbox_of_kind(rng, rows, cols, kind)
        if bbox is None:
            continue
        if all(not _boxes_too_close(bbox, other, PLACEMENT_GAP) for other in existing):
            return bbox
    return None


def _make_overlap_pair(rng: np.random.Generator, rows: int, cols: int) -> tuple[BBox, BBox] | None:
    mode = str(rng.choice(["share_top", "share_left", "cross"]))

    for _ in range(BOX_RETRIES):
        if mode == "share_top":
            h1 = _randint(rng, 2, 7)
            w1 = _randint(rng, 3, 8)
            h2 = _randint(rng, 2, 7)
            w2 = _randint(rng, 3, 8)
            delta = _randint(rng, -(w2 - 2), w1 - 2)
            c0_min = max(0, -delta)
            c0_max = min(cols - w1, cols - w2 - delta)
            r0_max = rows - max(h1, h2)
            if c0_min > c0_max or r0_max < 0:
                continue
            r0 = _randint(rng, 0, r0_max)
            c0 = _randint(rng, c0_min, c0_max)
            bbox_a = (r0, c0, r0 + h1 - 1, c0 + w1 - 1)
            bbox_b = (r0, c0 + delta, r0 + h2 - 1, c0 + delta + w2 - 1)
        elif mode == "share_left":
            h1 = _randint(rng, 3, 8)
            w1 = _randint(rng, 2, 7)
            h2 = _randint(rng, 3, 8)
            w2 = _randint(rng, 2, 7)
            delta = _randint(rng, -(h2 - 2), h1 - 2)
            r0_min = max(0, -delta)
            r0_max = min(rows - h1, rows - h2 - delta)
            c0_max = cols - max(w1, w2)
            if r0_min > r0_max or c0_max < 0:
                continue
            r0 = _randint(rng, r0_min, r0_max)
            c0 = _randint(rng, 0, c0_max)
            bbox_a = (r0, c0, r0 + h1 - 1, c0 + w1 - 1)
            bbox_b = (r0 + delta, c0, r0 + delta + h2 - 1, c0 + w2 - 1)
        else:
            h1 = _randint(rng, 4, 8)
            w1 = _randint(rng, 4, 8)
            h2 = _randint(rng, 3, 8)
            w2 = _randint(rng, 2, 7)
            if h1 > rows or w1 > cols or h2 > rows or w2 > cols:
                continue
            r0_a = _randint(rng, 0, rows - h1)
            c0_a = _randint(rng, 0, cols - w1)
            if w1 < 3:
                continue
            cross_col = _randint(rng, c0_a + 1, c0_a + w1 - 2)
            r0_b_min = max(0, r0_a - h2 + 1)
            r0_b_max = min(rows - h2, r0_a)
            if r0_b_min > r0_b_max or cross_col + w2 > cols:
                continue
            r0_b = _randint(rng, r0_b_min, r0_b_max)
            bbox_a = (r0_a, c0_a, r0_a + h1 - 1, c0_a + w1 - 1)
            bbox_b = (r0_b, cross_col, r0_b + h2 - 1, cross_col + w2 - 1)

        if bbox_a == bbox_b:
            continue
        if not _border_cells(bbox_a).intersection(_border_cells(bbox_b)):
            continue
        return bbox_a, bbox_b

    return None


def _edge_candidates(
    bbox: BBox, used: set[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    top = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in used]
    bottom = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in used]
    left = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in used]
    right = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in used]
    return top, bottom, left, right


def _sample_pixels_in_bbox(
    rng: np.random.Generator, bbox: BBox, used: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    available = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in used
    ]
    if len(available) < PIXEL_COUNT_RANGE[0]:
        return None

    if r0 == r1:
        required = {(r0, c0), (r0, c1)}
    elif c0 == c1:
        required = {(r0, c0), (r1, c0)}
    else:
        top, bottom, left, right = _edge_candidates(bbox, used)
        if not top or not bottom or not left or not right:
            return None
        required = {
            top[_randint(rng, 0, len(top) - 1)],
            bottom[_randint(rng, 0, len(bottom) - 1)],
            left[_randint(rng, 0, len(left) - 1)],
            right[_randint(rng, 0, len(right) - 1)],
        }

    if any(cell in used for cell in required):
        return None

    max_pixels = min(PIXEL_COUNT_RANGE[1], len(available))
    min_pixels = max(PIXEL_COUNT_RANGE[0], len(required))
    if min_pixels > max_pixels:
        return None
    target_count = _randint(rng, min_pixels, max_pixels)

    chosen = set(required)
    remaining = [cell for cell in available if cell not in chosen]

    interior = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in used and (r, c) not in chosen
    ]
    if interior and len(chosen) < target_count and rng.random() < 0.65:
        cell = interior[_randint(rng, 0, len(interior) - 1)]
        chosen.add(cell)
        remaining = [pos for pos in remaining if pos != cell]

    while len(chosen) < target_count:
        if not remaining:
            return None
        idx = _randint(rng, 0, len(remaining) - 1)
        chosen.add(remaining.pop(idx))

    coords = sorted(chosen)
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    if (min(rows), min(cols), max(rows), max(cols)) != bbox:
        return None
    return coords


def _build_specs(
    rng: np.random.Generator, rows: int, cols: int, colors: list[int]
) -> tuple[list[dict[str, object]], str, bool] | None:
    relation = RELATION_OVERLAP if rng.random() < 0.5 else RELATION_SEPARATE
    need_line = bool(rng.integers(0, 2))
    if relation == RELATION_OVERLAP and need_line and len(colors) < 3:
        return None

    boxes: list[BBox] = []
    if relation == RELATION_OVERLAP:
        pair = _make_overlap_pair(rng, rows, cols)
        if pair is None:
            return None
        boxes.extend(pair)
    else:
        for i in range(2):
            if need_line and i == 1 and len(boxes) == 1:
                kind = str(rng.choice(LINE_KINDS))
            else:
                kind = KIND_RECT
            bbox = _place_separate_bbox(rng, rows, cols, kind, boxes)
            if bbox is None:
                return None
            boxes.append(bbox)

    line_added = any(_bbox_height(box) == 1 or _bbox_width(box) == 1 for box in boxes)
    while len(boxes) < len(colors):
        if need_line and not line_added and len(boxes) == len(colors) - 1:
            kind = str(rng.choice(LINE_KINDS))
        elif need_line and not line_added and rng.random() < 0.35:
            kind = str(rng.choice(LINE_KINDS))
        else:
            kind = KIND_RECT
        bbox = _place_separate_bbox(rng, rows, cols, kind, boxes)
        if bbox is None:
            return None
        boxes.append(bbox)
        line_added = line_added or _bbox_height(bbox) == 1 or _bbox_width(bbox) == 1

    if need_line and not line_added:
        return None

    if relation == RELATION_OVERLAP:
        overlap_found = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _border_cells(boxes[i]).intersection(_border_cells(boxes[j])):
                    overlap_found = True
                    break
            if overlap_found:
                break
        if not overlap_found:
            return None
    else:
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _border_cells(boxes[i]).intersection(_border_cells(boxes[j])):
                    return None

    specs = [{"color": int(color), "bbox": bbox} for color, bbox in zip(colors, boxes)]
    return specs, relation, need_line


def _render_input(
    rng: np.random.Generator, shape: tuple[int, int], bg: int, specs: list[dict[str, object]]
) -> np.ndarray | None:
    grid = np.full(shape, int(bg), dtype=int)
    used: set[tuple[int, int]] = set()

    order = list(rng.permutation(len(specs)))
    for idx in order:
        spec = specs[idx]
        coords = _sample_pixels_in_bbox(rng, spec["bbox"], used)
        if coords is None:
            return None
        spec["coords"] = coords
        for r, c in coords:
            used.add((r, c))
            grid[r, c] = int(spec["color"])

    return grid


def _valid_scene(
    inp: np.ndarray, bg: int, specs: list[dict[str, object]], relation: str, need_line: bool
) -> bool:
    if _dominant_color(inp) != int(bg):
        return False

    colors_in_input = _non_background_colors_in_scan_order(inp, bg)
    if set(colors_in_input) != {int(spec["color"]) for spec in specs}:
        return False

    if need_line and not any(
        _bbox_height(spec["bbox"]) == 1 or _bbox_width(spec["bbox"]) == 1 for spec in specs
    ):
        return False

    for spec in specs:
        coords = spec.get("coords")
        if coords is None:
            return False
        if not (PIXEL_COUNT_RANGE[0] <= len(coords) <= PIXEL_COUNT_RANGE[1]):
            return False
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        if bbox != spec["bbox"]:
            return False

    border_overlap = False
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if _border_cells(specs[i]["bbox"]).intersection(_border_cells(specs[j]["bbox"])):
                border_overlap = True
                break
        if border_overlap:
            break
    if relation == RELATION_OVERLAP and not border_overlap:
        return False
    if relation == RELATION_SEPARATE and border_overlap:
        return False

    expected = _expected_from_specs(inp.shape, bg, specs, colors_in_input)
    return np.array_equal(main(inp), expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_colors = _randint(rng, *COLOR_COUNT_RANGE)
    rows, cols = _choose_grid_shape(rng, n_colors)
    bg = _randint(rng, 0, 9)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    colors = palette[:n_colors]

    built = _build_specs(rng, rows, cols, colors)
    if built is None:
        return None
    specs, relation, need_line = built

    inp = _render_input(rng, (rows, cols), bg, specs)
    if inp is None:
        return None
    if not _valid_scene(inp, bg, specs, relation, need_line):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input grid")

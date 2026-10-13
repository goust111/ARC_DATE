import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RECT_COUNT_RANGE = (3, 5)
TOP_SIZE_RANGE = (2, 6)
EXPAND_MAX = 3
MAIN_MARGIN_RANGE = (0, 3)
LEGEND_TOP_RANGE = (0, 2)
LEGEND_BOTTOM_RANGE = (0, 2)
MAIN_TOP_RANGE = (0, 3)
MAX_OUTER_SIZE = 18
GEN_ATTEMPTS = 400


def _color_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(grid == int(color))
    assert rows.size > 0, "each rectangle color must remain visible"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _legend_and_main(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 3, "expected main area, separator, and legend column"
    assert bool(np.all(g[:, -2] == BG)), "the separator column must be black"

    main = g[:, :-2]
    legend = [int(v) for v in g[:, -1].tolist() if int(v) != BG]
    assert legend, "legend column must contain at least one colored cell"
    return main, legend


def _stack_order(main: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int, int, int]]]:
    colors = [int(c) for c in np.unique(main) if int(c) != BG]
    assert len(colors) >= 2, "expected at least two rectangle colors"

    bboxes = {color: _color_bbox(main, color) for color in colors}
    edges = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}

    for lower in colors:
        r0, c0, r1, c1 = bboxes[lower]
        window_colors = {
            int(c)
            for c in np.unique(main[r0 : r1 + 1, c0 : c1 + 1]).tolist()
            if int(c) not in (BG, lower)
        }
        for upper in window_colors:
            if upper not in edges[lower]:
                edges[lower].add(upper)
                indeg[upper] += 1

    q = deque(sorted([color for color in colors if indeg[color] == 0]))
    order: list[int] = []
    while q:
        assert len(q) == 1, "overlap relations must determine a unique stack order"
        color = q.popleft()
        order.append(color)
        for nxt in sorted(edges[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    assert len(order) == len(colors), "stack relation graph must be acyclic"
    return order, bboxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden rectangles from overlap evidence, infer the stack order, and
    extract the legend-selected layer.

    Input structure
    - Background is black. The main area contains 3 to 5 overlapping axis-aligned
      monochrome rectangles with distinct colors.
    - The rightmost column is a legend column, separated from the main area by one
      black column. Each legend mark is a 1x1 colored cell.

    Rule
    - For each visible color in the main area, recover the full rectangle by taking
      the bounding box of that color's visible pixels.
    - If another color appears inside a recovered rectangle, that other color must be
      above it. Use all such cover relations to recover the unique bottom-to-top order.
    - Read the legend top-to-bottom. The 1st legend color refers to the bottom layer,
      the 2nd to the next layer, and so on; therefore the number of legend entries
      determines which depth to extract.
    - Output only the recovered rectangle at that depth, cropped to its full size.

    Key insight
    The visible fragments are only clues. First recover full rectangles, then infer
    the stack order, and only then decode the legend depth.
    """
    main_area, legend = _legend_and_main(np.asarray(grid, dtype=int))
    order, bboxes = _stack_order(main_area)
    assert len(legend) == len(set(legend)), "legend colors must be unique"
    assert set(legend).issubset(set(order)), "legend colors must come from the rectangle colors"

    target_depth = len(legend)
    assert 1 <= target_depth <= len(order), "legend depth must point to an existing layer"
    target_color = int(order[target_depth - 1])
    r0, c0, r1, c1 = bboxes[target_color]
    out_h = int(r1 - r0 + 1)
    out_w = int(c1 - c0 + 1)
    return np.full((out_h, out_w), target_color, dtype=int)


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _sample_nested_boxes(n_rects: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(120):
        top_h = random.randint(*TOP_SIZE_RANGE)
        top_w = random.randint(*TOP_SIZE_RANGE)
        top_box = (0, 0, top_h - 1, top_w - 1)
        boxes_top_to_bottom = [top_box]

        cur = top_box
        ok = True
        for _ in range(n_rects - 1):
            placed = False
            for _ in range(80):
                up = random.randint(0, EXPAND_MAX)
                down = random.randint(0, EXPAND_MAX)
                left = random.randint(0, EXPAND_MAX)
                right = random.randint(0, EXPAND_MAX)
                if up + down == 0 or left + right == 0:
                    continue

                nxt = (cur[0] - up, cur[1] - left, cur[2] + down, cur[3] + right)
                h = nxt[2] - nxt[0] + 1
                w = nxt[3] - nxt[1] + 1
                if h > MAX_OUTER_SIZE or w > MAX_OUTER_SIZE:
                    continue
                boxes_top_to_bottom.append(nxt)
                cur = nxt
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        outer = boxes_top_to_bottom[-1]
        shift_r = -outer[0]
        shift_c = -outer[1]
        normalized = [_shift_box(box, shift_r, shift_c) for box in boxes_top_to_bottom]
        boxes_bottom_to_top = list(reversed(normalized))
        outer_h = boxes_bottom_to_top[0][2] + 1
        outer_w = boxes_bottom_to_top[0][3] + 1
        if 6 <= outer_h <= MAX_OUTER_SIZE and 6 <= outer_w <= MAX_OUTER_SIZE:
            return boxes_bottom_to_top
    return None


def _render_main(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    canvas = np.full(shape, BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, boxes, strict=False):
        canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return canvas


def _expected_output(
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    target_depth: int,
) -> np.ndarray:
    target_color = int(colors[target_depth - 1])
    r0, c0, r1, c1 = boxes[target_depth - 1]
    return np.full((r1 - r0 + 1, c1 - c0 + 1), target_color, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    boxes = _sample_nested_boxes(n_rects)
    if boxes is None:
        return None

    colors = random.sample(PALETTE, n_rects)
    target_depth = random.randint(1, n_rects)

    outer_r0, outer_c0, outer_r1, outer_c1 = boxes[0]
    outer_h = outer_r1 - outer_r0 + 1
    outer_w = outer_c1 - outer_c0 + 1

    main_top_margin = random.randint(*MAIN_MARGIN_RANGE)
    main_bottom_margin = random.randint(*MAIN_MARGIN_RANGE)
    main_left_margin = random.randint(*MAIN_MARGIN_RANGE)
    main_right_margin = random.randint(*MAIN_MARGIN_RANGE)
    main_h = outer_h + main_top_margin + main_bottom_margin
    main_w = outer_w + main_left_margin + main_right_margin
    shifted_boxes = [
        _shift_box(box, main_top_margin, main_left_margin) for box in boxes
    ]

    legend_top = random.randint(*LEGEND_TOP_RANGE)
    legend_bottom = random.randint(*LEGEND_BOTTOM_RANGE)
    total_h = max(
        main_h + random.randint(*MAIN_TOP_RANGE),
        legend_top + target_depth + legend_bottom,
    )
    main_row = random.randint(0, total_h - main_h)

    full_main = np.full((total_h, main_w), BG, dtype=int)
    rendered = _render_main((main_h, main_w), shifted_boxes, colors)
    full_main[main_row : main_row + main_h, :] = rendered

    grid = np.full((total_h, main_w + 2), BG, dtype=int)
    grid[:, :main_w] = full_main
    legend_colors = colors[:target_depth]
    for i, color in enumerate(legend_colors):
        grid[legend_top + i, -1] = int(color)

    expected = _expected_output(shifted_boxes, colors, target_depth)

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.array_equal(out, expected):
        return None

    main_area, legend = _legend_and_main(grid)
    order, bboxes = _stack_order(main_area)
    if order != colors:
        return None
    if len(legend) != target_depth:
        return None
    actual_boxes = [_shift_box(box, main_row, 0) for box in shifted_boxes]
    for color, box in zip(colors, actual_boxes, strict=False):
        if bboxes[int(color)] != box:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")

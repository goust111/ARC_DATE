import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_rectangles(grid: np.ndarray, bg: int) -> dict[int, dict[str, int]]:
    body = grid[:, :-1]
    rectangles: dict[int, dict[str, int]] = {}
    for color in sorted(int(v) for v in np.unique(body) if int(v) != bg):
        coords = np.argwhere(body == color)
        assert coords.size > 0, "each rectangle color must appear in the body"
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        block = body[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(block == color), "each object must be a solid rectangle"
        assert 1 <= h <= 3 and 2 <= w <= 3, "rectangle size out of range"
        rectangles[color] = {"r0": r0, "c0": c0, "h": h, "w": w}

    rowspans = sorted(
        (rect["r0"], rect["r0"] + rect["h"] - 1, color) for color, rect in rectangles.items()
    )
    for (_, end_a, _), (start_b, _, _) in zip(rowspans, rowspans[1:]):
        assert end_a < start_b, "rectangle row intervals must be disjoint"
    return rectangles


def _marker_order(grid: np.ndarray, bg: int) -> list[int]:
    order = [int(v) for v in grid[:, -1] if int(v) != bg]
    assert len(order) == len(set(order)) and order, "marker column must contain unique colors"
    return order


def _cascade_layout(
    rectangles: dict[int, dict[str, int]], order: list[int], total_width: int
) -> dict[int, int]:
    barrier = total_width - 1
    left_by_color: dict[int, int] = {}
    for color in order:
        width = rectangles[color]["w"]
        left = barrier - width
        assert left >= 0, "cascade placement ran off the left edge"
        left_by_color[color] = left
        barrier = left
    return left_by_color


def _render(
    shape: tuple[int, int],
    rectangles: dict[int, dict[str, int]],
    left_by_color: dict[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color, rect in rectangles.items():
        r0 = rect["r0"]
        h = rect["h"]
        w = rect["w"]
        c0 = left_by_color[color]
        out[r0 : r0 + h, c0 : c0 + w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide colored rectangles rightward in the marker-column order so they form a global cascade of blockers.

    Input structure:
    - The background is black. The last column is a sparse order-marker column containing one singleton pixel for each rectangle color.
    - In the body of the grid, each non-background color forms exactly one solid filled rectangle of height 1-3 and width 2-3.
    - Rectangle row intervals are pairwise disjoint, and every rectangle lies at least three columns left of the marker column.

    Rule:
    - Read the non-background colors in the last column from top to bottom; this is the processing order.
    - Process rectangles in that order. Each rectangle slides horizontally to the right until its right edge is flush against the nearest current blocker: initially the free-space boundary just left of the marker column, then the left edge of any earlier-placed rectangle.
    - After each placement, that rectangle's left edge becomes the next blocker. When all rectangles are placed, erase the entire marker column to background.

    Key insight:
    - The marker column does not encode the final output directly; it schedules a cascade of placements, and early rectangles shrink the remaining horizontal space for later ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 5
    bg = _get_bg(g)
    rectangles = _extract_rectangles(g, bg)
    order = _marker_order(g, bg)
    assert set(order) == set(rectangles), "marker colors must match rectangle colors exactly"
    left_by_color = _cascade_layout(rectangles, order, g.shape[1])
    return _render(g.shape, rectangles, left_by_color, bg)


GEN_ATTEMPTS = 400
GRID_H_RANGE = (16, 23)
RECT_COUNT_CHOICES = (4, 5)
RECT_H_RANGE = (1, 3)
RECT_W_RANGE = (2, 3)
GAP_RANGE = (1, 3)
INPUT_SHIFT_MIN = 1


def _pick_rectangle_rows(heights: list[int], total_h: int) -> list[int] | None:
    need = sum(heights) + (len(heights) - 1)
    if need > total_h:
        return None
    extra = total_h - need
    gaps = [1] * (len(heights) - 1)
    top_margin = random.randint(0, extra)
    extra -= top_margin
    bottom_margin = random.randint(0, extra)
    extra -= bottom_margin
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    rows = []
    cur = top_margin
    for i, h in enumerate(heights):
        rows.append(cur)
        cur += h
        if i < len(gaps):
            cur += gaps[i]
    return rows


def _random_order(colors: list[int], rect_rows: dict[int, int], rect_widths: dict[int, int]) -> list[int]:
    row_order = [color for color, _ in sorted(rect_rows.items(), key=lambda item: item[1])]
    width_order = [color for color, _ in sorted(rect_widths.items(), key=lambda item: (item[1], item[0]))]
    for _ in range(200):
        order = colors[:]
        random.shuffle(order)
        if order != row_order and order != width_order and order != list(reversed(row_order)):
            return order
    return colors[:]


def _alternative_orders(
    colors: list[int], rect_rows: dict[int, int], rect_widths: dict[int, int], rect_cols: dict[int, int]
) -> list[list[int]]:
    row_order = [color for color, _ in sorted(rect_rows.items(), key=lambda item: item[1])]
    width_order = [color for color, _ in sorted(rect_widths.items(), key=lambda item: (item[1], item[0]))]
    left_order = [color for color, _ in sorted(rect_cols.items(), key=lambda item: item[1])]
    alts = [row_order, list(reversed(row_order)), width_order, left_order, list(reversed(colors))]
    out: list[list[int]] = []
    for order in alts:
        if order and order not in out:
            out.append(order)
    return out


def _independent_slide(rectangles: dict[int, dict[str, int]], total_width: int) -> dict[int, int]:
    return {color: total_width - 1 - rect["w"] for color, rect in rectangles.items()}


def _build_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_CHOICES)
    heights = [random.randint(*RECT_H_RANGE) for _ in range(rect_count)]
    min_h = max(GRID_H_RANGE[0], sum(heights) + rect_count)
    if min_h > GRID_H_RANGE[1]:
        return None
    total_h = random.randint(min_h, GRID_H_RANGE[1])
    top_rows = _pick_rectangle_rows(heights, total_h)
    if top_rows is None:
        return None

    widths = [random.randint(*RECT_W_RANGE) for _ in range(rect_count)]
    if len(set(widths)) == 1 and rect_count >= 4:
        widths[random.randrange(rect_count)] = 5 - widths[0]

    colors = random.sample(range(1, 10), rect_count)
    random.shuffle(colors)
    rect_rows = {color: row for color, row in zip(colors, top_rows)}
    rect_widths = {color: width for color, width in zip(colors, widths)}

    width_sum = sum(widths)
    total_w = random.randint(max(18, width_sum + 6), min(28, width_sum + 11))
    if total_w > 30:
        return None

    rectangles: dict[int, dict[str, int]] = {}
    for color, height, width, row in zip(colors, heights, widths, top_rows):
        rectangles[color] = {"r0": row, "c0": 0, "h": height, "w": width}

    order = _random_order(colors, rect_rows, rect_widths)
    final_left = _cascade_layout(rectangles, order, total_w)
    if min(final_left.values()) < INPUT_SHIFT_MIN:
        return None

    input_left: dict[int, int] = {}
    moved = 0
    for color in colors:
        max_left = min(final_left[color] - INPUT_SHIFT_MIN, total_w - 4 - rectangles[color]["w"])
        if max_left < 0:
            return None
        left = random.randint(0, max_left)
        input_left[color] = left
        moved += int(left < final_left[color])
    if moved < rect_count:
        return None

    rect_cols = dict(input_left)
    for color in colors:
        rectangles[color]["c0"] = input_left[color]

    out = np.zeros((total_h, total_w), dtype=int)
    for color, rect in rectangles.items():
        r0 = rect["r0"]
        c0 = rect["c0"]
        out[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = color

    marker_rows = sorted(random.sample(range(total_h), rect_count))
    for marker_row, color in zip(marker_rows, order):
        out[marker_row, total_w - 1] = color

    if np.any(out[:, total_w - 4 : total_w - 1] != 0):
        return None

    oracle = main(out)
    expected = _render(out.shape, rectangles, final_left, 0)
    if not np.array_equal(oracle, expected):
        return None

    independent = _render(out.shape, rectangles, _independent_slide(rectangles, total_w), 0)
    if np.array_equal(independent, expected):
        return None

    for alt_order in _alternative_orders(colors, rect_rows, rect_widths, rect_cols):
        if alt_order == order or set(alt_order) != set(order):
            continue
        alt_left = _cascade_layout(rectangles, alt_order, total_w)
        alt_out = _render(out.shape, rectangles, alt_left, 0)
        if np.array_equal(alt_out, expected):
            return None

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")

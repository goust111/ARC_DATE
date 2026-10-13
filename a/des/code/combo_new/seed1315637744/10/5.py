from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 8
STRIP_BG = 5
LINE_COLOR_POOL = (0, 1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _order_from_strip(strip: np.ndarray) -> tuple[int, list[int]]:
    sep = _most_common(strip)
    order: list[int] = []
    prev_sep = True
    prev = None
    for value in map(int, strip):
        if value == sep:
            prev_sep = True
            continue
        if prev_sep or value != prev:
            order.append(value)
        prev_sep = False
        prev = value
    assert 2 <= len(order) <= 4
    assert len(set(order)) == len(order)
    return sep, order


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        count += 1
        stack = [(rr, cc)]
        seen[rr, cc] = True
        while stack:
            cr, cc = stack.pop()
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _is_single_polyline(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0 or _count_components(mask) != 1:
        return False

    endpoints = 0
    h, w = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        deg = 0
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                deg += 1
        if deg == 1:
            endpoints += 1
        elif deg != 2:
            return False
    return endpoints in (0, 2)


def _recover_line(canvas: np.ndarray, color: int, bg: int, line_colors: list[int]) -> np.ndarray:
    mask = canvas == color
    bridgeable = np.isin(canvas, line_colors) & (canvas != color)

    changed = True
    while changed:
        changed = False
        add = np.zeros_like(mask, dtype=bool)
        for r in range(1, canvas.shape[0] - 1):
            for c in range(1, canvas.shape[1] - 1):
                if mask[r, c] or not bridgeable[r, c]:
                    continue
                vertical = bool(mask[r - 1, c] and mask[r + 1, c])
                horizontal = bool(mask[r, c - 1] and mask[r, c + 1])
                if vertical ^ horizontal:
                    add[r, c] = True
        if np.any(add):
            mask |= add
            changed = True

    assert _is_single_polyline(mask), "each recovered color must form one simple polyline"
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the full colored polylines at hidden crossings, then redraw them using the bottom strip's layer order.

    Input structure:
    - The background is azure.
    - The main canvas contains 2-4 differently colored one-pixel-wide polylines made of horizontal and vertical segments.
    - Polylines may cross; at each crossing, exactly one line color is visible while the other line continues underneath.
    - The bottom row is a grey separator strip containing the polyline colors from left to right in bottom-to-top order.

    Rule:
    - Read the bottom strip to obtain the correct z-order of the polyline colors.
    - For each polyline color, recover hidden crossing cells where that color must continue straight through a differently colored visible crossing.
    - Clear the canvas to azure and redraw the recovered full polylines from the strip's bottom layer to its top layer.
    - Remove the bottom order strip from the output.

    Key insight:
    - The visible crossing colors in the input are not the trusted layering information; the true layer order comes from the bottom strip, so the solver must first reconstruct each complete path and only then rerender the crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3

    canvas = g[:-1].copy()
    strip = g[-1]
    _sep, order = _order_from_strip(strip)

    bg = _most_common(canvas.ravel())
    assert bg not in order

    masks = {color: _recover_line(canvas, color, bg, order) for color in order}
    out = np.full(canvas.shape, bg, dtype=int)
    for color in order:
        out[masks[color]] = color
    return out


def _sample_orientations(n_lines: int) -> tuple[int, int]:
    n_horizontal = random.randint(1, n_lines - 1)
    return n_horizontal, n_lines - n_horizontal


def _sample_types(n_horizontal: int, n_vertical: int) -> tuple[list[str], list[str]] | None:
    want_bent = random.random() < 0.5
    h_types = [random.choice(("straight", "bent")) for _ in range(n_horizontal)]
    v_types = [random.choice(("straight", "bent")) for _ in range(n_vertical)]
    bent_count = sum(t == "bent" for t in h_types + v_types)
    if want_bent and bent_count == 0:
        return None
    if not want_bent and bent_count == n_horizontal + n_vertical:
        return None

    rows_needed = sum(2 if t == "bent" else 1 for t in h_types) + sum(t == "bent" for t in v_types)
    cols_needed = sum(t == "bent" for t in h_types) + sum(2 if t == "bent" else 1 for t in v_types)
    if rows_needed > 7 or cols_needed > 7:
        return None
    return h_types, v_types


def _take(pool: list[int], count: int) -> list[int]:
    picks = random.sample(pool, count)
    for value in picks:
        pool.remove(value)
    return picks


def _polyline_from_points(points: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            c_start, c_end = sorted((c0, c1))
            mask[r0, c_start : c_end + 1] = True
        else:
            r_start, r_end = sorted((r0, r1))
            mask[r_start : r_end + 1, c0] = True
    assert _is_single_polyline(mask)
    return mask


def _build_lines() -> tuple[dict[int, np.ndarray], list[int], tuple[int, int]] | None:
    n_lines = random.randint(2, 4)
    n_horizontal, n_vertical = _sample_orientations(n_lines)

    typed = _sample_types(n_horizontal, n_vertical)
    if typed is None:
        return None
    h_types, v_types = typed

    rows_needed = sum(2 if t == "bent" else 1 for t in h_types) + sum(t == "bent" for t in v_types)
    cols_needed = sum(t == "bent" for t in h_types) + sum(2 if t == "bent" else 1 for t in v_types)
    n_row_tracks = random.randint(max(4, rows_needed), 7)
    n_col_tracks = random.randint(max(4, cols_needed), 7)

    canvas_h = 3 * n_row_tracks + 2
    canvas_w = 3 * n_col_tracks + 2
    rows = [2 + 3 * i for i in range(n_row_tracks)]
    cols = [2 + 3 * i for i in range(n_col_tracks)]
    free_rows = rows[:]
    free_cols = cols[:]

    colors = random.sample(LINE_COLOR_POOL, n_lines)
    random.shuffle(h_types)
    random.shuffle(v_types)

    masks: dict[int, np.ndarray] = {}
    color_iter = iter(colors)
    for line_type in h_types:
        color = next(color_iter)
        if line_type == "straight":
            row = _take(free_rows, 1)[0]
            points = [(row, 1), (row, canvas_w - 2)]
        else:
            row_a, row_b = _take(free_rows, 2)
            turn_col = _take(free_cols, 1)[0]
            points = [(row_a, 1), (row_a, turn_col), (row_b, turn_col), (row_b, canvas_w - 2)]
        masks[color] = _polyline_from_points(points, (canvas_h, canvas_w))

    for line_type in v_types:
        color = next(color_iter)
        if line_type == "straight":
            col = _take(free_cols, 1)[0]
            points = [(1, col), (canvas_h - 2, col)]
        else:
            col_a, col_b = _take(free_cols, 2)
            turn_row = _take(free_rows, 1)[0]
            points = [(1, col_a), (turn_row, col_a), (turn_row, col_b), (canvas_h - 2, col_b)]
        masks[color] = _polyline_from_points(points, (canvas_h, canvas_w))

    occupancy: dict[tuple[int, int], list[int]] = {}
    for color, mask in masks.items():
        for r, c in np.argwhere(mask):
            cell = (int(r), int(c))
            occupancy.setdefault(cell, []).append(color)

    intersections = [cell for cell, owners in occupancy.items() if len(owners) == 2]
    if not intersections:
        return None
    if any(len(owners) > 2 for owners in occupancy.values()):
        return None

    intersection_counts = {color: 0 for color in masks}
    for cell in intersections:
        for color in occupancy[cell]:
            intersection_counts[color] += 1
    if any(count == 0 for count in intersection_counts.values()):
        return None

    return masks, colors, (canvas_h, canvas_w)


def _render_output(masks: dict[int, np.ndarray], order: list[int], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        out[masks[color]] = color
    return out


def _composition(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _make_strip(width: int, order: list[int]) -> np.ndarray:
    strip = np.full(width, STRIP_BG, dtype=int)
    min_width = 2 * len(order) - 1
    assert width >= min_width
    extras = _composition(width - min_width, len(order) + 1)
    col = extras[0]
    for i, color in enumerate(order):
        strip[col] = color
        col += 1
        if i + 1 < len(order):
            col += 1 + extras[i + 1]
    return strip


def _make_input(masks: dict[int, np.ndarray], order: list[int], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    canvas_h, canvas_w = shape
    occupancy: dict[tuple[int, int], list[int]] = {}
    for color, mask in masks.items():
        for r, c in np.argwhere(mask):
            occupancy.setdefault((int(r), int(c)), []).append(color)

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    order_index = {color: i for i, color in enumerate(order)}
    intersections: list[tuple[int, int]] = []
    chosen: dict[tuple[int, int], int] = {}

    for cell, owners in occupancy.items():
        if len(owners) == 1:
            chosen[cell] = owners[0]
            continue
        intersections.append(cell)
        chosen[cell] = random.choice(owners)

    assert intersections
    mismatches = []
    for cell in intersections:
        owners = occupancy[cell]
        top = max(owners, key=lambda color: order_index[color])
        if chosen[cell] != top:
            mismatches.append(cell)
    if not mismatches:
        cell = random.choice(intersections)
        owners = occupancy[cell]
        top = max(owners, key=lambda color: order_index[color])
        chosen[cell] = owners[0] if owners[1] == top else owners[1]

    for cell, color in chosen.items():
        canvas[cell] = color

    strip = _make_strip(canvas_w, order)
    inp = np.vstack([canvas, strip])
    out = _render_output(masks, order, shape)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    built = _build_lines()
    if built is None:
        return None
    masks, colors, shape = built
    order = random.sample(colors, len(colors))
    inp, expected = _make_input(masks, order, shape)

    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(inp[:-1], expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid crossing-reorder puzzle")

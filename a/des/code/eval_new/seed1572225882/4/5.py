from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PAINT_COLORS = (1, 2, 3, 4, 6, 7, 8)
SPECIAL_COLORS = {BLACK, GREY, MAROON}
_RNG = np.random.default_rng()


def _paint_color(value):
    return int(value) not in SPECIAL_COLORS


def _find_towers(grid):
    h, w = grid.shape
    towers = []
    for r in range(1, h - 2):
        for c in range(w - 2):
            color = int(grid[r, c])
            if not _paint_color(color):
                continue
            block = grid[r : r + 3, c : c + 3]
            if np.all(block == color) and int(grid[r - 1, c + 1]) == MAROON:
                cells = tuple((rr, cc) for rr in range(r, r + 3) for cc in range(c, c + 3))
                towers.append({"color": color, "cells": cells, "box": (r, c)})
    return towers


def _find_receivers(grid):
    h, w = grid.shape
    receivers = []
    for r in range(h - 2):
        for c in range(w - 2):
            border = (
                (r, c),
                (r, c + 1),
                (r, c + 2),
                (r + 1, c),
                (r + 1, c + 2),
                (r + 2, c),
                (r + 2, c + 1),
                (r + 2, c + 2),
            )
            if all(int(grid[rr, cc]) == GREY for rr, cc in border) and int(grid[r + 1, c + 1]) == BLACK:
                receivers.append({"border": border, "center": (r + 1, c + 1), "box": (r, c)})
    return receivers


def _find_filters(grid):
    h, w = grid.shape
    filters = []
    used = set()

    def add_filter(cells, in_color, out_color, orientation):
        if any(cell in used for cell in cells):
            return
        filters.append(
            {
                "cells": tuple(cells),
                "in": int(in_color),
                "out": int(out_color),
                "orientation": orientation,
            }
        )
        used.update(cells)

    for r in range(h):
        for c in range(w - 2):
            vals = [int(grid[r, c + i]) for i in range(3)]
            if vals[0] == vals[2] and vals[1] != vals[0] and all(_paint_color(v) for v in vals):
                add_filter(((r, c), (r, c + 1), (r, c + 2)), vals[0], vals[1], "h")

    for r in range(h - 2):
        for c in range(w):
            vals = [int(grid[r + i, c]) for i in range(3)]
            if vals[0] == vals[2] and vals[1] != vals[0] and all(_paint_color(v) for v in vals):
                add_filter(((r, c), (r + 1, c), (r + 2, c)), vals[0], vals[1], "v")

    return filters


def _trace_path(grid, tower, receivers, filters):
    h, w = grid.shape
    passable = set(zip(*np.where(grid == GREY)))
    filter_cell_to_id = {}
    for idx, flt in enumerate(filters):
        for cell in flt["cells"]:
            passable.add(cell)
            filter_cell_to_id[cell] = idx

    starts = set(tower["cells"])
    passable.update(starts)

    receiver_by_border = {}
    for idx, receiver in enumerate(receivers):
        for cell in receiver["border"]:
            receiver_by_border[cell] = idx

    prev = {}
    queue = deque()
    for cell in starts:
        prev[cell] = None
        queue.append(cell)

    found = None
    while queue:
        cell = queue.popleft()
        if cell in receiver_by_border:
            found = cell
            break
        r, c = cell
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and nxt in passable and nxt not in prev:
                prev[nxt] = cell
                queue.append(nxt)

    if found is None:
        return None, []

    path = []
    cur = found
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    order = []
    previous_filter = None
    for cell in path:
        flt_id = filter_cell_to_id.get(cell)
        if flt_id is None:
            previous_filter = None
            continue
        if flt_id != previous_filter and (not order or order[-1] != flt_id):
            order.append(flt_id)
        previous_filter = flt_id

    return receiver_by_border[found], order


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate each tower signal through grey wires and filters, then fill its receiver.

    Input structure:
    The background is black. Signal towers are 3x3 solid colored squares with a single
    maroon marker directly above their top-center cell. Receivers are 3x3 grey hollow
    frames with a black center. Filters are 1x3 or 3x1 colored bars whose two end cells
    share an input color and whose middle cell gives an output color. Thin grey
    horizontal or vertical wires connect towers, filters, and receivers into unambiguous
    paths.

    Rule:
    For each marked tower, follow its connected grey-and-filter path to a receiver.
    The signal starts as the tower color. Whenever the path crosses a filter, change
    the signal to the filter middle color only if the current signal equals the
    filter end color; otherwise keep the signal unchanged. Fill the black center of
    the reached receiver with the final signal color, leaving all wires, towers,
    filters, markers, and receiver frames unchanged.

    Key insight:
    The grey wires define topology, each colored bar encodes a conditional color
    rewrite, and the receiver color is obtained by applying those rewrites in path
    order rather than by comparing only the endpoint objects.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    towers = _find_towers(grid)
    receivers = _find_receivers(grid)
    filters = _find_filters(grid)

    for tower in towers:
        receiver_id, filter_order = _trace_path(grid, tower, receivers, filters)
        if receiver_id is None:
            continue
        signal = tower["color"]
        for flt_id in filter_order:
            flt = filters[flt_id]
            if signal == flt["in"]:
                signal = flt["out"]
        rr, cc = receivers[receiver_id]["center"]
        out[rr, cc] = signal

    return out


def _choice(rng, values):
    return values[int(rng.integers(0, len(values)))]


def _different_color(rng, excluded):
    blocked = set(int(v) for v in excluded)
    choices = [c for c in PAINT_COLORS if c not in blocked]
    if not choices:
        choices = list(PAINT_COLORS)
    return _choice(rng, choices)


def _spaced_starts(rng, lo, hi, count):
    if count <= 0 or lo > hi:
        return []
    values = list(range(lo, hi + 1))
    for _ in range(200):
        starts = sorted(int(v) for v in rng.choice(values, size=count, replace=False))
        if all(starts[i + 1] - starts[i] >= 5 for i in range(len(starts) - 1)):
            return starts
    rng.shuffle(values)
    chosen = []
    for value in values:
        if all(abs(value - other) >= 5 for other in chosen):
            chosen.append(value)
            if len(chosen) == count:
                return sorted(chosen)
    return sorted(chosen)


def _draw_tower(grid, top, left, color):
    grid[top : top + 3, left : left + 3] = color
    grid[top - 1, left + 1] = MAROON


def _draw_receiver(grid, top, left):
    grid[top : top + 3, left : left + 3] = GREY
    grid[top + 1, left + 1] = BLACK


def _place_horizontal_filters(grid, y, starts, path_order, tower_color, rng):
    current = int(tower_color)
    start_set = set(starts)
    for start in path_order:
        if int(rng.integers(0, 2)) == 0:
            in_color = current
        else:
            in_color = _different_color(rng, {current})
        out_color = _different_color(rng, {in_color})
        grid[y, start] = in_color
        grid[y, start + 1] = out_color
        grid[y, start + 2] = in_color
        if current == in_color:
            current = out_color
    return start_set


def _place_vertical_filters(grid, x, starts, tower_color, rng):
    current = int(tower_color)
    for start in starts:
        if int(rng.integers(0, 2)) == 0:
            in_color = current
        else:
            in_color = _different_color(rng, {current})
        out_color = _different_color(rng, {in_color})
        grid[start, x] = in_color
        grid[start + 1, x] = out_color
        grid[start + 2, x] = in_color
        if current == in_color:
            current = out_color


def _generate_horizontal(rng):
    path_count = int(rng.integers(1, 4))
    gap = int(rng.integers(6, 9))
    first_y = int(rng.integers(3, 5))
    centers = [first_y + i * gap for i in range(path_count)]
    height = int(rng.integers(centers[-1] + 3, 31))
    width = int(rng.integers(20, 31))
    grid = np.zeros((height, width), dtype=int)
    tower_colors = [int(v) for v in rng.choice(PAINT_COLORS, size=path_count, replace=False)]

    for y, tower_color in zip(centers, tower_colors):
        direction = 1 if int(rng.integers(0, 2)) == 0 else -1
        if direction == 1:
            tower_left = int(rng.integers(1, 5))
            receiver_left = int(rng.integers(tower_left + 12, width - 2))
            line_start = tower_left + 3
            line_end = receiver_left - 1
            path_order_reverse = False
        else:
            receiver_left = int(rng.integers(1, 5))
            tower_left = int(rng.integers(receiver_left + 12, width - 2))
            line_start = receiver_left + 3
            line_end = tower_left - 1
            path_order_reverse = True

        _draw_tower(grid, y - 1, tower_left, tower_color)
        _draw_receiver(grid, y - 1, receiver_left)
        grid[y, line_start : line_end + 1] = GREY

        line_len = line_end - line_start + 1
        max_filters = min(3, max(0, (line_len - 2) // 5))
        filter_count = int(rng.integers(0, max_filters + 1))
        starts = _spaced_starts(rng, line_start + 1, line_end - 3, filter_count)
        path_order = list(reversed(starts)) if path_order_reverse else starts
        _place_horizontal_filters(grid, y, starts, path_order, tower_color, rng)

    return grid


def _generate_vertical(rng):
    path_count = int(rng.integers(1, 4))
    gap = int(rng.integers(6, 9))
    first_x = int(rng.integers(3, 5))
    centers = [first_x + i * gap for i in range(path_count)]
    width = int(rng.integers(centers[-1] + 3, 31))
    height = int(rng.integers(20, 31))
    grid = np.zeros((height, width), dtype=int)
    tower_colors = [int(v) for v in rng.choice(PAINT_COLORS, size=path_count, replace=False)]

    for x, tower_color in zip(centers, tower_colors):
        tower_top = int(rng.integers(2, 5))
        receiver_top = int(rng.integers(tower_top + 12, height - 2))
        line_start = tower_top + 3
        line_end = receiver_top - 1

        _draw_tower(grid, tower_top, x - 1, tower_color)
        _draw_receiver(grid, receiver_top, x - 1)
        grid[line_start : line_end + 1, x] = GREY

        line_len = line_end - line_start + 1
        max_filters = min(3, max(0, (line_len - 2) // 5))
        filter_count = int(rng.integers(0, max_filters + 1))
        starts = _spaced_starts(rng, line_start + 1, line_end - 3, filter_count)
        _place_vertical_filters(grid, x, starts, tower_color, rng)

    return grid


def generate_input() -> np.ndarray:
    if int(_RNG.integers(0, 2)) == 0:
        return _generate_horizontal(_RNG)
    return _generate_vertical(_RNG)

from __future__ import annotations

import random

import numpy as np


BG = 8
GREY = 5
AXIS_COLOR_POOL = [0, 1, 2, 3, 4, 6, 7, 9]
MIN_STRIP = 2
MIN_GRID = 16
MAX_GRID = 30
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _col_lines(g: np.ndarray, bg: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for c in range(g.shape[1]):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            assert color not in out, "each vertical axis color must be unique"
            out[color] = int(c)
    return out


def _row_lines(g: np.ndarray, bg: int, blocked_cols: set[int]) -> dict[int, int]:
    out: dict[int, int] = {}
    keep_cols = [c for c in range(g.shape[1]) if c not in blocked_cols]
    assert keep_cols, "horizontal axes need non-intersection cells"
    for r in range(g.shape[0]):
        row = g[r, keep_cols]
        color = int(row[0])
        if color != bg and np.all(row == color):
            assert color not in out, "each horizontal axis color must be unique"
            out[color] = int(r)
    return out


def _extract_order(g: np.ndarray, axis_colors: set[int], first_row: int, first_col: int) -> list[int]:
    corner = g[:first_row, :first_col]
    order_items = []
    for color in axis_colors:
        coords = np.argwhere(corner == int(color))
        assert coords.size > 0, "every axis color must appear in the corner code"
        r0, c0 = coords.min(axis=0)
        order_items.append((int(r0), int(c0), int(color)))
    order_items.sort()
    return [color for _r, _c, color in order_items]


def _apply_fold(
    state: np.ndarray,
    bg: int,
    orientation: str,
    pos: int,
    bounds: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bounds
    out = state.copy()

    if orientation == "h":
        top = int(pos - r0)
        bottom = int(r1 - pos)
        assert top > 0 and bottom > 0 and top != bottom
        if top < bottom:
            src = state[r0:pos, c0 : c1 + 1]
            flipped = src[::-1, :]
            dst = out[pos + 1 : pos + 1 + top, c0 : c1 + 1]
            mask = flipped != bg
            dst[mask] = flipped[mask]
            out[r0:pos, c0 : c1 + 1] = bg
            return out, (int(pos), c0, r1, c1)

        src = state[pos + 1 : r1 + 1, c0 : c1 + 1]
        flipped = src[::-1, :]
        dst = out[pos - bottom : pos, c0 : c1 + 1]
        mask = flipped != bg
        dst[mask] = flipped[mask]
        out[pos + 1 : r1 + 1, c0 : c1 + 1] = bg
        return out, (r0, c0, int(pos), c1)

    assert orientation == "v"
    left = int(pos - c0)
    right = int(c1 - pos)
    assert left > 0 and right > 0 and left != right
    if left < right:
        src = state[r0 : r1 + 1, c0:pos]
        flipped = src[:, ::-1]
        dst = out[r0 : r1 + 1, pos + 1 : pos + 1 + left]
        mask = flipped != bg
        dst[mask] = flipped[mask]
        out[r0 : r1 + 1, c0:pos] = bg
        return out, (r0, int(pos), r1, c1)

    src = state[r0 : r1 + 1, pos + 1 : c1 + 1]
    flipped = src[:, ::-1]
    dst = out[r0 : r1 + 1, pos - right : pos]
    mask = flipped != bg
    dst[mask] = flipped[mask]
    out[r0 : r1 + 1, pos + 1 : c1 + 1] = bg
    return out, (r0, c0, r1, int(pos))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the smaller side of each full-row/full-column colored axis in the order
    encoded by the top-left corner column, then crop the remaining pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)

    v_axes = _col_lines(g, bg)
    h_axes = _row_lines(g, bg, set(v_axes.values()))
    assert h_axes and v_axes, "generator uses both horizontal and vertical axes"
    axis_colors = set(h_axes) | set(v_axes)
    assert len(axis_colors) == len(h_axes) + len(v_axes), "axis colors must be unique"

    first_row = min(h_axes.values())
    first_col = min(v_axes.values())
    order = _extract_order(g, axis_colors, first_row, first_col)
    assert set(order) == axis_colors and len(order) == len(axis_colors)

    content = g.copy()
    axis_mask = np.isin(content, np.array(sorted(axis_colors), dtype=int))
    content[(content == bg) | axis_mask] = bg

    bounds = (0, 0, g.shape[0] - 1, g.shape[1] - 1)
    axis_info = {color: ("h", pos) for color, pos in h_axes.items()}
    axis_info.update({color: ("v", pos) for color, pos in v_axes.items()})

    for color in order:
        orientation, pos = axis_info[int(color)]
        r0, c0, r1, c1 = bounds
        if orientation == "h":
            assert r0 < pos < r1
        else:
            assert c0 < pos < c1
        content, bounds = _apply_fold(content, bg, orientation, int(pos), bounds)

    box = _bbox(content != bg)
    assert box is not None, "final pattern must be non-empty"
    r0, c0, r1, c1 = box
    return content[r0 : r1 + 1, c0 : c1 + 1]


def _sample_desc_positions(curr_bound: int, count: int, inner_min: int) -> list[int] | None:
    if count == 0:
        return []
    if count == 1:
        lo = max(inner_min, curr_bound // 2 + 1)
        hi = curr_bound - MIN_STRIP
        if lo > hi:
            return None
        return [int(random.randint(lo, hi))]

    lo = max(inner_min + MIN_STRIP * (count - 1), curr_bound // 2 + 1)
    hi = curr_bound - MIN_STRIP
    if lo > hi:
        return None

    candidates = list(range(lo, hi + 1))
    random.shuffle(candidates)
    for pos in candidates:
        tail = _sample_desc_positions(int(pos), count - 1, inner_min)
        if tail is not None:
            return [int(pos)] + tail
    return None


def _interleave_axes(h_rows: list[int], v_cols: list[int], colors: list[int]) -> list[tuple[int, str, int]]:
    h_queue = list(zip(colors[: len(h_rows)], ["h"] * len(h_rows), h_rows))
    v_queue = list(zip(colors[len(h_rows) :], ["v"] * len(v_cols), v_cols))
    order: list[tuple[int, str, int]] = []

    while h_queue or v_queue:
        if not h_queue:
            order.append(v_queue.pop(0))
        elif not v_queue:
            order.append(h_queue.pop(0))
        elif random.choice([False, True]):
            order.append(h_queue.pop(0))
        else:
            order.append(v_queue.pop(0))
    return [(int(color), orient, int(pos)) for color, orient, pos in order]


def _bands(size: int, cuts: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [int(cut) + 1 for cut in cuts]
    ends = [int(cut) - 1 for cut in cuts] + [size - 1]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if s <= e]


def _neighbors4(r: int, c: int, r0: int, c0: int, r1: int, c1: int):
    if r > r0:
        yield r - 1, c
    if r < r1:
        yield r + 1, c
    if c > c0:
        yield r, c - 1
    if c < c1:
        yield r, c + 1


def _grow_shape(
    allowed: np.ndarray,
    region: tuple[int, int, int, int],
    target_area: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = region
    cells = np.argwhere(allowed)
    if cells.size == 0:
        return None

    start = tuple(int(v) for v in cells[random.randrange(len(cells))])
    shape = {start}
    frontier = set()
    for nr, nc in _neighbors4(start[0], start[1], r0, c0, r1, c1):
        if allowed[nr, nc]:
            frontier.add((int(nr), int(nc)))

    while len(shape) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if not allowed[cell]:
            continue
        shape.add(cell)
        for nr, nc in _neighbors4(cell[0], cell[1], r0, c0, r1, c1):
            if allowed[nr, nc] and (nr, nc) not in shape:
                frontier.add((int(nr), int(nc)))

    if not shape:
        return None
    return sorted(shape)


def _place_region_shape(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    reserved: np.ndarray,
    force: bool,
) -> bool:
    r0, c0, r1, c1 = region
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = h * w
    if area <= 0:
        return False
    if not force and random.random() < 0.35:
        return False

    allowed = np.zeros_like(grid, dtype=bool)
    allowed[r0 : r1 + 1, c0 : c1 + 1] = True
    allowed &= grid == BG
    allowed &= ~reserved

    count = int(allowed[r0 : r1 + 1, c0 : c1 + 1].sum())
    if count == 0:
        return False

    upper = min(7, count, max(1, area // 3))
    lower = 2 if upper >= 2 and force else 1
    if lower > upper:
        lower = upper

    for _ in range(20):
        target_area = random.randint(lower, upper)
        shape = _grow_shape(allowed, region, target_area)
        if shape is None:
            continue
        for r, c in shape:
            grid[r, c] = GREY
        return True
    return False


def _build_grid() -> np.ndarray | None:
    total_axes = random.randint(2, 4)
    h_count = random.randint(1, total_axes - 1)
    v_count = total_axes - h_count

    code_height = 2 * total_axes - 1
    inner_row_min = code_height + 2
    inner_col_min = 4

    H = random.randint(MIN_GRID, MAX_GRID)
    W = random.randint(MIN_GRID, MAX_GRID)
    h_rows = _sample_desc_positions(H - 1, h_count, inner_row_min)
    v_cols = _sample_desc_positions(W - 1, v_count, inner_col_min)
    if h_rows is None or v_cols is None:
        return None

    colors = random.sample(AXIS_COLOR_POOL, total_axes)
    order = _interleave_axes(h_rows, v_cols, colors)
    grid = np.full((H, W), BG, dtype=int)

    row_positions = sorted(h_rows)
    col_positions = sorted(v_cols)
    row_bands = _bands(H, row_positions)
    col_bands = _bands(W, col_positions)
    first_row = row_positions[0]
    first_col = col_positions[0]

    reserved = np.zeros((H, W), dtype=bool)
    for r, c in [(2 * i, 0) for i in range(total_axes)]:
        reserved[r, c] = True
    if first_col >= 2:
        reserved[:first_row, 0:2] = True

    for color, orient, pos in order:
        if orient == "h":
            grid[pos, :] = int(color)
    for color, orient, pos in order:
        if orient == "v":
            grid[:, pos] = int(color)

    for i, (color, _orient, _pos) in enumerate(order):
        grid[2 * i, 0] = int(color)

    placed_any = False
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            region = (int(r0), int(c0), int(r1), int(c1))
            force = ri == 0 and ci == 0
            if force and c1 - c0 + 1 <= 2:
                return None
            if _place_region_shape(grid, region, reserved, force):
                placed_any = True

    if not placed_any:
        return None

    if _mode_color(grid) != BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape[0] < 2 or out.shape[1] < 2:
        return None
    if int(np.count_nonzero(out != BG)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fold-axis puzzle input")

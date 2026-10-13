import random
from typing import Optional

import numpy as np


def _legend_run(arr: np.ndarray, bg: int) -> Optional[list[int]]:
    idx = np.flatnonzero(arr != bg)
    if len(idx) < 2:
        return None
    start = int(idx[0])
    end = int(idx[-1])
    segment = arr[start : end + 1]
    if np.any(segment == bg):
        return None
    colors = [int(v) for v in segment]
    if len(set(colors)) != len(colors):
        return None
    return colors


def _extract_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[str, np.ndarray, list[int]]] = []

    bottom_colors = _legend_run(g[-1, :], bg)
    if bottom_colors is not None:
        body = g[:-1, :]
        if body.size and set(bottom_colors) == set(map(int, np.unique(body[body != bg]))):
            candidates.append(("bottom", body, bottom_colors))

    right_colors = _legend_run(g[:, -1], bg)
    if right_colors is not None:
        body = g[:, :-1]
        if body.size and set(right_colors) == set(map(int, np.unique(body[body != bg]))):
            candidates.append(("right", body, right_colors))

    assert len(candidates) == 1, "legend strip must be uniquely identifiable"
    _kind, body, colors = candidates[0]
    return body, colors


def _recover_full_path(canvas: np.ndarray, color: int, bg: int) -> np.ndarray:
    visible = canvas == color
    recovered = visible.copy()
    foreign = (canvas != bg) & (canvas != color)

    while True:
        vertical = np.zeros_like(recovered)
        horizontal = np.zeros_like(recovered)
        vertical[1:-1, :] = recovered[:-2, :] & recovered[2:, :]
        horizontal[:, 1:-1] = recovered[:, :-2] & recovered[:, 2:]
        add = foreign & (vertical | horizontal) & ~recovered
        if not np.any(add):
            return recovered
        recovered |= add


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct each colored path through crossings, then redraw the scene using
    the layer order encoded by the bottom-row or right-column legend strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = 0
    assert np.any(g == bg), "expected black background"

    canvas, order = _extract_legend(g, bg)
    present = set(map(int, np.unique(canvas[canvas != bg])))
    assert set(order) == present

    recovered_masks = {color: _recover_full_path(canvas, color, bg) for color in order}

    out = np.full_like(canvas, bg)
    for color in order:
        out[recovered_masks[color]] = color
    return out


# --------------- generator ---------------

BACKGROUND = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

CANVAS_H_RANGE = (10, 15)
CANVAS_W_RANGE = (10, 15)
LINE_GAP = 2
H_COUNT_RANGE = (1, 3)
V_COUNT_RANGE = (1, 3)
GEN_RETRIES = 300


def _sample_spaced(low: int, high: int, count: int, gap: int) -> Optional[list[int]]:
    values = list(range(low, high + 1))
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(200):
        sample = sorted(random.sample(values, count))
        if all(b - a >= gap for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _render(shape: tuple[int, int], masks: dict[int, np.ndarray], order: list[int]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for color in order:
        out[masks[color]] = color
    return out


def _embed_legend(canvas: np.ndarray, order: list[int], use_bottom: bool) -> np.ndarray:
    h, w = canvas.shape
    if use_bottom:
        grid = np.full((h + 1, w), BACKGROUND, dtype=int)
        grid[:-1, :] = canvas
        start = random.randint(0, w - len(order))
        grid[-1, start : start + len(order)] = order
        return grid

    grid = np.full((h, w + 1), BACKGROUND, dtype=int)
    grid[:, :-1] = canvas
    start = random.randint(0, h - len(order))
    grid[start : start + len(order), -1] = np.array(order, dtype=int)
    return grid


def _gen_attempt() -> Optional[np.ndarray]:
    h = random.randint(*CANVAS_H_RANGE)
    w = random.randint(*CANVAS_W_RANGE)

    row_low, row_high = 2, h - 3
    col_low, col_high = 2, w - 3
    if row_low > row_high or col_low > col_high:
        return None

    max_h = min(H_COUNT_RANGE[1], len(range(row_low, row_high + 1, LINE_GAP)))
    max_v = min(V_COUNT_RANGE[1], len(range(col_low, col_high + 1, LINE_GAP)))
    if max_h < 1 or max_v < 1:
        return None

    h_count = random.randint(H_COUNT_RANGE[0], max_h)
    v_count = random.randint(V_COUNT_RANGE[0], max_v)

    rows = _sample_spaced(row_low, row_high, h_count, LINE_GAP)
    cols = _sample_spaced(col_low, col_high, v_count, LINE_GAP)
    if rows is None or cols is None:
        return None

    left_choices = list(range(1, min(cols)))
    right_choices = list(range(max(cols) + 1, w - 1))
    top_choices = list(range(1, min(rows)))
    bottom_choices = list(range(max(rows) + 1, h - 1))
    if not left_choices or not right_choices or not top_choices or not bottom_choices:
        return None

    total = h_count + v_count
    colors = random.sample(COLOR_POOL, total)
    horiz_colors = colors[:h_count]
    vert_colors = colors[h_count:]

    masks: dict[int, np.ndarray] = {}
    for row, color in zip(rows, horiz_colors):
        c0 = random.choice(left_choices)
        c1 = random.choice(right_choices)
        mask = np.zeros((h, w), dtype=bool)
        mask[row, c0 : c1 + 1] = True
        masks[color] = mask

    for col, color in zip(cols, vert_colors):
        r0 = random.choice(top_choices)
        r1 = random.choice(bottom_choices)
        mask = np.zeros((h, w), dtype=bool)
        mask[r0 : r1 + 1, col] = True
        masks[color] = mask

    legend_order = random.sample(colors, len(colors))
    output_canvas = _render((h, w), masks, legend_order)

    input_order = legend_order[:]
    for _ in range(80):
        trial = random.sample(colors, len(colors))
        if trial == legend_order:
            continue
        input_canvas = _render((h, w), masks, trial)
        if not np.array_equal(input_canvas, output_canvas):
            input_order = trial
            break
    else:
        return None

    use_bottom = random.random() < 0.5
    if use_bottom and w < len(legend_order):
        return None
    if (not use_bottom) and h < len(legend_order):
        return None

    grid = _embed_legend(input_canvas, legend_order, use_bottom)
    if not np.array_equal(main(grid), output_canvas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

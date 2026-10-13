from __future__ import annotations

import random

import numpy as np


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _paint_rectangles(
    shape: tuple[int, int],
    bg: int,
    color_to_box: dict[int, tuple[int, int, int, int]],
    top_to_bottom: list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in reversed(top_to_bottom):
        r0, c0, r1, c1 = color_to_box[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint overlapping rectangles in the main area using the right-side legend as the new depth order.

    Input structure:
    - The rightmost column is a 1-pixel-wide legend whose non-background pixels list rectangle colors from top to bottom.
    - The legend is separated from the main scene by one full background column.
    - The main scene contains one visible object per legend color; each object is the visible part of an axis-aligned rectangle under the current occlusion order.

    Rule:
    - Drop the separator column and the legend column from the output.
    - For each legend color, recover that rectangle's full extent as the bounding box of its visible pixels in the main scene.
    - Repaint those recovered rectangles from bottom to top according to the legend order, where higher legend entries are painted later and overwrite lower ones.

    Key insight:
    - The input already shows each rectangle's color and enough exposed boundary to recover its full bounding box; the task is to ignore the old occlusion order and rerender the same rectangles with the legend-defined stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert w >= 3

    sep_col = g[:, -2]
    vals, counts = np.unique(sep_col, return_counts=True)
    bg = int(vals[counts.argmax()])

    main_area = g[:, :-2]
    legend = [int(g[r, -1]) for r in range(h) if int(g[r, -1]) != bg]
    assert legend
    assert len(legend) == len(set(legend))

    main_colors = set(map(int, np.unique(main_area[main_area != bg])))
    assert main_colors == set(legend)

    color_to_box: dict[int, tuple[int, int, int, int]] = {}
    for color in legend:
        coords = np.argwhere(main_area == int(color))
        assert coords.size > 0
        color_to_box[int(color)] = _bbox_from_coords(coords)

    return _paint_rectangles(main_area.shape, bg, color_to_box, legend)


# --------------- generator ---------------

COLOR_POOL = tuple(range(10))
RECT_COUNT_RANGE = (3, 5)
GRID_HEIGHT_MAX = 26
MAIN_WIDTH_MAX = 26
INNER_MIN_SIDE = 2
GENERATION_RETRIES = 400


def _sample_nonadjacent_rows(h: int, n: int) -> list[int] | None:
    if h < 2 * n - 1:
        return None
    base = sorted(random.sample(range(h - n + 1), n))
    return [value + i for i, value in enumerate(base)]


def _sample_nested_boxes(h: int, w: int, n: int) -> list[tuple[int, int, int, int]] | None:
    min_outer_h = INNER_MIN_SIDE + 2 * (n - 1)
    min_outer_w = INNER_MIN_SIDE + 2 * (n - 1)
    if h < min_outer_h or w < min_outer_w:
        return None

    outer_h = random.randint(min_outer_h, h)
    outer_w = random.randint(min_outer_w, w)
    outer_r0 = random.randint(0, h - outer_h)
    outer_c0 = random.randint(0, w - outer_w)
    boxes = [(outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)]

    for depth in range(1, n):
        prev_r0, prev_c0, prev_r1, prev_c1 = boxes[-1]
        remaining_after = n - depth - 1
        min_h = INNER_MIN_SIDE + 2 * remaining_after
        min_w = INNER_MIN_SIDE + 2 * remaining_after

        top_lo = prev_r0 + 1
        top_hi = prev_r1 - min_h + 1
        left_lo = prev_c0 + 1
        left_hi = prev_c1 - min_w + 1
        if top_lo > top_hi or left_lo > left_hi:
            return None

        new_r0 = random.randint(top_lo, top_hi)
        new_c0 = random.randint(left_lo, left_hi)

        bottom_lo = new_r0 + min_h - 1
        bottom_hi = prev_r1 - 1
        right_lo = new_c0 + min_w - 1
        right_hi = prev_c1 - 1
        if bottom_lo > bottom_hi or right_lo > right_hi:
            return None

        new_r1 = random.randint(bottom_lo, bottom_hi)
        new_c1 = random.randint(right_lo, right_hi)
        boxes.append((new_r0, new_c0, new_r1, new_c1))

    return boxes


def _visible_boxes_match(rendered: np.ndarray, color_to_box: dict[int, tuple[int, int, int, int]]) -> bool:
    for color, box in color_to_box.items():
        coords = np.argwhere(rendered == int(color))
        if coords.size == 0 or _bbox_from_coords(coords) != box:
            return False
    return True


def _nontrivial_output(inp: np.ndarray, out: np.ndarray, bg: int) -> bool:
    if np.array_equal(inp, out):
        return False
    visible_colors = set(map(int, np.unique(out[out != bg])))
    return len(visible_colors) >= 2


def _choose_new_order(
    colors_bottom_to_top: list[int],
    color_to_box: dict[int, tuple[int, int, int, int]],
    shape: tuple[int, int],
    bg: int,
    current_main: np.ndarray,
) -> tuple[list[int], np.ndarray] | None:
    current_top_to_bottom = list(reversed(colors_bottom_to_top))
    for _ in range(80):
        if random.random() < 0.5:
            candidate = current_top_to_bottom.copy()
            i, j = random.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            candidate = random.sample(colors_bottom_to_top, len(colors_bottom_to_top))
        if candidate == current_top_to_bottom:
            continue

        repainted = _paint_rectangles(shape, bg, color_to_box, candidate)
        if _nontrivial_output(current_main, repainted, bg):
            return candidate, repainted
    return None


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    min_h = max(2 * n_rects - 1, INNER_MIN_SIDE + 2 * (n_rects - 1))
    h = random.randint(max(10, min_h), GRID_HEIGHT_MAX)
    w_main = random.randint(max(10, INNER_MIN_SIDE + 2 * (n_rects - 1)), MAIN_WIDTH_MAX)
    if w_main + 2 > 30:
        return None

    bg = random.choice(COLOR_POOL)
    colors = random.sample([c for c in COLOR_POOL if c != bg], n_rects)

    boxes = _sample_nested_boxes(h, w_main, n_rects)
    if boxes is None:
        return None

    color_to_box = {int(color): box for color, box in zip(colors, boxes)}
    current_main = _paint_rectangles((h, w_main), bg, color_to_box, list(reversed(colors)))
    if not _visible_boxes_match(current_main, color_to_box):
        return None

    new_order_and_output = _choose_new_order(colors, color_to_box, (h, w_main), bg, current_main)
    if new_order_and_output is None:
        return None
    legend_order, expected_output = new_order_and_output

    legend_rows = _sample_nonadjacent_rows(h, n_rects)
    if legend_rows is None:
        return None

    grid = np.full((h, w_main + 2), bg, dtype=int)
    grid[:, :w_main] = current_main
    grid[legend_rows, w_main + 1] = np.asarray(legend_order, dtype=int)

    if not np.array_equal(main(grid), expected_output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")

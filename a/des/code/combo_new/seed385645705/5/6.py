from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    assert len(rows) > 0, "expected a non-empty mask"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for col in range(grid.shape[1]):
        vals = np.unique(grid[:, col])
        if len(vals) == 1 and int(vals[0]) != bg:
            candidates.append((col, int(vals[0])))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return candidates[0]


def _indicator_order(left: np.ndarray, bg: int) -> list[int]:
    blocks = []
    for color in sorted(set(map(int, left.flatten())) - {bg}):
        mask = left == color
        r0, c0, r1, c1 = _bbox_from_mask(mask)
        assert np.all(left[r0 : r1 + 1, c0 : c1 + 1] == color), "indicator blocks must be solid rectangles"
        blocks.append((r0, color))
    assert blocks, "expected at least one indicator block"
    blocks.sort()
    return [color for _row, color in blocks]


def _render_canvas(
    shape: tuple[int, int], rects: dict[int, tuple[int, int, int, int]], draw_order: list[int], bg: int
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in draw_order:
        r0, c0, r1, c1 = rects[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _candidate_rectangles(visible: np.ndarray, higher_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    assert visible.shape == higher_mask.shape
    if not visible.any():
        return []

    base = _bbox_from_mask(visible)
    r0, c0, r1, c1 = base
    interior_visible = visible[r0 : r1 + 1, c0 : c1 + 1]
    interior_higher = higher_mask[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(~interior_visible & ~interior_higher):
        return []

    h, w = visible.shape
    found: set[tuple[int, int, int, int]] = set()

    @lru_cache(None)
    def expand(a: int, b: int, c: int, d: int) -> None:
        found.add((a, b, c, d))
        if a > 0 and np.all(higher_mask[a - 1, b : d + 1]):
            expand(a - 1, b, c, d)
        if c + 1 < h and np.all(higher_mask[c + 1, b : d + 1]):
            expand(a, b, c + 1, d)
        if b > 0 and np.all(higher_mask[a : c + 1, b - 1]):
            expand(a, b - 1, c, d)
        if d + 1 < w and np.all(higher_mask[a : c + 1, d + 1]):
            expand(a, b, c, d + 1)

    expand(*base)
    return sorted(found, key=lambda box: ((box[2] - box[0] + 1) * (box[3] - box[1] + 1), box))


def _recover_rectangles(
    canvas: np.ndarray, top_to_bottom: list[int], bg: int
) -> dict[int, tuple[int, int, int, int]]:
    visible_masks = {int(color): canvas == int(color) for color in top_to_bottom}

    def dfs(
        idx: int, higher_mask: np.ndarray, rects: dict[int, tuple[int, int, int, int]]
    ) -> dict[int, tuple[int, int, int, int]] | None:
        if idx == len(top_to_bottom):
            expected = _render_canvas(canvas.shape, rects, list(reversed(top_to_bottom)), bg)
            return rects.copy() if np.array_equal(expected, canvas) else None

        color = int(top_to_bottom[idx])
        visible = visible_masks[color]
        if not visible.any() or np.any(visible & higher_mask):
            return None

        for rect in _candidate_rectangles(visible, higher_mask):
            r0, c0, r1, c1 = rect
            next_mask = higher_mask.copy()
            next_mask[r0 : r1 + 1, c0 : c1 + 1] = True
            rects[color] = rect
            solved = dfs(idx + 1, next_mask, rects)
            if solved is not None:
                return solved
        rects.pop(color, None)
        return None

    solution = dfs(0, np.zeros_like(canvas, dtype=bool), {})
    assert solution is not None, "failed to recover a consistent rectangle layering"
    return solution


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the visible stacking order of overlapping rectangles using the left-side layer legend.

    Input structure:
    - The background is black.
    - A full-height separator column splits the grid into a left legend area and a right canvas area.
    - The legend area contains vertically stacked solid color blocks whose top-to-bottom order encodes rectangle layers from topmost to bottommost.
    - The canvas area shows one solid axis-aligned rectangle per legend color, with overlaps rendered by normal top-layer occlusion.

    Rule:
    - Read the legend blocks from top to bottom to obtain the current top-to-bottom layer order.
    - Recover each canvas rectangle's fixed position and size by treating lower layers as rectangles partially hidden only by already identified higher layers.
    - Repaint the same rectangles in the reverse visibility order: draw the former top block's rectangle first and the former bottom block's rectangle last.
    - Keep the legend area, separator column, and grid size unchanged.

    Key insight:
    - The left column does not label targets individually; it encodes a global z-order, so the task is to infer the hidden full rectangles on the right and then replay them with that order inverted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep_col, _sep_color = _find_separator(g, bg)
    top_to_bottom = _indicator_order(g[:, :sep_col], bg)

    canvas = g[:, sep_col + 1 :]
    canvas_colors = sorted(set(map(int, canvas.flatten())) - {bg})
    assert set(canvas_colors) == set(top_to_bottom), "legend colors and canvas colors must match exactly"

    rects = _recover_rectangles(canvas, top_to_bottom, bg)
    repainted = _render_canvas(canvas.shape, rects, top_to_bottom, bg)

    out = g.copy()
    out[:, sep_col + 1 :] = repainted
    return out


# --------------- generator ---------------

BG_COLOR = 0
SEP_COLOR = 5
RECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

LAYER_COUNT_RANGE = (2, 5)
BLOCK_SIZE_CHOICES = (2, 3)
BLOCK_GAP_RANGE = (1, 2)
GRID_HEIGHT_RANGE = (12, 28)
RIGHT_WIDTH_RANGE = (10, 18)
RECT_MIN_SIDE = 4
CORE_SIZE_RANGE = (2, 4)

RECTANGLE_ATTEMPTS = 120
GENERATION_ATTEMPTS = 500


def _random_rectangles(
    shape: tuple[int, int], top_to_bottom: list[int]
) -> dict[int, tuple[int, int, int, int]] | None:
    h, w = shape
    max_core_h = min(CORE_SIZE_RANGE[1], h - 5)
    max_core_w = min(CORE_SIZE_RANGE[1], w - 5)
    if max_core_h < CORE_SIZE_RANGE[0] or max_core_w < CORE_SIZE_RANGE[0]:
        return None

    for _ in range(RECTANGLE_ATTEMPTS):
        core_h = random.randint(CORE_SIZE_RANGE[0], max_core_h)
        core_w = random.randint(CORE_SIZE_RANGE[0], max_core_w)
        core_r0 = random.randint(2, h - core_h - 3)
        core_c0 = random.randint(2, w - core_w - 3)
        core_r1 = core_r0 + core_h - 1
        core_c1 = core_c0 + core_w - 1

        rects: dict[int, tuple[int, int, int, int]] = {}
        used: set[tuple[int, int, int, int]] = set()
        ok = True
        for color in top_to_bottom:
            placed = False
            for _inner in range(60):
                r0 = random.randint(1, core_r0)
                c0 = random.randint(1, core_c0)
                r1 = random.randint(core_r1, h - 2)
                c1 = random.randint(core_c1, w - 2)
                if r1 - r0 + 1 < RECT_MIN_SIDE or c1 - c0 + 1 < RECT_MIN_SIDE:
                    continue
                box = (r0, c0, r1, c1)
                if box in used:
                    continue
                rects[int(color)] = box
                used.add(box)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and _scene_is_valid(shape, rects, top_to_bottom):
            return rects
    return None


def _scene_is_valid(
    shape: tuple[int, int], rects: dict[int, tuple[int, int, int, int]], top_to_bottom: list[int]
) -> bool:
    input_canvas = _render_canvas(shape, rects, list(reversed(top_to_bottom)), BG_COLOR)
    output_canvas = _render_canvas(shape, rects, top_to_bottom, BG_COLOR)
    if np.array_equal(input_canvas, output_canvas):
        return False

    higher = np.zeros(shape, dtype=bool)
    for color in top_to_bottom:
        visible = input_canvas == int(color)
        if not visible.any():
            return False
        candidates = _candidate_rectangles(visible, higher)
        if candidates != [rects[int(color)]]:
            return False
        r0, c0, r1, c1 = rects[int(color)]
        higher[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _build_grid(
    height: int,
    right_width: int,
    top_to_bottom: list[int],
    rects: dict[int, tuple[int, int, int, int]],
    block_size: int,
    gaps: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    left_margin = random.randint(1, 2)
    legend_gap = random.randint(1, 2)
    left_width = left_margin + block_size + legend_gap
    total_width = left_width + 1 + right_width
    assert total_width <= 30

    input_canvas = _render_canvas((height, right_width), rects, list(reversed(top_to_bottom)), BG_COLOR)
    output_canvas = _render_canvas((height, right_width), rects, top_to_bottom, BG_COLOR)

    indicator_height = len(top_to_bottom) * block_size + sum(gaps)
    top_offset = random.randint(1, height - indicator_height - 1)

    inp = np.full((height, total_width), BG_COLOR, dtype=int)
    out = np.full_like(inp, BG_COLOR)
    sep_col = left_width
    inp[:, sep_col] = SEP_COLOR
    out[:, sep_col] = SEP_COLOR

    r = top_offset
    for idx, color in enumerate(top_to_bottom):
        inp[r : r + block_size, left_margin : left_margin + block_size] = int(color)
        out[r : r + block_size, left_margin : left_margin + block_size] = int(color)
        r += block_size
        if idx + 1 < len(top_to_bottom):
            r += gaps[idx]

    inp[:, sep_col + 1 :] = input_canvas
    out[:, sep_col + 1 :] = output_canvas
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_layers = random.randint(*LAYER_COUNT_RANGE)
    colors = random.sample(RECT_COLORS, n_layers)
    random.shuffle(colors)
    top_to_bottom = [int(color) for color in colors]

    block_size = random.choice(BLOCK_SIZE_CHOICES)
    gaps = [random.randint(*BLOCK_GAP_RANGE) for _ in range(n_layers - 1)]
    indicator_height = n_layers * block_size + sum(gaps)

    min_height = max(GRID_HEIGHT_RANGE[0], indicator_height + 2)
    if min_height > GRID_HEIGHT_RANGE[1]:
        return None
    height = random.randint(min_height, GRID_HEIGHT_RANGE[1])

    left_width = random.randint(1, 2) + block_size + random.randint(1, 2)
    max_right_width = min(RIGHT_WIDTH_RANGE[1], 30 - left_width - 1)
    if max_right_width < RIGHT_WIDTH_RANGE[0]:
        return None
    right_width = random.randint(RIGHT_WIDTH_RANGE[0], max_right_width)

    rects = _random_rectangles((height, right_width), top_to_bottom)
    if rects is None:
        return None

    inp, out = _build_grid(height, right_width, top_to_bottom, rects, block_size, gaps)
    bg_count = int(np.count_nonzero(inp == BG_COLOR))
    color_counts = [int(np.count_nonzero(inp == color)) for color in np.unique(inp) if int(color) != BG_COLOR]
    if not color_counts or bg_count <= max(color_counts):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

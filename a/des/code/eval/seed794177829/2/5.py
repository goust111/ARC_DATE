from __future__ import annotations

import itertools
import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
GRID_MIN_SIZE = 22
GRID_MAX_SIZE = 28
RECT_COUNT_OPTIONS = (3, 4)
STEP_MIN = 1
STEP_MAX = 2
TOP_RECT_SIDE_RANGE = (3, 4)
LOWER_EXTRA_RANGE = (0, 1)
PLACEMENT_MARGIN = 5
MAX_HIDDEN_PIXELS = 8
PALETTE = tuple(range(1, 10))


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != 0]


def _bbox_for_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(grid == color)
    assert rows.size > 0, "foreground color must be visible"
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for color in _foreground_colors(grid):
        r0, c0, r1, c1 = _bbox_for_color(grid, color)
        visible = int(np.count_nonzero(grid == color))
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        assert area >= visible > 0
        rects.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "visible": visible,
                "hidden": area - visible,
            }
        )
    assert 3 <= len(rects) <= 4, "expected 3-4 rectangles"
    return rects


def _render_rectangles(
    shape: tuple[int, int], rects: list[dict], order: tuple[int, ...]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = rects[idx]["color"]
    return out


def _find_unique_order(grid: np.ndarray, rects: list[dict]) -> tuple[int, ...]:
    matches = []
    for perm in itertools.permutations(range(len(rects))):
        if np.array_equal(_render_rectangles(grid.shape, rects, perm), grid):
            matches.append(perm)
    assert len(matches) == 1, "layer order must be uniquely recoverable"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapped solid rectangles, then expand each by its hidden-pixel count.

    The input contains 3-4 differently colored opaque rectangles on a black
    background. Each rectangle's full bounds are inferred from the visible part,
    the unique stacking order is recovered by replaying candidate layer orders,
    and every rectangle is then expanded outward by N cells on all sides, where
    N is the number of pixels hidden by higher layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rects = _extract_rectangles(g)
    order = _find_unique_order(g, rects)

    out = np.zeros_like(g)
    height, width = map(int, g.shape)
    for idx in order:
        rect = rects[idx]
        n_hidden = int(rect["hidden"])
        r0, c0, r1, c1 = rect["bbox"]
        rr0 = max(0, r0 - n_hidden)
        cc0 = max(0, c0 - n_hidden)
        rr1 = min(height - 1, r1 + n_hidden)
        cc1 = min(width - 1, c1 + n_hidden)
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = rect["color"]

    return out


def _has_visible_corner(
    grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = bbox
    return any(
        np.all(grid[r_edge, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c_edge] == color)
        for r_edge in (r0, r1)
        for c_edge in (c0, c1)
    )


def _build_canonical_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    rect_count = random.choice(RECT_COUNT_OPTIONS)

    row_steps = [random.randint(STEP_MIN, STEP_MAX) for _ in range(rect_count - 1)]
    col_steps = [random.randint(STEP_MIN, STEP_MAX) for _ in range(rect_count - 1)]

    row_offsets = [0]
    col_offsets = [0]
    for dr, dc in zip(row_steps, col_steps):
        row_offsets.append(row_offsets[-1] + dr)
        col_offsets.append(col_offsets[-1] + dc)

    top_row_offset = row_offsets[-1]
    top_col_offset = col_offsets[-1]
    heights: list[int] = []
    widths: list[int] = []
    for i in range(rect_count):
        if i == rect_count - 1:
            heights.append(random.randint(*TOP_RECT_SIDE_RANGE))
            widths.append(random.randint(*TOP_RECT_SIDE_RANGE))
        else:
            min_h = max(3, top_row_offset - row_offsets[i] + 1)
            min_w = max(3, top_col_offset - col_offsets[i] + 1)
            heights.append(min_h + random.randint(*LOWER_EXTRA_RANGE))
            widths.append(min_w + random.randint(*LOWER_EXTRA_RANGE))

    if height <= max(heights) + 2 * PLACEMENT_MARGIN:
        return None
    if width <= max(widths) + 2 * PLACEMENT_MARGIN:
        return None

    base_r = random.randint(PLACEMENT_MARGIN, height - max(heights) - PLACEMENT_MARGIN - 1)
    base_c = random.randint(PLACEMENT_MARGIN, width - max(widths) - PLACEMENT_MARGIN - 1)
    colors = random.sample(PALETTE, rect_count)

    original_rects: list[dict] = []
    for i, color in enumerate(colors):
        r0 = base_r + row_offsets[i]
        c0 = base_c + col_offsets[i]
        original_rects.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r0 + heights[i] - 1, c0 + widths[i] - 1),
            }
        )

    inp = _render_rectangles((height, width), original_rects, tuple(range(rect_count)))
    rects = _extract_rectangles(inp)
    order = _find_unique_order(inp, rects)
    if order != tuple(range(rect_count)):
        return None

    hidden_counts = [int(rect["hidden"]) for rect in rects]
    if hidden_counts[-1] != 0 or any(count <= 0 for count in hidden_counts[:-1]):
        return None
    if max(hidden_counts) > MAX_HIDDEN_PIXELS:
        return None

    for rect in rects:
        if not _has_visible_corner(inp, rect["color"], rect["bbox"]):
            return None
        r0, c0, r1, c1 = rect["bbox"]
        n_hidden = int(rect["hidden"])
        if r0 - n_hidden < 0 or c0 - n_hidden < 0:
            return None
        if r1 + n_hidden >= height or c1 + n_hidden >= width:
            return None

    out = main(inp)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_canonical_case()
        if built is None:
            continue
        inp, expected = built
        rotation = random.randint(0, 3)
        inp = np.rot90(inp, rotation)
        expected = np.rot90(expected, rotation)
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid overlapping-rectangle sample")

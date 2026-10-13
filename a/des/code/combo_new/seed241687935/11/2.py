from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
CANVAS_SIZE = 30
GENERATION_RETRIES = 500
CHAIN_RETRIES = 200
PLACEMENT_RETRIES = 80

UNIQUE_RECT_COUNT_RANGE = (3, 5)
OUTER_HEIGHT_RANGE = (6, 18)
OUTER_WIDTH_RANGE = (6, 18)
TEMPLATE_OUTER_HEIGHT_RANGE = (4, 12)
TEMPLATE_OUTER_WIDTH_RANGE = (4, 12)


def _components(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    rects: list[dict[str, int | tuple[int, int, int, int]]] = []

    for start_r in range(height):
        for start_c in range(width):
            color = int(g[start_r, start_c])
            if color == BG or seen[start_r, start_c]:
                continue

            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            area = rect_h * rect_w
            window = g[r0 : r1 + 1, c0 : c1 + 1]
            assert len(cells) == area
            assert bool(np.all(window == color)), "every object must be a solid rectangle"

            rects.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "height": rect_h,
                    "width": rect_w,
                    "area": area,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the singleton rectangles as centered concentric layers ordered from largest area to smallest.

    Input structure:
    - The background is black.
    - The grid contains several disjoint monochrome solid rectangles.
    - Exactly one color appears on two separate rectangles; all other colors appear once.

    Rule:
    - Find the duplicated-color pair and treat it as the teaching example for centered nesting.
    - Ignore that duplicated pair, then sort every singleton-color rectangle by area from largest to smallest.
    - Create an output whose size matches the largest singleton rectangle, draw that rectangle first, and place each smaller rectangle centered inside the previous layer, breaking odd offsets toward the top-left.

    Key insight:
    - The repeated color does not contribute material to the answer; it only signals that the remaining rectangles must be stacked by area as concentric centered overlays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _components(g)
    assert rects, "expected at least one rectangle"

    by_color: dict[int, list[dict[str, int | tuple[int, int, int, int]]]] = {}
    for rect in rects:
        color = int(rect["color"])
        by_color.setdefault(color, []).append(rect)

    repeated_colors = [color for color, items in by_color.items() if len(items) == 2]
    assert len(repeated_colors) == 1, "expected exactly one duplicated color"
    repeated_color = repeated_colors[0]
    assert all(len(items) == 1 or color == repeated_color for color, items in by_color.items())

    template_pair = sorted(by_color[repeated_color], key=lambda rect: int(rect["area"]))
    assert int(template_pair[0]["area"]) < int(template_pair[1]["area"])
    assert int(template_pair[0]["height"]) <= int(template_pair[1]["height"])
    assert int(template_pair[0]["width"]) <= int(template_pair[1]["width"])

    layers = [items[0] for color, items in by_color.items() if color != repeated_color]
    assert layers, "expected singleton rectangles to reconstruct"
    areas = [int(rect["area"]) for rect in layers]
    assert len(set(areas)) == len(areas), "singleton rectangle areas must define a strict order"

    layers.sort(key=lambda rect: int(rect["area"]), reverse=True)
    outer = layers[0]
    out = np.full((int(outer["height"]), int(outer["width"])), int(outer["color"]), dtype=int)

    cur_h, cur_w = out.shape
    for rect in layers[1:]:
        rect_h = int(rect["height"])
        rect_w = int(rect["width"])
        assert rect_h <= cur_h and rect_w <= cur_w, "nested rectangles must fit inside the previous layer"
        r0 = (cur_h - rect_h) // 2
        c0 = (cur_w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(rect["color"])

    return out


def _sample_chain(count: int) -> list[tuple[int, int]]:
    for _ in range(CHAIN_RETRIES):
        outer_h = random.randint(*OUTER_HEIGHT_RANGE)
        outer_w = random.randint(*OUTER_WIDTH_RANGE)
        if count > outer_h or count > outer_w:
            continue
        heights = sorted(random.sample(range(1, outer_h + 1), count), reverse=True)
        widths = sorted(random.sample(range(1, outer_w + 1), count), reverse=True)
        sizes = list(zip(heights, widths))
        areas = [h * w for h, w in sizes]
        if len(set(areas)) != count:
            continue
        return sizes
    raise RuntimeError("failed to sample a valid nesting chain")


def _sample_template_pair() -> tuple[tuple[int, int], tuple[int, int]]:
    outer_h = random.randint(*TEMPLATE_OUTER_HEIGHT_RANGE)
    outer_w = random.randint(*TEMPLATE_OUTER_WIDTH_RANGE)
    inner_h = random.randint(1, outer_h - 1)
    inner_w = random.randint(1, outer_w - 1)
    return (outer_h, outer_w), (inner_h, inner_w)


def _valid_positions(blocked: np.ndarray, rect_h: int, rect_w: int) -> list[tuple[int, int]]:
    size = blocked.shape[0]
    candidates: list[tuple[int, int]] = []
    for r in range(size - rect_h + 1):
        rr0 = max(0, r - 1)
        rr1 = min(size, r + rect_h + 1)
        for c in range(size - rect_w + 1):
            cc0 = max(0, c - 1)
            cc1 = min(size, c + rect_w + 1)
            if not blocked[rr0:rr1, cc0:cc1].any():
                candidates.append((r, c))
    return candidates


def _place_rectangles(rects: list[tuple[int, int, int]]) -> np.ndarray | None:
    order = sorted(rects, key=lambda item: item[0] * item[1], reverse=True)
    for _ in range(PLACEMENT_RETRIES):
        grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
        blocked = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
        ok = True

        for rect_h, rect_w, color in order:
            candidates = _valid_positions(blocked, rect_h, rect_w)
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
            blocked[max(0, r0 - 1) : min(CANVAS_SIZE, r0 + rect_h + 1),
                    max(0, c0 - 1) : min(CANVAS_SIZE, c0 + rect_w + 1)] = True

        if not ok:
            continue

        coords = np.argwhere(grid != BG)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        top_margin = random.randint(0, int(r0))
        left_margin = random.randint(0, int(c0))
        bottom_margin = random.randint(0, int(CANVAS_SIZE - 1 - r1))
        right_margin = random.randint(0, int(CANVAS_SIZE - 1 - c1))
        return grid[
            int(r0 - top_margin) : int(r1 + bottom_margin) + 1,
            int(c0 - left_margin) : int(c1 + right_margin) + 1,
        ]

    return None


def _expected_from_sizes(sizes: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    out = np.full(sizes[0], colors[0], dtype=int)
    cur_h, cur_w = out.shape
    for (rect_h, rect_w), color in zip(sizes[1:], colors[1:]):
        r0 = (cur_h - rect_h) // 2
        c0 = (cur_w - rect_w) // 2
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    unique_count = random.randint(*UNIQUE_RECT_COUNT_RANGE)
    unique_sizes = _sample_chain(unique_count)
    template_outer, template_inner = _sample_template_pair()

    colors = random.sample(COLOR_POOL, unique_count + 1)
    template_color = colors[0]
    layer_colors = colors[1:]

    rects: list[tuple[int, int, int]] = [
        (template_outer[0], template_outer[1], template_color),
        (template_inner[0], template_inner[1], template_color),
    ]
    rects.extend((rect_h, rect_w, color) for (rect_h, rect_w), color in zip(unique_sizes, layer_colors))

    grid = _place_rectangles(rects)
    if grid is None:
        return None

    expected = _expected_from_sizes(unique_sizes, layer_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid centered-nesting input")

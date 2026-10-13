from __future__ import annotations

import random

import numpy as np


Rect = tuple[int, int, int, int, int]  # (color, r0, c0, r1, c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "each rectangle color must remain visible"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != 0)


def _overlap_box(a: Rect, b: Rect) -> tuple[int, int, int, int] | None:
    r0 = max(a[1], b[1])
    c0 = max(a[2], b[2])
    r1 = min(a[3], b[3])
    c1 = min(a[4], b[4])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _contains(rect: Rect, r: int, c: int) -> bool:
    return rect[1] <= r <= rect[3] and rect[2] <= c <= rect[4]


def _paint_rect(out: np.ndarray, rect: Rect, *, expand: bool) -> None:
    color, r0, c0, r1, c1 = rect
    h, w = out.shape
    out[r0 : r1 + 1, c0 : c1 + 1] = color
    if not expand:
        return
    if r0 > 0:
        out[r0 - 1, c0 : c1 + 1] = color
    if r1 + 1 < h:
        out[r1 + 1, c0 : c1 + 1] = color
    if c0 > 0:
        out[r0 : r1 + 1, c0 - 1] = color
    if c1 + 1 < w:
        out[r0 : r1 + 1, c1 + 1] = color


def _render(shape: tuple[int, int], rects: list[Rect], *, expand: bool) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rect in rects:
        _paint_rect(out, rect, expand=expand)
    return out


def _expanded_mask(shape: tuple[int, int], rect: Rect) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _paint_rect(mask, rect, expand=True)
    return mask


def _rect_mask(shape: tuple[int, int], rect: Rect) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _, r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _infer_rectangles(grid: np.ndarray) -> list[Rect]:
    rects = []
    for color in _colors(grid):
        rects.append((color, *_bbox(grid == color)))
    assert 3 <= len(rects) <= 5, "expected 3 to 5 visible rectangle colors"
    return rects


def _infer_order(grid: np.ndarray, rects: list[Rect]) -> list[Rect]:
    edges: dict[int, set[int]] = {rect[0]: set() for rect in rects}
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            covering = [rect[0] for rect in rects if _contains(rect, r, c)]
            visible = int(grid[r, c])
            if not covering:
                assert visible == 0, "background cannot appear inside any reconstructed rectangle"
                continue

            assert visible in covering, "a covered cell must show one of its covering rectangles"
            for color in covering:
                if color != visible:
                    edges[color].add(visible)

    indegree = {color: 0 for color in edges}
    for lower, uppers in edges.items():
        assert lower not in uppers, "layer graph cannot contain self-loops"
        for upper in uppers:
            indegree[upper] += 1

    rect_by_color = {rect[0]: rect for rect in rects}
    remaining = set(rect_by_color)
    order: list[Rect] = []

    while remaining:
        bottoms = sorted(color for color in remaining if indegree[color] == 0)
        assert len(bottoms) == 1, "original overlaps must determine a unique total layer order"
        bottom = bottoms[0]
        order.append(rect_by_color[bottom])
        remaining.remove(bottom)
        for upper in edges[bottom]:
            indegree[upper] -= 1

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden rectangle extents, infer their layer order from occlusions, then extrude each rectangle by one pixel.

    Input structure:
    - Black is background and every non-black color belongs to one solid axis-aligned rectangle.
    - Rectangles partially overlap, so a lower rectangle may be visible only as a clipped remnant around higher ones.
    - Each color remains visible somewhere, and the visible remnant of each color still pins down that rectangle's full bbox.

    Rule:
    - Recover each rectangle as the bbox of its visible color.
    - For every input cell, compare the visible color with every reconstructed rectangle covering that cell; any covered color that is not visible is below the visible one.
    - Take the unique bottom-to-top order implied by those original overlap constraints.
    - Repaint the rectangles in that order on a black canvas, but use each rectangle's one-pixel edge extrusion as its painted shape: original body plus top, bottom, left, and right outward strips, with the four new outer corners left black.

    Key insight:
    - The same inferred layer order resolves both the original occlusions and the new clashes created by the one-pixel extrusions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _infer_rectangles(g)
    order = _infer_order(g, rects)
    return _render(g.shape, order, expand=True)


# --------------- generator ---------------

GRID_H_RANGE = (16, 20)
GRID_W_RANGE = (16, 20)
RECT_COUNT_RANGE = (3, 5)
RECT_H_RANGE = (3, 5)
RECT_W_RANGE = (4, 6)
STEP_CHOICES = (-2, -1, 1, 2)
PALETTE = (1, 2, 3, 4, 5)

MIN_VISIBLE_PIXELS = 5
MAX_OVERLAP_FRACTION = 0.5
PLACEMENT_TRIES = 80
GENERATION_RETRIES = 800


def _random_rect(color: int, h: int, w: int, r0: int, c0: int) -> Rect:
    return color, r0, c0, r0 + h - 1, c0 + w - 1


def _random_size() -> tuple[int, int]:
    return random.randint(*RECT_H_RANGE), random.randint(*RECT_W_RANGE)


def _sample_rectangles(shape: tuple[int, int]) -> list[Rect] | None:
    h, w = shape
    count = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(PALETTE, count)
    rects: list[Rect] = []

    first_h, first_w = _random_size()
    first_r0 = random.randint(4, h - first_h - 5)
    first_c0 = random.randint(4, w - first_w - 5)
    rects.append(_random_rect(colors[0], first_h, first_w, first_r0, first_c0))

    for i in range(1, count):
        prev = rects[-1]
        placed = False
        for _ in range(PLACEMENT_TRIES):
            rect_h, rect_w = _random_size()
            r0 = prev[1] + random.choice(STEP_CHOICES)
            c0 = prev[2] + random.choice(STEP_CHOICES)
            if not (1 <= r0 <= h - rect_h - 2 and 1 <= c0 <= w - rect_w - 2):
                continue

            rect = _random_rect(colors[i], rect_h, rect_w, r0, c0)
            if any(rect[1:] == other[1:] for other in rects):
                continue

            overlap = _overlap_box(prev, rect)
            if overlap is None:
                continue
            rr0, cc0, rr1, cc1 = overlap
            if (rr1 - rr0 + 1) * (cc1 - cc0 + 1) < 2:
                continue

            rects.append(rect)
            placed = True
            break

        if not placed:
            return None

    return rects


def _validate_input(grid: np.ndarray, rects: list[Rect]) -> bool:
    for rect in rects:
        color = rect[0]
        if not np.any(grid == color):
            return False
        if _bbox(grid == color) != rect[1:]:
            return False
        if int(np.count_nonzero(grid == color)) < MIN_VISIBLE_PIXELS:
            return False

    coverage = np.zeros(grid.shape, dtype=int)
    for rect in rects:
        _, r0, c0, r1, c1 = rect
        coverage[r0 : r1 + 1, c0 : c1 + 1] += 1

    overlap_cells = int(np.count_nonzero(coverage >= 2))
    colored_cells = int(np.count_nonzero(grid))
    if overlap_cells > max(8, int(colored_cells * MAX_OVERLAP_FRACTION)):
        return False

    for i in range(1, len(rects)):
        lower = rects[i - 1]
        upper = rects[i]
        overlap = _overlap_box(lower, upper)
        if overlap is None:
            return False
        r0, c0, r1, c1 = overlap
        if not np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == upper[0]):
            return False

        expanded_overlap = _expanded_mask(grid.shape, lower) & _expanded_mask(grid.shape, upper)
        original_overlap = _rect_mask(grid.shape, lower) & _rect_mask(grid.shape, upper)
        if not np.any(expanded_overlap & ~original_overlap):
            return False

    return True


def _attempt() -> np.ndarray | None:
    shape = (random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE))
    rects = _sample_rectangles(shape)
    if rects is None:
        return None

    inp = _render(shape, rects, expand=False)
    if not _validate_input(inp, rects):
        return None

    expected = _render(shape, rects, expand=True)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATION_RETRIES} attempts")

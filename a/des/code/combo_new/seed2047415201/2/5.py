from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
MIN_GRID_SIZE = 12
MAX_GRID_SIZE = 30
GENERATION_ATTEMPTS = 300


BBox = tuple[int, int, int, int]


def _foreground_colors(g: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in g.flat:
        color = int(value)
        if color == BG or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _bbox(mask: np.ndarray) -> BBox | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rect_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}

    for lower, higher in edges:
        assert lower != higher, "a layer cannot be both above and below itself"
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must give one layer order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the bottom-to-top stacking order of overlapping transparent solid rectangles.

    Input structure:
    - The grid background is Black.
    - Each foreground color is one complete filled rectangle, though parts of it may be hidden by rectangles above it.
    - Every rectangle has enough visible pixels to reveal its full bounding box, and every color is visible at least once.

    Rule:
    - Reconstruct each colored rectangle as the filled bounding box of its visible pixels.
    - At any cell covered by two or more reconstructed rectangles, the visible color is above all other rectangles covering that cell.
    - Topologically sort these pairwise blocking constraints and return a one-column grid from bottom color to top color.

    Key insight: The spatial image is only evidence for hidden filled rectangles; their overlap cells reveal a pure layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    colors = _foreground_colors(g)
    assert colors, "input must contain at least one colored rectangle"

    rect_masks: dict[int, np.ndarray] = {}
    for color in colors:
        box = _bbox(g == color)
        assert box is not None, "each foreground color must be visible"
        rect_masks[color] = _rect_mask(g.shape, box)

    edges: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if rect_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "an overlap cell must show one of its covering rectangles"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    if len(colors) > 1:
        assert edges, "multiple rectangles need overlap evidence"
    order = _unique_layer_order(colors, edges)
    return np.array(order, dtype=int).reshape(-1, 1)


# --------------- generator ---------------


def _sample_step(rect_len: int) -> int:
    return random.randint(2, max(2, rect_len - 2))


def _make_diagonal_stack() -> tuple[int, int, list[BBox]] | None:
    count = random.randint(3, 6)
    rect_h = random.randint(5, 10)
    rect_w = random.randint(5, 11)
    step_r = _sample_step(rect_h)
    step_c = _sample_step(rect_w)

    span_h = rect_h + (count - 1) * step_r
    span_w = rect_w + (count - 1) * step_c
    if span_h > MAX_GRID_SIZE or span_w > MAX_GRID_SIZE:
        return None

    min_h = max(MIN_GRID_SIZE, span_h + 1)
    min_w = max(MIN_GRID_SIZE, span_w + 1)
    if min_h > MAX_GRID_SIZE or min_w > MAX_GRID_SIZE:
        return None

    height = random.randint(min_h, MAX_GRID_SIZE)
    width = random.randint(min_w, MAX_GRID_SIZE)
    base_r = random.randint(0, height - span_h)
    base_c = random.randint(0, width - span_w)
    sign_r = random.choice((-1, 1))
    sign_c = random.choice((-1, 1))

    if sign_r > 0:
        first_r = base_r
    else:
        first_r = base_r + (count - 1) * step_r

    if sign_c > 0:
        first_c = base_c
    else:
        first_c = base_c + (count - 1) * step_c

    boxes: list[BBox] = []
    for i in range(count):
        r0 = first_r + sign_r * i * step_r
        c0 = first_c + sign_c * i * step_c
        boxes.append((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))

    return height, width, boxes


def _render_stack(height: int, width: int, colors: list[int], boxes: list[BBox]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, boxes):
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _visible_bboxes_match(grid: np.ndarray, colors: list[int], boxes: list[BBox]) -> bool:
    for color, box in zip(colors, boxes):
        if _bbox(grid == int(color)) != box:
            return False
    return True


def _solver_matches(grid: np.ndarray, colors: list[int]) -> bool:
    try:
        expected = np.array(colors, dtype=int).reshape(-1, 1)
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    placed = _make_diagonal_stack()
    if placed is None:
        return None
    height, width, boxes = placed

    colors = random.sample(list(COLORS), len(boxes))
    random.shuffle(colors)
    grid = _render_stack(height, width, colors, boxes)

    if not _visible_bboxes_match(grid, colors, boxes):
        return None
    if not _solver_matches(grid, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")

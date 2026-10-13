from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


BG = 0
COLOR_POOL = tuple(range(1, 10))

# Generator configuration.
GRID_MIN_SIZE = 16
GRID_MAX_SIZE = 28
RECT_COUNT_CHOICES = (3, 4, 5)
CORE_SIDE_MIN = 2
CORE_SIDE_MAX = 4
MAX_MARGIN = 8
MIN_RECT_SIDE = 5
MIN_RECT_AREA = 24
MIN_VISIBLE_PIXELS = 4
GENERATION_RETRIES = 400
BOX_NEAR_DUPLICATE_DISTANCE = 4


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _bbox_intersection(a: BBox, b: BBox) -> BBox | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _contains(a: BBox, b: BBox) -> bool:
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _render_scene(shape: tuple[int, int], rectangles: dict[int, BBox], order: tuple[int, ...]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rectangles[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _recover_rectangles(grid: np.ndarray) -> dict[int, BBox]:
    rectangles: dict[int, BBox] = {}
    colors = sorted(set(map(int, grid.ravel())) - {BG})
    assert colors, "expected at least one visible rectangle"
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size, "every rectangle color must remain visible"
        rectangles[color] = _bbox_from_coords(coords)
    return rectangles


def _visibility_dag(grid: np.ndarray, rectangles: dict[int, BBox]) -> tuple[dict[int, set[int]], dict[int, int]] | None:
    colors = sorted(rectangles)
    edges = {color: set() for color in colors}  # lower -> direct/indirect known uppers
    indegree = {color: 0 for color in colors}

    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            inter = _bbox_intersection(rectangles[a], rectangles[b])
            if inter is None:
                continue

            r0, c0, r1, c1 = inter
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            sees_a = bool(np.any(region == a))
            sees_b = bool(np.any(region == b))

            if sees_a and sees_b:
                return None
            if sees_a:
                lower, upper = b, a
            elif sees_b:
                lower, upper = a, b
            else:
                continue

            if upper not in edges[lower]:
                edges[lower].add(upper)
                indegree[upper] += 1

    return edges, indegree


def _consistent_orders(
    grid: np.ndarray,
    rectangles: dict[int, BBox],
    limit: int = 2,
) -> list[tuple[int, ...]]:
    dag = _visibility_dag(grid, rectangles)
    if dag is None:
        return []

    edges, indegree = dag
    remaining = set(rectangles)
    partial: list[int] = []
    orders: list[tuple[int, ...]] = []

    def backtrack() -> None:
        if len(orders) >= limit:
            return
        if not remaining:
            candidate = tuple(partial)
            if np.array_equal(_render_scene(grid.shape, rectangles, candidate), grid):
                orders.append(candidate)
            return

        available = sorted(color for color in remaining if indegree[color] == 0)
        for color in available:
            remaining.remove(color)
            partial.append(color)
            lowered: list[int] = []
            for upper in edges[color]:
                indegree[upper] -= 1
                lowered.append(upper)

            backtrack()

            for upper in lowered:
                indegree[upper] += 1
            partial.pop()
            remaining.add(color)

            if len(orders) >= limit:
                return

    backtrack()
    return orders


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a stack of partially hidden solid rectangles, move the bottom rectangle to the top, and redraw the overlaps.

    Input structure:
    - The background is black.
    - Each non-background color belongs to one solid axis-aligned rectangle, but rectangles may overlap and hide parts of one another.
    - The visible pixels of a rectangle still reveal its full bounding box: the bbox of each color's visible cells is that rectangle's true extent.
    - Overlap cells show only the color of the upper rectangle in the current drawing order.

    Rule:
    - For every visible color, infer its complete rectangle from the bbox of its visible cells.
    - Recover the current bottom-to-top stacking order by checking which rectangle colors are visible inside pairwise overlap regions and keeping only orders whose redraw exactly matches the input.
    - Take the current bottom rectangle and move it to the top of the stack.
    - Redraw all inferred full rectangles on a black canvas using that updated order.

    Key insight:
    - The geometry and the layer order must both be reconstructed: visible fragments give each full rectangle, while overlap regions reveal which rectangle sits above which other one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 in set(map(int, g.ravel())), "expected black background cells"

    rectangles = _recover_rectangles(g)
    orders = _consistent_orders(g, rectangles, limit=2)
    assert len(orders) == 1, "expected a unique stacking order"

    order = orders[0]
    new_order = order[1:] + order[:1]
    return _render_scene(g.shape, rectangles, new_order)


def _has_non_nested_pair(rectangles: dict[int, BBox]) -> bool:
    boxes = list(rectangles.values())
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if not _contains(a, b) and not _contains(b, a):
                return True
    return False


def _visible_bboxes_match(grid: np.ndarray, rectangles: dict[int, BBox]) -> bool:
    for color, box in rectangles.items():
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return False
        if _bbox_from_coords(coords) != box:
            return False
        if int(coords.shape[0]) < MIN_VISIBLE_PIXELS:
            return False
    return True


def _sample_rectangles() -> tuple[tuple[int, int], dict[int, BBox], tuple[int, ...]] | None:
    height = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    width = random.randint(GRID_MIN_SIZE, GRID_MAX_SIZE)
    n_rects = random.choice(RECT_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, n_rects)

    core_h = random.randint(CORE_SIDE_MIN, min(CORE_SIDE_MAX, height - 8))
    core_w = random.randint(CORE_SIDE_MIN, min(CORE_SIDE_MAX, width - 8))
    core_r0 = random.randint(3, height - core_h - 4)
    core_c0 = random.randint(3, width - core_w - 4)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rectangles: dict[int, BBox] = {}
    boxes: list[BBox] = []
    for color in colors:
        for _ in range(80):
            up = random.randint(1, min(MAX_MARGIN, core_r0))
            down = random.randint(1, min(MAX_MARGIN, height - 1 - core_r1))
            left = random.randint(1, min(MAX_MARGIN, core_c0))
            right = random.randint(1, min(MAX_MARGIN, width - 1 - core_c1))

            box = (core_r0 - up, core_c0 - left, core_r1 + down, core_c1 + right)
            rect_h = box[2] - box[0] + 1
            rect_w = box[3] - box[1] + 1
            if rect_h < MIN_RECT_SIDE or rect_w < MIN_RECT_SIDE:
                continue
            if rect_h * rect_w < MIN_RECT_AREA:
                continue
            if box in boxes:
                continue
            if any(
                sum(abs(a - b) for a, b in zip(box, prev)) < BOX_NEAR_DUPLICATE_DISTANCE
                for prev in boxes
            ):
                continue

            rectangles[color] = box
            boxes.append(box)
            break
        else:
            return None

    if not _has_non_nested_pair(rectangles):
        return None

    order = tuple(random.sample(colors, n_rects))
    return (height, width), rectangles, order


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sampled = _sample_rectangles()
        if sampled is None:
            continue

        shape, rectangles, order = sampled
        inp = _render_scene(shape, rectangles, order)
        if not _visible_bboxes_match(inp, rectangles):
            continue

        orders = _consistent_orders(inp, rectangles, limit=2)
        if orders != [order]:
            continue

        out = _render_scene(shape, rectangles, order[1:] + order[:1])
        if np.array_equal(inp, out):
            continue

        if not np.array_equal(main(inp), out):
            continue
        return inp

    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_RETRIES} attempts")

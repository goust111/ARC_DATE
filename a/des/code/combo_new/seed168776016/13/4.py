import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

# Generator bounds.
RECT_COUNT_CHOICES = (3, 4)
SCENE_HEIGHT_RANGE = (12, 24)
SCENE_WIDTH_RANGE = (12, 24)
RECT_SIDE_RANGE = (5, 10)
STEP_RANGE = (1, 3)
MIN_VISIBLE_AREA = 3
MAX_GENERATION_ATTEMPTS = 400


def _extract_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    for c in range(w - 1, -1, -1):
        rows = np.flatnonzero(grid[:, c] != BG)
        if rows.size == 0:
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(colors) < 2 or len(set(colors)) != len(colors):
            continue
        if all(int(np.count_nonzero(grid == color)) == 1 for color in colors):
            return c, colors
    raise AssertionError("failed to find the palette column")


def _unique_topo_sort(edges: dict[int, set[int]]) -> list[int]:
    indeg = {node: 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            assert dst in indeg and dst != src, "invalid overlap relation"
            indeg[dst] += 1

    ready = [node for node, degree in indeg.items() if degree == 0]
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "stacking order must be uniquely determined"
        node = ready.pop()
        order.append(node)
        for dst in edges[node]:
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)

    assert len(order) == len(edges), "overlap relations must form a DAG"
    return order


def _infer_scene(scene: np.ndarray) -> tuple[dict[int, tuple[int, int, int, int]], list[int]]:
    colors = [int(color) for color in np.unique(scene) if int(color) != BG]
    assert colors, "expected at least one rectangle"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    edges: dict[int, set[int]] = {color: set() for color in colors}
    for color in colors:
        pts = np.argwhere(scene == color)
        assert pts.size > 0
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        box = scene[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(box != BG), "recoverable rectangle bbox must stay fully covered"
        bboxes[color] = (r0, c0, r1, c1)
        for higher in np.unique(box):
            higher = int(higher)
            if higher not in (BG, color):
                edges[color].add(higher)

    order = _unique_topo_sort(edges)
    return bboxes, order


def _paint_scene(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
    colors: list[int],
) -> np.ndarray:
    scene = np.full(shape, BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, rectangles):
        scene[r0 : r1 + 1, c0 : c1 + 1] = color
    return scene


def _sample_rectangles(
    h: int,
    w: int,
    n_rects: int,
) -> list[tuple[int, int, int, int]] | None:
    dr = random.choice((-1, 1))
    dc = random.choice((-1, 1))
    step_r = random.randint(*STEP_RANGE)
    step_c = random.randint(*STEP_RANGE)

    max_h = min(RECT_SIDE_RANGE[1], h - (n_rects - 1) * step_r)
    max_w = min(RECT_SIDE_RANGE[1], w - (n_rects - 1) * step_c)
    min_h = max(RECT_SIDE_RANGE[0], step_r + 2)
    min_w = max(RECT_SIDE_RANGE[0], step_c + 2)
    if max_h < min_h or max_w < min_w:
        return None

    heights = [random.randint(min_h, max_h) for _ in range(n_rects)]
    widths = [random.randint(min_w, max_w) for _ in range(n_rects)]

    top_margin = (n_rects - 1) * step_r if dr < 0 else 0
    left_margin = (n_rects - 1) * step_c if dc < 0 else 0
    bottom_room = h - max(heights) - ((n_rects - 1) * step_r if dr > 0 else 0)
    right_room = w - max(widths) - ((n_rects - 1) * step_c if dc > 0 else 0)
    if bottom_room < top_margin or right_room < left_margin:
        return None

    start_r = random.randint(top_margin, bottom_room)
    start_c = random.randint(left_margin, right_room)

    rectangles: list[tuple[int, int, int, int]] = []
    for idx in range(n_rects):
        r0 = start_r + idx * dr * step_r
        c0 = start_c + idx * dc * step_c
        r1 = r0 + heights[idx] - 1
        c1 = c0 + widths[idx] - 1
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            return None
        rectangles.append((r0, c0, r1, c1))

    for idx in range(n_rects - 1):
        ar0, ac0, ar1, ac1 = rectangles[idx]
        br0, bc0, br1, bc1 = rectangles[idx + 1]
        overlap_h = min(ar1, br1) - max(ar0, br0) + 1
        overlap_w = min(ac1, bc1) - max(ac0, bc0) + 1
        if overlap_h <= 0 or overlap_w <= 0:
            return None

    return rectangles


def _build_input(
    scene: np.ndarray,
    palette_colors: list[int],
) -> np.ndarray:
    h, w = scene.shape
    rows = sorted(random.sample(range(h), len(palette_colors)))
    grid = np.zeros((h, w + 1), dtype=int)
    grid[:, :w] = scene
    for row, color in zip(rows, palette_colors):
        grid[row, w] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.choice(RECT_COUNT_CHOICES)
    scene_h = random.randint(*SCENE_HEIGHT_RANGE)
    scene_w = random.randint(*SCENE_WIDTH_RANGE)
    if scene_w + 1 > 30:
        return None

    rectangles = _sample_rectangles(scene_h, scene_w, n_rects)
    if rectangles is None:
        return None

    chosen = random.sample(COLOR_POOL, 2 * n_rects)
    scene_colors = chosen[:n_rects]
    palette_colors = chosen[n_rects:]

    scene = _paint_scene((scene_h, scene_w), rectangles, scene_colors)
    visible_counts = {color: int(np.count_nonzero(scene == color)) for color in scene_colors}
    if min(visible_counts.values()) < MIN_VISIBLE_AREA:
        return None
    if any(
        visible_counts[color] == (r1 - r0 + 1) * (c1 - c0 + 1)
        for color, (r0, c0, r1, c1) in zip(scene_colors[:-1], rectangles[:-1])
    ):
        return None

    try:
        inferred_bboxes, inferred_order = _infer_scene(scene)
    except AssertionError:
        return None

    actual_bbox_map = {color: rect for color, rect in zip(scene_colors, rectangles)}
    if inferred_order != scene_colors:
        return None
    if any(inferred_bboxes[color] != actual_bbox_map[color] for color in scene_colors):
        return None

    return _build_input(scene, palette_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the stacked solid rectangles according to the right-side depth palette.

    Input structure:
    - The background is black.
    - The left part of the grid contains several overlapping solid axis-aligned rectangles, each with its own color.
    - The rightmost column contains one single-pixel palette swatch per rectangle, ordered from top to bottom.

    Rule:
    - Remove the palette column from the scene.
    - Recover each rectangle's full bounding box from its visible footprint, then infer the stacking DAG: if another visible rectangle color appears inside a rectangle's bounding box, that other rectangle must be above it.
    - Use the uniquely determined bottom-to-top order to repaint the rectangles from bottom to top with the palette colors read from top to bottom, while preserving the same rectangle boxes and painter-style occlusion.

    Key insight:
    - The original rectangle colors are irrelevant; the visible interruptions reveal the depth order, and the palette column remaps that order to a new bottom-to-top coloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_col, palette_colors = _extract_palette(g)
    scene = g[:, :palette_col]
    bboxes, order = _infer_scene(scene)
    assert len(order) == len(palette_colors), "palette size must match rectangle count"

    out = np.full(scene.shape, BG, dtype=int)
    for depth, color in enumerate(order):
        r0, c0, r1, c1 = bboxes[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = palette_colors[depth]
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacked-rectangle ARC input")

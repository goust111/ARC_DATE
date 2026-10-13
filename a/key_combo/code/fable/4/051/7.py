from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _draw_border(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_rectangles(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = _get_bg(grid)
    rectangles: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, f"missing visible cells for color {color}"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each object must be a hollow rectangle"
        mask = _border_mask(grid.shape, box)
        assert np.all(mask[coords[:, 0], coords[:, 1]]), "visible pixels must lie on one rectangle border"
        rectangles.append({"color": color, "bbox": box, "mask": mask})
    assert len(rectangles) >= 2, "expected multiple rectangles"
    return bg, rectangles


def _infer_bottom_to_top(grid: np.ndarray, rectangles: list[dict[str, object]]) -> list[int]:
    n = len(rectangles)
    edges = [set() for _ in range(n)]
    indegree = [0] * n

    for i in range(n):
        for j in range(i + 1, n):
            ci = int(rectangles[i]["color"])
            cj = int(rectangles[j]["color"])
            overlap = rectangles[i]["mask"] & rectangles[j]["mask"]
            assert np.any(overlap), "every rectangle pair must overlap"
            overlap_colors = {int(v) for v in np.unique(grid[overlap])}
            assert overlap_colors <= {ci, cj}, "pairwise overlap cannot be hidden by a third rectangle"
            assert len(overlap_colors) == 1, "one rectangle must consistently occlude the other"
            winner = ci if ci in overlap_colors else cj
            loser_idx, winner_idx = (j, i) if winner == ci else (i, j)
            if winner_idx not in edges[loser_idx]:
                edges[loser_idx].add(winner_idx)
                indegree[winner_idx] += 1

    order: list[int] = []
    frontier = [idx for idx, deg in enumerate(indegree) if deg == 0]
    while frontier:
        assert len(frontier) == 1, "occlusion clues must define a unique global order"
        cur = frontier.pop()
        order.append(cur)
        for nxt in sorted(edges[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)
    assert len(order) == n, "occlusion graph must be acyclic"
    return order


def _render(shape: tuple[int, int], bg: int, rectangles: list[dict[str, object]], order: list[int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for idx in order:
        rect = rectangles[idx]
        _draw_border(out, rect["bbox"], int(rect["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden z-order of overlapping hollow rectangles, then redraw them in the exact reverse order.

    Input structure:
    - The grid has a plain background and several differently colored hollow axis-aligned rectangles.
    - Rectangles overlap each other only through border crossings, so lower borders are broken where higher borders pass.
    - Every rectangle color is unique, and the overlap clues induce one global bottom-to-top order.

    Rule:
    - Reconstruct each rectangle from the visible cells of its color by taking the full border of that color's bounding box.
    - For every pair of rectangles, inspect their reconstructed border intersections; the visible color at those shared cells is the one that was above the other.
    - Build the full z-order from these pairwise occlusion clues, reverse that order, and redraw all original rectangle borders on the same background.

    Key insight:
    - The missing border segments are not noise: each break identifies which rectangle sat on top at that crossing, and reversing the whole hidden stack swaps every crossing color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    bg, rectangles = _extract_rectangles(g)
    bottom_to_top = _infer_bottom_to_top(g, rectangles)
    return _render(g.shape, bg, rectangles, list(reversed(bottom_to_top)))


def _sample_geometry() -> tuple[list[BBox], tuple[int, int]]:
    n_rects = random.randint(2, 4)
    step_r = random.randint(1, 2)
    step_c = random.randint(1, 2)
    max_shift_r = (n_rects - 1) * step_r
    max_shift_c = (n_rects - 1) * step_c

    rect_h = random.randint(max_shift_r + 5, min(14, max_shift_r + 9))
    rect_w = random.randint(max_shift_c + 5, min(14, max_shift_c + 9))

    sign_r = random.choice((-1, 1))
    sign_c = random.choice((-1, 1))
    offsets_r = [sign_r * step_r * i for i in range(n_rects)]
    offsets_c = [sign_c * step_c * i for i in range(n_rects)]

    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)

    min_off_r, max_off_r = min(offsets_r), max(offsets_r)
    min_off_c, max_off_c = min(offsets_c), max(offsets_c)
    height = rect_h + (max_off_r - min_off_r) + margin_top + margin_bottom
    width = rect_w + (max_off_c - min_off_c) + margin_left + margin_right
    assert height <= 30 and width <= 30

    base_r0 = margin_top - min_off_r
    base_c0 = margin_left - min_off_c
    rectangles = []
    for dr, dc in zip(offsets_r, offsets_c):
        r0 = base_r0 + dr
        c0 = base_c0 + dc
        rectangles.append((r0, c0, r0 + rect_h - 1, c0 + rect_w - 1))
    return rectangles, (height, width)


def _has_only_pairwise_crossings(rectangles: list[BBox], shape: tuple[int, int]) -> bool:
    masks = [_border_mask(shape, box) for box in rectangles]
    coverage = np.zeros(shape, dtype=int)
    for mask in masks:
        coverage += mask.astype(int)
    if np.any(coverage > 2):
        return False
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            if not np.any(masks[i] & masks[j]):
                return False
    return True


def _render_layered(shape: tuple[int, int], bg: int, rectangles: list[BBox], colors: list[int], order: list[int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for idx in order:
        _draw_border(out, rectangles[idx], colors[idx])
    return out


def _gen_attempt() -> np.ndarray | None:
    rectangles, shape = _sample_geometry()
    if not _has_only_pairwise_crossings(rectangles, shape):
        return None

    bg = 0
    palette = list(range(1, 10))
    random.shuffle(palette)
    colors = palette[: len(rectangles)]
    bottom_to_top = random.sample(range(len(rectangles)), k=len(rectangles))

    inp = _render_layered(shape, bg, rectangles, colors, bottom_to_top)
    expected = _render_layered(shape, bg, rectangles, colors, list(reversed(bottom_to_top)))

    visible_counts = [int(np.sum(inp == color)) for color in colors]
    if min(visible_counts) < 8:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

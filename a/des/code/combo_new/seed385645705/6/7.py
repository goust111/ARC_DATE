from __future__ import annotations

import random

import numpy as np


def _ring_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _paint_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _visible_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"color {color} must be visible"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return tuple(map(int, (r0, c0, r1, c1)))


def _recover_rings(
    grid: np.ndarray,
) -> tuple[list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray], dict[int, int]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != 0]
    assert len(colors) >= 2, "expected at least two rings"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    masks: dict[int, np.ndarray] = {}
    areas: dict[int, int] = {}
    widths: dict[int, int] = {}
    heights: dict[int, int] = {}

    for color in colors:
        bbox = _visible_bbox(grid, color)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "every ring must have a hollow interior"
        bboxes[color] = bbox
        masks[color] = _ring_mask(grid.shape, bbox)
        heights[color] = int(r1 - r0 + 1)
        widths[color] = int(c1 - c0 + 1)
        areas[color] = int(heights[color] * widths[color])

    assert len(set(heights.values())) == len(colors), "ring heights must be distinct"
    assert len(set(widths.values())) == len(colors), "ring widths must be distinct"
    assert len(set(areas.values())) == len(colors), "ring areas must be distinct"
    return colors, bboxes, masks, areas


def _stack_order(grid: np.ndarray, colors: list[int], masks: dict[int, np.ndarray]) -> list[int]:
    above: dict[int, set[int]] = {int(color): set() for color in colors}
    indegree: dict[int, int] = {int(color): 0 for color in colors}

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = masks[int(color_a)] & masks[int(color_b)]
            if not bool(overlap.any()):
                continue
            visible = {
                int(value)
                for value in np.unique(grid[overlap])
                if int(value) in {int(color_a), int(color_b)}
            }
            if not visible:
                continue
            assert len(visible) == 1, "every pairwise crossing must expose one top ring"
            top = visible.pop()
            under = int(color_b) if top == int(color_a) else int(color_a)
            if top not in above[under]:
                above[under].add(top)
                indegree[top] += 1

    ordered: list[int] = []
    remaining = set(int(color) for color in colors)
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1, "crossings must imply one total bottom-to-top order"
        color = int(ready[0])
        remaining.remove(color)
        ordered.append(color)
        for nxt in above[color]:
            indegree[nxt] -= 1
    return ordered


def _canonical_output(shape: tuple[int, int], colors_outer_to_inner: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=int)
    n = int(len(colors_outer_to_inner))
    center_r = h // 2
    center_c = w // 2
    assert center_r >= n and center_c >= n
    assert center_r + n < h and center_c + n < w

    for slot_index, color in enumerate(colors_outer_to_inner):
        radius = int(n - slot_index)
        bbox = (
            int(center_r - radius),
            int(center_c - radius),
            int(center_r + radius),
            int(center_c + radius),
        )
        _paint_ring(out, bbox, int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapped rectangular rings, read their occlusion order, and redraw them as centered concentric rings.

    Input structure:
    - Black background with several differently colored one-pixel-thick hollow rectangular rings.
    - Rings appear at arbitrary positions and partially overlap, with later-painted rings hiding earlier ones on shared border cells.
    - Each color keeps enough visible pixels that its full rectangle is still recoverable from the visible bbox.

    Rule:
    - For each non-black color, infer the ring's complete rectangle from the bbox of its visible pixels.
    - At every theoretical border crossing of two recovered rings, the visible color identifies which ring lies above the other; combine these pairwise relations into one bottom-to-top stacking order.
    - Sort the recovered rectangles by area to define the outer-to-inner physical slots, then reverse-map the stacking order onto those slots: the bottom ring color goes to the outermost slot and the top ring color goes to the innermost slot.
    - Clear the grid to black and redraw centered concentric one-pixel rings with equal spacing, using half-size ranks N, N-1, ..., 1 for N rings.

    Key insight:
    - The puzzle hides two independent orderings: geometry recovers the ring set and slot count, while visible crossings recover the occlusion hierarchy that is reassigned, in reverse, onto the centered canonical slots.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors, _bboxes, masks, areas = _recover_rings(g)
    bottom_to_top = _stack_order(g, colors, masks)

    # Once the size-ranked slots are established, the redraw uses canonical spacing.
    physical_slots = sorted(colors, key=lambda color: -areas[int(color)])
    colors_outer_to_inner = [int(mapped_color) for _slot, mapped_color in zip(physical_slots, bottom_to_top)]
    return _canonical_output(g.shape, colors_outer_to_inner)


# --------------- generator (o5x) ---------------

GRID_SIZE_CHOICES = (19, 21, 23, 25, 27, 29)
RING_COUNT_RANGE = (3, 5)
MIN_RING_SIDE = 8
GEN_RETRIES = 400
GEOMETRY_RETRIES = 80


def _sample_crossing_bboxes(
    rng: random.Random, h: int, w: int, n: int
) -> list[tuple[int, int, int, int]] | None:
    top_candidates = list(range(1, h // 2 - 3))
    bottom_candidates = list(range(h // 2 + 3, h - 1))
    left_candidates = list(range(1, w // 2 - 3))
    right_candidates = list(range(w // 2 + 3, w - 1))
    if min(len(top_candidates), len(bottom_candidates), len(left_candidates), len(right_candidates)) < n:
        return None

    for _ in range(GEOMETRY_RETRIES):
        tops = sorted(rng.sample(top_candidates, n))
        bottoms = sorted(rng.sample(bottom_candidates, n))
        lefts = sorted(rng.sample(left_candidates, n), reverse=True)
        rights = sorted(rng.sample(right_candidates, n), reverse=True)

        bboxes = [
            (int(tops[i]), int(lefts[i]), int(bottoms[i]), int(rights[i]))
            for i in range(n)
        ]
        heights = [int(r1 - r0 + 1) for r0, _c0, r1, _c1 in bboxes]
        widths = [int(c1 - c0 + 1) for _r0, c0, _r1, c1 in bboxes]
        areas = [int(hh * ww) for hh, ww in zip(heights, widths)]
        if min(heights) < MIN_RING_SIDE or min(widths) < MIN_RING_SIDE:
            continue
        if len(set(heights)) != n or len(set(widths)) != n or len(set(areas)) != n:
            continue

        masks = [_ring_mask((h, w), bbox) for bbox in bboxes]
        coverage = np.zeros((h, w), dtype=int)
        for mask in masks:
            coverage += mask.astype(int)
        if int(coverage.max()) > 2:
            continue

        good = True
        for i in range(n):
            for j in range(i + 1, n):
                overlap = masks[i] & masks[j]
                if int(overlap.sum()) < 2:
                    good = False
                    break
            if not good:
                break
        if good:
            return bboxes
    return None


def _apply_random_transform(rng: random.Random, grid: np.ndarray) -> np.ndarray:
    transforms = (
        lambda x: x,
        np.flipud,
        np.fliplr,
        lambda x: np.flipud(np.fliplr(x)),
        np.transpose,
        lambda x: np.flipud(np.transpose(x)),
        lambda x: np.fliplr(np.transpose(x)),
        lambda x: np.flipud(np.fliplr(np.transpose(x))),
    )
    transformed = transforms[rng.randrange(len(transforms))](grid)
    return np.array(transformed, dtype=int, copy=True)


def _validate_base_grid(
    grid: np.ndarray,
    rings: list[tuple[int, tuple[int, int, int, int]]],
    render_order: list[int],
) -> bool:
    try:
        for color, bbox in rings:
            if _visible_bbox(grid, int(color)) != tuple(map(int, bbox)):
                return False

        colors, _bboxes, masks, _areas = _recover_rings(grid)
        if set(colors) != {int(color) for color, _bbox in rings}:
            return False
        if _stack_order(grid, colors, masks) != [int(color) for color in render_order]:
            return False

        expected = _canonical_output(grid.shape, [int(color) for color in render_order])
        return bool(np.array_equal(main(grid), expected))
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = int(rng.randint(RING_COUNT_RANGE[0], RING_COUNT_RANGE[1]))
    h = int(rng.choice(GRID_SIZE_CHOICES))
    w = int(rng.choice(GRID_SIZE_CHOICES))
    if min(h, w) < 2 * n + 3:
        return None

    bboxes = _sample_crossing_bboxes(rng, h, w, n)
    if bboxes is None:
        return None

    colors = [int(color) for color in rng.sample(list(range(1, 10)), n)]
    rings = list(zip(colors, bboxes))
    render_rings = rings[:]
    rng.shuffle(render_rings)
    render_order = [int(color) for color, _bbox in render_rings]

    base = np.zeros((h, w), dtype=int)
    for color, bbox in render_rings:
        _paint_ring(base, bbox, int(color))
    if not _validate_base_grid(base, rings, render_order):
        return None

    transformed = _apply_random_transform(rng, base)
    expected = _canonical_output(transformed.shape, render_order)
    try:
        if not np.array_equal(main(transformed), expected):
            return None
    except Exception:
        return None
    return transformed


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")

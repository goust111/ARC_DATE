from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _within_bbox(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return (
        (coords[:, 0] >= r0)
        & (coords[:, 0] <= r1)
        & (coords[:, 1] >= c0)
        & (coords[:, 1] <= c1)
    )


def _descendants(colors: list[int], edges: dict[int, set[int]]) -> dict[int, set[int]]:
    memo: dict[int, set[int]] = {}

    def dfs(color: int, path: set[int]) -> set[int]:
        if color in memo:
            return memo[color]
        assert color not in path, "occlusion graph must be acyclic"
        seen: set[int] = set()
        for lower in edges[color]:
            seen.add(lower)
            seen |= dfs(lower, path | {color})
        memo[color] = seen
        return seen

    for color in colors:
        dfs(color, set())
    return memo


def _front_to_back_order(
    colors: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    edges: dict[int, set[int]],
) -> list[int]:
    descendants = _descendants(colors, edges)
    indegree = {color: 0 for color in colors}
    for color in colors:
        for lower in edges[color]:
            indegree[lower] += 1

    order: list[int] = []
    ready = [color for color in colors if indegree[color] == 0]
    while ready:
        ready.sort(
            key=lambda color: (
                -len(descendants[color]),
                bboxes[color][0],
                bboxes[color][1],
                -_bbox_shape(bboxes[color])[0] * _bbox_shape(bboxes[color])[1],
            )
        )
        color = ready.pop(0)
        order.append(color)
        for lower in sorted(edges[color]):
            indegree[lower] -= 1
            if indegree[lower] == 0:
                ready.append(lower)

    assert len(order) == len(colors), "every rectangle must receive a depth rank"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover occluded rectangles and lay them out from front to back.

    Input structure:
    - The background is Black.
    - Several distinct-colored solid rectangles overlap as opaque layers, so lower rectangles may only remain partially visible.
    - Each rectangle color appears at least once in the visible input.
    - The bottom row is a one-cell-high Grey divider spanning the full width.

    Rule:
    - Remove the Grey divider row.
    - For each visible non-background rectangle color, take the bbox of all visible pixels of that color; because every object is an axis-aligned rectangle, that bbox is the rectangle's full shape.
    - Infer layer order: if visible pixels of color A appear inside color B's bbox, then A is above B.
    - Repaint the recovered rectangles on a new Black canvas in front-to-back order from left to right, with one background column between neighbors and top-aligned placement.

    Key insight:
    The visible fragments already encode both missing geometry and depth: bbox extremes recover each whole rectangle, and pixels of one color inside another color's bbox reveal which rectangle sat on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 1
    assert np.all(g[-1, :] == DIVIDER), "expected a full-width Grey divider at the bottom"

    scene = g[:-1, :]
    colors = [int(color) for color in np.unique(scene) if int(color) not in (BG, DIVIDER)]
    assert colors, "expected at least one visible rectangle"

    coords_by_color: dict[int, np.ndarray] = {}
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(scene == color)
        assert coords.size > 0
        coords_by_color[color] = coords
        bboxes[color] = _bbox_from_coords(coords)

    edges = {color: set() for color in colors}
    for upper in colors:
        upper_coords = coords_by_color[upper]
        for lower in colors:
            if upper == lower:
                continue
            if bool(_within_bbox(upper_coords, bboxes[lower]).any()):
                edges[upper].add(lower)

    order = _front_to_back_order(colors, bboxes, edges)
    dims = [_bbox_shape(bboxes[color]) for color in order]
    out_h = max(h for h, _w in dims)
    out_w = sum(w for _h, w in dims) + len(dims) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for color, (height, width) in zip(order, dims):
        out[:height, col : col + width] = color
        col += width + 1
    return out


def _render_stack(
    body_shape: tuple[int, int],
    colors: list[int],
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray:
    body = np.full(body_shape, BG, dtype=int)
    for color, (r0, c0, r1, c1) in zip(reversed(colors), reversed(rects)):
        body[r0 : r1 + 1, c0 : c1 + 1] = color

    grid = np.full((body_shape[0] + 1, body_shape[1]), BG, dtype=int)
    grid[:-1, :] = body
    grid[-1, :] = DIVIDER
    return grid


def _render_expected(colors: list[int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    dims = [(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in rects]
    out_h = max(h for h, _w in dims)
    out_w = sum(w for _h, w in dims) + len(dims) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for color, (height, width) in zip(colors, dims):
        out[:height, col : col + width] = color
        col += width + 1
    return out


def _sample_nested_stack() -> tuple[list[tuple[int, int, int, int]], list[int], tuple[int, int]] | None:
    count = random.randint(2, 5)
    top_h = random.randint(1, 6)
    top_w = random.randint(1, 6)

    rects: list[tuple[int, int, int, int]] = [(0, 0, top_h - 1, top_w - 1)]
    for _ in range(1, count):
        prev_r0, prev_c0, prev_r1, prev_c1 = rects[-1]
        expand_top = random.randint(1, 3)
        expand_bottom = random.randint(1, 3)
        expand_left = random.randint(1, 3)
        expand_right = random.randint(1, 3)
        rects.append(
            (
                prev_r0 - expand_top,
                prev_c0 - expand_left,
                prev_r1 + expand_bottom,
                prev_c1 + expand_right,
            )
        )

    min_r = min(r0 for r0, _c0, _r1, _c1 in rects)
    min_c = min(c0 for _r0, c0, _r1, _c1 in rects)
    shifted = [
        (r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c)
        for r0, c0, r1, c1 in rects
    ]

    outer_h = max(r1 for _r0, _c0, r1, _c1 in shifted) + 1
    outer_w = max(c1 for _r0, _c0, _r1, c1 in shifted) + 1
    widths = [c1 - c0 + 1 for _r0, c0, _r1, c1 in shifted]
    output_w = sum(widths) + len(widths) - 1
    if outer_h + 1 > ARC_MAX or outer_w > ARC_MAX or output_w > ARC_MAX:
        return None

    pad_top = random.randint(0, min(3, ARC_MAX - 1 - outer_h))
    pad_left = random.randint(0, min(3, ARC_MAX - outer_w))
    pad_bottom = random.randint(0, min(3, ARC_MAX - 1 - outer_h - pad_top))
    pad_right = random.randint(0, min(3, ARC_MAX - outer_w - pad_left))

    body_h = outer_h + pad_top + pad_bottom
    body_w = outer_w + pad_left + pad_right
    if body_h + 1 > ARC_MAX or body_w > ARC_MAX:
        return None

    placed = [
        (r0 + pad_top, c0 + pad_left, r1 + pad_top, c1 + pad_left)
        for r0, c0, r1, c1 in shifted
    ]
    colors = random.sample(list(RECT_COLORS), len(placed))
    return placed, colors, (body_h, body_w)


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_nested_stack()
    if sampled is None:
        return None

    rects, colors, body_shape = sampled
    inp = _render_stack(body_shape, colors, rects)
    expected = _render_expected(colors, rects)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-rectangle scene")

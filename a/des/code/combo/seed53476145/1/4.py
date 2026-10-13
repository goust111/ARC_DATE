from __future__ import annotations

import random

import numpy as np


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one visible pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersection(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int]:
    indegree = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indegree[dst] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1, "layer order must be uniquely determined"
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for dst in edges[color]:
            indegree[dst] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each colored rectangle, infer the occlusion order, and emit the colors
    from bottom to top as a single output column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert len(colors) >= 2, "expected at least two rectangles"

    bboxes = {color: _bbox_from_mask(g == color) for color in colors}
    above: dict[int, set[int]] = {color: set() for color in colors}

    for i, color_a in enumerate(colors):
        bbox_a = bboxes[color_a]
        for color_b in colors[i + 1 :]:
            bbox_b = bboxes[color_b]
            overlap = _intersection(bbox_a, bbox_b)
            if overlap is None:
                continue

            r0, c0, r1, c1 = overlap
            window = g[r0 : r1 + 1, c0 : c1 + 1]
            a_visible = bool(np.any(window == color_a))
            b_visible = bool(np.any(window == color_b))
            assert not (a_visible and b_visible), "pairwise overlap implies one fixed winner"

            if a_visible:
                above[color_b].add(color_a)
            elif b_visible:
                above[color_a].add(color_b)

    ordered = _topological_order(colors, above)
    return np.array(ordered, dtype=int).reshape(-1, 1)


def _render_scene(
    rects: list[tuple[int, int, int, int]], colors: list[int], shape: tuple[int, int]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _has_overlap(rects: list[tuple[int, int, int, int]]) -> bool:
    for i, rect_a in enumerate(rects):
        for rect_b in rects[i + 1 :]:
            if _intersection(rect_a, rect_b) is not None:
                return True
    return False


def _visible_sides_ok(
    grid: np.ndarray, rect: tuple[int, int, int, int], color: int
) -> bool:
    r0, c0, r1, c1 = rect
    mask = grid == color
    if not bool(mask.any()):
        return False
    if _bbox_from_mask(mask) != rect:
        return False
    coords = np.argwhere(mask)
    return bool(
        np.any(coords[:, 0] == r0)
        and np.any(coords[:, 0] == r1)
        and np.any(coords[:, 1] == c0)
        and np.any(coords[:, 1] == c1)
    )


def _validate_scene(
    grid: np.ndarray, rects: list[tuple[int, int, int, int]], colors: list[int]
) -> bool:
    if not _has_overlap(rects):
        return False

    for color, rect in zip(colors, rects):
        if not _visible_sides_ok(grid, rect, color):
            return False

    expected = np.array(colors, dtype=int).reshape(-1, 1)
    try:
        return np.array_equal(main(grid), expected)
    except AssertionError:
        return False


def _shift_rects(
    rects: list[tuple[int, int, int, int]], pad_top: int, pad_left: int
) -> list[tuple[int, int, int, int]]:
    min_r = min(rect[0] for rect in rects)
    min_c = min(rect[1] for rect in rects)
    dr = pad_top - min_r
    dc = pad_left - min_c
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _nested_rects(n_layers: int) -> list[tuple[int, int, int, int]] | None:
    top_h = random.randint(4, 7)
    top_w = random.randint(4, 7)
    current = (0, 0, top_h - 1, top_w - 1)
    rects_top_to_bottom = [current]

    for _ in range(n_layers - 1):
        expand_top = random.randint(1, 3)
        expand_bottom = random.randint(1, 3)
        expand_left = random.randint(1, 3)
        expand_right = random.randint(1, 3)
        current = (
            current[0] - expand_top,
            current[1] - expand_left,
            current[2] + expand_bottom,
            current[3] + expand_right,
        )
        rects_top_to_bottom.append(current)

    rects = list(reversed(rects_top_to_bottom))
    rects = _shift_rects(rects, random.randint(1, 3), random.randint(1, 3))
    max_r = max(rect[2] for rect in rects)
    max_c = max(rect[3] for rect in rects)
    if max_r + 2 > 29 or max_c + 2 > 29:
        return None
    return rects


def _sample_anchor(rect: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = rect
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    return random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1)


def _cross_chain_rects(n_layers: int) -> list[tuple[int, int, int, int]] | None:
    orientation = random.choice(("H", "V"))
    thickness = random.randint(3, 4)
    length = random.randint(8, 13)

    if orientation == "H":
        rects = [(0, 0, thickness - 1, length - 1)]
    else:
        rects = [(0, 0, length - 1, thickness - 1)]

    for _ in range(1, n_layers):
        prev = rects[-1]
        anchor = _sample_anchor(prev)
        if anchor is None:
            return None
        ar, ac = anchor
        orientation = "V" if orientation == "H" else "H"
        thickness = random.randint(3, 4)
        length = random.randint(7, 13)

        if orientation == "H":
            if length < 6:
                return None
            r0 = ar - random.randint(1, thickness - 2)
            c0 = ac - random.randint(2, length - 3)
            rect = (r0, c0, r0 + thickness - 1, c0 + length - 1)
        else:
            if length < 6:
                return None
            r0 = ar - random.randint(2, length - 3)
            c0 = ac - random.randint(1, thickness - 2)
            rect = (r0, c0, r0 + length - 1, c0 + thickness - 1)
        rects.append(rect)

    rects = _shift_rects(rects, random.randint(1, 4), random.randint(1, 4))
    max_r = max(rect[2] for rect in rects)
    max_c = max(rect[3] for rect in rects)
    if max_r + 2 > 29 or max_c + 2 > 29:
        return None
    return rects


def _generate_attempt() -> np.ndarray | None:
    n_layers = random.randint(2, 5)
    colors = random.sample(range(1, 10), n_layers)

    if random.random() < 0.5:
        rects = _nested_rects(n_layers)
    else:
        rects = _cross_chain_rects(n_layers)
    if rects is None:
        return None

    height = max(rect[2] for rect in rects) + random.randint(2, 4)
    width = max(rect[3] for rect in rects) + random.randint(2, 4)
    if height > 30 or width > 30:
        return None

    grid = _render_scene(rects, colors, (height, width))
    if not _validate_scene(grid, rects, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-layer input")

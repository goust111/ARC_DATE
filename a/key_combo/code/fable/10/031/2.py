from __future__ import annotations

from collections import deque
import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden rectangles and summarize their hidden areas as a depth-ordered histogram.

    Input structure:
    - A black background contains several solid axis-aligned rectangles, each in a distinct color.
    - The rectangles overlap as a front-to-back stack, so only the frontmost color is visible at each cell.
    - Every rectangle still shows two opposite corners, which lets its full bounding box be recovered from
      the visible pixels of that color.

    Rule:
    - For each visible color, recover the rectangle's full extent as the bounding box of all visible pixels
      of that color.
    - If some visible pixels of color B lie inside the recovered extent of rectangle A, then B is in front of A.
      Use these visible overlap relations to recover the unique total depth order from front to back.
    - For each rectangle, count how many cells of its full bounding box are hidden by rectangles in front of it.
    - Return a black grid whose width is the number of rectangles and whose height is the largest hidden-cell
      count; column k corresponds to the kth rectangle in front-to-back order and is filled from the top with
      that rectangle's color for exactly its hidden-cell count.

    Key insight:
    - The visible fragments reveal each full rectangle through its diagonal corners, and the colors appearing
      inside another rectangle's reconstructed bbox reveal depth order; the histogram counts the invisible part,
      not the visible one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = 0
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one rectangle"

    rects = []
    for color in colors:
        points = np.argwhere(g == color)
        assert len(points) > 0
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        rects.append(
            {
                "color": color,
                "points": points,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "visible": int(len(points)),
            }
        )

    order = _depth_order(rects)
    hidden_counts = []
    for rect in order:
        r0, c0, r1, c1 = rect["bbox"]
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        hidden = area - rect["visible"]
        assert hidden >= 0, "visible pixels cannot exceed recovered area"
        hidden_counts.append(hidden)

    max_hidden = max(hidden_counts)
    assert max_hidden > 0, "expected at least one occluded rectangle"

    out = np.full((max_hidden, len(order)), bg, dtype=int)
    for col, (rect, hidden) in enumerate(zip(order, hidden_counts)):
        if hidden > 0:
            out[:hidden, col] = int(rect["color"])
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid遮挡深度直方图 input")


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    step = random.choice((3, 4))
    max_height = 6 if step == 3 else 7
    heights = [random.randint(step + 1, max_height) for _ in range(count)]
    widths = [random.randint(6, 9) for _ in range(count)]

    base_h = max(18, count * step + max(heights) + 3)
    base_h = min(base_h, 24)
    base_w = random.randint(14, 20)
    if max(widths) + 7 > base_w:
        return None

    start_row = random.randint(1, 3)
    left_max = max(2, base_w // 4)
    right_min = max(left_max + 2, base_w // 2 - 1)
    right_max = min(base_w - max(widths), right_min + 3)
    if right_max < right_min:
        return None

    start_side = random.randint(0, 1)
    rects = []
    for idx in range(count):
        side = (start_side + idx) % 2
        height = heights[idx]
        width = widths[idx]
        row0 = start_row + idx * step + random.randint(0, 1)
        if row0 + height > base_h:
            return None
        if side == 0:
            col0 = random.randint(0, left_max)
        else:
            col0 = random.randint(right_min, right_max)
        if col0 + width > base_w:
            return None
        rects.append((row0, col0, row0 + height - 1, col0 + width - 1))

    if not _valid_overlap_chain(rects):
        return None

    transform_id = random.randrange(8)
    rects = _transform_rects(rects, base_h, base_w, transform_id)
    rects = _pack_with_margin(rects)

    colors = random.sample(range(1, 10), count)
    grid = _render_stack(rects, colors)
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None

    if not _visible_bbox_matches(grid, rects, colors):
        return None
    if not all(_has_visible_opposite_corners(grid, rect, color) for rect, color in zip(rects, colors)):
        return None

    try:
        expected = _expected_histogram(rects, colors)
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    hidden_counts = [int(np.count_nonzero(expected[:, col] == colors[col])) for col in range(expected.shape[1])]
    if hidden_counts == sorted(hidden_counts) or hidden_counts == sorted(hidden_counts, reverse=True):
        return None

    areas = [_area(rect) for rect in rects]
    if areas == sorted(areas) or areas == sorted(areas, reverse=True):
        return None

    return grid


def _depth_order(rects: list[dict]) -> list[dict]:
    count = len(rects)
    edges = [set() for _ in range(count)]
    indegree = [0] * count

    for back_idx, back in enumerate(rects):
        br0, bc0, br1, bc1 = back["bbox"]
        for front_idx, front in enumerate(rects):
            if front_idx == back_idx:
                continue
            if not _overlap(back["bbox"], front["bbox"]):
                continue
            points = front["points"]
            inside = (
                (points[:, 0] >= br0)
                & (points[:, 0] <= br1)
                & (points[:, 1] >= bc0)
                & (points[:, 1] <= bc1)
            )
            if bool(inside.any()) and back_idx not in edges[front_idx]:
                edges[front_idx].add(back_idx)
                indegree[back_idx] += 1

    queue = deque(idx for idx, deg in enumerate(indegree) if deg == 0)
    order = []
    while queue:
        assert len(queue) == 1, "depth order must be uniquely determined"
        idx = queue.popleft()
        order.append(rects[idx])
        for nxt in sorted(edges[idx]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    assert len(order) == count, "occlusion graph must form a complete chain"
    return order


def _valid_overlap_chain(rects: list[tuple[int, int, int, int]]) -> bool:
    for idx in range(len(rects) - 1):
        if not _overlap(rects[idx], rects[idx + 1]):
            return False
    for i in range(len(rects)):
        for j in range(i + 2, len(rects)):
            if _overlap(rects[i], rects[j]):
                return False
    return True


def _render_stack(rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    height = 1 + max(rect[2] for rect in rects)
    width = 1 + max(rect[3] for rect in rects)
    grid = np.zeros((height, width), dtype=int)
    for rect, color in zip(reversed(rects), reversed(colors)):
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _expected_histogram(rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    hidden_counts = []
    occluded = np.zeros((1 + max(r1 for _, _, r1, _ in rects), 1 + max(c1 for _, _, _, c1 in rects)), dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect
        region = occluded[r0 : r1 + 1, c0 : c1 + 1]
        hidden_counts.append(int(np.count_nonzero(region)))
        region[:, :] = True
    max_hidden = max(hidden_counts)
    out = np.zeros((max_hidden, len(rects)), dtype=int)
    for col, (color, hidden) in enumerate(zip(colors, hidden_counts)):
        if hidden > 0:
            out[:hidden, col] = int(color)
    return out


def _visible_bbox_matches(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    colors: list[int],
) -> bool:
    for rect, color in zip(rects, colors):
        points = np.argwhere(grid == color)
        if len(points) == 0:
            return False
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != rect:
            return False
    return True


def _has_visible_opposite_corners(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> bool:
    r0, c0, r1, c1 = rect
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    visible = {point for point in corners if int(grid[point]) == int(color)}
    return ({corners[0], corners[3]} <= visible) or ({corners[1], corners[2]} <= visible)


def _pack_with_margin(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    min_r = min(rect[0] for rect in rects)
    min_c = min(rect[1] for rect in rects)
    max_r = max(rect[2] for rect in rects)
    max_c = max(rect[3] for rect in rects)

    margin_top = random.randint(1, 3)
    margin_left = random.randint(1, 3)

    packed = []
    for r0, c0, r1, c1 in rects:
        packed.append(
            (
                r0 - min_r + margin_top,
                c0 - min_c + margin_left,
                r1 - min_r + margin_top,
                c1 - min_c + margin_left,
            )
        )

    height = 1 + max(rect[2] for rect in packed) + random.randint(1, 3)
    width = 1 + max(rect[3] for rect in packed) + random.randint(1, 3)
    assert height <= 30 and width <= 30
    return packed


def _transform_rects(
    rects: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    transform_id: int,
) -> list[tuple[int, int, int, int]]:
    transformed = []
    for rect in rects:
        corners = np.array(
            [
                (rect[0], rect[1]),
                (rect[0], rect[3]),
                (rect[2], rect[1]),
                (rect[2], rect[3]),
            ],
            dtype=int,
        )
        mapped = np.array([_transform_point(tuple(point), height, width, transform_id) for point in corners], dtype=int)
        r0, c0 = mapped.min(axis=0)
        r1, c1 = mapped.max(axis=0)
        transformed.append((int(r0), int(c0), int(r1), int(c1)))
    return transformed


def _transform_point(
    point: tuple[int, int],
    height: int,
    width: int,
    transform_id: int,
) -> tuple[int, int]:
    r, c = point
    if transform_id == 0:
        return r, c
    if transform_id == 1:
        return c, height - 1 - r
    if transform_id == 2:
        return height - 1 - r, width - 1 - c
    if transform_id == 3:
        return width - 1 - c, r
    if transform_id == 4:
        return r, width - 1 - c
    if transform_id == 5:
        return height - 1 - r, c
    if transform_id == 6:
        return c, r
    if transform_id == 7:
        return width - 1 - c, height - 1 - r
    raise ValueError(f"unknown transform_id {transform_id}")


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _area(rect: tuple[int, int, int, int]) -> int:
    return (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)

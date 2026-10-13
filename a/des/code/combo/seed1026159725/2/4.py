import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _infer_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rects = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "each foreground color must remain visible"
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        rects.append(
            {
                "color": color,
                "coords": coords,
                "bbox": bbox,
                "visible_area": int(len(coords)),
                "occluded_area": int(area - len(coords)),
            }
        )
    rects.sort(key=lambda rect: rect["bbox"])
    return rects


def _bottom_to_top_order(grid: np.ndarray, rects: list[dict]) -> list[int]:
    n = len(rects)
    edges = [set() for _ in range(n)]
    indeg = [0] * n
    by_color = {int(rect["color"]): idx for idx, rect in enumerate(rects)}

    for rect in rects:
        color = int(rect["color"])
        top_idx = by_color[color]
        for r, c in rect["coords"]:
            point = (int(r), int(c))
            for lower_idx, lower in enumerate(rects):
                if lower_idx == top_idx:
                    continue
                if _inside_bbox(point, lower["bbox"]) and top_idx not in edges[lower_idx]:
                    edges[lower_idx].add(top_idx)
                    indeg[top_idx] += 1

    ready = [
        idx
        for idx in range(n)
        if indeg[idx] == 0
    ]
    ready.sort(key=lambda idx: rects[idx]["bbox"])

    order = []
    queue = deque(ready)
    while queue:
        idx = queue.popleft()
        order.append(idx)
        released = []
        for nxt in sorted(edges[idx], key=lambda j: rects[j]["bbox"]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                released.append(nxt)
        released.sort(key=lambda idx: rects[idx]["bbox"])
        queue.extend(released)

    if len(order) != n:
        return list(range(n))
    return order


def _paint(shape: tuple[int, int], bg: int, rects: list[dict], order: list[int]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rects[idx]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the stacking priority of overlapping rectangles by promoting the most occluded rectangle.

    Input structure:
    - The background is the most common color.
    - Each non-background color belongs to one solid axis-aligned rectangle.
    - Rectangles may overlap, and only the top color remains visible in overlap cells.
    - Every rectangle keeps at least one visible pixel on each side, so its full bbox is recoverable.

    Rule:
    - Reconstruct each rectangle's bbox from its visible pixels.
    - Occluded area = bbox area - visible area.
    - Redraw the rectangles so larger occluded area means a higher final layer.

    Key insight:
    - Hiddenness, not color or size, determines the new z-order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    rects = _infer_rectangles(g, bg)
    base_order = _bottom_to_top_order(g, rects)
    base_rank = {idx: rank for rank, idx in enumerate(base_order)}

    redraw_order = sorted(
        range(len(rects)),
        key=lambda idx: (
            int(rects[idx]["occluded_area"]),
            int(base_rank.get(idx, idx)),
            rects[idx]["bbox"],
        ),
    )
    return _paint(g.shape, bg, rects, redraw_order)


_GEN_RETRIES = 500


def _sample_core_rectangles() -> tuple[np.ndarray, list[dict], list[int], int] | None:
    n = random.randint(3, 4)
    bg = random.randint(0, 9)
    colors = random.sample([c for c in range(10) if c != bg], n)

    core_h = random.randint(2, 4)
    core_w = random.randint(2, 4)
    corners = ["TL", "TR", "BR", "BL"]
    step = random.choice([1, -1])
    start = random.randrange(len(corners))
    patterns = [corners[(start + step * i) % len(corners)] for i in range(n)]

    side_specs = []
    for idx in range(n):
        depth = n - 1 - idx
        long_a = random.randint(4 + depth, 6 + depth)
        long_b = random.randint(4 + depth, 6 + depth)
        short_a = random.randint(1 + depth // 2, 2 + depth)
        short_b = random.randint(1 + depth // 2, 2 + depth)
        pattern = patterns[idx]
        if pattern == "TL":
            top, bottom, left, right = long_a, short_a, long_b, short_b
        elif pattern == "TR":
            top, bottom, left, right = long_a, short_a, short_b, long_b
        elif pattern == "BR":
            top, bottom, left, right = short_a, long_a, short_b, long_b
        else:
            top, bottom, left, right = short_a, long_a, long_b, short_b
        side_specs.append((top, bottom, left, right))

    max_top = max(spec[0] for spec in side_specs)
    max_bottom = max(spec[1] for spec in side_specs)
    max_left = max(spec[2] for spec in side_specs)
    max_right = max(spec[3] for spec in side_specs)

    h_min = core_h + max_top + max_bottom + 3
    w_min = core_w + max_left + max_right + 3
    h = random.randint(max(17, h_min), max(17, h_min) + 4)
    w = random.randint(max(17, w_min), max(17, w_min) + 4)
    core_r0 = random.randint(max_top + 1, h - core_h - max_bottom - 2)
    core_c0 = random.randint(max_left + 1, w - core_w - max_right - 2)
    core_r1 = core_r0 + core_h - 1
    core_c1 = core_c0 + core_w - 1

    rects = []
    seen_bboxes = set()
    for idx in range(n):
        top_pad, bottom_pad, left_pad, right_pad = side_specs[idx]
        r0 = core_r0 - top_pad
        c0 = core_c0 - left_pad
        r1 = core_r1 + bottom_pad
        c1 = core_c1 + right_pad
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            return None
        bbox = (int(r0), int(c0), int(r1), int(c1))
        if bbox in seen_bboxes:
            return None
        seen_bboxes.add(bbox)
        rects.append({"color": int(colors[idx]), "bbox": bbox})

    order = list(range(n))
    grid = np.full((h, w), bg, dtype=int)
    for idx in order:
        r0, c0, r1, c1 = rects[idx]["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rects[idx]["color"])

    return grid, rects, order, bg


def _scene_stats(
    grid: np.ndarray, rects: list[dict], order: list[int], bg: int
) -> tuple[bool, list[int], np.ndarray]:
    visible_counts = {}
    inferred = _infer_rectangles(grid, bg)
    inferred_by_color = {int(rect["color"]): rect for rect in inferred}

    occluded = []
    any_hidden = False
    for rect in rects:
        color = int(rect["color"])
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return False, [], grid
        vis_bbox = _bbox_from_coords(coords)
        if vis_bbox != rect["bbox"]:
            return False, [], grid
        r0, c0, r1, c1 = rect["bbox"]
        visible_counts[color] = int(len(coords))
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        hidden = int(area - len(coords))
        occluded.append(hidden)
        any_hidden |= hidden > 0

        side_checks = [
            np.any(grid[r0, c0 : c1 + 1] == color),
            np.any(grid[r1, c0 : c1 + 1] == color),
            np.any(grid[r0 : r1 + 1, c0] == color),
            np.any(grid[r0 : r1 + 1, c1] == color),
        ]
        if not all(bool(x) for x in side_checks):
            return False, [], grid

        if inferred_by_color[color]["occluded_area"] != hidden:
            return False, [], grid

    if not any_hidden:
        return False, [], grid
    if len(set(occluded)) != len(occluded):
        return False, [], grid
    if list(occluded) != sorted(occluded, reverse=True):
        return False, [], grid
    for i, rect_i in enumerate(rects):
        a0, b0, a1, b1 = rect_i["bbox"]
        for j, rect_j in enumerate(rects):
            if i == j:
                continue
            c0, d0, c1, d1 = rect_j["bbox"]
            if a0 <= c0 and b0 <= d0 and a1 >= c1 and b1 >= d1:
                return False, [], grid
    if _get_bg(grid) != bg:
        return False, [], grid
    if np.count_nonzero(grid != bg) >= grid.size // 2:
        return False, [], grid

    expected_order = sorted(
        range(len(rects)),
        key=lambda idx: (occluded[idx], idx),
    )
    expected = np.full(grid.shape, bg, dtype=int)
    for idx in expected_order:
        r0, c0, r1, c1 = rects[idx]["bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = int(rects[idx]["color"])

    visible_output_colors = set(int(x) for x in np.unique(expected) if int(x) != bg)
    if len(visible_output_colors) < min(len(rects), 3):
        return False, [], grid
    if np.array_equal(grid, expected):
        return False, [], grid

    return True, occluded, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        sample = _sample_core_rectangles()
        if sample is None:
            continue
        grid, rects, order, bg = sample
        ok, _occluded, expected = _scene_stats(grid, rects, order, bg)
        if not ok:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid rectangle-stacking input")

from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_SIDE = 30

# Generator configuration.
WORKSPACE_SHAPE = (24, 24)
RECT_COUNT_RANGE = (2, 5)
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 8
RECT_AREA_MIN = 9
RECT_AREA_MAX = 28
MIN_OVERLAP_SPAN = 2
PLACEMENT_ATTEMPTS = 60
GENERATION_ATTEMPTS = 600


def _scan_order_colors(grid: np.ndarray, bg: int = BG) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    for value in map(int, grid.ravel()):
        if value == bg or value in seen:
            continue
        seen.add(value)
        colors.append(value)
    return colors


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_rectangles(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    rects: list[dict[str, object]] = []
    for color in _scan_order_colors(grid, bg):
        mask = grid == int(color)
        assert bool(mask.any()), "each non-background color must remain visible"
        bbox = _bbox_from_mask(mask)
        r0, c0, r1, c1 = bbox
        rects.append(
            {
                "color": int(color),
                "bbox": bbox,
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                "anchor": (int(r0), int(c0)),
            }
        )
    return rects


def _intersection(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(int(bbox_a[0]), int(bbox_b[0]))
    c0 = max(int(bbox_a[1]), int(bbox_b[1]))
    r1 = min(int(bbox_a[2]), int(bbox_b[2]))
    c1 = min(int(bbox_a[3]), int(bbox_b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _infer_occlusion_edges(
    grid: np.ndarray, rects: list[dict[str, object]]
) -> tuple[dict[int, set[int]], dict[tuple[int, int], int]]:
    n = len(rects)
    edges = {i: set() for i in range(n)}
    supports: dict[tuple[int, int], int] = {}

    for i in range(n):
        color_i = int(rects[i]["color"])
        bbox_i = rects[i]["bbox"]
        for j in range(i + 1, n):
            color_j = int(rects[j]["color"])
            bbox_j = rects[j]["bbox"]
            inter = _intersection(bbox_i, bbox_j)
            if inter is None:
                continue

            r0, c0, r1, c1 = inter
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            sees_i = bool(np.any(window == color_i))
            sees_j = bool(np.any(window == color_j))

            if sees_i and sees_j:
                raise AssertionError("overlap cannot show both rectangles at once")
            if not sees_i and not sees_j:
                continue

            if sees_i:
                upper, lower, upper_color = i, j, color_i
            else:
                upper, lower, upper_color = j, i, color_j

            if lower not in edges[upper]:
                edges[upper].add(lower)
                supports[(upper, lower)] = int(np.count_nonzero(window == upper_color))

    return edges, supports


def _descendant_counts(edges: dict[int, set[int]]) -> list[int]:
    memo: dict[int, set[int]] = {}
    active: set[int] = set()

    def visit(node: int) -> set[int]:
        if node in memo:
            return memo[node]
        if node in active:
            raise AssertionError("occlusion graph must be acyclic")
        active.add(node)
        reach: set[int] = set()
        for nxt in edges[node]:
            reach.add(nxt)
            reach |= visit(nxt)
        active.remove(node)
        memo[node] = reach
        return reach

    return [len(visit(i)) for i in range(len(edges))]


def _topological_order(
    rects: list[dict[str, object]], edges: dict[int, set[int]], supports: dict[tuple[int, int], int]
) -> tuple[list[int], bool]:
    n = len(rects)
    indeg = [0] * n
    for src, dsts in edges.items():
        for dst in dsts:
            indeg[dst] += 1

    descendant_counts = _descendant_counts(edges)
    support_strength = [0] * n
    for (src, _dst), weight in supports.items():
        support_strength[src] += int(weight)

    available = [i for i in range(n) if indeg[i] == 0]
    order: list[int] = []
    unique = True

    while available:
        if len(available) > 1:
            unique = False
        available.sort(
            key=lambda idx: (
                -descendant_counts[idx],
                -support_strength[idx],
                rects[idx]["anchor"][0],
                rects[idx]["anchor"][1],
            )
        )
        current = available.pop(0)
        order.append(current)
        for nxt in edges[current]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                available.append(nxt)

    if len(order) != n:
        raise AssertionError("failed to recover a complete layer order")
    return order, unique


def _solve(grid: np.ndarray) -> tuple[list[dict[str, object]], list[int], bool]:
    rects = _extract_rectangles(grid, bg=BG)
    assert rects, "expected at least one rectangle"
    edges, supports = _infer_occlusion_edges(grid, rects)
    order, unique = _topological_order(rects, edges, supports)
    return rects, order, unique


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially occluded rectangle, turn its full area into a one-row strip, and stack the strips by depth.

    Input structure:
    - The background is black.
    - Each non-black color belongs to one solid axis-aligned rectangle, but higher rectangles may hide part of lower rectangles.
    - Rectangle colors are unique, so each visible color fragment identifies exactly one rectangle.

    Rule:
    - For each color, infer the rectangle's full bounding box from the visible extremal rows and columns, then compute its area as height times width.
    - Compare reconstructed rectangles pairwise: whenever two boxes overlap, whichever color is still visible inside the shared region is the upper rectangle.
    - Topologically sort these occlusion relations from front to back.
    - Output one left-aligned row per rectangle in that depth order; each row is a solid strip of the rectangle's color whose length equals its area, padded with black to the longest strip.

    Key insight:
    The visible fragment of each color still exposes the rectangle's complete row span and column span, so geometric recovery and depth inference can be solved separately and then combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects, order, _unique = _solve(g)
    ordered_rects = [rects[idx] for idx in order]
    max_area = max(int(rect["area"]) for rect in ordered_rects)

    out = np.full((len(ordered_rects), max_area), BG, dtype=int)
    for out_r, rect in enumerate(ordered_rects):
        out[out_r, : int(rect["area"])] = int(rect["color"])
    return out


def _sample_dims() -> tuple[int, int]:
    mode = random.choice(("square", "wide", "tall"))
    for _ in range(80):
        if mode == "square":
            side = random.randint(RECT_SIDE_MIN, min(RECT_SIDE_MAX, 6))
            h = side
            w = side
        elif mode == "wide":
            h = random.randint(RECT_SIDE_MIN, 5)
            w = random.randint(max(h + 1, RECT_SIDE_MIN + 1), RECT_SIDE_MAX)
        else:
            w = random.randint(RECT_SIDE_MIN, 5)
            h = random.randint(max(w + 1, RECT_SIDE_MIN + 1), RECT_SIDE_MAX)

        area = h * w
        if RECT_AREA_MIN <= area <= RECT_AREA_MAX:
            return int(h), int(w)
        mode = random.choice(("square", "wide", "tall"))

    raise RuntimeError("failed to sample rectangle dimensions")


def _sample_bottom_rect() -> tuple[int, int, int, int]:
    h, w = _sample_dims()
    max_r = WORKSPACE_SHAPE[0] - h - 2
    max_c = WORKSPACE_SHAPE[1] - w - 2
    r0 = random.randint(2, max(2, max_r))
    c0 = random.randint(2, max(2, max_c))
    return int(r0), int(c0), int(h), int(w)


def _sample_overlap_position(
    lower_rect: tuple[int, int, int, int], height: int, width: int
) -> tuple[int, int] | None:
    lr0, lc0, lh, lw = lower_rect
    lr1 = lr0 + lh - 1
    lc1 = lc0 + lw - 1

    row_lo = max(0, lr0 - height + MIN_OVERLAP_SPAN)
    row_hi = min(WORKSPACE_SHAPE[0] - height, lr1 - MIN_OVERLAP_SPAN + 1)
    col_lo = max(0, lc0 - width + MIN_OVERLAP_SPAN)
    col_hi = min(WORKSPACE_SHAPE[1] - width, lc1 - MIN_OVERLAP_SPAN + 1)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        if (r0, c0, height, width) != lower_rect:
            return int(r0), int(c0)
    return None


def _render(rects: list[dict[str, int]], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])
        color = int(rect["color"])
        grid[r0 : r0 + h, c0 : c0 + w] = color
    return grid


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    top_room = r0
    left_room = c0
    bottom_room = grid.shape[0] - 1 - r1
    right_room = grid.shape[1] - 1 - c1

    top = random.randint(1 if top_room > 0 else 0, min(3, top_room))
    left = random.randint(1 if left_room > 0 else 0, min(3, left_room))
    bottom = random.randint(1 if bottom_room > 0 else 0, min(3, bottom_room))
    right = random.randint(1 if right_room > 0 else 0, min(3, right_room))

    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _augment(grid: np.ndarray) -> np.ndarray:
    transforms = (
        lambda x: x,
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        lambda x: np.flipud(np.fliplr(x)),
        lambda x: x.T,
        lambda x: np.flipud(x.T),
        lambda x: np.fliplr(x.T),
        lambda x: np.flipud(np.fliplr(x.T)),
    )
    return np.asarray(random.choice(transforms)(grid), dtype=int)


def _expected_output(rects: list[dict[str, int]]) -> np.ndarray:
    ordered = list(reversed(rects))
    max_area = max(int(rect["h"]) * int(rect["w"]) for rect in ordered)
    out = np.full((len(ordered), max_area), BG, dtype=int)
    for r, rect in enumerate(ordered):
        out[r, : int(rect["h"]) * int(rect["w"])] = int(rect["color"])
    return out


def _visible_bboxes_match(grid: np.ndarray, rects: list[dict[str, int]]) -> bool:
    for rect in rects:
        color = int(rect["color"])
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            return False
        bbox = _bbox_from_mask(grid == color)
        expected = (
            int(rect["r0"]),
            int(rect["c0"]),
            int(rect["r0"]) + int(rect["h"]) - 1,
            int(rect["c0"]) + int(rect["w"]) - 1,
        )
        if bbox != expected:
            return False
    return True


def _attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    colors = random.sample(PALETTE, count)

    rect_specs: list[tuple[int, int, int, int]] = [_sample_bottom_rect()]
    for _ in range(1, count):
        h, w = _sample_dims()
        pos = _sample_overlap_position(rect_specs[-1], h, w)
        if pos is None:
            return None
        rect_specs.append((pos[0], pos[1], h, w))

    rects = [
        {"r0": r0, "c0": c0, "h": h, "w": w, "color": color}
        for (r0, c0, h, w), color in zip(rect_specs, colors, strict=False)
    ]

    raw_grid = _render(rects, WORKSPACE_SHAPE)
    if not _visible_bboxes_match(raw_grid, rects):
        return None
    cropped = _crop_with_margin(raw_grid)
    grid = _augment(cropped)
    expected = _expected_output(rects)

    if max(grid.shape) > MAX_SIDE or max(expected.shape) > MAX_SIDE:
        return None

    try:
        inferred_rects, order, unique = _solve(grid)
    except AssertionError:
        return None

    inferred_colors = [int(inferred_rects[idx]["color"]) for idx in order]
    expected_colors = [int(rect["color"]) for rect in reversed(rects)]
    if not unique or inferred_colors != expected_colors:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")

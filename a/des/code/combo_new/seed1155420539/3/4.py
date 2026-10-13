from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
ARC_MAX = 30
RECT_COUNT_RANGE = (3, 5)
SIDE_RANGE = (3, 8)
SHIFT_MAX = 3
GEN_RETRIES = 800


def _color_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(g == int(color))
    assert pts.size, "each rectangle color must be visible"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangles_from_visible(g: np.ndarray) -> list[dict[str, int]]:
    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert 3 <= len(colors) <= 5, "expected three to five colored rectangles"
    rects = []
    for color in colors:
        r0, c0, r1, c1 = _color_bbox(g, color)
        rects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
            }
        )
    return rects


def _topological_bottom_to_top(g: np.ndarray, rects: list[dict[str, int]]) -> list[dict[str, int]]:
    n = len(rects)
    edges: list[set[int]] = [set() for _ in range(n)]  # lower index -> upper indices

    for i in range(n):
        a = rects[i]
        for j in range(i + 1, n):
            b = rects[j]
            r0 = max(int(a["r0"]), int(b["r0"]))
            c0 = max(int(a["c0"]), int(b["c0"]))
            r1 = min(int(a["r1"]), int(b["r1"]))
            c1 = min(int(a["c1"]), int(b["c1"]))
            if r0 > r1 or c0 > c1:
                continue

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            a_visible = bool((patch == int(a["color"])).any())
            b_visible = bool((patch == int(b["color"])).any())
            assert not (a_visible and b_visible), "overlap cannot expose both pair colors"
            if a_visible:
                edges[j].add(i)
            elif b_visible:
                edges[i].add(j)

    indeg = [0] * n
    for i in range(n):
        for j in edges[i]:
            indeg[j] += 1

    order: list[int] = []
    ready = [i for i in range(n) if indeg[i] == 0]
    while ready:
        assert len(ready) == 1, "stacking order must be uniquely determined"
        cur = ready.pop()
        order.append(cur)
        for nxt in sorted(edges[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)

    assert len(order) == n, "stacking constraints must be acyclic and complete"
    return [rects[i] for i in order]


def _pack_rectangles(rects: list[dict[str, int]]) -> np.ndarray:
    out_h = max(int(rect["h"]) for rect in rects)
    out_w = sum(int(rect["w"]) for rect in rects) + len(rects) - 1
    assert out_h <= ARC_MAX and out_w <= ARC_MAX

    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for rect in rects:
        h = int(rect["h"])
        w = int(rect["w"])
        out[:h, col : col + w] = int(rect["color"])
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unstack overlapping colored rectangles into bottom-to-top horizontal order.

    Input structure:
    - Black background with three to five differently colored solid axis-aligned rectangles.
    - Rectangles overlap like stacked cards: higher rectangles hide lower pixels in shared cells.
    - For every color, the visible pixels still reach the true row and column extremes of that rectangle.

    Rule:
    - For each non-black color, take the bounding box of all visible pixels as that rectangle's complete hidden-and-visible box.
    - For every pair of reconstructed boxes that overlap, the pair color visible inside the overlap is above the other color.
    - Topologically sort those above/below constraints from bottom to top.
    - Draw the complete rectangles left to right in that bottom-to-top order, top-aligned, with one black separator column.

    Key insight:
    - Visible color extrema recover the covered rectangle sizes, and visible pixels inside pairwise overlaps reveal which card covers which.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rects = _rectangles_from_visible(g)
    ordered = _topological_bottom_to_top(g, rects)
    return _pack_rectangles(ordered)


def _sample_dims(n: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        dims = [
            (random.randint(*SIDE_RANGE), random.randint(*SIDE_RANGE))
            for _ in range(n)
        ]
        if sum(w for _h, w in dims) + n - 1 <= ARC_MAX:
            return dims
    return None


def _next_origin(
    prev: dict[str, int],
    h: int,
    w: int,
    row_dir: int,
    col_dir: int,
) -> tuple[int, int]:
    row_shift = random.randint(1, min(SHIFT_MAX, int(prev["h"]) - 1))
    col_shift = random.randint(1, min(SHIFT_MAX, int(prev["w"]) - 1))

    if row_dir > 0:
        r0 = int(prev["r0"]) + row_shift
    else:
        r1 = int(prev["r1"]) - row_shift
        r0 = r1 - int(h) + 1

    if col_dir > 0:
        c0 = int(prev["c0"]) + col_shift
    else:
        c1 = int(prev["c1"]) - col_shift
        c0 = c1 - int(w) + 1

    return int(r0), int(c0)


def _make_chain(dims: list[tuple[int, int]], row_dir: int, col_dir: int) -> list[dict[str, int]]:
    rects: list[dict[str, int]] = []
    first_h, first_w = dims[0]
    rects.append(
        {
            "r0": 0,
            "c0": 0,
            "r1": int(first_h - 1),
            "c1": int(first_w - 1),
            "h": int(first_h),
            "w": int(first_w),
        }
    )

    for h, w in dims[1:]:
        r0, c0 = _next_origin(rects[-1], int(h), int(w), row_dir, col_dir)
        rects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + h - 1),
                "c1": int(c0 + w - 1),
                "h": int(h),
                "w": int(w),
            }
        )
    return rects


def _normalize_and_pad(rects: list[dict[str, int]]) -> tuple[list[dict[str, int]], int, int] | None:
    min_r = min(int(rect["r0"]) for rect in rects)
    min_c = min(int(rect["c0"]) for rect in rects)
    max_r = max(int(rect["r1"]) for rect in rects)
    max_c = max(int(rect["c1"]) for rect in rects)
    span_h = int(max_r - min_r + 1)
    span_w = int(max_c - min_c + 1)
    if span_h > ARC_MAX or span_w > ARC_MAX:
        return None

    extra_h = ARC_MAX - span_h
    extra_w = ARC_MAX - span_w
    top_margin = random.randint(0, min(4, extra_h))
    left_margin = random.randint(0, min(4, extra_w))
    bottom_margin = random.randint(0, min(4, extra_h - top_margin))
    right_margin = random.randint(0, min(4, extra_w - left_margin))

    dr = int(top_margin - min_r)
    dc = int(left_margin - min_c)
    shifted = []
    for rect in rects:
        new = dict(rect)
        new["r0"] = int(rect["r0"] + dr)
        new["r1"] = int(rect["r1"] + dr)
        new["c0"] = int(rect["c0"] + dc)
        new["c1"] = int(rect["c1"] + dc)
        shifted.append(new)

    return shifted, int(span_h + top_margin + bottom_margin), int(span_w + left_margin + right_margin)


def _render(rects: list[dict[str, int]], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for rect in rects:
        grid[int(rect["r0"]) : int(rect["r1"]) + 1, int(rect["c0"]) : int(rect["c1"]) + 1] = int(rect["color"])
    return grid


def _rects_overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    return bool(
        max(int(a["r0"]), int(b["r0"])) <= min(int(a["r1"]), int(b["r1"]))
        and max(int(a["c0"]), int(b["c0"])) <= min(int(a["c1"]), int(b["c1"]))
    )


def _has_adjacent_visible_edges(grid: np.ndarray, rect: dict[str, int]) -> bool:
    color = int(rect["color"])
    r0, c0, r1, c1 = (int(rect[k]) for k in ("r0", "c0", "r1", "c1"))
    top = bool((grid[r0, c0 : c1 + 1] == color).all())
    bottom = bool((grid[r1, c0 : c1 + 1] == color).all())
    left = bool((grid[r0 : r1 + 1, c0] == color).all())
    right = bool((grid[r0 : r1 + 1, c1] == color).all())
    return bool((top and left) or (top and right) or (bottom and left) or (bottom and right))


def _visible_extrema_are_true(grid: np.ndarray, rect: dict[str, int]) -> bool:
    try:
        return _color_bbox(grid, int(rect["color"])) == (
            int(rect["r0"]),
            int(rect["c0"]),
            int(rect["r1"]),
            int(rect["c1"]),
        )
    except AssertionError:
        return False


def _expected_output(rects: list[dict[str, int]]) -> np.ndarray:
    return _pack_rectangles(rects)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*RECT_COUNT_RANGE)
    dims = _sample_dims(n)
    if dims is None:
        return None

    row_dir = random.choice((-1, 1))
    col_dir = random.choice((-1, 1))
    rects = _make_chain(dims, row_dir, col_dir)
    placed = _normalize_and_pad(rects)
    if placed is None:
        return None
    rects, height, width = placed
    if height > ARC_MAX or width > ARC_MAX:
        return None

    colors = random.sample(COLORS, n)
    for rect, color in zip(rects, colors):
        rect["color"] = int(color)

    for a, b in zip(rects, rects[1:]):
        if not _rects_overlap(a, b):
            return None

    grid = _render(rects, height, width)
    if not all(_visible_extrema_are_true(grid, rect) for rect in rects):
        return None
    if not all(_has_adjacent_visible_edges(grid, rect) for rect in rects):
        return None

    expected = _expected_output(rects)
    if expected.shape[0] > ARC_MAX or expected.shape[1] > ARC_MAX:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")

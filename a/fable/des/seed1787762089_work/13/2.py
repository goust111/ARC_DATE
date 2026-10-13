from __future__ import annotations

import random

import numpy as np

BBox = tuple[int, int, int, int]

BG = 0
CARD_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (24, 28)
CARD_COUNT_RANGE = (3, 6)
BASE_H_RANGE = {
    3: (7, 11),
    4: (7, 10),
    5: (7, 9),
    6: (7, 9),
}
BASE_W_RANGE = {
    3: (8, 12),
    4: (8, 11),
    5: (7, 10),
    6: (7, 10),
}
GEN_RETRIES = 500
STEP_RETRIES = 80


def _bbox_from_mask(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _bbox_from_color(grid: np.ndarray, color: int) -> BBox | None:
    return _bbox_from_mask(grid == int(color))


def _dims(bbox: BBox) -> tuple[int, int]:
    return int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)


def _intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _intersection(a: BBox, b: BBox) -> BBox | None:
    if not _intersects(a, b):
        return None
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def _render_rings(order: list[int]) -> np.ndarray:
    n = len(order)
    size = 2 * n - 1
    out = np.full((size, size), int(order[0]), dtype=int)
    for layer, color in enumerate(order):
        out[layer : size - layer, layer : size - layer] = int(color)
    return out


def _infer_order(grid: np.ndarray) -> list[int]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    assert 3 <= len(colors) <= 6, "expected 3 to 6 card colors"

    rects: dict[int, BBox] = {}
    for color in colors:
        bbox = _bbox_from_color(grid, color)
        assert bbox is not None, "each card color must remain visible"
        rects[color] = bbox

    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            overlap = _intersection(rects[color_a], rects[color_b])
            if overlap is None:
                continue
            r0, c0, r1, c1 = overlap
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            visible = {int(v) for v in np.unique(patch) if int(v) in {color_a, color_b}}
            assert len(visible) <= 1, "overlap evidence cannot support both orders"
            if not visible:
                continue
            top = visible.pop()
            under = color_b if top == color_a else color_a
            if top not in above[under]:
                above[under].add(top)
                indegree[top] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1, "cover relations must imply a unique total order"
        color = ready[0]
        remaining.remove(color)
        order.append(color)
        for nxt in above[color]:
            indegree[nxt] -= 1

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover full solid rectangles from visible fragments, infer the unique
    bottom-to-top cover order, and render that order as concentric square rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    order = _infer_order(g)
    return _render_rings(order)


def _sample_inner_span(start: int, end: int) -> tuple[int, int] | None:
    total = int(end - start + 1)
    if total < 6:
        return None
    span_lo = max(4, total - 4)
    span_hi = total - 2
    if span_lo > span_hi:
        return None
    span = random.randint(span_lo, span_hi)
    offset = random.randint(1, total - span - 1)
    first = start + offset
    return int(first), int(first + span - 1)


def _sample_next_rect(prev: BBox, older: list[BBox], extra_cap: int) -> BBox | None:
    pr0, pc0, pr1, pc1 = prev
    ph, pw = _dims(prev)

    directions: list[str] = []
    if ph >= 6:
        directions.extend(["left", "right"])
    if pw >= 6:
        directions.extend(["up", "down"])
    random.shuffle(directions)

    for direction in directions:
        for _ in range(STEP_RETRIES):
            if direction in {"left", "right"}:
                row_span = _sample_inner_span(pr0, pr1)
                if row_span is None:
                    break
                nr0, nr1 = row_span
                overlap_max = min(pw - 1, 5)
                overlap_w = random.randint(2, overlap_max)
                min_w = max(4, overlap_w + 1)
                max_w = overlap_w + extra_cap
                if min_w > max_w:
                    continue
                new_w = random.randint(min_w, max_w)
                if direction == "right":
                    nc0 = pc1 - overlap_w + 1
                    nc1 = nc0 + new_w - 1
                else:
                    nc1 = pc0 + overlap_w - 1
                    nc0 = nc1 - new_w + 1
            else:
                col_span = _sample_inner_span(pc0, pc1)
                if col_span is None:
                    break
                nc0, nc1 = col_span
                overlap_max = min(ph - 1, 5)
                overlap_h = random.randint(2, overlap_max)
                min_h = max(4, overlap_h + 1)
                max_h = overlap_h + extra_cap
                if min_h > max_h:
                    continue
                new_h = random.randint(min_h, max_h)
                if direction == "down":
                    nr0 = pr1 - overlap_h + 1
                    nr1 = nr0 + new_h - 1
                else:
                    nr1 = pr0 + overlap_h - 1
                    nr0 = nr1 - new_h + 1

            cand = (int(nr0), int(nc0), int(nr1), int(nc1))
            if any(_intersects(cand, bbox) for bbox in older[:-1]):
                continue
            return cand
    return None


def _shift_rects(rects: list[BBox], height: int, width: int) -> list[BBox] | None:
    min_r = min(b[0] for b in rects)
    max_r = max(b[2] for b in rects)
    min_c = min(b[1] for b in rects)
    max_c = max(b[3] for b in rects)

    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    if span_h + 2 > height or span_w + 2 > width:
        return None

    top = random.randint(1, height - span_h - 1)
    left = random.randint(1, width - span_w - 1)
    dr = top - min_r
    dc = left - min_c
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _paint_cards(rects: list[BBox], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for bbox, color in zip(rects, colors, strict=False):
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _corners_visible(grid: np.ndarray, bbox: BBox, color: int) -> bool:
    r0, c0, r1, c1 = bbox
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    return all(int(grid[r, c]) == int(color) for r, c in corners)


def _is_valid_instance(grid: np.ndarray, rects: list[BBox], colors: list[int]) -> bool:
    for bbox, color in zip(rects, colors, strict=False):
        visible_bbox = _bbox_from_color(grid, color)
        if visible_bbox != bbox:
            return False
        if not _corners_visible(grid, bbox, color):
            return False

    for i in range(len(rects) - 1):
        if _intersection(rects[i], rects[i + 1]) is None:
            return False
    for i in range(len(rects)):
        for j in range(i + 2, len(rects)):
            if _intersects(rects[i], rects[j]):
                return False

    try:
        out = main(grid)
    except Exception:
        return False
    return np.array_equal(out, _render_rings(colors))


def _generate_attempt() -> np.ndarray | None:
    n_cards = random.randint(*CARD_COUNT_RANGE)
    base_h = random.randint(*BASE_H_RANGE[n_cards])
    base_w = random.randint(*BASE_W_RANGE[n_cards])
    extra_cap = 4 if n_cards <= 4 else 3

    rects: list[BBox] = [(0, 0, base_h - 1, base_w - 1)]
    for _ in range(1, n_cards):
        nxt = _sample_next_rect(rects[-1], rects, extra_cap)
        if nxt is None:
            return None
        rects.append(nxt)

    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    placed = _shift_rects(rects, height, width)
    if placed is None:
        return None

    colors = random.sample(list(CARD_COLORS), n_cards)
    grid = _paint_cards(placed, colors, (height, width))
    if not _is_valid_instance(grid, placed, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangle instance")

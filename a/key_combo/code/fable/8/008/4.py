from __future__ import annotations

import itertools

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
FRAGMENT_LEN = 2
MIN_SPAN = 6
MAX_SPAN = 12
GEN_ATTEMPTS = 600

Rect = tuple[int, int, int, int]
Span = tuple[int, int]


def _edge_spans(line: np.ndarray, bg: int = BG) -> dict[int, Span]:
    spans: dict[int, Span] = {}
    for color0 in np.unique(line):
        color = int(color0)
        if color == bg:
            continue
        positions = np.flatnonzero(line == color)
        spans[color] = (int(positions.min()), int(positions.max()))
    return spans


def _paired_rectangles(grid: np.ndarray) -> tuple[dict[int, Span], dict[int, Span], dict[int, Rect]]:
    g = np.asarray(grid, dtype=int)
    top = _edge_spans(g[0], BG)
    left = _edge_spans(g[:, 0], BG)
    rects: dict[int, Rect] = {}
    for color in sorted(set(top) & set(left)):
        c0, c1 = top[color]
        r0, r1 = left[color]
        rects[color] = (r0, c0, r1, c1)
    return top, left, rects


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _intersection(a: Rect, b: Rect) -> Rect | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (r0, c0, r1, c1)


def _paint_rectangles(grid: np.ndarray, rects: dict[int, Rect], *, smaller_on_top: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    paint_order = sorted(
        rects.items(),
        key=lambda item: (_rect_area(item[1]), item[0]),
        reverse=bool(smaller_on_top),
    )
    for color, (r0, c0, r1, c1) in paint_order:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    out[0, :] = grid[0, :]
    out[:, 0] = grid[:, 0]
    return out


def _paint_by_edge_order(grid: np.ndarray) -> np.ndarray:
    top, left, _ = _paired_rectangles(grid)
    top_items = sorted(top.items(), key=lambda item: (item[1][0], item[1][1], item[0]))
    left_items = sorted(left.items(), key=lambda item: (item[1][0], item[1][1], item[0]))

    rects: dict[int, Rect] = {}
    for (top_color, (c0, c1)), (_left_color, (r0, r1)) in zip(top_items, left_items):
        rects[int(top_color)] = (int(r0), int(c0), int(r1), int(c1))
    return _paint_rectangles(grid, rects, smaller_on_top=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild filled rectangles from matching top-edge and left-edge color spans.

    Input structure:
    - The background is black.
    - The top row and the left column contain colored border fragments.
    - For any color, all of its top-row cells jointly define one column span, and all of its left-column cells jointly define one row span.
    - Some colors appear on both edges, while distractor colors appear on only one edge; the paired rectangle areas are all distinct.

    Rule:
    - For each color present on both edges, take the inclusive column span from the top row and the inclusive row span from the left column.
    - Fill the rectangle at the intersection of those two spans with that color.
    - If filled rectangles overlap, the smaller-area rectangle overwrites the larger one.
    - Keep every original top-edge and left-edge fragment unchanged.

    Key insight:
    - The interior rectangles are invisible and must be reconstructed from two orthogonal edge projections; color matching selects the valid pairs, and area decides the overlap stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    _top, _left, rects = _paired_rectangles(g)
    return _paint_rectangles(g, rects, smaller_on_top=True)


def _fragment_cells(start: int, end: int) -> list[int]:
    return [start, start + 1, end - 1, end]


def _can_place_span(occupied: np.ndarray, start: int, end: int) -> bool:
    if start < 1 or end >= occupied.size or end - start + 1 < MIN_SPAN:
        return False
    cells = _fragment_cells(start, end)
    if len(set(cells)) != len(cells):
        return False
    return all(not occupied[idx] for idx in cells)


def _mark_span(occupied: np.ndarray, start: int, end: int) -> None:
    for idx in _fragment_cells(start, end):
        occupied[idx] = True


def _paint_top_fragments(grid: np.ndarray, color: int, span: Span) -> None:
    start, end = span
    grid[0, start : start + FRAGMENT_LEN] = int(color)
    grid[0, end - FRAGMENT_LEN + 1 : end + 1] = int(color)


def _paint_left_fragments(grid: np.ndarray, color: int, span: Span) -> None:
    start, end = span
    grid[start : start + FRAGMENT_LEN, 0] = int(color)
    grid[end - FRAGMENT_LEN + 1 : end + 1, 0] = int(color)


def _sample_overlap_pair(edge_len: int, small_span: int, large_span: int, offset: int) -> tuple[Span, Span] | None:
    total = max(small_span, offset + large_span)
    if total >= edge_len:
        return None
    start_small_hi = edge_len - total
    if start_small_hi < 1:
        return None
    start_small = int(np.random.randint(1, start_small_hi + 1))
    start_large = start_small + offset
    return (start_small, start_small + small_span - 1), (start_large, start_large + large_span - 1)


def _reserve_overlap_pair(
    occupied: np.ndarray,
    small_span: int,
    large_span: int,
    offset_lo: int,
    offset_hi: int,
) -> tuple[Span, Span] | None:
    for _ in range(120):
        if offset_lo > offset_hi:
            return None
        offset = int(np.random.randint(offset_lo, offset_hi + 1))
        pair = _sample_overlap_pair(occupied.size, small_span, large_span, offset)
        if pair is None:
            continue
        span_a, span_b = pair
        if not _can_place_span(occupied, *span_a):
            continue
        if not _can_place_span(occupied, *span_b):
            continue
        _mark_span(occupied, *span_a)
        _mark_span(occupied, *span_b)
        return span_a, span_b
    return None


def _sample_free_span(occupied: np.ndarray, min_span: int = MIN_SPAN, max_span: int = MAX_SPAN) -> Span | None:
    span_hi = min(int(max_span), occupied.size - 1)
    for _ in range(200):
        span = int(np.random.randint(int(min_span), span_hi + 1))
        start = int(np.random.randint(1, occupied.size - span + 1))
        end = start + span - 1
        if not _can_place_span(occupied, start, end):
            continue
        _mark_span(occupied, start, end)
        return (start, end)
    return None


def _visible_interior_mask(rect: Rect, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mask[0, :] = False
    mask[:, 0] = False
    return mask


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        top, left, rects = _paired_rectangles(grid)
        if len(rects) < 2:
            return False

        top_only = set(top) - set(left)
        left_only = set(left) - set(top)
        if not top_only or not left_only:
            return False

        areas = {color: _rect_area(rect) for color, rect in rects.items()}
        if len(set(areas.values())) != len(areas):
            return False

        overlaps: list[tuple[int, int]] = []
        for ca, cb in itertools.combinations(sorted(rects), 2):
            if _intersection(rects[ca], rects[cb]) is not None:
                overlaps.append((ca, cb))
        if not overlaps:
            return False

        expected = _paint_rectangles(grid, rects, smaller_on_top=True)
        produced = main(grid)
        if not np.array_equal(produced, expected):
            return False
        if np.array_equal(produced, grid):
            return False

        wrong_overlap = _paint_rectangles(grid, rects, smaller_on_top=False)
        if np.array_equal(wrong_overlap, expected):
            return False

        wrong_order = _paint_by_edge_order(grid)
        if np.array_equal(wrong_order, expected):
            return False

        for color, rect in rects.items():
            interior = _visible_interior_mask(rect, grid.shape)
            if not np.any((produced == color) & interior):
                return False

        for color in top_only | left_only:
            mask = produced == int(color)
            if np.any(mask[1:, 1:]):
                return False

        smaller_wins = False
        for ca, cb in overlaps:
            ia = areas[ca]
            ib = areas[cb]
            inter = _intersection(rects[ca], rects[cb])
            assert inter is not None
            r0, c0, r1, c1 = inter
            seen = set(np.unique(produced[r0 : r1 + 1, c0 : c1 + 1]).tolist())
            if ia < ib and seen == {ca}:
                smaller_wins = True
            if ib < ia and seen == {cb}:
                smaller_wins = True
        if not smaller_wins:
            return False
    except Exception:
        return False
    return True


def _sample_cross_overlap_dimensions() -> tuple[int, int, int, int] | None:
    for _ in range(200):
        h_small = int(np.random.randint(4, 7))
        h_large = int(np.random.randint(h_small + 1, min(10, h_small + 4)))
        w_large = int(np.random.randint(4, 7))
        w_small = int(np.random.randint(w_large + 1, min(10, w_large + 4)))
        if h_small * w_small >= h_large * w_large:
            continue
        return h_small, h_large, w_small, w_large
    return None


def _gen_attempt() -> np.ndarray | None:
    n_pairs = int(np.random.randint(2, 4))
    total_colors = n_pairs + 2
    colors = [int(c) for c in np.random.choice(np.array(PALETTE, dtype=int), size=total_colors, replace=False)]
    pair_colors = colors[:n_pairs]
    top_only_color = colors[-2]
    left_only_color = colors[-1]

    height = int(np.random.randint(24, 31))
    width = int(np.random.randint(24, 31))

    top_occ = np.zeros(width, dtype=bool)
    left_occ = np.zeros(height, dtype=bool)

    dims = _sample_cross_overlap_dimensions()
    if dims is None:
        return None
    h_small, h_large, w_small, w_large = dims

    row_offset_hi = h_small - 2
    col_offset_lo = w_small - w_large + 1
    col_offset_hi = w_small - 2
    if col_offset_lo > col_offset_hi:
        return None

    row_pair = _reserve_overlap_pair(left_occ, h_small, h_large, 1, row_offset_hi)
    col_pair = _reserve_overlap_pair(top_occ, w_small, w_large, col_offset_lo, col_offset_hi)
    if row_pair is None or col_pair is None:
        return None

    top_spans: dict[int, Span] = {
        pair_colors[0]: col_pair[0],
        pair_colors[1]: col_pair[1],
    }
    left_spans: dict[int, Span] = {
        pair_colors[0]: row_pair[0],
        pair_colors[1]: row_pair[1],
    }
    used_areas = {
        _rect_area((left_spans[color][0], top_spans[color][0], left_spans[color][1], top_spans[color][1]))
        for color in pair_colors[:2]
    }

    for color in pair_colors[2:]:
        placed = False
        for _ in range(200):
            top_span = _sample_free_span(top_occ)
            if top_span is None:
                return None
            left_span = _sample_free_span(left_occ)
            if left_span is None:
                return None
            area = (top_span[1] - top_span[0] + 1) * (left_span[1] - left_span[0] + 1)
            if area in used_areas:
                for idx in _fragment_cells(*top_span):
                    top_occ[idx] = False
                for idx in _fragment_cells(*left_span):
                    left_occ[idx] = False
                continue
            top_spans[color] = top_span
            left_spans[color] = left_span
            used_areas.add(area)
            placed = True
            break
        if not placed:
            return None

    top_distractor = _sample_free_span(top_occ)
    left_distractor = _sample_free_span(left_occ)
    if top_distractor is None or left_distractor is None:
        return None
    top_spans[top_only_color] = top_distractor
    left_spans[left_only_color] = left_distractor

    grid = np.zeros((height, width), dtype=int)
    for color, span in top_spans.items():
        _paint_top_fragments(grid, color, span)
    for color, span in left_spans.items():
        _paint_left_fragments(grid, color, span)
    grid[0, 0] = BG

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

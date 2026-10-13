from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
PAIR_COUNT_RANGE = (2, 5)
RECT_H_RANGE = (2, 6)
RECT_W_RANGE = (2, 7)
RECT_AREA_RANGE = (6, 24)
PARTITION_RETRIES = 200
PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    return arr - arr.min(axis=0, keepdims=True)


def _bbox_shape(coords: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].max()) + 1, int(arr[:, 1].max()) + 1


def _shape_set(coords: np.ndarray) -> set[tuple[int, int]]:
    arr = np.asarray(coords, dtype=int)
    return {tuple(map(int, pt)) for pt in arr}


def _all_rectangle_dims(shape_a: np.ndarray, shape_b: np.ndarray) -> set[tuple[int, int]]:
    a = _shape_set(_normalize(shape_a))
    b = _shape_set(_normalize(shape_b))
    total = len(a) + len(b)
    dims: set[tuple[int, int]] = set()
    for dr in range(-total, total + 1):
        for dc in range(-total, total + 1):
            shifted_b = {(r + dr, c + dc) for r, c in b}
            if a & shifted_b:
                continue
            union = a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            h = max(rows) - min(rows) + 1
            w = max(cols) - min(cols) + 1
            if h * w == total:
                dims.add((int(h), int(w)))
    return dims


def _find_rectangle_dims(shape_a: np.ndarray, shape_b: np.ndarray) -> tuple[int, int]:
    dims = _all_rectangle_dims(shape_a, shape_b)
    assert dims, "same-color pieces must complement to at least one rectangle"
    return min(dims, key=lambda hw: (hw[0] * hw[1], hw[0], hw[1]))


def _render_rectangles(rectangles: list[tuple[int, int, int]]) -> np.ndarray:
    out_h = sum(h for _color, h, _w in rectangles) + max(0, len(rectangles) - 1)
    out_w = max(w for _color, _h, w in rectangles)
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)
    row = 0
    for idx, (color, h, w) in enumerate(rectangles):
        out[row : row + h, :w] = int(color)
        row += h
        if idx != len(rectangles) - 1:
            row += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-colored fragments, recover each completed rectangle, and stack the rectangles vertically."""
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert colors, "expected at least one non-background color"

    rectangles: list[tuple[int, int, int, int, int]] = []
    for color in colors:
        comps = _components(g == color)
        assert len(comps) == 2, "each color must form exactly two disconnected fragments"
        top_row = min(int(comp[:, 0].min()) for comp in comps)
        left_col = min(int(comp[:, 1].min()) for comp in comps)
        h, w = _find_rectangle_dims(_normalize(comps[0]), _normalize(comps[1]))
        rectangles.append((top_row, left_col, color, h, w))

    rectangles.sort(key=lambda item: (item[0], item[1]))
    return _render_rectangles([(color, h, w) for _top, _left, color, h, w in rectangles])


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(bounds[0], bounds[1])


def _rect_dims() -> tuple[int, int]:
    for _ in range(100):
        h = _randint(RECT_H_RANGE)
        w = _randint(RECT_W_RANGE)
        area = h * w
        if RECT_AREA_RANGE[0] <= area <= RECT_AREA_RANGE[1]:
            return h, w
    raise RuntimeError("failed to sample rectangle dimensions")


def _rect_neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _shape_is_rectangle(coords: np.ndarray) -> bool:
    h, w = _bbox_shape(coords)
    return h * w == int(len(coords))


def _shape_is_line(coords: np.ndarray) -> bool:
    h, w = _bbox_shape(coords)
    return h == 1 or w == 1


def _grow_partition(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]
    total = len(cells)
    for _ in range(PARTITION_RETRIES):
        seed_a, seed_b = random.sample(cells, 2)
        regions = [set([seed_a]), set([seed_b])]
        owner = {seed_a: 0, seed_b: 1}
        frontiers = [
            {nbr for nbr in _rect_neighbors(seed_a, rect_h, rect_w) if nbr != seed_b},
            {nbr for nbr in _rect_neighbors(seed_b, rect_h, rect_w) if nbr != seed_a},
        ]
        target_a = random.randint(max(2, total // 3), min(total - 2, (2 * total) // 3))

        while len(owner) < total:
            options = [idx for idx in (0, 1) if frontiers[idx]]
            if not options:
                break
            if len(options) == 1:
                chosen = options[0]
            else:
                need_a = target_a - len(regions[0])
                need_b = (total - target_a) - len(regions[1])
                if need_a > need_b and need_a > 0:
                    chosen = 0
                elif need_b > need_a and need_b > 0:
                    chosen = 1
                else:
                    chosen = random.choice(options)

            cell = random.choice(tuple(frontiers[chosen]))
            owner[cell] = chosen
            regions[chosen].add(cell)
            frontiers[0].discard(cell)
            frontiers[1].discard(cell)
            for nbr in _rect_neighbors(cell, rect_h, rect_w):
                if nbr not in owner:
                    frontiers[chosen].add(nbr)
            frontiers[0] = {pt for pt in frontiers[0] if pt not in owner}
            frontiers[1] = {pt for pt in frontiers[1] if pt not in owner}

        if len(owner) != total:
            continue

        shape_a = _normalize(np.asarray(sorted(regions[0]), dtype=int))
        shape_b = _normalize(np.asarray(sorted(regions[1]), dtype=int))
        if min(len(shape_a), len(shape_b)) < 2:
            continue
        if _shape_is_rectangle(shape_a) and _shape_is_rectangle(shape_b):
            continue
        if _shape_is_line(shape_a) and _shape_is_line(shape_b):
            continue
        if _all_rectangle_dims(shape_a, shape_b) != {(rect_h, rect_w)}:
            continue
        return shape_a, shape_b
    return None


def _make_pair_shapes() -> tuple[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None:
    for _ in range(120):
        rect_h, rect_w = _rect_dims()
        partition = _grow_partition(rect_h, rect_w)
        if partition is not None:
            return (rect_h, rect_w), partition
    return None


def _mark_forbidden(forbidden: np.ndarray, coords: np.ndarray) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                forbidden[nr, nc] = True


def _place_piece(
    shape: np.ndarray,
    color: int,
    grid: np.ndarray,
    forbidden: np.ndarray,
) -> tuple[int, int] | None:
    h, w = _bbox_shape(shape)
    grid_h, grid_w = grid.shape
    if h > grid_h or w > grid_w:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(0, grid_h - h)
        c0 = random.randint(0, grid_w - w)
        placed = shape + np.array([r0, c0], dtype=int)
        if forbidden[placed[:, 0], placed[:, 1]].any():
            continue
        grid[placed[:, 0], placed[:, 1]] = int(color)
        _mark_forbidden(forbidden, placed)
        return int(r0), int(c0)
    return None


def _canvas_shape(pairs: list[dict[str, object]]) -> tuple[int, int]:
    piece_boxes = []
    for pair in pairs:
        for shape in pair["shapes"]:
            piece_boxes.append(_bbox_shape(shape))
    total_box_area = sum(h * w for h, w in piece_boxes)
    max_h = max(h for h, _w in piece_boxes)
    max_w = max(w for _h, w in piece_boxes)
    base = int(np.ceil(np.sqrt(total_box_area * 2.6)))
    h_lo = max(12, max_h + 2, base)
    w_lo = max(12, max_w + 2, base)
    h_hi = max(h_lo + 2, h_lo + random.randint(3, 8))
    w_hi = max(w_lo + 2, w_lo + random.randint(3, 8))
    return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)


def _expected_from_pairs(
    pairs: list[dict[str, object]],
    top_rows: list[int],
    left_cols: list[int],
) -> np.ndarray:
    order = sorted(
        range(len(pairs)),
        key=lambda idx: (int(top_rows[idx]), int(left_cols[idx])),
    )
    rectangles = []
    for idx in order:
        rect_h, rect_w = pairs[idx]["rect"]
        rectangles.append((int(pairs[idx]["color"]), int(rect_h), int(rect_w)))
    return _render_rectangles(rectangles)


def _gen_attempt() -> np.ndarray | None:
    pair_count = _randint(PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, pair_count)
    pairs: list[dict[str, object]] = []
    for color in colors:
        pair_data = _make_pair_shapes()
        if pair_data is None:
            return None
        rect_shape, shapes = pair_data
        pairs.append({"color": int(color), "rect": rect_shape, "shapes": shapes})

    grid_h, grid_w = _canvas_shape(pairs)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    forbidden = np.zeros((grid_h, grid_w), dtype=bool)

    top_rows = [0] * pair_count
    left_cols = [0] * pair_count
    seen_pair = [False] * pair_count

    items: list[tuple[int, int, float, int, np.ndarray]] = []
    for idx, pair in enumerate(pairs):
        for shape in pair["shapes"]:
            bbox_h, bbox_w = _bbox_shape(shape)
            items.append((len(shape), bbox_h * bbox_w, random.random(), idx, shape))
    items.sort(key=lambda item: (-item[1], -item[0], item[2]))

    placed_counts = [0] * pair_count
    for _area, _cells, _rand, pair_idx, shape in items:
        anchor = _place_piece(shape, int(pairs[pair_idx]["color"]), grid, forbidden)
        if anchor is None:
            return None
        r0, c0 = anchor
        if not seen_pair[pair_idx]:
            top_rows[pair_idx] = r0
            left_cols[pair_idx] = c0
            seen_pair[pair_idx] = True
        else:
            top_rows[pair_idx] = min(top_rows[pair_idx], r0)
            left_cols[pair_idx] = min(left_cols[pair_idx], c0)
        placed_counts[pair_idx] += 1

    if any(count != 2 for count in placed_counts):
        return None
    if len(set(top_rows)) != len(top_rows):
        return None

    expected = _expected_from_pairs(pairs, top_rows, left_cols)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

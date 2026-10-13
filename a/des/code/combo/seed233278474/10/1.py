from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
FILL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 20
GRID_MAX = 30
LEGEND_COUNT_MIN = 3
LEGEND_COUNT_MAX = 4
QUERY_EXTRA_MAX = 3
SHAPE_H_MIN = 4
SHAPE_H_MAX = 6
SHAPE_W_MIN = 4
SHAPE_W_MAX = 6
LEGEND_GAP_MIN = 1
LEGEND_GAP_MAX = 3
FIELD_GAP_MIN = 4
FIELD_GAP_MAX = 6
OUTER_MARGIN = 1
MAX_SHAPE_ATTEMPTS = 400
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _normalized_signature(points: list[tuple[int, int]]) -> tuple[tuple[int, ...], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    arr = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=np.uint8)
    for r, c in points:
        arr[r - r0, c - c0] = 1
    return tuple(int(x) for x in arr.ravel()), (arr.shape[0], arr.shape[1])


def _enclosed_black_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    walls = np.zeros((h + 2, w + 2), dtype=bool)
    for r, c in points:
        walls[r - r0 + 1, c - c0 + 1] = True

    outside = np.zeros_like(walls, dtype=bool)
    q = deque([(0, 0)])
    outside[0, 0] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not walls[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    fill = (~walls) & (~outside)
    return [
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(fill[1:-1, 1:-1])
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each grey outline on the right to the same normalized outline in the left legend,
    then fill the enclosed black interior with the legend's color while keeping the outline grey.
    """
    g = np.asarray(grid, dtype=int)
    grey_comps = _components(g == GREY)
    color_comps = _components((g != BLACK) & (g != GREY))

    assert grey_comps, "expected grey outline components"
    assert color_comps, "expected colored legend blocks"

    grey_boxes = [_bbox(comp) for comp in grey_comps]
    legend_ids: set[int] = set()
    shape_to_color: dict[tuple[tuple[int, ...], tuple[int, int]], int] = {}

    for block in color_comps:
        colors = {int(g[r, c]) for r, c in block}
        assert len(colors) == 1, "legend block must be monochrome"
        color = next(iter(colors))

        br0, bc0, br1, bc1 = _bbox(block)
        candidates: list[tuple[int, int, int]] = []
        for idx, (gr0, gc0, gr1, gc1) in enumerate(grey_boxes):
            gap = bc0 - gc1
            overlap = min(br1, gr1) - max(br0, gr0) + 1
            if 1 <= gap <= 2 and overlap > 0:
                center_delta = abs((br0 + br1) - (gr0 + gr1))
                candidates.append((gap, center_delta, idx))

        assert candidates, "each color block must pair with one grey legend outline"
        candidates.sort()
        comp_idx = candidates[0][2]
        assert comp_idx not in legend_ids, "legend outline cannot have multiple color blocks"
        legend_ids.add(comp_idx)

        sig = _normalized_signature(grey_comps[comp_idx])
        assert sig not in shape_to_color, "legend shapes must be unique"
        shape_to_color[sig] = color

    out = g.copy()
    for idx, comp in enumerate(grey_comps):
        if idx in legend_ids:
            continue
        sig = _normalized_signature(comp)
        assert sig in shape_to_color, "query outline must match a legend shape"
        fill_color = shape_to_color[sig]
        for r, c in _enclosed_black_cells(comp):
            if out[r, c] == BLACK:
                out[r, c] = fill_color
    return out


def _crop_tight(mask: np.ndarray) -> np.ndarray:
    points = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    r0, c0, r1, c1 = _bbox(points)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _boundary_mask(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    out = np.zeros_like(filled, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and filled[nr, nc]):
                    out[r, c] = True
                    break
    return out


def _sample_shape() -> tuple[np.ndarray, np.ndarray, int] | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = random.randint(SHAPE_H_MIN, SHAPE_H_MAX)
        w = random.randint(SHAPE_W_MIN, SHAPE_W_MAX)

        left = [0] * h
        right = [0] * h
        length = random.randint(3, w)
        left[0] = random.randint(0, w - length)
        right[0] = left[0] + length - 1

        ok = True
        for r in range(1, h):
            placed = False
            for _ in range(50):
                l = max(0, min(w - 3, left[r - 1] + random.choice((-1, 0, 1))))
                rr = max(l + 2, min(w - 1, right[r - 1] + random.choice((-1, 0, 1))))
                if min(rr, right[r - 1]) - max(l, left[r - 1]) >= 0:
                    left[r] = l
                    right[r] = rr
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        filled = np.zeros((h, w), dtype=bool)
        for r in range(h):
            filled[r, left[r] : right[r] + 1] = True

        filled = _crop_tight(filled)
        boundary = _boundary_mask(filled)
        interior = filled & ~boundary
        if int(interior.sum()) == 0:
            continue
        if len(_components(boundary)) != 1:
            continue

        block_rows = [
            r
            for r in range(boundary.shape[0] - 1)
            if boundary[r, -1] and boundary[r + 1, -1]
        ]
        if not block_rows:
            continue
        return boundary, interior, random.choice(block_rows)
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = color


def _try_generate() -> np.ndarray | None:
    n_shapes = random.randint(LEGEND_COUNT_MIN, LEGEND_COUNT_MAX)
    colors = random.sample(FILL_COLORS, n_shapes)

    shapes: list[tuple[np.ndarray, np.ndarray, int]] = []
    seen_signatures: set[tuple[tuple[int, ...], tuple[int, int]]] = set()
    for _ in range(n_shapes):
        placed = False
        for _ in range(MAX_SHAPE_ATTEMPTS):
            sampled = _sample_shape()
            if sampled is None:
                break
            boundary, interior, block_row = sampled
            sig = _normalized_signature([(int(r), int(c)) for r, c in np.argwhere(boundary)])
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            shapes.append((boundary, interior, block_row))
            placed = True
            break
        if not placed:
            return None

    legend_positions: list[tuple[int, int]] = []
    row = OUTER_MARGIN
    max_entry_width = 0
    for boundary, _interior, _block_row in shapes:
        h, w = boundary.shape
        legend_positions.append((row, OUTER_MARGIN))
        max_entry_width = max(max_entry_width, w + 1)
        row += h + random.randint(LEGEND_GAP_MIN, LEGEND_GAP_MAX)

    min_height = row
    if min_height > GRID_MAX - 1:
        return None

    n_queries = random.randint(n_shapes, min(n_shapes + QUERY_EXTRA_MAX, 7))
    assignments = list(range(n_shapes))
    if n_queries > n_shapes:
        assignments += random.choices(range(n_shapes), k=n_queries - n_shapes)
    random.shuffle(assignments)

    query_col0 = max_entry_width + random.randint(FIELD_GAP_MIN, FIELD_GAP_MAX)
    tallest_query = max(shapes[idx][0].shape[0] for idx in assignments)
    min_height = max(min_height, tallest_query + 2)
    min_width = max(query_col0 + 10, GRID_MIN)
    if min_height > GRID_MAX or min_width > GRID_MAX:
        return None

    h = random.randint(min_height, GRID_MAX)
    w = random.randint(min_width, GRID_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for (boundary, _interior, block_row), color, (r0, c0) in zip(shapes, colors, legend_positions, strict=False):
        bh, bw = boundary.shape
        _paint_mask(grid, boundary, r0, c0, GREY)
        grid[r0 + block_row : r0 + block_row + 2, c0 + bw] = color
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + bh + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + bw + 2)
        reserved[rr0:rr1, cc0:cc1] = True

    query_positions: list[tuple[int, int] | None] = [None] * n_queries
    order = sorted(
        range(n_queries),
        key=lambda i: int(np.prod(shapes[assignments[i]][0].shape)),
        reverse=True,
    )
    for query_idx in order:
        boundary, _interior, _block_row = shapes[assignments[query_idx]]
        bh, bw = boundary.shape
        options: list[tuple[int, int]] = []
        for r0 in range(1, h - bh):
            for c0 in range(query_col0, w - bw):
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r0 + bh + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c0 + bw + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                options.append((r0, c0))
        if not options:
            return None
        r0, c0 = random.choice(options)
        query_positions[query_idx] = (r0, c0)
        reserved[max(0, r0 - 1) : min(h, r0 + bh + 1), max(0, c0 - 1) : min(w, c0 + bw + 1)] = True
        _paint_mask(grid, boundary, r0, c0, GREY)

    out = grid.copy()
    for query_idx, pos in enumerate(query_positions):
        assert pos is not None
        r0, c0 = pos
        _boundary, interior, _block_row = shapes[assignments[query_idx]]
        fill_color = colors[assignments[query_idx]]
        _paint_mask(out, interior, r0, c0, fill_color)

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")

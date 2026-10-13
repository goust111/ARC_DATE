from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MIN_RECT_GAP = 5
GRID_MIN = 20
GRID_MAX = 28
RECT_DIMS = tuple((h, w) for h in range(2, 5) for w in range(2, 5))


BBox = tuple[int, int, int, int]
Rect = tuple[int, int, int, int]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, clip: bool = False) -> None:
    h, w = patch.shape
    dh, dw = dst.shape
    if not clip:
        dst[r0 : r0 + h, c0 : c0 + w] = patch
        return
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dh, int(r0) + h)
    cc1 = min(dw, int(c0) + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - int(r0)
    pc0 = cc0 - int(c0)
    dst[rr0:rr1, cc0:cc1] = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]


def _rect_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _boxes_far_enough(a: Rect, b: Rect, gap: int = MIN_RECT_GAP) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    row_gap = _rect_gap(ar0, ar1, br0, br1)
    col_gap = _rect_gap(ac0, ac1, bc0, bc1)
    return row_gap >= gap or col_gap >= gap


def _solid_rectangles(grid: np.ndarray) -> list[Rect]:
    comps = _components(grid == BLUE)
    rects: list[Rect] = []
    for coords in comps:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if int(coords.shape[0]) != h * w:
            raise AssertionError("blue components must be solid rectangles")
        rects.append((r0, c0, h, w))
    rects.sort()
    return rects


def _stamped_canvas(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h_total, w_total = out.shape
    for rect in _solid_rectangles(out):
        r0, c0, h, w = rect
        sprite = _crop_bbox(out, (r0, c0, r0 + h - 1, c0 + w - 1))
        for dc in (-w, w):
            rr = r0 + h
            cc = c0 + dc
            while rr < h_total and cc < w_total and cc + w > 0:
                _paste(out, sprite, rr, cc, clip=True)
                rr += h
                cc += dc
    return out


def _exterior_mask(stamped: np.ndarray) -> np.ndarray:
    h, w = stamped.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if stamped[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if stamped[r, c] == BG and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and stamped[nr, nc] == BG and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _enclosed_mask(stamped: np.ndarray) -> np.ndarray:
    return (stamped == BG) & ~_exterior_mask(stamped)


def _render_input(rects: list[Rect], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for r0, c0, rh, rw in rects:
        grid[r0 : r0 + rh, c0 : c0 + rw] = BLUE
    return grid


def _hole_count_for_pair(dim_a: tuple[int, int], dim_b: tuple[int, int], gap: int) -> int:
    h1, w1 = dim_a
    h2, w2 = dim_b
    grid = _render_input([(2, 2, h1, w1), (2, 2 + w1 + gap, h2, w2)], (26, 26))
    return int(_enclosed_mask(_stamped_canvas(grid)).sum())


def _build_primary_pair_options() -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, ...]]]:
    options: list[tuple[tuple[int, int], tuple[int, int], tuple[int, ...]]] = []
    for dim_a in RECT_DIMS:
        for dim_b in RECT_DIMS:
            if dim_a == dim_b:
                continue
            good_gaps = [
                gap
                for gap in range(MIN_RECT_GAP, 11)
                if _hole_count_for_pair(dim_a, dim_b, gap) >= 3
            ]
            if good_gaps:
                options.append((dim_a, dim_b, tuple(good_gaps)))
    return options


PRIMARY_PAIR_OPTIONS = _build_primary_pair_options()


def _grid_is_valid(grid: np.ndarray) -> bool:
    rects = _solid_rectangles(grid)
    if len(rects) not in (2, 3):
        return False
    for i, rect in enumerate(rects):
        for other in rects[i + 1 :]:
            if not _boxes_far_enough(rect, other):
                return False

    stamped = _stamped_canvas(grid)
    if int(_enclosed_mask(stamped).sum()) <= 0:
        return False

    single_holes = [
        int(_enclosed_mask(_stamped_canvas(_render_input([rect], grid.shape))).sum())
        for rect in rects
    ]
    if max(single_holes) != 0:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each blue solid rectangle along both downward diagonals, then fill the background regions those stamp-tracks enclose.

    Input structure:
    - The background is black.
    - Two or three separated blue 4-connected objects are present.
    - Each blue object is a filled axis-aligned rectangle.

    Rule:
    - Keep every original blue rectangle.
    - For a rectangle of height h and width w, stamp identical blue copies at offsets (h, w), (2h, 2w), ... and (h, -w), (2h, -2w), ..., clipping any off-grid part and stopping a direction once the next copy is fully outside.
    - Treat every non-black pixel after stamping as a wall.
    - Flood 4-connected black background from the grid border to mark the exterior.
    - Recolor every remaining unmarked black region red, while keeping the blue walls and exterior background unchanged.

    Key insight:
    - The enclosing walls are not explicit in the input; they emerge only after each rectangle repeats with a stride equal to its own size, and the enclosed background is then found by border flood-fill.
    """
    stamped = _stamped_canvas(np.asarray(grid, dtype=int))
    out = stamped.copy()
    out[_enclosed_mask(stamped)] = RED
    return out


def _sample_primary_pair() -> tuple[tuple[int, int], tuple[int, int], int]:
    dim_a, dim_b, gaps = random.choice(PRIMARY_PAIR_OPTIONS)
    return dim_a, dim_b, random.choice(gaps)


def _try_add_third_rect(
    rects: list[Rect],
    shape: tuple[int, int],
    forbidden_dims: set[tuple[int, int]],
) -> list[Rect] | None:
    h_total, w_total = shape
    choices = [dim for dim in RECT_DIMS if dim not in forbidden_dims]
    if not choices:
        return None

    for _ in range(120):
        rh, rw = random.choice(choices)
        row_lo = max(h_total // 2 - 2, 0)
        row_hi = h_total - rh
        if row_lo > row_hi:
            continue
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(0, w_total - rw)
        candidate = (r0, c0, rh, rw)
        if all(_boxes_far_enough(candidate, rect) for rect in rects):
            return rects + [candidate]
    return None


def _gen_attempt() -> np.ndarray | None:
    (h1, w1), (h2, w2), gap = _sample_primary_pair()
    required_w = w1 + gap + w2 + 2
    w_total = random.randint(max(GRID_MIN, required_w), GRID_MAX)
    h_total = random.randint(GRID_MIN, GRID_MAX)

    row_max = min(5, h_total - max(h1, h2) - 8)
    if row_max < 1:
        return None
    r0 = random.randint(1, row_max)

    left_max = w_total - (w1 + gap + w2) - 1
    if left_max < 1:
        return None
    c1 = random.randint(1, left_max)
    c2 = c1 + w1 + gap
    rects: list[Rect] = [(r0, c1, h1, w1), (r0, c2, h2, w2)]

    if random.random() < 0.55:
        extended = _try_add_third_rect(rects, (h_total, w_total), {(h1, w1), (h2, w2)})
        if extended is not None:
            rects = extended

    grid = _render_input(rects, (h_total, w_total))
    if not _grid_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-stamp enclosure input")

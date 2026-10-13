from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_RECTS = 3
MAX_RECTS = 5
MIN_GRID_SIDE = 16
MAX_GRID_SIDE = 30
MIN_RECT_SIDE = 2
MAX_RECT_SIDE = 13
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 300


@dataclass(frozen=True)
class Rect:
    r0: int
    c0: int
    h: int
    w: int

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    @property
    def area(self) -> int:
        return self.h * self.w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = map(int, mask.shape)

    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue

        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _grey_rectangles(g: np.ndarray) -> list[Rect]:
    rects: list[Rect] = []
    for coords in _components(g == GREY):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert int(coords.shape[0]) == h * w, "grey components must be solid rectangles"
        assert r1 < g.shape[0] - 1, "grey rectangles must not occupy the legend row"
        rects.append(Rect(r0, c0, h, w))

    assert rects, "expected at least one grey rectangle"
    areas = [rect.area for rect in rects]
    assert len(set(areas)) == len(areas), "grey rectangle areas must be distinct"
    return rects


def _legend_colors(g: np.ndarray, count: int) -> list[int]:
    bottom = g[-1, :]
    cols = np.where(bottom != BLACK)[0]
    assert len(cols) == count, "legend color count must match the rectangle count"
    assert np.all(np.diff(cols) == 1), "legend colors must form one contiguous bottom-row strip"

    colors = [int(bottom[c]) for c in cols]
    assert GREY not in colors, "legend colors must be non-grey"
    assert len(set(colors)) == len(colors), "legend colors must be distinct"
    return colors


def _centered_slice(center2: int, length: int, limit: int) -> slice:
    start = (int(center2) - int(length) + 1) // 2
    stop = start + int(length)
    assert 0 <= start and stop <= int(limit), "centered rectangle must remain in bounds"
    return slice(start, stop)


def _draw_centered_stack(shape: tuple[int, int], rects: list[Rect], colors: list[int]) -> np.ndarray:
    ordered = sorted(rects, key=lambda rect: -rect.area)
    assert len(ordered) == len(colors)

    largest = ordered[0]
    center2_r = largest.r0 + largest.r1
    center2_c = largest.c0 + largest.c1

    out = np.full(shape, BLACK, dtype=int)
    for rect, color in zip(ordered, colors):
        rs = _centered_slice(center2_r, rect.h, shape[0])
        cs = _centered_slice(center2_c, rect.w, shape[1])
        out[rs, cs] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack scattered grey rectangles at the largest rectangle's center and recolor them from the bottom legend.

    Input structure:
    - The background is black.
    - Several non-overlapping grey solid rectangles are scattered above the bottom row, with distinct areas.
    - The bottom row contains one contiguous strip of distinct non-grey color pixels; its length equals the rectangle count.

    Rule:
    - Find every grey rectangle and sort them by area from largest to smallest.
    - Read the bottom legend strip from left to right and pair those colors with the sorted rectangles.
    - Clear the original rectangles and the legend strip to black.
    - Redraw the rectangles in descending area order, keeping each width and height, all centered on the largest rectangle's original center.

    Key insight:
    The grey rectangles are size-ranked masks, while the bottom legend supplies the recoloring order; their positions are discarded except for the largest rectangle's center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    rects = _grey_rectangles(g)
    colors = _legend_colors(g, len(rects))
    return _draw_centered_stack(g.shape, rects, colors)


def _sample_dimension_sequence(count: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        max_h = random.randint(7, MAX_RECT_SIDE)
        max_w = random.randint(7, MAX_RECT_SIDE)
        dims = [(max_h, max_w)]
        used_areas = {max_h * max_w}
        prev_h, prev_w = max_h, max_w
        prev_area = max_h * max_w

        ok = True
        for _rank in range(1, count):
            candidates: list[tuple[int, int]] = []
            for h in range(MIN_RECT_SIDE, prev_h + 1):
                if h % 2 != max_h % 2:
                    continue
                for w in range(MIN_RECT_SIDE, prev_w + 1):
                    if w % 2 != max_w % 2:
                        continue
                    area = h * w
                    if area < prev_area and area not in used_areas:
                        candidates.append((h, w))
            if not candidates:
                ok = False
                break

            candidates.sort(key=lambda item: item[0] * item[1], reverse=True)
            pool = candidates[: max(1, len(candidates) // 2)]
            h, w = random.choice(pool)
            dims.append((h, w))
            used_areas.add(h * w)
            prev_h, prev_w = h, w
            prev_area = h * w

        if ok:
            return dims
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    height, width = map(int, occupied.shape)
    r1 = int(r0 + h)
    c1 = int(c0 + w)
    if r0 < 0 or c0 < 0 or r1 > height - 1 or c1 > width:
        return False

    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(height, r1 + 1)
    pc1 = min(width, c1 + 1)
    return not bool(np.any(occupied[pr0:pr1, pc0:pc1]))


def _place_rectangles(height: int, width: int, dims: list[tuple[int, int]]) -> list[Rect] | None:
    occupied = np.zeros((height, width), dtype=bool)
    occupied[-1, :] = True
    rects: list[Rect | None] = [None] * len(dims)
    order = list(range(len(dims)))
    random.shuffle(order)

    for idx in order:
        h, w = dims[idx]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            max_r0 = height - h - 1
            max_c0 = width - w
            if max_r0 < 0 or max_c0 < 0:
                return None
            r0 = random.randint(0, max_r0)
            c0 = random.randint(0, max_c0)
            if _can_place(occupied, r0, c0, h, w):
                occupied[r0 : r0 + h, c0 : c0 + w] = True
                rects[idx] = Rect(r0, c0, h, w)
                placed = True
                break
        if not placed:
            return None

    return [rect for rect in rects if rect is not None]


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(MIN_RECTS, MAX_RECTS)
    dims = _sample_dimension_sequence(count)
    if dims is None:
        return None

    max_h = max(h for h, _w in dims)
    max_w = max(w for _h, w in dims)
    height = random.randint(max(MIN_GRID_SIDE, max_h + 4), MAX_GRID_SIDE)
    width = random.randint(max(MIN_GRID_SIDE, max_w + count + 2), MAX_GRID_SIDE)

    rects = _place_rectangles(height, width, dims)
    if rects is None or len(rects) != count:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    for rect in rects:
        grid[rect.r0 : rect.r1 + 1, rect.c0 : rect.c1 + 1] = GREY

    colors = random.sample(PALETTE, count)
    legend_start = random.randint(0, width - count)
    grid[-1, legend_start : legend_start + count] = np.array(colors, dtype=int)

    expected = _draw_centered_stack(grid.shape, rects, colors)
    try:
        result = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(result, expected):
        return None
    if max(grid.shape) > 30 or max(result.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed807260302/1/8 input")

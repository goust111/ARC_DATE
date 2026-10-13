from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


BLACK = 0
RED = 2
RECT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

GRID_MIN = 8
GRID_MAX = 28
RECT_COUNT_RANGE = (3, 5)
AXIS_SIZE_RANGE = (2, 5)
BAND_SIZE_RANGE = (1, 4)
EXTEND_RANGE = (0, 4)
MARGIN_RANGE = (1, 3)
MAX_GAP = 6
GEN_ATTEMPTS = 500


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width


@dataclass(frozen=True)
class PairInfo:
    first: Rect
    second: Rect
    gap: int
    overlap: int


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel())
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[Rect]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[Rect] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert area == len(coords), "each object must be a solid rectangle"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle interior must be monochrome"
            rects.append(Rect(color=color, r0=r0, c0=c0, r1=r1, c1=c1))

    colors = [rect.color for rect in rects]
    assert len(rects) >= RECT_COUNT_RANGE[0], "expected at least three rectangles"
    assert len(rects) <= RECT_COUNT_RANGE[1], "expected at most five rectangles"
    assert len(colors) == len(set(colors)), "rectangle colors must be distinct"
    return rects


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _horizontal_pairs(rects: list[Rect]) -> list[PairInfo] | None:
    ordered = sorted(rects, key=lambda rect: (rect.c0, rect.r0))
    pairs: list[PairInfo] = []
    for left, right in zip(ordered, ordered[1:]):
        gap = right.c0 - left.c1 - 1
        if gap <= 0:
            return None
        overlap = _interval_overlap(left.r0, left.r1, right.r0, right.r1)
        if overlap <= 0:
            return None
        pairs.append(PairInfo(first=left, second=right, gap=gap, overlap=overlap))
    return pairs


def _vertical_pairs(rects: list[Rect]) -> list[PairInfo] | None:
    ordered = sorted(rects, key=lambda rect: (rect.r0, rect.c0))
    pairs: list[PairInfo] = []
    for top, bottom in zip(ordered, ordered[1:]):
        gap = bottom.r0 - top.r1 - 1
        if gap <= 0:
            return None
        overlap = _interval_overlap(top.c0, top.c1, bottom.c0, bottom.c1)
        if overlap <= 0:
            return None
        pairs.append(PairInfo(first=top, second=bottom, gap=gap, overlap=overlap))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw one red bridge across the uniquely smallest gap between adjacent aligned rectangles.

    Input structure:
    - Background is black.
    - The grid contains 3-5 separated solid monochrome rectangles of distinct colors.
    - Rectangles are arranged either left-to-right with overlapping row ranges,
      or top-to-bottom with overlapping column ranges.

    Rule:
    - For each adjacent pair along the shared axis, measure the background gap.
    - Select the unique pair with the smallest gap.
    - Fill the entire gap between that pair with a red rectangle whose thickness
      equals the overlap length on the perpendicular axis.
    - Keep every original rectangle unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    rects = _extract_rectangles(g, bg)

    axis_candidates: list[tuple[str, list[PairInfo]]] = []
    horizontal = _horizontal_pairs(rects)
    if horizontal is not None:
        axis_candidates.append(("horizontal", horizontal))
    vertical = _vertical_pairs(rects)
    if vertical is not None:
        axis_candidates.append(("vertical", vertical))
    assert len(axis_candidates) == 1, "layout must be aligned on exactly one axis"

    axis, pairs = axis_candidates[0]
    min_gap = min(info.gap for info in pairs)
    best = [info for info in pairs if info.gap == min_gap]
    assert len(best) == 1, "smallest adjacent gap must be unique"
    chosen = best[0]

    out = g.copy()
    if axis == "horizontal":
        r0 = max(chosen.first.r0, chosen.second.r0)
        r1 = min(chosen.first.r1, chosen.second.r1)
        c0 = chosen.first.c1 + 1
        c1 = chosen.second.c0 - 1
        out[r0 : r1 + 1, c0 : c1 + 1] = RED
    else:
        c0 = max(chosen.first.c0, chosen.second.c0)
        c1 = min(chosen.first.c1, chosen.second.c1)
        r0 = chosen.first.r1 + 1
        r1 = chosen.second.r0 - 1
        out[r0 : r1 + 1, c0 : c1 + 1] = RED
    return out


def _sample_distinct_gaps(count: int) -> list[int]:
    return random.sample(range(1, MAX_GAP + 1), count)


def _horizontal_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    widths = [random.randint(*AXIS_SIZE_RANGE) for _ in range(rect_count)]
    band = random.randint(*BAND_SIZE_RANGE)
    above = [random.randint(*EXTEND_RANGE) for _ in range(rect_count)]
    below = [random.randint(*EXTEND_RANGE) for _ in range(rect_count)]
    gaps = _sample_distinct_gaps(rect_count - 1)

    max_above = max(above)
    max_below = max(below)
    content_h = max_above + band + max_below
    content_w = sum(widths) + sum(gaps)

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    h = content_h + top + bottom
    w = content_w + left + right
    if not (GRID_MIN <= h <= GRID_MAX and GRID_MIN <= w <= GRID_MAX):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    colors = random.sample(RECT_COLORS, rect_count)

    c = left
    for i, color in enumerate(colors):
        r0 = top + (max_above - above[i])
        r1 = top + (max_above + band + below[i] - 1)
        c1 = c + widths[i] - 1
        grid[r0 : r1 + 1, c : c1 + 1] = color
        c = c1 + 1
        if i < rect_count - 1:
            c += gaps[i]
    return grid


def _vertical_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    heights = [random.randint(*AXIS_SIZE_RANGE) for _ in range(rect_count)]
    band = random.randint(*BAND_SIZE_RANGE)
    left_ext = [random.randint(*EXTEND_RANGE) for _ in range(rect_count)]
    right_ext = [random.randint(*EXTEND_RANGE) for _ in range(rect_count)]
    gaps = _sample_distinct_gaps(rect_count - 1)

    max_left = max(left_ext)
    max_right = max(right_ext)
    content_h = sum(heights) + sum(gaps)
    content_w = max_left + band + max_right

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)
    h = content_h + top + bottom
    w = content_w + left + right
    if not (GRID_MIN <= h <= GRID_MAX and GRID_MIN <= w <= GRID_MAX):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    colors = random.sample(RECT_COLORS, rect_count)

    r = top
    for i, color in enumerate(colors):
        c0 = left + (max_left - left_ext[i])
        c1 = left + (max_left + band + right_ext[i] - 1)
        r1 = r + heights[i] - 1
        grid[r : r1 + 1, c0 : c1 + 1] = color
        r = r1 + 1
        if i < rect_count - 1:
            r += gaps[i]
    return grid


def _generated_grid_ok(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    except Exception:
        return False

    diff = (out != grid)
    if not np.any(diff):
        return False
    if not np.all(out[diff] == RED):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        if random.choice((False, True)):
            grid = _horizontal_attempt()
        else:
            grid = _vertical_attempt()
        if grid is not None and _generated_grid_ok(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")

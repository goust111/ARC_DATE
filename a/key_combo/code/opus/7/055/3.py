from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
BARRIER = 5
RED = 2
GREEN = 3
YELLOW = 4

GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _count_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _top_segment_span(grid: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(grid[0] == YELLOW)
    if cols.size == 0:
        raise ValueError("expected a yellow segment on the top edge")
    c0 = int(cols.min())
    c1 = int(cols.max())
    if not np.all(grid[0, c0 : c1 + 1] == YELLOW):
        raise ValueError("top yellow segment must be contiguous")
    return c0, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the teal background regions and use that count as the stripe cycle length.

    Input structure:
    - The grid uses teal (8) as background, grey (5) as a thin barrier network,
      red (2) as small interior blotches, and one short yellow (4) segment on the
      top border.
    - The grey barrier does not touch the top edge and splits the teal background
      into several 4-connected regions.

    Rule:
    - Count the number N of 4-connected teal background regions.
    - Starting from the top yellow segment, paint every other row downward.
    - Stripe colors follow a cycle of length N: the first N-1 stripes are yellow
      and only fill the segment's column span, while the Nth stripe is green and
      fills the entire row.
    - A green stripe recolors everything on that row, including grey barrier pixels
      and red blotches, to green.

    Key insight:
    - The barrier's structural complexity is encoded as a temporal period: more
      teal regions mean green flood rows appear less often.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    c0, c1 = _top_segment_span(g)
    period = _count_components(g == BG)

    out = g.copy()
    stripe_index = 0
    for row in range(0, h, 2):
        stripe_index += 1
        if stripe_index % period == 0:
            out[row, :] = GREEN
        else:
            out[row, c0 : c1 + 1] = YELLOW
    return out


def _rect_ring_cells(top: int, left: int, height: int, width: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    bottom = top + height - 1
    right = left + width - 1
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _safe_for_red(
    grid: np.ndarray,
    occupied: np.ndarray,
    cells: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BG or occupied[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BARRIER:
                return False
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _place_red_blob(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    *,
    target_row: int | None = None,
) -> bool:
    patterns = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]
    rng.shuffle(patterns)
    h, w = grid.shape

    if target_row is not None:
        candidates = [(target_row, c) for c in range(1, w - 1)]
        rng.shuffle(candidates)
        for r, c in candidates:
            if _safe_for_red(grid, occupied, [(r, c)]):
                grid[r, c] = RED
                occupied[r, c] = True
                return True
        return False

    for pattern in patterns:
        ph = max(dr for dr, _ in pattern) + 1
        pw = max(dc for _, dc in pattern) + 1
        row_choices = list(range(1, h - ph))
        col_choices = list(range(1, w - pw))
        rng.shuffle(row_choices)
        rng.shuffle(col_choices)
        for top in row_choices:
            for left in col_choices:
                cells = [(top + dr, left + dc) for dr, dc in pattern]
                if _safe_for_red(grid, occupied, cells):
                    for r, c in cells:
                        grid[r, c] = RED
                        occupied[r, c] = True
                    return True
    return False


def _build_barrier(
    rng: random.Random,
    n_regions: int,
) -> tuple[np.ndarray, int] | None:
    loop_count = n_regions - 1
    first_green_row = 2 * n_regions - 2
    h_min = max(13, first_green_row + 4)
    h_max = min(20, first_green_row + 8)
    if h_min > h_max:
        return None
    h = rng.randint(h_min, h_max)

    widths: list[int] = []
    heights: list[int] = []
    for _ in range(loop_count):
        widths.append(rng.randint(4, 5 if loop_count >= 4 else 6))
        heights.append(rng.randint(4, 6))

    gaps = [rng.randint(1, 2 if loop_count >= 4 else 3) for _ in range(loop_count - 1)]
    extent = sum(widths) + sum(gaps)
    w_min = max(17, extent + 3)
    w_max = min(30, extent + 7)
    if w_min > w_max:
        return None
    w = rng.randint(w_min, w_max)

    grid = np.full((h, w), BG, dtype=int)
    barrier = np.zeros((h, w), dtype=bool)

    extra = w - extent
    start_col = rng.randint(0, extra)
    tops: list[int] = []
    lefts: list[int] = []

    col = start_col
    for idx in range(loop_count):
        height = heights[idx]
        width = widths[idx]
        low = max(2, first_green_row - (height - 2))
        high = min(first_green_row - 1, h - height - 1)
        if low > high:
            return None
        top = rng.randint(low, high)
        tops.append(top)
        lefts.append(col)
        ring = _rect_ring_cells(top, col, height, width)
        for r, c in ring:
            barrier[r, c] = True
        col += width
        if idx < loop_count - 1:
            col += gaps[idx]

    for idx in range(loop_count - 1):
        left_loop_right = lefts[idx] + widths[idx] - 1
        right_loop_left = lefts[idx + 1]
        for c in range(left_loop_right, right_loop_left + 1):
            barrier[first_green_row, c] = True

    if barrier[0].any():
        return None
    grid[barrier] = BARRIER
    if _count_components(grid == BG) != n_regions:
        return None
    if not np.any(grid[first_green_row] == BARRIER):
        return None
    return grid, first_green_row


def _segment_candidates(grid: np.ndarray, length: int) -> list[int]:
    _, w = grid.shape
    good = []
    for start in range(0, w - length + 1):
        cols = range(start, start + length)
        if all(not np.any(grid[:, c] == BARRIER) for c in cols):
            good.append(start)
    return good


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(3, 6)
    built = _build_barrier(rng, n_regions)
    if built is None:
        return None
    grid, first_green_row = built
    h, w = grid.shape

    lengths = [2, 3]
    rng.shuffle(lengths)
    seg_start = None
    seg_len = None
    for length in lengths:
        starts = _segment_candidates(grid, length)
        if starts:
            seg_start = rng.choice(starts)
            seg_len = length
            break
    if seg_start is None or seg_len is None:
        return None
    grid[0, seg_start : seg_start + seg_len] = YELLOW

    red_occupied = np.zeros((h, w), dtype=bool)
    if not _place_red_blob(rng, grid, red_occupied, target_row=first_green_row):
        return None

    extra_blobs = rng.randint(1, 4)
    placed = 0
    for _ in range(extra_blobs * 5):
        if placed >= extra_blobs:
            break
        if _place_red_blob(rng, grid, red_occupied):
            placed += 1
    if placed < extra_blobs:
        return None

    if _count_components(grid == BG) != n_regions:
        return None
    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid != BG):
        return None
    if np.any(grid[0] == BARRIER):
        return None
    if np.any(grid[:, seg_start : seg_start + seg_len] == BARRIER):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.any((grid == BARRIER) & (out == GREEN)):
        return None
    if not np.any((grid == RED) & (out == GREEN)):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None

    green_rows = list(range(first_green_row, h, 2 * n_regions))
    if not green_rows:
        return None
    if not all(np.all(out[r] == GREEN) for r in green_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")

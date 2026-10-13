from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
FALLING_COLORS = (1, 2, 4, 6, 7, 8, 9)

GEN_RETRIES = 400
MIN_H = 15
MAX_H = 24
MIN_W = 12
MAX_W = 22
MIN_TERRAIN = 3
MAX_TERRAIN = 9
MAX_CAVITIES = 3
MAX_STACK = 3


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _edge_reachable_black(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if grid[r, c] == BLACK and not reachable[r, c]:
            reachable[r, c] = True
            q.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))
    return reachable


def _settle_columns(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    out = np.zeros_like(g)
    out[g == GREEN] = GREEN

    for c in range(w):
        blockers = np.flatnonzero(g[:, c] == GREEN).tolist()
        bounds = [-1, *blockers, h]
        for lo, hi in zip(bounds, bounds[1:]):
            start = lo + 1
            stop = hi
            if start >= stop:
                continue
            falling = [
                int(g[r, c])
                for r in range(stop - 1, start - 1, -1)
                if int(g[r, c]) not in (BLACK, GREEN)
            ]
            dest = stop - 1
            for color in falling:
                out[dest, c] = color
                dest -= 1
    return out


def _recolor_over_cavities(settled: np.ndarray) -> np.ndarray:
    reachable = _edge_reachable_black(settled)
    enclosed = (settled == BLACK) & ~reachable
    out = settled.copy()
    h, w = map(int, settled.shape)

    for c in range(w):
        has_enclosed_below = False
        for r in range(h - 1, -1, -1):
            cell = int(settled[r, c])
            if cell == BLACK:
                has_enclosed_below = has_enclosed_below or bool(enclosed[r, c])
            elif cell != GREEN and has_enclosed_below:
                out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each colored pixel straight down by column, then recolor pixels grey when an enclosed cavity lies somewhere below them in the same column.

    Input structure:
    - The grid has a black background with green wall terrain occupying lower cells in each column.
    - Some black pockets are completely enclosed by green terrain and do not reach the grid border.
    - Non-black, non-green single pixels are scattered above the terrain and may share columns.

    Rule:
    - Treat every column independently and let every non-black, non-green pixel fall vertically within its own open vertical segment.
    - Pixels stack from the bottom of the segment upward, stopping on the grid bottom, on green terrain, or on previously settled pixels; green cells never move.
    - After all pixels settle, inspect each settled colored pixel and look at every black cell below it in the same column.
    - If at least one of those lower black cells belongs to a 4-connected black region that cannot reach the grid border, recolor that pixel grey; otherwise keep its color.

    Key insight:
    - The visible motion is only the first step; the final recoloring depends on whether the column's supporting ground hides a topological hole rather than on the landing height alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    settled = _settle_columns(g)
    return _recolor_over_cavities(settled)


def _gen_heights(h: int, w: int) -> np.ndarray:
    low = MIN_TERRAIN
    high = min(MAX_TERRAIN, h - 5)
    heights = np.zeros(w, dtype=int)
    current = random.randint(low, high)
    c = 0
    while c < w:
        seg_len = random.randint(2, 5)
        if c > 0:
            current = max(low, min(high, current + random.choice((-2, -1, 0, 1, 2))))
        heights[c : min(w, c + seg_len)] = current
        c += seg_len
    return heights


def _build_terrain(h: int, w: int, heights: np.ndarray) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for c, height in enumerate(heights.tolist()):
        grid[h - height :, c] = GREEN
    return grid


def _grow_connected_mask(box_h: int, box_w: int) -> np.ndarray | None:
    area = box_h * box_w
    target = random.randint(max(2, area // 2), area)
    mask = np.zeros((box_h, box_w), dtype=bool)
    seed = (random.randrange(box_h), random.randrange(box_w))
    mask[seed] = True

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        coords = np.argwhere(mask)
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True

    cols_used = len({int(c) for _r, c in np.argwhere(mask)})
    if cols_used < min(2, box_w):
        return None
    return mask


def _carve_cavities(grid: np.ndarray, heights: np.ndarray) -> np.ndarray | None:
    out = grid.copy()
    h, w = map(int, out.shape)
    used = np.zeros((h, w), dtype=bool)
    target_cavities = random.randint(1, min(MAX_CAVITIES, max(1, w // 6)))
    carved = 0

    for _ in range(120):
        if carved >= target_cavities:
            break
        support_w = random.randint(4, min(8, w))
        support_c0 = random.randint(0, w - support_w)
        carve_c0 = support_c0 + 1
        carve_w = support_w - 2
        if carve_w < 2:
            continue
        box_h = random.randint(1, min(3, h - 4))
        support_cols = range(support_c0, support_c0 + support_w)
        surface = max(h - int(heights[c]) for c in support_cols)
        top_lo = surface + 1
        top_hi = h - box_h - 2
        if top_lo > top_hi:
            continue
        r0 = random.randint(top_lo, top_hi)
        r1 = r0 + box_h - 1
        if used[max(0, r0 - 1) : min(h, r1 + 2), max(0, carve_c0 - 1) : min(w, carve_c0 + carve_w + 1)].any():
            continue
        shape = _grow_connected_mask(box_h, carve_w)
        if shape is None:
            continue

        rr, cc = np.where(shape)
        abs_r = r0 + rr
        abs_c = carve_c0 + cc
        if np.any(out[abs_r, abs_c] != GREEN):
            continue
        out[abs_r, abs_c] = BLACK
        used[max(0, r0 - 1) : min(h, r1 + 2), max(0, carve_c0 - 1) : min(w, carve_c0 + carve_w + 1)] = True
        carved += 1

    if carved == 0:
        return None

    reachable = _edge_reachable_black(out)
    enclosed = (out == BLACK) & ~reachable
    if not np.any(enclosed):
        return None
    return out


def _surface_rows(terrain: np.ndarray) -> np.ndarray:
    h, w = map(int, terrain.shape)
    surface = np.full(w, h, dtype=int)
    for c in range(w):
        green_rows = np.flatnonzero(terrain[:, c] == GREEN)
        if green_rows.size:
            surface[c] = int(green_rows[0])
    return surface


def _pick_drop_columns(surface: np.ndarray, enclosed: np.ndarray) -> list[int] | None:
    w = int(surface.shape[0])
    grey_cols = [c for c in range(w) if surface[c] >= 1 and np.any(enclosed[:, c])]
    clear_cols = [c for c in range(w) if surface[c] >= 1 and not np.any(enclosed[:, c])]
    if not grey_cols or len(clear_cols) < 2:
        return None

    grey_take = random.randint(1, min(3, len(grey_cols)))
    clear_take = random.randint(2, min(5, len(clear_cols)))
    cols = random.sample(grey_cols, grey_take) + random.sample(clear_cols, clear_take)
    random.shuffle(cols)
    return cols


def _final_rows(surface_row: int, count: int) -> list[int]:
    return list(range(surface_row - count, surface_row))


def _place_settled_pixels(terrain: np.ndarray, enclosed: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = map(int, terrain.shape)
    surface = _surface_rows(terrain)
    cols = _pick_drop_columns(surface, enclosed)
    if cols is None:
        return None

    color_pool = random.sample(FALLING_COLORS, random.randint(3, min(5, len(FALLING_COLORS))))
    settled = terrain.copy()
    expected = terrain.copy()
    input_grid = terrain.copy()
    moved_any = False
    total_pixels = 0
    multi_stack_cols = 0

    for c in cols:
        max_count = min(MAX_STACK, int(surface[c]))
        if max_count <= 0:
            return None
        if max_count >= 2 and random.random() < 0.4:
            count = random.randint(2, max_count)
        else:
            count = 1
        total_pixels += count
        multi_stack_cols += int(count >= 2)
        bottom_to_top = [random.choice(color_pool) for _ in range(count)]
        distinct_rows = sorted(random.sample(range(int(surface[c])), count))
        final_rows = _final_rows(int(surface[c]), count)
        moved_any = moved_any or distinct_rows != final_rows

        for row, color in zip(reversed(final_rows), bottom_to_top):
            settled[row, c] = color
            expected[row, c] = GREY if np.any(enclosed[row + 1 :, c]) else color

        top_to_bottom_colors = list(reversed(bottom_to_top))
        for row, color in zip(distinct_rows, top_to_bottom_colors):
            input_grid[row, c] = color

    if total_pixels < 5 or total_pixels > 12:
        return None
    if not moved_any:
        return None
    if multi_stack_cols == 0 and any(surface[c] >= 2 for c in cols):
        return None

    colored_out = expected[(expected != BLACK) & (expected != GREEN)]
    if np.count_nonzero(colored_out == GREY) == 0:
        return None
    if np.count_nonzero((colored_out != GREY)) == 0:
        return None
    if len(set(map(int, input_grid[(input_grid != BLACK) & (input_grid != GREEN)].tolist()))) < 3:
        return None
    return input_grid, expected


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_H, MAX_H)
    w = random.randint(MIN_W, MAX_W)
    heights = _gen_heights(h, w)
    terrain = _build_terrain(h, w, heights)
    terrain = _carve_cavities(terrain, heights)
    if terrain is None:
        return None

    reachable = _edge_reachable_black(terrain)
    enclosed = (terrain == BLACK) & ~reachable
    placed = _place_settled_pixels(terrain, enclosed)
    if placed is None:
        return None
    input_grid, expected = placed

    actual = main(input_grid)
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")

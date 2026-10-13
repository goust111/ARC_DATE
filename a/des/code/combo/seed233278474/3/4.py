from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RIGHT_TURNS = {
    ((0, 1), (1, 0)),
    ((1, 0), (0, -1)),
    ((0, -1), (-1, 0)),
    ((-1, 0), (0, 1)),
}

RNG = np.random.default_rng()
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _top_segments(mask: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    top_r = int(np.where(mask.any(axis=1))[0][0])
    cols = np.where(mask[top_r])[0]
    segments: list[tuple[int, int]] = []
    seg_start = int(cols[0])
    prev = int(cols[0])
    for c in cols[1:]:
        c = int(c)
        if c == prev + 1:
            prev = c
            continue
        segments.append((seg_start, prev))
        seg_start = prev = c
    segments.append((seg_start, prev))
    return top_r, segments


def _vertical_run(mask: np.ndarray, r: int, c: int) -> int:
    run = 1
    rr = r + 1
    while rr < mask.shape[0] and mask[rr, c]:
        run += 1
        rr += 1
    return run


def _choose_start(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = mask.shape
    top_r, segments = _top_segments(mask)
    candidates: list[tuple[int, int, int, int, tuple[int, int], tuple[int, int]]] = []
    for c0, c1 in segments:
        for c, dir0, prefer_left in ((c0, (0, 1), 1), (c1, (0, -1), 0)):
            nr, nc = top_r + dir0[0], c + dir0[1]
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                continue
            if top_r + 1 >= h or not mask[top_r + 1, c]:
                continue
            run = _vertical_run(mask, top_r, c)
            # Prefer the endpoint whose downward arm is longer. Ties use the widest
            # top segment and then a stable left-before-right fallback.
            stable_col = -c if prefer_left else c
            candidates.append((run, c1 - c0, prefer_left, stable_col, (top_r, c), dir0))
    if not candidates:
        cols = np.where(mask[top_r])[0]
        left = int(cols[0])
        right = int(cols[-1])
        if right > left:
            return (top_r, left), (0, 1)
        return (top_r, right), (0, -1)
    _, _, _, _, start, dir0 = max(candidates)
    return start, dir0


def _classify_loop(mask: np.ndarray) -> bool:
    pts = set(map(tuple, np.argwhere(mask)))
    start, step = _choose_start(mask)
    prev = start
    cur = (start[0] + step[0], start[1] + step[1])
    if cur not in pts:
        return step == (0, 1)

    while True:
        next_pts = []
        for dr, dc in DIR4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in pts and nxt != prev:
                next_pts.append(nxt)
        if len(next_pts) > 1 and start in next_pts:
            next_pts = [nxt for nxt in next_pts if nxt != start]
        if not next_pts:
            return step == (0, 1)
        nxt = next_pts[0]
        new_step = (nxt[0] - cur[0], nxt[1] - cur[1])
        if new_step != step:
            return (step, new_step) in RIGHT_TURNS
        prev, cur, step = cur, nxt, new_step


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    clockwise_color = int(g[h - 2, w - 1])
    counterclockwise_color = int(g[h - 1, w - 1])

    out = g.copy()
    blue_mask = g == BLUE
    blue_components = _components(blue_mask)
    blue_labels = np.full((h, w), -1, dtype=int)
    loop_is_clockwise: list[bool] = []

    for idx, comp in enumerate(blue_components):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        submask = blue_mask[r0 : r1 + 1, c0 : c1 + 1]
        loop_is_clockwise.append(_classify_loop(submask))
        for r, c in comp:
            blue_labels[r, c] = idx

    for comp in _components(g == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        adjacent_loops = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and blue_labels[nr, nc] >= 0:
                    adjacent_loops.add(int(blue_labels[nr, nc]))
        if len(adjacent_loops) != 1:
            continue
        loop_idx = next(iter(adjacent_loops))
        fill = clockwise_color if loop_is_clockwise[loop_idx] else counterclockwise_color
        for r, c in comp:
            out[r, c] = fill

    return out


def _histogram_cells(heights: list[int]) -> set[tuple[int, int]]:
    return {(r, c) for c, height in enumerate(heights) for r in range(int(height))}


def _boundary_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells) + 1
    max_c = max(c for _, c in cells) + 1
    mask = np.zeros((2 * max_r + 1, 2 * max_c + 1), dtype=bool)
    for r, c in cells:
        if (r - 1, c) not in cells:
            mask[2 * r, 2 * c : 2 * c + 3] = True
        if (r + 1, c) not in cells:
            mask[2 * r + 2, 2 * c : 2 * c + 3] = True
        if (r, c - 1) not in cells:
            mask[2 * r : 2 * r + 3, 2 * c] = True
        if (r, c + 1) not in cells:
            mask[2 * r : 2 * r + 3, 2 * c + 2] = True
    return mask


def _stretch_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return mask.copy()
    h, w = mask.shape
    out = np.zeros(((h - 1) * scale + 1, (w - 1) * scale + 1), dtype=bool)
    pts = np.argwhere(mask)
    for r, c in pts:
        out[r * scale, c * scale] = True
    for r, c in pts:
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                continue
            sr, sc = r * scale, c * scale
            er, ec = nr * scale, nc * scale
            if sr == er:
                lo, hi = sorted((sc, ec))
                out[sr, lo : hi + 1] = True
            else:
                lo, hi = sorted((sr, er))
                out[lo : hi + 1, sc] = True
    return out


def _random_heights(width: int, max_height: int) -> list[int]:
    heights = [int(RNG.integers(2, max_height + 1)) for _ in range(width)]
    if len(set(heights)) == 1:
        idx = int(RNG.integers(width))
        choices = [v for v in range(2, max_height + 1) if v != heights[idx]]
        heights[idx] = int(RNG.choice(choices))
    if heights[0] == heights[-1]:
        if heights[0] < max_height and (heights[-1] == 2 or RNG.random() < 0.5):
            heights[0] += 1
        else:
            heights[-1] -= 1
    return heights


def _make_loop(clockwise: bool, count: int, grid_h: int, grid_w: int) -> np.ndarray | None:
    max_scale = 2 if count <= 2 and min(grid_h, grid_w) >= 22 else 1
    scale = int(RNG.integers(1, max_scale + 1))
    max_width = 6 if count == 1 else 5 if count == 2 else 4
    max_height = 6 if count == 1 else 5 if count == 2 else 4
    width = int(RNG.integers(3, max_width + 1))
    heights = _random_heights(width, max_height)
    if heights[0] == heights[-1]:
        return None
    if clockwise and heights[0] < heights[-1]:
        heights = heights[::-1]
    if not clockwise and heights[0] > heights[-1]:
        heights = heights[::-1]
    if heights[0] == heights[-1]:
        return None

    mask = _stretch_mask(_boundary_mask(_histogram_cells(heights)), scale)
    if _classify_loop(mask) != clockwise:
        return None
    return mask


def _can_place(occupied: np.ndarray, shape: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = shape.shape
    r0 = max(0, top - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c0 = max(0, left - pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _count_enclosed_regions(grid: np.ndarray) -> int:
    h, w = grid.shape
    total = 0
    for comp in _components(grid == BLACK):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]))
    if count == 1:
        grid_h = int(RNG.integers(17, 25))
        grid_w = int(RNG.integers(17, 25))
    elif count == 2:
        grid_h = int(RNG.integers(20, 29))
        grid_w = int(RNG.integers(20, 29))
    elif count == 3:
        grid_h = int(RNG.integers(24, 33))
        grid_w = int(RNG.integers(24, 33))
    else:
        grid_h = int(RNG.integers(28, 35))
        grid_w = int(RNG.integers(28, 35))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    occupied[max(0, grid_h - 3) : grid_h, max(0, grid_w - 3) : grid_w] = True
    grid[grid_h - 2, grid_w - 1] = RED
    grid[grid_h - 1, grid_w - 1] = GREEN

    orientations = [bool(RNG.integers(2)) for _ in range(count)]
    if count > 1 and RNG.random() < 0.6:
        orientations[0] = True
        orientations[1] = False
        RNG.shuffle(orientations)

    placed = 0
    for clockwise in orientations:
        shape = None
        for _ in range(80):
            candidate = _make_loop(clockwise, count, grid_h, grid_w)
            if candidate is None:
                continue
            h, w = candidate.shape
            if h + 2 < grid_h and w + 2 < grid_w:
                shape = candidate
                break
        if shape is None:
            return None

        sh, sw = shape.shape
        spot_found = False
        for _ in range(200):
            top = int(RNG.integers(1, grid_h - sh))
            left = int(RNG.integers(1, grid_w - sw))
            if not _can_place(occupied, shape, top, left, pad=1):
                continue
            patch = grid[top : top + sh, left : left + sw]
            patch[shape] = BLUE
            occupied[max(0, top - 1) : min(grid_h, top + sh + 1), max(0, left - 1) : min(grid_w, left + sw + 1)] = True
            spot_found = True
            placed += 1
            break
        if not spot_found:
            return None

    if placed != count:
        return None
    if _count_enclosed_regions(grid) != count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[grid == BLUE], grid[grid == BLUE]):
        return None
    if int(out[grid_h - 2, grid_w - 1]) != RED or int(out[grid_h - 1, grid_w - 1]) != GREEN:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid winding-loop ARC input")

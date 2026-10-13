from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEAL = 8
INF = 10**9
GEN_ATTEMPTS = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bfs_distances(grid: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), INF, dtype=int)
    if int(grid[seed]) == TEAL:
        return dist
    queue = deque([seed])
    dist[seed] = 0

    while queue:
        r, c = queue.popleft()
        next_dist = int(dist[r, c]) + 1
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != TEAL and int(dist[nr, nc]) == INF:
                dist[nr, nc] = next_dist
                queue.append((nr, nc))
    return dist


def _meeting_data(grid: np.ndarray):
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    assert len(seeds) >= 2, "expected at least two grey seeds"

    dist_stack = np.stack([_bfs_distances(grid, seed) for seed in seeds], axis=0)
    min_dist = dist_stack.min(axis=0)
    reachable = min_dist < INF
    winner_count = np.sum(dist_stack == min_dist[None, :, :], axis=0)
    red_mask = (grid == BLACK) & reachable & (winner_count >= 2)
    return seeds, dist_stack, min_dist, reachable, red_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Mark the wall-aware equal-arrival frontier where multiple seed halos first meet.

    Input structure:
    - A black background contains several grey single-pixel seeds and one or more cyan wall segments.
    - Walls are impassable, while every non-wall cell is reachable from at least one seed.

    Rule:
    - All seeds expand synchronously through the 8-neighborhood, advancing one step per round and never entering cyan wall cells.
    - A background cell turns red iff two or more different seeds first reach that cell in the same round.
    - Preserve the original grey seeds and cyan walls; leave every other cell black.

    Key insight:
    - The red pattern is the geodesic equidistant frontier under king-move distance around obstacles, so walls bend the meeting line away from the naive straight midpoint set.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _, _, _, _, red_mask = _meeting_data(g)

    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY
    out[g == TEAL] = TEAL
    out[red_mask] = RED
    return out


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _chebyshev_ties(shape: tuple[int, int], seeds: list[tuple[int, int]], bg_mask: np.ndarray) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    dist_stack = np.stack(
        [np.maximum(np.abs(rows - r), np.abs(cols - c)) for r, c in seeds],
        axis=0,
    )
    min_dist = dist_stack.min(axis=0)
    return bg_mask & (np.sum(dist_stack == min_dist[None, :, :], axis=0) >= 2)


def _build_layout(h: int, w: int, rng: np.random.Generator):
    orient = str(rng.choice(["v", "h"]))
    layout = str(rng.choice(["single", "hook"], p=[0.55, 0.45]))

    if orient == "v":
        c = int(rng.integers(4, w - 4))
        min_len = max(6, h // 2)
        max_len = h - 4
        if min_len > max_len:
            return None
        length = int(rng.integers(min_len, max_len + 1))
        r0 = int(rng.integers(2, h - length - 1))
        primary = [(r0 + i, c) for i in range(length)]
        segments = [primary]

        if layout == "hook":
            attach_top = bool(rng.integers(2))
            side = int(rng.choice([-1, 1]))
            rr = r0 if attach_top else r0 + length - 1
            max_hook = c - 3 if side < 0 else w - c - 4
            if max_hook < 4:
                return None
            hook_len = int(rng.integers(4, min(8, max_hook) + 1))
            if side < 0:
                hook = [(rr, c - d) for d in range(1, hook_len + 1)]
            else:
                hook = [(rr, c + d) for d in range(1, hook_len + 1)]
            segments.append(hook)

        mid = r0 + length // 2
        left_hi = min(5, c - 1)
        right_hi = min(5, w - c - 1)
        if left_hi < 2 or right_hi < 2:
            return None
        seed_a = (
            int(np.clip(mid + int(rng.integers(-2, 3)), 1, h - 2)),
            c - int(rng.integers(2, left_hi + 1)),
        )
        seed_b = (
            int(np.clip(mid + int(rng.integers(-2, 3)), 1, h - 2)),
            c + int(rng.integers(2, right_hi + 1)),
        )
        return segments, [seed_a, seed_b]

    r = int(rng.integers(4, h - 4))
    min_len = max(6, w // 2)
    max_len = w - 4
    if min_len > max_len:
        return None
    length = int(rng.integers(min_len, max_len + 1))
    c0 = int(rng.integers(2, w - length - 1))
    primary = [(r, c0 + i) for i in range(length)]
    segments = [primary]

    if layout == "hook":
        attach_left = bool(rng.integers(2))
        side = int(rng.choice([-1, 1]))
        cc = c0 if attach_left else c0 + length - 1
        max_hook = r - 3 if side < 0 else h - r - 4
        if max_hook < 4:
            return None
        hook_len = int(rng.integers(4, min(8, max_hook) + 1))
        if side < 0:
            hook = [(r - d, cc) for d in range(1, hook_len + 1)]
        else:
            hook = [(r + d, cc) for d in range(1, hook_len + 1)]
        segments.append(hook)

    mid = c0 + length // 2
    up_hi = min(5, r - 1)
    down_hi = min(5, h - r - 1)
    if up_hi < 2 or down_hi < 2:
        return None
    seed_a = (
        r - int(rng.integers(2, up_hi + 1)),
        int(np.clip(mid + int(rng.integers(-2, 3)), 1, w - 2)),
    )
    seed_b = (
        r + int(rng.integers(2, down_hi + 1)),
        int(np.clip(mid + int(rng.integers(-2, 3)), 1, w - 2)),
    )
    return segments, [seed_a, seed_b]


def _place_extra_seeds(grid: np.ndarray, count: int, rng: np.random.Generator) -> bool:
    if count == 0:
        return True

    h, w = grid.shape
    wall_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == TEAL)]
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)]
    regions = [
        (1, max(2, h // 3), 1, max(2, w // 3)),
        (1, max(2, h // 3), max(1, 2 * w // 3), w - 2),
        (max(1, 2 * h // 3), h - 2, 1, max(2, w // 3)),
        (max(1, 2 * h // 3), h - 2, max(1, 2 * w // 3), w - 2),
    ]
    rng.shuffle(regions)

    placed = 0
    tries = 0
    while placed < count and tries < 220:
        tries += 1
        if tries <= len(regions) * 24:
            r0, r1, c0, c1 = regions[(tries - 1) % len(regions)]
            if r0 > r1 or c0 > c1:
                continue
            r = int(rng.integers(r0, r1 + 1))
            c = int(rng.integers(c0, c1 + 1))
        else:
            r = int(rng.integers(1, h - 1))
            c = int(rng.integers(1, w - 1))

        if int(grid[r, c]) != BLACK:
            continue
        if any(max(abs(r - sr), abs(c - sc)) < 5 for sr, sc in seeds):
            continue
        if any(max(abs(r - wr), abs(c - wc)) < 2 for wr, wc in wall_cells):
            continue

        grid[r, c] = GREY
        seeds.append((r, c))
        placed += 1
    return placed == count


def _validate_candidate(grid: np.ndarray, segments: list[list[tuple[int, int]]]) -> bool:
    seeds, _, _, reachable, red_mask = _meeting_data(grid)
    free_mask = grid != TEAL
    bg_mask = grid == BLACK

    if not bool(np.all(reachable[free_mask])):
        return False

    red_count = int(red_mask.sum())
    if red_count < 4 or red_count > int(max(30, free_mask.sum() // 4)):
        return False

    for comp in _components8(free_mask):
        if not any(int(grid[r, c]) == GREY for r, c in comp):
            return False

    if np.array_equal(_chebyshev_ties(grid.shape, seeds, bg_mask), red_mask):
        return False

    for segment in segments:
        test_grid = grid.copy()
        for r, c in segment:
            test_grid[r, c] = BLACK
        changed_red = _meeting_data(test_grid)[4] & bg_mask
        if np.array_equal(changed_red, red_mask):
            return False

    for seed in seeds:
        test_grid = grid.copy()
        test_grid[seed] = BLACK
        if int(np.count_nonzero(test_grid == GREY)) < 2:
            continue
        changed_red = _meeting_data(test_grid)[4] & bg_mask
        if np.array_equal(changed_red, red_mask):
            return False

    return True


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 27))
    w = int(rng.integers(16, 27))
    built = _build_layout(h, w, rng)
    if built is None:
        return None
    segments, seeds = built

    grid = np.zeros((h, w), dtype=int)
    for segment in segments:
        for r, c in segment:
            if int(grid[r, c]) != BLACK:
                return None
            grid[r, c] = TEAL

    for seed in seeds:
        if int(grid[seed]) != BLACK:
            return None
        grid[seed] = GREY

    target_seed_count = int(rng.choice([2, 3, 4], p=[0.45, 0.35, 0.20]))
    if not _place_extra_seeds(grid, target_seed_count - 2, rng):
        return None
    if not _validate_candidate(grid, segments):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid受阻光晕样例")

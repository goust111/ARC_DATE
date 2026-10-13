from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
STRIPE = 2
WALL = 3
FILL = 4
OBSTACLE = 5

MAX_GENERATION_ATTEMPTS = 500


def _bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int, int]:
    blue_mask = grid == FRAME
    r0, c0, r1, c1 = _bbox_of(blue_mask)
    assert r1 - r0 >= 6 and c1 - c0 >= 5

    expected = np.zeros_like(blue_mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(blue_mask, expected)

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape[0] >= 5 and inner.shape[1] >= 3
    stripe_rows = []
    for rr in range(inner.shape[0]):
        row = inner[rr]
        if np.all(row == STRIPE):
            stripe_rows.append(r0 + 1 + rr)
        else:
            assert np.all(row == BG)

    assert len(stripe_rows) >= 2
    diffs = np.diff(stripe_rows)
    assert np.all(diffs == diffs[0])
    period = int(diffs[0])
    assert period > 0
    return (r0, c0, r1, c1), int(stripe_rows[0]), period


def _find_wall_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    wall_mask = grid == WALL
    r0, c0, r1, c1 = _bbox_of(wall_mask)
    assert r1 - r0 >= 5 and c1 - c0 >= 4

    expected = np.zeros_like(wall_mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    assert np.array_equal(wall_mask, expected)
    return r0, c0, r1, c1


def _seed_coord(grid: np.ndarray) -> tuple[int, int]:
    seeds = np.argwhere(grid == FILL)
    assert len(seeds) == 1
    return int(seeds[0, 0]), int(seeds[0, 1])


def _row_active_mask(height: int, anchor_row: int, period: int) -> np.ndarray:
    rows = np.arange(height)
    return ((rows - anchor_row) % period) == 0


def _raw_fill_mask(grid: np.ndarray, wall_bbox: tuple[int, int, int, int], seed: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    r0, c0, r1, c1 = wall_bbox
    sr, sc = seed
    assert r0 < sr < r1 and c0 < sc < c1

    passable = (grid == BG) | (grid == FILL)
    seen = np.zeros_like(grid, dtype=bool)
    q = deque([seed])
    seen[sr, sc] = True

    def inside_window(r: int, c: int) -> bool:
        return r0 < r < r1 and c0 < c <= c1

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if seen[nr, nc] or not passable[nr, nc] or not inside_window(nr, nc):
                continue
            seen[nr, nc] = True
            q.append((nr, nc))

    out = seen.copy()
    exit_rows = [r for r in range(r0 + 1, r1) if seen[r, c1]]
    for r in exit_rows:
        for c in range(c1 + 1, w):
            if grid[r, c] != BG:
                break
            out[r, c] = True
    return out


def _apply_transform(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    anchor_row: int,
    period: int,
    wall_bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
) -> np.ndarray:
    raw_fill = _raw_fill_mask(grid, wall_bbox, seed)
    active_rows = _row_active_mask(grid.shape[0], anchor_row, period)
    keep = raw_fill & active_rows[:, None]

    out = np.array(grid, copy=True)
    out[raw_fill] = BG
    out[keep] = FILL

    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a distant framed stripe code to mask a right-radiating flood-fill from a U-shaped wall.

    Input structure:
    - The background is black. In the upper-right sits a one-cell-thick blue rectangle whose interior contains full-width red horizontal stripes.
    - In the lower-left sits a green U-shaped wall open to the right, with one yellow seed inside and one or two grey obstacles on the outward beam rows.

    Rule:
    - Read the rectangle's interior stripe rows, infer their constant row period, and extend that row mask to the full grid using the first stripe row as the anchor.
    - Flood-fill the seed's enclosed region, then radiate rightward through the wall opening until the grid edge or the first non-background cell in each beam row; grey obstacles stop fill on their row and cast a shadow behind themselves.
    - Keep only flood-filled yellow cells on active mask rows, erase the blue frame and all red stripe cells, and preserve the green wall plus grey obstacles.

    Key insight:
    - The blue rectangle is only an encoder: its separated stripe spacing controls which rows survive from the wall-driven flood-fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_bbox, anchor_row, period = _find_frame(g)
    wall_bbox = _find_wall_bbox(g)
    seed = _seed_coord(g)
    return _apply_transform(g, frame_bbox, anchor_row, period, wall_bbox, seed)


def _expected_from_params(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    wall_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    fr0, _fc0, _fr1, _fc1 = frame_bbox
    anchor_row = fr0 + 2
    period = 3
    seed = _seed_coord(grid)
    return _apply_transform(grid, frame_bbox, anchor_row, period, wall_bbox, seed)


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    inner_h = r1 - r0 - 1
    for local_r in range(1, inner_h, 3):
        grid[r0 + 1 + local_r, c0 + 1 : c1] = STRIPE


def _draw_wall(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _build_input_from_params(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    wall_bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    obstacles: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    _draw_frame(grid, frame_bbox)
    _draw_wall(grid, wall_bbox)
    sr, sc = seed
    grid[sr, sc] = FILL
    for r, c in obstacles:
        grid[r, c] = OBSTACLE
    return grid


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    wall_bbox: tuple[int, int, int, int],
    blocked_rows: set[int],
) -> bool:
    fr0, _fc0, _fr1, _fc1 = frame_bbox
    wr0, wc0, wr1, wc1 = wall_bbox
    anchor = fr0 + 2
    active_rows = _row_active_mask(grid.shape[0], anchor, 3)
    raw_fill = _raw_fill_mask(grid, wall_bbox, _seed_coord(grid))
    final_fill = expected == FILL

    interior_rows = list(range(wr0 + 1, wr1))
    active_inside = [r for r in interior_rows if active_rows[r]]
    inactive_inside = [r for r in interior_rows if not active_rows[r]]
    if len(active_inside) < len(blocked_rows) + 1 or not inactive_inside:
        return False

    if raw_fill.sum() <= final_fill.sum() or final_fill.sum() == 0:
        return False

    if not np.any(raw_fill[~active_rows, :]):
        return False
    if not np.any(final_fill[active_rows, :]):
        return False

    if np.any(expected[frame_bbox[0] : frame_bbox[2] + 1, frame_bbox[1] : frame_bbox[3] + 1] != BG):
        return False

    clear_active_rows = [r for r in active_inside if r not in blocked_rows]
    if not clear_active_rows:
        return False
    if not any(np.any(final_fill[r, wc1 + 1 :]) for r in clear_active_rows):
        return False

    for r in blocked_rows:
        obstacle_cols = np.flatnonzero(grid[r] == OBSTACLE)
        if len(obstacle_cols) != 1:
            return False
        oc = int(obstacle_cols[0])
        if oc >= grid.shape[1] - 1 or oc <= wc1 + 1:
            return False
        if np.any(final_fill[r, oc + 1 :]):
            return False
        if not np.any(final_fill[r, wc1 + 1 : oc]):
            return False

    if not np.array_equal(main(grid), expected):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 24)
    w = rng.randint(19, 26)

    frame_h = rng.randint(7, 10)
    frame_w = rng.randint(6, 9)
    fr0 = rng.randint(1, 4)
    max_fc0 = w - frame_w - 1
    min_fc0 = max(w - frame_w - 4, 1)
    if min_fc0 > max_fc0:
        return None
    fc0 = rng.randint(min_fc0, max_fc0)
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    obstacle_count = rng.choice([1, 2])
    inner_h_choices = [4, 5, 6, 7]
    if obstacle_count == 2:
        inner_h_choices = [v for v in inner_h_choices if v >= 6]
    wall_inner_h = rng.choice(inner_h_choices)
    wall_inner_w = rng.randint(3, 5)
    wall_h = wall_inner_h + 2
    wall_w = wall_inner_w + 2

    wr0_min = fr0 + frame_h + 2
    wr0_max = h - wall_h - 1
    if wr0_min > wr0_max:
        return None
    wr0 = rng.randint(wr0_min, wr0_max)
    wc0_max = min(4, fc0 - wall_w - 3)
    if wc0_max < 1:
        return None
    wc0 = rng.randint(1, wc0_max)
    wall_bbox = (wr0, wc0, wr0 + wall_h - 1, wc0 + wall_w - 1)

    if not _boxes_separated(frame_bbox, wall_bbox):
        return None

    anchor = fr0 + 2
    interior_rows = list(range(wr0 + 1, wr0 + wall_inner_h + 1))
    active_rows = [r for r in interior_rows if (r - anchor) % 3 == 0]
    inactive_rows = [r for r in interior_rows if (r - anchor) % 3 != 0]
    if len(active_rows) < obstacle_count + 1 or not inactive_rows:
        return None

    blocked_rows = set(rng.sample(active_rows, obstacle_count))
    clear_active_rows = [r for r in active_rows if r not in blocked_rows]
    if not clear_active_rows:
        return None

    wc1 = wall_bbox[3]
    obstacle_col_lo = wc1 + 2
    obstacle_col_hi = w - 2
    if obstacle_col_lo > obstacle_col_hi:
        return None

    obstacles = []
    used_cols = set()
    for r in sorted(blocked_rows):
        choices = [c for c in range(obstacle_col_lo, obstacle_col_hi + 1) if c not in used_cols]
        if not choices:
            return None
        c = rng.choice(choices)
        obstacles.append((r, c))
        used_cols.add(c)

    seed_row_pool = inactive_rows if rng.random() < 0.5 else interior_rows
    sr = rng.choice(seed_row_pool)
    sc = rng.randint(wc0 + 1, wc1 - 1)
    seed = (sr, sc)

    grid = _build_input_from_params((h, w), frame_bbox, wall_bbox, seed, obstacles)
    expected = _expected_from_params(grid, frame_bbox, wall_bbox)
    if not _semantic_checks(grid, expected, frame_bbox, wall_bbox, blocked_rows):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")

"""ARC Solver for puzzle 5.
This module provides:
- main(grid: np.ndarray) -> np.ndarray
- generate_input() -> np.ndarray
"""

from __future__ import annotations

import numpy as np
from collections import deque
from collections.abc import Iterable

BG = 0
WALL = 5
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return a 1-based label image and list of components for True cells."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    visited = np.zeros((h, w), dtype=np.bool_)
    comps: list[np.ndarray] = []

    comp_id = 0
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue

            comp_id += 1
            visited[r, c] = True
            q.append((r, c))
            cells: list[tuple[int, int]] = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not mask[nr, nc] or visited[nr, nc]:
                        continue
                    visited[nr, nc] = True
                    labels[nr, nc] = comp_id
                    q.append((nr, nc))
                    cells.append((nr, nc))

            for rr, cc in cells:
                labels[rr, cc] = comp_id
            comps.append(np.asarray(cells, dtype=np.int64))

    return labels, comps


def _find_cavity_components(grid: np.ndarray, wall_labels: np.ndarray) -> dict[int, np.ndarray]:
    """Collect enclosed background components and map each to a wall id."""
    h, w = grid.shape
    walkable = grid != WALL
    outside = np.zeros((h, w), dtype=np.bool_)
    q: deque[tuple[int, int]] = deque()

    for c in range(w):
        if walkable[0, c] and not outside[0, c]:
            outside[0, c] = True
            q.append((0, c))
        if walkable[h - 1, c] and not outside[h - 1, c]:
            outside[h - 1, c] = True
            q.append((h - 1, c))

    for r in range(1, h - 1):
        if walkable[r, 0] and not outside[r, 0]:
            outside[r, 0] = True
            q.append((r, 0))
        if walkable[r, w - 1] and not outside[r, w - 1]:
            outside[r, w - 1] = True
            q.append((r, w - 1))

    while q:
        cr, cc = q.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not walkable[nr, nc] or outside[nr, nc]:
                continue
            outside[nr, nc] = True
            q.append((nr, nc))

    cavity_mask = (grid == BG) & (~outside)
    if not cavity_mask.any():
        return {}

    # Label cavity components (max size expected <=9 by task definition).
    _, cavity_groups = _connected_components(cavity_mask)

    mapping: dict[int, np.ndarray] = {}
    for comp in cavity_groups:
        if len(comp) == 0:
            continue

        # Count wall ids touching this component.
        counts: dict[int, int] = {}
        for r, c in comp:
            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                wall_id = int(wall_labels[nr, nc])
                if wall_id == 0:
                    continue
                counts[wall_id] = counts.get(wall_id, 0) + 1

        if not counts:
            continue

        # Most-supported wall owns this cavity.
        wall_id = max(counts.items(), key=lambda item: item[1])[0]
        if wall_id not in mapping or len(comp) > len(mapping[wall_id]):
            mapping[wall_id] = comp

    return mapping


def _find_nearest_wall_for_seed(
    r: int,
    c: int,
    grid: np.ndarray,
    wall_labels: np.ndarray,
) -> tuple[int, int, int] | None:
    """Find nearest enclosing wall for a seed by ray-casting up/down/left/right."""
    h, w = grid.shape

    best = None
    for dr, dc in DIRECTIONS:
        rr, cc = r + dr, c + dc
        dist = 1
        while 0 <= rr < h and 0 <= cc < w:
            v = grid[rr, cc]
            if v == WALL:
                wall_id = int(wall_labels[rr, cc])
                if wall_id != 0:
                    if best is None or dist < best[0]:
                        best = (dist, wall_id, dr, dc)
                break
            if v != BG:
                break
            rr += dr
            cc += dc
            dist += 1

    return None if best is None else (best[1], best[2], best[3])


def main(grid: np.ndarray) -> np.ndarray:
    """Transform wall cavities into color patterns around seeds.

    1) Detect all grey walls and index each wall component.
    2) Compute each enclosed background cavity and associate it to its wall.
    3) For every seed pixel, select the nearest wall by direct up/down/left/right scan.
    4) Place a translated copy of that wall cavity: center -> seed then shift 1 step farther from wall.
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("main expects a numpy.ndarray input")

    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("main expects a 2D grid")

    h, w = grid.shape
    arr = np.array(grid, dtype=np.int64)
    out = arr.copy()

    wall_mask = arr == WALL
    if not wall_mask.any():
        return out

    wall_labels, wall_components = _connected_components(wall_mask)
    if not wall_components:
        return out

    cavity_by_wall = _find_cavity_components(arr, wall_labels)

    # Seed pixels are any non-zero and non-wall color.
    seeds = np.argwhere((arr != BG) & (arr != WALL))

    for sr, sc in seeds:
        sr_i, sc_i = int(sr), int(sc)
        pair = _find_nearest_wall_for_seed(sr_i, sc_i, arr, wall_labels)
        if pair is None:
            continue

        wall_id, away_r, away_c = pair
        cavity = cavity_by_wall.get(wall_id)
        if cavity is None or len(cavity) == 0:
            continue

        seed_color = int(arr[sr_i, sc_i])
        # Align geometry center to the seed, then shift one step outward from the wall.
        ctr = cavity.mean(axis=0)
        shift_r = int(np.rint(sr_i - ctr[0] + away_r))
        shift_c = int(np.rint(sc_i - ctr[1] + away_c))

        for r, c in cavity:
            nr, nc = int(r + shift_r), int(c + shift_c)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if out[nr, nc] != BG:
                continue
            out[nr, nc] = seed_color

    return out


def _draw_rect_wall(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    """Draw a 1-pixel-thick rectangular wall."""
    grid[r0, c0:c1 + 1] = WALL
    grid[r1, c0:c1 + 1] = WALL
    grid[r0:r1 + 1, c0] = WALL
    grid[r0:r1 + 1, c1] = WALL


def generate_input() -> np.ndarray:
    """Generate one random valid input example following the puzzle constraints."""
    rng = np.random.default_rng()
    h = int(rng.integers(12, 30))
    w = int(rng.integers(12, 30))
    grid = np.zeros((h, w), dtype=np.int64)

    seed_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    color_idx = 0

    wall_count = int(rng.integers(1, 4))
    placed = 0

    for _ in range(wall_count):
        for _ in range(200):
            inner_h = int(rng.integers(1, 4))
            inner_w = int(rng.integers(1, 4))
            wall_h = inner_h + 2
            wall_w = inner_w + 2
            if h <= wall_h + 4 or w <= wall_w + 4:
                continue

            r0 = int(rng.integers(2, h - wall_h - 1))
            c0 = int(rng.integers(2, w - wall_w - 1))
            r1 = r0 + wall_h - 1
            c1 = c0 + wall_w - 1

            if np.any(grid[r0 - 1:r1 + 2, c0 - 1:c1 + 2] != BG):
                continue

            _draw_rect_wall(grid, r0, c0, r1, c1)

            directions = [
                (-1, 0, "up", r0 - 1, range(c0, c1 + 1)),
                (1, 0, "down", r1 + 1, range(c0, c1 + 1)),
                (0, -1, "left", c0 - 1, range(r0, r1 + 1)),
                (0, 1, "right", c1 + 1, range(r0, r1 + 1)),
            ]
            rng.shuffle(directions)

            placed_seed = False
            for dr, dc, _name, fixed, variable in directions:
                vars_list = list(variable)
                if not vars_list:
                    continue
                for _ in range(6):
                    v = int(rng.choice(vars_list))
                    if dr == 0:
                        sr, sc = fixed, v
                    else:
                        sr, sc = v, fixed
                    if not (0 <= sr < h and 0 <= sc < w):
                        continue
                    if grid[sr, sc] != BG:
                        continue
                    # Keep seed exactly one cell away from that wall side.
                    if dr != 0 and ((dc != 0)):
                        pass
                    grid[sr, sc] = seed_colors[color_idx % len(seed_colors)]
                    color_idx += 1
                    placed_seed = True
                    break
                if placed_seed:
                    break

            if not placed_seed:
                # Clean up and try another placement.
                grid[r0 : r1 + 1, c0 : c1 + 1] = BG
            else:
                placed += 1
                break

    # If anything failed to place, still return the best effort layout.
    if placed == 0:
        # Fallback single small sample (always valid).
        h2, w2 = max(h, 12), max(w, 12)
        if h2 > h or w2 > w:
            grid = np.zeros((h2, w2), dtype=np.int64)
        h, w = grid.shape
        r0, c0, r1, c1 = 3, 3, 5, 6
        _draw_rect_wall(grid, r0, c0, r1, c1)
        grid[r0 - 1, c0 + 1] = seed_colors[0]

    return grid


def _quick_self_check() -> None:
    """Run small internal sanity checks for shape constraints and invariants."""
    for _ in range(4):
        inp = generate_input()
        out = main(inp)

        assert inp.shape == out.shape
        assert np.array_equal((inp == WALL), (out == WALL))

        seeds = np.argwhere((inp != BG) & (inp != WALL))
        for sr, sc in seeds:
            sr_i, sc_i = int(sr), int(sc)
            assert inp[sr_i, sc_i] == out[sr_i, sc_i]

    print("quick self-check passed")


if __name__ == "__main__":
    _quick_self_check()

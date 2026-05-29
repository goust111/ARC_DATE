# Variant of dd6b8c4b
# Dimension: D4 — 填充目标区域的方式 (path)
# Change: row_major_fill——按行优先顺序从绿色方块内部左上角开始，逐个填入收集到的栗色像素... → 列优先填充

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the D4 variant
# ---
# source_task: none
# concrete_behavior:
#   - The scene keeps the original 11x11 layout: orange floor, optional magenta walls,
#     a fixed 3x3 green target with a red center at (5,5), and scattered maroon tokens.
#   - The only intended semantic change is how the destination block is filled after the
#     reachable maroons are selected. Instead of scanning the 3x3 block row by row
#     [(4,4),(4,5),(4,6),(5,4),...], this variant scans by columns
#     [(4,4),(5,4),(6,4),(4,5),...].
# behavior_trace:
#   - Example with k=4 reachable maroons:
#     original output fills {(4,4),(4,5),(4,6),(5,4)}
#     variant  output fills {(4,4),(5,4),(6,4),(4,5)}
#     difference: the fourth token moves down the first column instead of extending across
#     the top row, so the shape becomes a left-heavy "L" rather than a top-heavy "L".
# complexity_calibration:
#   - Reachability and distance ranking stay unchanged, preserving the need to read wall
#     corridors. The only branch we expose is the fill path, and it has exactly two
#     visually distinct candidates: row-major vs column-major.
#   - Full 3x3 fill (k=9) would erase the distinction, so generation is restricted to k<=8.
# implicit_assumption_audit:
#   - Original `dest_cells.sort()` quietly encoded row-major order. In this variant, the
#     destination block still needs an unordered set for adjacency logic, but the fill path
#     must be built explicitly in column-major order.
#   - Original `for r, c in dest_cells[:k]` relied on that row-major ordering. It must be
#     replaced with `fill_cells[:k]` where `fill_cells` is column-major.
#   - Distance ranking of source maroons by `(dist,row,col)` remains valid because the
#     variant changes only the destination placement order, not source selection.
# placement_legality:
#   - The 3x3 target stays centered, so no destination cell goes out of bounds.
#   - Maroons are placed only on floor cells, never on walls or inside the target.
#   - Distractors are sampled only from BFS-unreachable floor cells, guaranteeing that the
#     solver will move exactly the intended reachable maroons.
# observability_and_cleanliness:
#   - Restricting k to 2..8 ensures most outputs differ from the original row-major task on
#     the same input. The color palette and wall structure stay unchanged, so the new signal
#     is a clean geometric difference inside the target block rather than extra clutter.
# color_reasoning:
#   - No new color mapping is introduced. Colors keep their original roles, so the human
#     solver only needs to infer a changed fill trajectory.
# generator_guarantees_for_main:
#   - `main()` assumes exactly one red center and at least one passable neighbor near the
#     target. The generator preserves both conditions and validates every sample by running
#     the solver before returning it.


def _dest_cells(center_r: int, center_c: int, h: int, w: int, order: str) -> list[tuple[int, int]]:
    rows = range(center_r - 1, center_r + 2)
    cols = range(center_c - 1, center_c + 2)
    cells: list[tuple[int, int]] = []
    if order == "row_major":
        for r in rows:
            for c in cols:
                if 0 <= r < h and 0 <= c < w:
                    cells.append((r, c))
    elif order == "col_major":
        for c in cols:
            for r in rows:
                if 0 <= r < h and 0 <= c < w:
                    cells.append((r, c))
    else:
        raise ValueError(f"unknown order: {order}")
    return cells


def _solve_with_fill_order(grid: np.ndarray, fill_order: str) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w, "row_major")
    fill_cells = _dest_cells(cr, cc, h, w, fill_order)
    dest_set = set(dest_cells)

    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, (
        "ring around destination must contain passable (non-maroon, non-red, non-wall) "
        "cells to infer floor color"
    )
    vals, cnts = np.unique(cand_ring, return_counts=True)
    ring_blocker = int(vals[cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, ring_blocker)]
    floor = other[0] if len(other) == 1 else (
        max(other, key=lambda color: int((grid == color).sum())) if other else None
    )
    if floor is None:
        floor = ring_blocker

    passable = {floor, 9}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                dist[rr, cc2] = d + 1
                q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [triple for triple in maroons if triple[0] < inf]
    maroons.sort()

    k = min(len(fill_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in fill_cells[:k]:
        out[r, c] = 9

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the nearest reachable maroon pixels into the 3x3 block around the lone red center, filling that block by columns.

    Input structure:
    - Orange floor cells surround a fixed green 3x3 target, with a unique red center and optional magenta wall barriers nearby.
    - Maroon source pixels are scattered on floor cells outside the target, with some reachable through corridor openings and others potentially trapped as distractors.

    Rule:
    - Flood-fill outward from passable cells adjacent to the target through floor and maroon cells; only reachable maroons are eligible.
    - Rank eligible maroons by BFS distance, breaking ties by row then column, erase the closest ones from their source positions, and place them into the target in column-major order from the top-left.
    - Any destination cells beyond the number of moved maroons keep their original green/red colors, and unreachable maroons stay in place.

    Key insight:
    - The walls still determine which maroons can be harvested at all, but this variant is identified by the destination pattern: it fills downward through the left column before moving to the next column.
    """

    return _solve_with_fill_order(grid, "col_major")


# --------------- generator (c) ---------------

# ---- CFG: tunable generation parameters ----
GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

FILL_K_RANGE = (2, 8)            # cap at 8 so column-major differs visibly from row-major
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 2
GEN_RETRIES = 300


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited = set()
    queue = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in dest_set:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    if r < CENTER_R:
        dirs.append((-1, 0))
    if r > CENTER_R:
        dirs.append((1, 0))
    if c < CENTER_C:
        dirs.append((0, -1))
    if c > CENTER_C:
        dirs.append((0, 1))
    return dirs


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    length = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(length):
        nr, nc = nr + dr, nc + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = set()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)

    num_gaps = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring, num_gaps))
    wall_ring = [cell for cell in ring if cell not in gap_cells]
    wall_cells = set(wall_ring)

    style = random.choice(WALL_STYLES)

    if style == "arms":
        for r, c in wall_ring:
            if random.random() < ARM_EXTEND_PROB:
                _extend_arm(wall_cells, dest_set, r, c, ARM_LENGTH_RANGE[1])
    elif style == "frame":
        dist2 = random.randint(*FRAME_DIST_RANGE)
        for r in range(CENTER_R - dist2, CENTER_R + dist2 + 1):
            for c in range(CENTER_C - dist2, CENTER_C + dist2 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in dest_set:
                    if r in (CENTER_R - dist2, CENTER_R + dist2) or c in (CENTER_C - dist2, CENTER_C + dist2):
                        wall_cells.add((r, c))
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr, nc = nr + dr, nc + dc
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            n_remove = random.randint(1, max(1, len(outer_frame) // 4))
            for rc in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, GRID_H - abs(r - CENTER_R) - 2)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W, "row_major")
    fill_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W, "col_major")
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    max_k = min(FILL_K_RANGE[1], len(reachable_floor))
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0

    total = k + num_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        elif max_k >= k + needed and k + needed <= FILL_K_RANGE[1]:
            k += needed
        else:
            return None
    if k + num_distractors > TOTAL_MAROON_RANGE[1]:
        num_distractors = max(0, TOTAL_MAROON_RANGE[1] - k)

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = grid.copy()
    for r, c in fill_cells[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON

    try:
        result = main(inp)
        row_major_result = _solve_with_fill_order(inp, "row_major")
        if np.array_equal(result, out) and not np.array_equal(result, row_major_result):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

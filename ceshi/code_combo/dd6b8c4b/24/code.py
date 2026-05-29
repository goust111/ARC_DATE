# Combo variant of dd6b8c4b (index 24)
# combo_spec: D7/2+D18/0
# Name: 复制而非移动+不可达像素擦除

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant:
# - Concrete behavior trace:
#   Original task: reachable maroon pixels were moved into the 3x3 target block, so the chosen
#   source cells turned back into orange floor; unreachable maroons stayed where they were.
#   This combo changes both ends at once: chosen reachable maroons are copied into the target
#   block but remain maroon at their original positions, while unreachable maroons are erased.
#   Example sketch: if two reachable maroons and one walled-off maroon are present, the original
#   output would keep only two maroons inside the block plus the unreachable distractor; the new
#   output keeps the two reachable maroons in place, adds two maroons inside the block, and removes
#   the unreachable distractor entirely.
# - Complexity calibration:
#   In this fused rule, the "closest k" ranking becomes visually irrelevant once all reachable
#   maroons stay in place and the destination uses a single color. To avoid hidden/unobservable
#   degrees of freedom, the generator only creates 2..9 reachable maroons, so every reachable
#   maroon is copied and none are silently excluded by the ranking step. It also forces at least
#   one unreachable distractor so D18/0 is visible instead of degenerating back toward the base task.
# - Assumption audit:
#   main() now assumes the original palette from the spec is fixed: center=2, target ring=3,
#   walls=6, floor=7, maroons=9. It also assumes one red center, a 3x3 target around it, at least
#   one passable opening adjacent to the block, and maroons placed only on floor cells outside the
#   block. generate_input() constructs only legal instances satisfying those assumptions.
# - Placement legality and cleanliness:
#   Walls are painted only outside the destination block; maroons are sampled only from floor cells,
#   partitioned into reachable and unreachable sets, so there is no overlap or out-of-bounds risk.
#   The output remains visually tidy because each region has a single role: target block shows the
#   copies, reachable corridors keep their maroons, and blocked regions become clean floor.
# - Human observability:
#   Across a few examples, a solver can see a stable contrast: maroons connected to a corridor
#   opening appear twice in the output, while maroons behind walls disappear. No hidden color-value
#   arithmetic or non-visual ordering is required.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

RING_GAP_RANGE = (1, 3)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

FILL_K_RANGE = (2, 9)
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 2
GEN_RETRIES = 500


def _dest_cells(center_r: int, center_c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(center_r - 1, center_r + 2):
        for c in range(center_c - 1, center_c + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _passable_starts(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    seen = set()
    dest_set = set(dest_cells)
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if (rr, cc) in dest_set:
                    continue
                if int(grid[rr, cc]) not in (FLOOR_COLOR, MAROON):
                    continue
                if (rr, cc) in seen:
                    continue
                seen.add((rr, cc))
                starts.append((rr, cc))
    return starts


def _reachable_maroons(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    h, w = grid.shape
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    starts = _passable_starts(grid, dest_cells)
    assert starts, "there must be at least one passable cell adjacent to the destination block"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    queue = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while queue:
        r, c = queue.popleft()
        for rr, cc2 in neighbors4(r, c):
            if not (0 <= rr < h and 0 <= cc2 < w):
                continue
            if int(grid[rr, cc2]) not in (FLOOR_COLOR, MAROON):
                continue
            if dist[rr, cc2] != inf:
                continue
            dist[rr, cc2] = dist[r, c] + 1
            queue.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    reachable = [item for item in maroons if item[0] < inf]
    reachable.sort()
    return reachable, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy reachable maroon pixels into the 3x3 block around the unique red center, while erasing blocked ones.

    Input structure:
    - Orange floor carries scattered maroon pixels, and magenta walls may carve the floor into connected and sealed-off regions.
    - A fixed 3x3 destination block is centered on the unique red pixel: eight green cells around one red center.

    Rule:
    - Flood-fill outward from floor or maroon cells touching the destination block, traversing only orange floor and maroon cells.
    - Maroon pixels reached by that search are the valid source set; the first k of them by distance, then row, then column are copied into the destination block in row-major order.
    - Those reachable sources stay maroon in place, but any maroon pixel not reached by the flood-fill is erased to orange floor.

    Key insight:
    - The output separates maroons by connectivity, not by raw proximity: corridor-connected maroons survive and also appear inside the block, while wall-isolated maroons disappear.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells(cr, cc, h, w)

    reachable, _dist = _reachable_maroons(grid)
    reachable_set = {(r, c) for _, r, c in reachable}
    all_maroons = {(int(r), int(c)) for r, c in np.argwhere(grid == MAROON)}
    unreachable = all_maroons - reachable_set

    for r, c in unreachable:
        out[r, c] = FLOOR_COLOR

    k = min(len(dest_cells), len(reachable))
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def _bfs_reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Reachable floor cells from openings adjacent to the destination block."""
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
    dirs = []
    if r < CENTER_R:
        dirs.append((-1, 0))
    if r > CENTER_R:
        dirs.append((1, 0))
    if c < CENTER_C:
        dirs.append((0, -1))
    if c > CENTER_C:
        dirs.append((0, 1))
    return dirs


def _extend_arm(
    wall_cells: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    r: int,
    c: int,
    max_len: int,
) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    nr, nc = r, c
    length = random.randint(1, max_len)
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


def _expected_output(
    base_grid: np.ndarray,
    source_cells: list[tuple[int, int]],
    dest_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = base_grid.copy()
    for r, c in source_cells:
        out[r, c] = MAROON
    for r, c in dest_cells[: len(source_cells)]:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable_floor(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor
    if not unreachable_floor:
        return None

    max_k = min(FILL_K_RANGE[1], len(reachable_floor), TOTAL_MAROON_RANGE[1] - 1)
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    max_total = min(TOTAL_MAROON_RANGE[1], k + min(DISTRACTOR_MAX, len(unreachable_floor)))
    min_total = max(TOTAL_MAROON_RANGE[0], k + 1)
    if min_total > max_total:
        return None
    total = random.randint(min_total, max_total)
    num_distractors = total - k

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(grid, source_cells, dest_cells)

    try:
        result = main(inp)
    except Exception:
        return None
    if np.array_equal(result, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

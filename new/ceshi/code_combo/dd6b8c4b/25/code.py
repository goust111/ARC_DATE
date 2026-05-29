# Combo variant of dd6b8c4b (index 25)
# combo_spec: D8/0+D11/1
# Name: 仅输出方块内部+跳过红色中心

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant
# - Concrete behavior trace:
#   In the original task, if six reachable maroon tokens are ranked closest first,
#   row-major filling writes into the full 3x3 block as:
#     (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),...
#   so a cropped view of the old output would contain a maroon center once k >= 5.
#   This combo changes two steps together:
#   1) only the 3x3 destination interior is emitted;
#   2) filling skips the red center, so the order becomes
#      (0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2).
#   For k = 6, the fused output is therefore:
#      [[9, 9, 9],
#       [9, 2, 9],
#       [9, 9, 3]]
#   instead of the old cropped behavior:
#      [[9, 9, 9],
#       [9, 9, 9],
#       [3, 3, 3]].
# - Complexity calibration:
#   The generator samples k from 5..8, not 0..8. This avoids degeneration:
#   if k <= 4, skipping the center would be invisible after the tight crop.
# - Implicit assumptions changed by the combo:
#   The solver now assumes the unique red center is the fixed anchor of a full 3x3
#   destination block and that at least one adjacent floor/maroon cell exists to seed
#   BFS. The generator guarantees this by keeping the center at (5,5), rejecting
#   layouts with too few reachable floor cells, and validating every sample through
#   main().
# - Placement legality:
#   Source maroons are sampled only from reachable floor cells outside the 3x3 block;
#   distractors are sampled only from unreachable floor cells; wall construction never
#   writes into the destination block.
# - Visual observability:
#   The output is intentionally tiny, so the center must act as a persistent reference
#   point. Every accepted sample keeps that red pixel visible and places at least five
#   maroons around it, making the "skip center" rule visible in 2-4 examples.
# - Cleanliness:
#   The cropped output contains only green, red, and maroon inside one 3x3 block, so
#   there is no overlapping annotation layer or ambiguous outside context.


GRID_H = 11
GRID_W = 11
CENTER_R = 5
CENTER_C = 5

FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
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

FILL_K_RANGE = (5, 8)
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (6, 16)
MIN_REACHABLE = 5
GEN_RETRIES = 400


def _dest_cells() -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(CENTER_R - 1, CENTER_R + 2)
        for c in range(CENTER_C - 1, CENTER_C + 2)
    ]


def _fill_slots() -> list[tuple[int, int]]:
    return [(r, c) for r, c in _dest_cells() if (r, c) != (CENTER_R, CENTER_C)]


def _rank_reachable_maroons(grid: np.ndarray) -> list[tuple[int, int, int]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])
    assert 0 < cr < h - 1 and 0 < cc < w - 1, "red center must support a full 3x3 block"

    dest_cells = [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]
    dest_set = set(dest_cells)
    passable = {FLOOR_COLOR, MAROON}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and (nr, nc) not in seen:
                seen.add((nr, nc))
                starts.append((nr, nc))
    assert starts, "there must be a passable entry adjacent to the destination block"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    ranked = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    ranked = [item for item in ranked if item[0] < inf]
    ranked.sort()
    return ranked


def _render_output_block(k: int, *, skip_center: bool) -> np.ndarray:
    out = np.full((3, 3), DEST_COLOR, dtype=int)
    out[1, 1] = RED_CENTER

    slots = _fill_slots() if skip_center else _dest_cells()
    for r, c in slots[: min(k, len(slots))]:
        out[r - (CENTER_R - 1), c - (CENTER_C - 1)] = MAROON
    return out


def _baseline_crop_without_skip(grid: np.ndarray) -> np.ndarray:
    ranked = _rank_reachable_maroons(grid)
    return _render_output_block(len(ranked), skip_center=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the task to the destination interior while keeping the red anchor untouched.

    Input structure:
    - The full 11x11 scene still contains orange floor, optional magenta walls, scattered maroon tokens, and one green 3x3 target block around a unique red center.
    - Maroon tokens never start inside the target block; the red center is the visual anchor of that block.

    Rule:
    - Flood-fill from passable cells adjacent to the target through orange and maroon cells, ignoring magenta walls.
    - Sort reachable maroon tokens by BFS distance, then by row and column.
    - Output only the 3x3 target interior.
    - Fill that 3x3 in row-major order but skip the red center cell, which always stays red.

    Key insight: The outside world is still needed to decide how many maroons qualify, but the returned grid is only the cropped destination state, with the center acting as a fixed reference point.
    """

    ranked = _rank_reachable_maroons(grid)
    k = min(len(_fill_slots()), len(ranked))
    return _render_output_block(k, skip_center=True)


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    queue = deque()

    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if int(grid[nr, nc]) == FLOOR_COLOR and (nr, nc) not in dest_set:
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
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
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
                    nr += dr
                    nc += dc
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


def _expected_output(k: int) -> np.ndarray:
    return _render_output_block(k, skip_center=True)


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set)

    reachable_floor = _bfs_reachable(base, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    max_k = min(FILL_K_RANGE[1], len(reachable_floor))
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(base[r, c]) == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = list(all_floor - reachable_floor)

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    if max_dist > 0 and random.random() < 0.8:
        num_distractors = random.randint(1, max_dist)
    else:
        num_distractors = 0

    total = k + num_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - total
        if max_dist >= num_distractors + needed:
            num_distractors += needed
        elif max_k >= k + needed:
            k += needed
        else:
            return None
    if k + num_distractors > TOTAL_MAROON_RANGE[1]:
        num_distractors = max(0, TOTAL_MAROON_RANGE[1] - k)

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(unreachable_floor, num_distractors)

    inp = base.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(k)
    if np.array_equal(out, _baseline_crop_without_skip(inp)):
        return None

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

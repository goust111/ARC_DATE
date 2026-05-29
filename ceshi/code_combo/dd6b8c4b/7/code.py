# Combo variant of dd6b8c4b (index 7)
# combo_spec: D4/2+D8/0
# Name: 螺旋填充+跳过红色中心
#
# Design analysis
# combo_variant:
#   invariant_structure:
#     grid: always 11x11
#     colors: floor=7, wall=6 or absent, destination ring=3, center marker=2, movable tokens=9
#     destination: fixed 3x3 block centered at (5,5)
#     source_selection: unchanged flood-fill reachability through floor/maroon via 4-neighbors
#   fused_change:
#     path:
#       old: row-major over all 9 destination cells
#       new: fixed clockwise spiral over the 3x3 block
#     transform:
#       old: the red center could be overwritten if enough maroons were moved
#       new: the center is a permanent hole; the spiral skips it instead of filling it last
#     effective_write_order:
#       - (cr-1, cc-1)
#       - (cr-1, cc)
#       - (cr-1, cc+1)
#       - (cr, cc+1)
#       - (cr+1, cc+1)
#       - (cr+1, cc)
#       - (cr+1, cc-1)
#       - (cr, cc-1)
#   behavior_trace:
#     original_with_5_reachable:
#       destination_after_move: top row, then middle-left, then center
#     combo_with_5_reachable:
#       destination_after_move: top row, then middle-right, then bottom-right
#     combo_with_10_reachable:
#       moved: only the closest 8 maroons enter the outer ring
#       preserved: the red center stays red and 2 farther reachable maroons remain outside
#     visible_difference:
#       the partial frontier bends around the perimeter, and saturated cases show a full maroon ring with a red hole
#   complexity_calibration:
#     fixed_orientation:
#       one canonical clockwise spiral is used; adding flipped or rotated spirals would create hidden branches with too little evidence
#     visibility_guard:
#       generation mixes two source-count regimes:
#       - 4..7 reachable maroons so the spiral prefix is visible
#       - 9..11 reachable maroons so skipping the center differs from the pure spiral variant that would fill it
#   implicit_assumption_audit:
#     still_required:
#       - exactly one red center exists
#       - the destination is the 3x3 block around that center
#       - at least one passable cell touches the destination so BFS can start
#     changed_code_paths:
#       - write capacity is now 8 cells rather than 9
#       - any code using sorted(dest_cells) as the final placement order is invalid
#       - output construction must preserve color 2 at (cr,cc) even when more than 8 reachable maroons exist
#     new_contracts_checked_in_generator:
#       - partial-mode examples never collapse to the original row-major prefix because k is kept at 4..7
#       - dense-mode examples have at least 9 reachable maroons so the center-skip behavior is exercised
#   placement_legality:
#     - source maroons are placed only on reachable floor cells, never in the destination or on walls
#     - extra reachable maroons in dense mode are chosen outside the first 8 ranked cells, so the solver deterministically leaves them behind
#     - distractor maroons are placed only on unreachable floor cells
#   visual_readability:
#     - all semantic cues remain local to the fixed 3x3 target, keeping the wall/corridor scene readable
#     - the red center is a stable landmark, so the skipped hole is visually obvious once the ring fills around it
#   inferability_check:
#     - no hidden use of color ids or randomness affects ranking; only geometry and visible reachability matter
#     - across a small training set, humans can see both the spiral perimeter order and the persistent red hole

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


def _dest_cells_row_major(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _dest_cells_spiral(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (0, 0),
    ]
    return [
        (cr + dr, cc + dc)
        for dr, dc in offsets
        if 0 <= cr + dr < h and 0 <= cc + dc < w
    ]


def _dest_cells_spiral_skip_center(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    return [cell for cell in _dest_cells_spiral(cr, cc, h, w) if cell != (cr, cc)]


def _infer_floor_color(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
) -> int:
    ring_colors: list[int] = []
    h, w = grid.shape
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, "ring around destination must contain passable cells to infer floor color"
    vals, cnts = np.unique(cand_ring, return_counts=True)
    ring_blocker = int(vals[cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, ring_blocker)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    if floor is None:
        floor = ring_blocker
    return floor


def _distance_map(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    passable: set[int],
) -> np.ndarray:
    h, w = grid.shape
    starts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    queue = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(queue):
        r, c = queue[qi]
        qi += 1
        d = int(dist[r, c])
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                dist[rr, cc2] = d + 1
                queue.append((rr, cc2))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the 3×3 block around the unique red center, tracing a spiral that leaves the center untouched.

    Input structure:
    - Orange floor cells contain scattered maroon tokens, while magenta walls may block some corridors near the target.
    - A single red pixel marks the middle of a green 3×3 destination block.

    Rule:
    - Flood-fill from passable cells touching the destination through floor/maroon cells; only reachable maroons are movable candidates.
    - Sort those candidates by BFS distance to the destination, then by row and column.
    - Erase the chosen sources back to floor and place them into the destination along the fixed clockwise spiral, skipping the red center so only the 8 outer cells can be filled.

    Key insight:
    - Two constraints interact: connectivity still decides which maroons qualify, but the destination now behaves like a spiral track wrapped around a permanent red hole.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells_row_major(cr, cc, h, w)
    dest_set = set(dest_cells)
    fill_order = _dest_cells_spiral_skip_center(cr, cc, h, w)

    floor = _infer_floor_color(grid, dest_cells, dest_set)
    dist = _distance_map(grid, dest_cells, dest_set, {floor, 9})
    inf = 10**9

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [triple for triple in maroons if triple[0] < inf]
    maroons.sort()

    k = min(len(fill_order), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in fill_order[:k]:
        out[r, c] = 9

    return out


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

PARTIAL_SOURCE_RANGE = (4, 7)
DENSE_SOURCE_RANGE = (9, 11)
DENSE_MODE_PROB = 0.55
DISTRACTOR_MAX = 8
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 4
GEN_RETRIES = 300


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """BFS from floor cells adjacent to dest through floor cells. Returns reachable floor cells."""
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
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
    length = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(length):
        nr, nc = nr + dr, nc + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    """Build wall barrier around dest with corridor openings, extending outward."""
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


def _ordered_reachable_floor(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    dist = _distance_map(grid, dest_cells, dest_set, {FLOOR_COLOR})
    inf = 10**9
    ordered = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == FLOOR_COLOR)]
    ordered = [triple for triple in ordered if triple[0] < inf and (triple[1], triple[2]) not in dest_set]
    ordered.sort()
    return [(r, c) for _, r, c in ordered]


def _choose_source_cells(ordered_reachable: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    partial_hi = min(PARTIAL_SOURCE_RANGE[1], len(ordered_reachable))
    if partial_hi < PARTIAL_SOURCE_RANGE[0]:
        raise ValueError("not enough reachable cells for partial mode")

    can_dense = len(ordered_reachable) >= DENSE_SOURCE_RANGE[0]
    use_dense = can_dense and random.random() < DENSE_MODE_PROB

    if use_dense:
        dense_hi = min(DENSE_SOURCE_RANGE[1], len(ordered_reachable), TOTAL_MAROON_RANGE[1])
        count = random.randint(DENSE_SOURCE_RANGE[0], dense_hi)
        extras = random.sample(ordered_reachable[8:], count - 8)
        source_cells = ordered_reachable[:8] + extras
        moved_cells = ordered_reachable[:8]
    else:
        count = random.randint(PARTIAL_SOURCE_RANGE[0], partial_hi)
        prefix_len = min(len(ordered_reachable), count + 5)
        source_cells = random.sample(ordered_reachable[:prefix_len], count)
        moved_cells = source_cells

    return source_cells, moved_cells


def _gen_attempt() -> np.ndarray | None:
    """Build a candidate input grid; returns the grid or None on failure."""
    dest_cells = _dest_cells_row_major(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    fill_order = _dest_cells_spiral_skip_center(CENTER_R, CENTER_C, GRID_H, GRID_W)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    ordered_reachable = _ordered_reachable_floor(grid, dest_cells, dest_set)
    if len(ordered_reachable) < MIN_REACHABLE:
        return None

    try:
        source_cells, moved_cells = _choose_source_cells(ordered_reachable)
    except ValueError:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - len(source_cells))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0
    if len(source_cells) + num_distractors < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - (len(source_cells) + num_distractors)
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        else:
            return None

    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = grid.copy()
    moved_set = set(moved_cells)
    stationary_sources = [cell for cell in source_cells if cell not in moved_set]
    for r, c in stationary_sources + distractor_cells:
        out[r, c] = MAROON
    for r, c in fill_order[: len(moved_cells)]:
        out[r, c] = MAROON

    try:
        result = main(inp)
        if np.array_equal(result, out):
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

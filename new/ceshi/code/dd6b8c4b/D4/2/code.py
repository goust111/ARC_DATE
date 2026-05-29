# Variant of dd6b8c4b
# Dimension: D4 — 填充目标区域的方式 (path)
# Change: row_major_fill——按行优先顺序从绿色方块内部左上角开始，逐个填入收集到的栗色像素... → 螺旋填充
#
# Design analysis
# variant_spec:
#   invariant_structure:
#     grid: always 11x11
#     colors: floor=7, wall=6 or absent, destination ring=3, center marker=2, movable tokens=9
#     destination: fixed 3x3 block centered at (5,5)
#     source_selection: unchanged from the original task
#   d4_change:
#     old_path: row-major over the 3x3 destination
#     new_path: a single fixed clockwise spiral that starts at the destination's top-left corner,
#       walks around the outer ring, and ends at the center
#     order:
#       - (cr-1, cc-1)
#       - (cr-1, cc)
#       - (cr-1, cc+1)
#       - (cr, cc+1)
#       - (cr+1, cc+1)
#       - (cr+1, cc)
#       - (cr+1, cc-1)
#       - (cr, cc-1)
#       - (cr, cc)
#   behavior_trace:
#     original_for_k5:
#       row_major_fill: fills top row, then middle-left, then center
#     variant_for_k5:
#       spiral_fill: fills top row, then right-middle, then bottom-right
#     visible_difference:
#       the fourth and fifth placements move to the right edge instead of dropping to the next row,
#       so partial fills form a bent spiral front rather than a rectangular prefix
#   complexity_calibration:
#     why_fixed_spiral:
#       a 3x3 destination is too small for many spiral subcases; adding orientation switches would
#       create hidden branches with weak evidence in only 2-4 ARC examples
#     visibility_guard:
#       generation restricts k to 4..8 so the produced outputs are not dominated by the degenerate
#       cases k in {1,2,3,9}, where row-major and the spiral are identical or nearly identical
#   implicit_assumption_audit:
#     preserved:
#       - unique red center still identifies the destination
#       - reachability still depends only on flood-fill through floor/maroon cells
#       - maroons are still ranked by distance, then row, then column
#     changed_code_path:
#       - any write to destination cells can no longer assume sorted(dest_cells) is the final path
#       - expected outputs in the generator must use the same spiral order; row-major output-first
#         construction would silently validate the wrong task
#     harmless_old_pattern_now_checked:
#       - overwriting the red center remains legal, but under the spiral it happens last rather than
#         fifth, so generator examples must include partial fills where the center often survives
#   placement_legality:
#     - source maroons are placed only on reachable floor cells, never in destination or wall cells
#     - distractor maroons are placed only on unreachable floor cells
#     - the wall-building logic is unchanged, so BFS assumptions remain valid
#   visual_readability:
#     - only one semantic change is introduced: the pattern inside the 3x3 destination
#     - the surrounding corridor / wall geometry remains as in the original task, avoiding clutter
#   color_reasoning:
#     - no color remapping is introduced
#     - ordering depends only on visible geometric position in the 3x3 block, never on color ids
#   solver_generator_contract:
#     - main() assumes at least one passable cell touches the destination and that the red center is unique
#     - generate_input() preserves both conditions and also guarantees k >= 4 reachable maroons

import numpy as np

from arc_utils import neighbors4
import random
from collections import deque


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the 3×3 block around the unique red center using a spiral fill.

    Input structure:
    - Orange floor cells hold scattered maroon tokens, and magenta walls may carve corridors around the target.
    - A single red pixel sits in the middle of a green 3×3 destination block.

    Rule:
    - Flood-fill outward from passable cells touching the destination; reachable maroon tokens are the movable candidates.
    - Sort candidates by BFS distance to the destination, then by row and column, and take as many as fit in the 3×3 block.
    - Erase those source maroons back to floor and write them into the destination along a clockwise outer-to-inner spiral.

    Key insight:
    - Connectivity still decides which maroons qualify, but the visible change is the write path: partial fills trace the destination's perimeter before reaching the center.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells_row_major(cr, cc, h, w)
    dest_set = set(dest_cells)
    fill_order = _dest_cells_spiral(cr, cc, h, w)

    # Infer floor color from the ring around the destination.
    ring_colors: list[int] = []
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

    passable = {floor, 9}

    # BFS start = passable cells adjacent to the destination.
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
    maroons = [t for t in maroons if t[0] < inf]
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

FILL_K_RANGE = (4, 8)
DISTRACTOR_MAX = 10
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


def _gen_attempt() -> np.ndarray | None:
    """Build a candidate input grid; returns the grid or None on failure."""
    dest_cells = _dest_cells_row_major(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    fill_order = _dest_cells_spiral(CENTER_R, CENTER_C, GRID_H, GRID_W)

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

    max_k = min(FILL_K_RANGE[1], len(reachable_floor), len(fill_order))
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
        elif max_k >= k + needed:
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
    for r, c in fill_order[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
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

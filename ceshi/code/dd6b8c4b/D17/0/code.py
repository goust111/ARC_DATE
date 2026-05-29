# Variant of dd6b8c4b
# Dimension: D17 — 红色中心像素的功能 (meta)
# Change: 绿色方块中心的红色像素（2）标识方块位置，可能在收集时被覆盖... → 红色像素作为排斥源
#
# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     unchanged:
#       - 11x11 canvas, same colors, same 3x3 green destination centered on the unique red pixel
#       - reachable maroon source selection still uses 4-neighbor flood-fill through floor+maroon
#       - source maroons are still ranked by BFS distance from the destination openings
#     changed:
#       - the destination write order is no longer row-major
#       - the red center now acts as a repulsion source, so maroons fill the cells farthest from red first
#       - deterministic tie break inside the same distance shell is row-major
#   concretization_and_trace:
#     destination_geometry:
#       - center at (5,5)
#       - corners have Manhattan distance 2 from red
#       - edge-centers have Manhattan distance 1
#       - the red center has distance 0 and is last
#     fill_order_example:
#       old_row_major:
#         - (4,4) (4,5) (4,6) (5,4) (5,5) (5,6) (6,4) (6,5) (6,6)
#       new_repulsion_order:
#         - (4,4) (4,6) (6,4) (6,6) (4,5) (5,4) (5,6) (6,5) (5,5)
#     concrete_behavior_trace:
#       - suppose BFS finds exactly k=4 reachable maroons
#       - original task writes them to the whole top row plus left-middle
#       - this variant writes them to the four corners instead
#       - therefore the moved tokens are the same, but the red pixel changes the visible pattern
#   complexity_calibration:
#     - the transformation still has one main decision axis: reachability through walls
#     - D17 adds one secondary visible axis: inside the 3x3 block, filling happens from outside-in
#     - generator caps k at 8 so the repulsion effect remains visible and the red center usually survives
#     - samples that would match the legacy row-major output are rejected to avoid silent degeneration
#   implicit_assumption_review:
#     - unique red center is still required; solver asserts exactly one red pixel
#     - destination is still the full 3x3 neighborhood around that red pixel
#     - floor inference still assumes colors 2/3/9 are special and any other dominant ring color can be treated as wall-like
#     - the old writeback `dest_cells[:k]` depended on red being passive; in this variant that assumption is false, so writeback must use a new order derived from distance to red
#     - source erasure still assumes reachable sources are outside the destination; generator enforces that
#     - BFS start still assumes at least one passable cell touches the destination; generator enforces that by rejecting blocked layouts
#   placement_legality:
#     - maroons are only placed on floor cells, never on wall cells or inside the 3x3 destination
#     - wall construction may restrict access but may not overwrite the destination
#     - reachable source count must support the chosen k; generator rejects layouts that cannot
#   visual_observability:
#     - using distance-to-red is visible because corners/edges/center are different geometric roles
#     - with k in [2,8], the output differs from the original row-major task on every accepted sample
#     - no invisible attribute such as color id or generation order is used in sorting
#   visual_cleanliness:
#     - only one 3x3 object is edited, so the new rule stays localized and readable
#     - keeping the color palette unchanged avoids introducing extra visual noise
#   color_inference:
#     - no new color mapping is introduced
#     - the only color behavior change is whether the red center remains visible until the block is almost full

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _repulsion_fill_order(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    dest_cells = _destination_cells(cr, cc, h, w)
    return sorted(dest_cells, key=lambda rc: (-abs(rc[0] - cr) - abs(rc[1] - cc), rc[0], rc[1]))


def _infer_floor_color(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> int:
    ring_colors: list[int] = []
    h, w = grid.shape
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    candidates = [x for x in ring_colors if x not in (2, 3, 9)]
    assert candidates, "ring around destination must contain passable cells to infer floor color"
    vals, cnts = np.unique(candidates, return_counts=True)
    wall_like = int(vals[cnts.argmax()])
    others = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, wall_like)]
    floor = others[0] if len(others) == 1 else (max(others, key=lambda c: int((grid == c).sum())) if others else None)
    return wall_like if floor is None else int(floor)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the 3×3 block around the unique red center, filling from the outside inward.

    Input structure:
    - The board contains floor cells, optional wall barriers, scattered maroon (9) source pixels, and one green 3×3 target block.
    - The target block has a unique red (2) center pixel, and that center is the reference point for the final placement order.

    Rule:
    - Flood-fill through floor/maroon cells starting from openings next to the target; only reachable maroons are eligible to move.
    - Rank eligible maroons by BFS distance to the target as before, but place the chosen ones into target cells ordered by decreasing distance from the red center.
    - Erase moved source maroons back to floor; any untouched target cells keep their original green/red colors.

    Key insight:
    - Walls still decide which maroons can be collected at all, but the red center is no longer a passive marker: it actively repels the fill so corners are occupied before edges, and the center is filled last.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w)
    dest_set = set(dest_cells)
    fill_order = _repulsion_fill_order(cr, cc, h, w)

    floor = _infer_floor_color(grid, dest_cells, dest_set)
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
    queue = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while queue:
        r, c = queue.popleft()
        d = int(dist[r, c])
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                dist[rr, cc2] = d + 1
                queue.append((rr, cc2))

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

FILL_K_RANGE = (2, 8)
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 2
GEN_RETRIES = 300


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """BFS from floor cells adjacent to dest through floor cells only."""
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
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


def _paint_output(
    base_grid: np.ndarray,
    fill_order: list[tuple[int, int]],
    k: int,
    distractor_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = base_grid.copy()
    for r, c in fill_order[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    return out


def _legacy_fill_order() -> list[tuple[int, int]]:
    return _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    repulsion_order = _repulsion_fill_order(CENTER_R, CENTER_C, GRID_H, GRID_W)

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
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set
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

    out = _paint_output(grid, repulsion_order, k, distractor_cells)
    legacy_out = _paint_output(grid, _legacy_fill_order(), k, distractor_cells)
    if np.array_equal(out, legacy_out):
        return None

    try:
        if np.array_equal(main(inp), out):
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

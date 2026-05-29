# Combo variant of dd6b8c4b (index 48)
# combo_spec: D16/2+D18/1
# Name: 背景色变化显示路径+不可达像素变色

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9
TRAIL_COLOR = 8
REJECT_COLOR = 1

RING_GAP_RANGE = (1, 3)
WALL_STYLES = ["arms", "frame", "mixed"]
ARM_EXTEND_PROB = 0.55
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.45
FRAME_GAP_REMOVE_PROB = 0.2
MIXED_EXTEND_PROB = 0.65

FILL_K_RANGE = (2, 9)
DISTRACTOR_RANGE = (1, 6)
MIN_REACHABLE_FLOOR = 5
GEN_RETRIES = 600


# Design analysis for the fusion:
# a) Behavior trace:
#    Original output only moved reachable maroon tokens into the 3x3 target.
#    In this combo, the same reachability computation still selects the moved
#    maroons, but its hidden BFS footprint becomes visible: every reachable
#    background cell turns from orange(7) to trail color(8), and every
#    unreachable maroon distractor turns blue(1). Example: if a maroon sits
#    behind a wall arm, original output left it maroon; new output recolors it
#    blue while leaving its surrounding unreachable floor orange. Chosen source
#    maroons are still removed, but their cells now become trail color rather
#    than plain floor because they belong to the reachable region.
# b) Complexity calibration:
#    The output uses only two new colors with stable semantics: 8 means
#    reachable path/background, 1 means rejected maroon. This is enough to make
#    the change obvious without introducing many color roles. The generator
#    forces both reachable and unreachable areas so orange(untouched floor),
#    azure trail, and blue rejected tokens co-exist in typical samples.
# c) Hidden assumptions broken by the combo:
#    The original solver inferred a single non-wall floor color and wrote only
#    that color back. That assumption no longer holds in output, because a
#    second background-like color appears. The new solver therefore computes
#    reachability from the input only and then writes either 7 or 8 explicitly.
#    It also cannot assume all non-moved maroons stay maroon: unreachable ones
#    must flip to 1.
# d) Placement legality:
#    The generator keeps the fixed 11x11 canvas and 3x3 center block, places
#    walls only outside the destination, and samples source/distractor tokens
#    from disjoint reachable/unreachable floor sets, so no overlaps or
#    out-of-bounds placements occur.
# e) Visual observability:
#    A human can now read the reachable component directly: trail cells form a
#    connected region from the target openings, while rejected blue maroons sit
#    in orange regions cut off by magenta walls. The ranking rule is still
#    visible because only the closest reachable maroons are transferred into the
#    destination in row-major order.
# f) Visual cleanliness:
#    Each cell has one role in the output: wall, untouched floor, trail floor,
#    rejected token, or destination fill. We avoid piling multiple markings onto
#    the same cell, so the board remains easy to parse.
# g) New solver assumptions and generator guarantees:
#    `main()` assumes exactly one red center, a 3x3 destination around it, at
#    least one passable opening next to that block, and no extra colors beyond
#    the fixed palette. `generate_input()` constructs layouts satisfying those
#    conditions and additionally guarantees non-empty reachable floor,
#    non-empty unreachable floor, and at least one unreachable maroon so the
#    fusion does not degenerate back to the original task.


def _dest_cells() -> list[tuple[int, int]]:
    cells = []
    for r in range(CENTER_R - 1, CENTER_R + 2):
        for c in range(CENTER_C - 1, CENTER_C + 2):
            cells.append((r, c))
    return cells


def _reachable_info(grid: np.ndarray, dest_set: set[tuple[int, int]]):
    passable = {FLOOR_COLOR, MAROON}
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_set:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and int(grid[rr, cc]) in passable:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    starts.append((rr, cc))
    assert starts, "destination must have at least one passable opening"

    inf = 10**9
    dist = np.full((GRID_H, GRID_W), inf, dtype=int)
    queue = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    reachable_passable = set(starts)
    while queue:
        r, c = queue.popleft()
        d = int(dist[r, c])
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and dist[rr, cc] == inf:
                if int(grid[rr, cc]) in passable:
                    dist[rr, cc] = d + 1
                    reachable_passable.add((rr, cc))
                    queue.append((rr, cc))

    reachable_floor = {
        (r, c) for r, c in reachable_passable if int(grid[r, c]) == FLOOR_COLOR
    }
    return dist, reachable_passable, reachable_floor


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


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int):
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


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]):
    ring = set()
    for r, c in dest_set:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and (rr, cc) not in dest_set:
                ring.add((rr, cc))
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
            n_remove = random.randint(1, max(1, len(outer_frame) // 5))
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


def _reachable_floor_on_base(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited = set()
    queue = deque()
    for r, c in dest_set:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and (rr, cc) not in dest_set:
                if int(grid[rr, cc]) == FLOOR_COLOR and (rr, cc) not in visited:
                    visited.add((rr, cc))
                    queue.append((rr, cc))
    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and (rr, cc) not in visited and (rr, cc) not in dest_set:
                if int(grid[rr, cc]) == FLOOR_COLOR:
                    visited.add((rr, cc))
                    queue.append((rr, cc))
    return visited


def _expected_output(base_grid: np.ndarray, dest_cells: list[tuple[int, int]], reachable_floor: set[tuple[int, int]], distractor_cells: list[tuple[int, int]], k: int) -> np.ndarray:
    out = base_grid.copy()
    for r, c in reachable_floor:
        out[r, c] = TRAIL_COLOR
    for r, c in distractor_cells:
        out[r, c] = REJECT_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the nearest reachable maroon pixels into the center block, while exposing the reachability map.

    Input structure:
    - The board uses orange floor, magenta walls, scattered maroon tokens, and a unique red center inside a 3x3 green destination block.
    - Walls carve the orange floor into accessible and blocked regions around that destination.

    Rule:
    - Flood-fill outward from passable cells adjacent to the destination, using orange and maroon cells as the traversable medium.
    - Sort reachable maroon tokens by BFS distance, then by row and column; move the closest ones into the destination in row-major order.
    - In the output, every reachable floor location is recolored to the trail color, chosen source tokens disappear into that trail, and unreachable maroon tokens are recolored blue.

    Key insight:
    - The destination fill still depends on the same reachability-and-distance logic as the original task, but this variant also paints the whole reachable component and explicitly marks rejected maroons, turning the hidden flood-fill partition into visible evidence.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"

    dest_cells = _dest_cells()
    dest_set = set(dest_cells)

    dist, _reachable_passable, reachable_floor = _reachable_info(grid, dest_set)

    reachable_maroons = []
    unreachable_maroons = []
    for r, c in np.argwhere(grid == MAROON):
        d = int(dist[r, c])
        item = (d, int(r), int(c))
        if d < 10**9:
            reachable_maroons.append(item)
        else:
            unreachable_maroons.append((int(r), int(c)))
    reachable_maroons.sort()

    k = min(len(dest_cells), len(reachable_maroons))
    chosen_sources = {(r, c) for _, r, c in reachable_maroons[:k]}

    for r, c in reachable_floor:
        out[r, c] = TRAIL_COLOR
    for r, c in chosen_sources:
        out[r, c] = TRAIL_COLOR
    for r, c in unreachable_maroons:
        out[r, c] = REJECT_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    _build_walls(base, dest_set)

    reachable_floor = _reachable_floor_on_base(base, dest_set)
    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(base[r, c]) == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    if len(reachable_floor) < MIN_REACHABLE_FLOOR:
        return None
    if not unreachable_floor:
        return None

    max_k = min(FILL_K_RANGE[1], len(reachable_floor) - 2)
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    max_distractors = min(DISTRACTOR_RANGE[1], len(unreachable_floor))
    if max_distractors < DISTRACTOR_RANGE[0]:
        return None
    num_distractors = random.randint(DISTRACTOR_RANGE[0], max_distractors)

    source_cells = random.sample(sorted(reachable_floor), k)
    distractor_cells = random.sample(sorted(unreachable_floor), num_distractors)

    inp = base.copy()
    for r, c in source_cells:
        inp[r, c] = MAROON
    for r, c in distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(base, dest_cells, reachable_floor, distractor_cells, k)

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, out):
        return None

    trail_count = int((out == TRAIL_COLOR).sum())
    blue_count = int((out == REJECT_COLOR).sum())
    untouched_floor_count = int((out == FLOOR_COLOR).sum())
    if trail_count < k + 2:
        return None
    if blue_count < 1 or untouched_floor_count < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

# Combo variant of dd6b8c4b (index 11)
# combo_spec: D7/0+D18/1
# Name: 原位痕迹色+不可达像素变色

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant:
# - Concrete behavior: keep the original reachability test and distance ranking, but expose the
#   decision in the output. Example: if three maroon tokens can be reached from the 3x3 target
#   block and two are trapped behind magenta walls, then the output shows three maroons inside
#   the destination, three gray source traces at the reachable source cells, and two blue marks
#   at the unreachable cells. Relative to the base task, only the classification colors change;
#   the ranking and fill order stay the same.
# - Complexity calibration: the variant uses exactly two added colors, gray(5) and blue(1).
#   This is enough to make the fusion visible without introducing ambiguous many-to-many maps.
#   Every generated example contains all three cues simultaneously: destination maroons,
#   gray reachable traces, and blue unreachable markers.
# - Hidden-assumption audit: the original solver inferred floor/wall colors from local context.
#   That becomes brittle once the output introduces extra recolor states, so this variant fixes
#   the task palette to the spec values used by the generator: floor=7, wall=6, dest=3, center=2,
#   maroon=9, trace=5, unreachable=1. The solver still only reads original input colors.
# - Placement legality: the generator keeps the 3x3 destination centered at (5,5), places source
#   maroons only on floor cells outside that block, and adds a separate sealed 4x4 wall pocket in
#   one corner so at least one unreachable maroon can always be placed without overlap.
# - Visual observability and neatness: the gray traces stay exactly at reachable source positions,
#   while blue markers remain inside an isolated walled pocket or another disconnected region.
#   Because the destination block is spatially distinct from both kinds of outside evidence, the
#   three signals do not visually collide.
# - Generator guarantee for full traceability: we cap reachable source count at the destination
#   capacity, so every reachable maroon is one of the collected tokens. That avoids leftover
#   reachable maroons outside the block and keeps the output limited to the intended three cues.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9
TRACE_COLOR = 5
UNREACHABLE_COLOR = 1

RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]
ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

FILL_K_RANGE = (2, 9)
DISTRACTOR_RANGE = (1, 4)
TOTAL_MAROON_RANGE = (4, 12)
MIN_REACHABLE = 2
GEN_RETRIES = 500
POCKET_TOP_LEFTS = [(0, 0), (0, 7), (7, 0), (7, 7)]


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _reachable_distances(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    dest_set = set(dest_cells)
    passable = {FLOOR_COLOR, MAROON}

    dist = np.full((h, w), 10**9, dtype=int)
    q = deque()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set and int(grid[rr, cc]) in passable:
                if dist[rr, cc] != 10**9:
                    continue
                dist[rr, cc] = 0
                q.append((rr, cc))

    assert q, "there must be at least one floor or maroon cell adjacent to the destination block"

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set and int(grid[rr, cc]) in passable:
                if dist[rr, cc] == 10**9:
                    dist[rr, cc] = dist[r, c] + 1
                    q.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify maroon tokens by reachability, move the reachable ones into the center block, and leave visible evidence behind.

    Input structure:
    - The board is an 11x11 orange floor with an always-present 3x3 destination block centered on one red cell.
    - Green cells form the rest of that destination block, magenta cells act as walls, and maroon cells are scattered tokens on the floor.

    Rule:
    - Flood-fill outward from floor-or-maroon cells touching the destination block, stopping at magenta walls.
    - Sort reachable maroon cells by BFS distance, then by row and column; place them into the destination in row-major order.
    - Each reachable maroon leaves a gray trace at its source position, while each unreachable maroon is recolored blue and stays in place.

    Key insight:
    - The output reveals the full connectivity judgment: maroon inside the block shows which tokens were chosen, gray shows which outside tokens were reachable, and blue shows which ones walls cut off.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells(cr, cc, h, w)

    dist = _reachable_distances(grid, dest_cells)
    maroons = [(int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    reachable = [(int(dist[r, c]), int(r), int(c)) for r, c in maroons if int(dist[r, c]) < 10**9]
    unreachable = [(int(r), int(c)) for r, c in maroons if int(dist[r, c]) >= 10**9]
    reachable.sort()

    for _d, r, c in reachable:
        out[r, c] = TRACE_COLOR
    for r, c in unreachable:
        out[r, c] = UNREACHABLE_COLOR

    k = min(len(dest_cells), len(reachable))
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def _bfs_reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and int(grid[nr, nc]) == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and int(grid[nr, nc]) == FLOOR_COLOR:
                if (nr, nc) not in visited:
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
    nr, nc = r, c
    for _ in range(random.randint(1, max_len)):
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring: set[tuple[int, int]] = set()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring_list = sorted(ring)

    gap_cells = set(random.sample(ring_list, random.randint(*RING_GAP_RANGE)))
    wall_ring = [cell for cell in ring_list if cell not in gap_cells]
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
            for rc in random.sample(outer_frame, random.randint(1, max(1, len(outer_frame) // 4))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, GRID_H - abs(r - CENTER_R) - 2)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _add_isolated_pocket(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    choices = POCKET_TOP_LEFTS[:]
    random.shuffle(choices)
    for r0, c0 in choices:
        border = {(r, c) for r in range(r0, r0 + 4) for c in range(c0, c0 + 4) if r in (r0, r0 + 3) or c in (c0, c0 + 3)}
        inside = {(r, c) for r in range(r0 + 1, r0 + 3) for c in range(c0 + 1, c0 + 3)}
        cells = border | inside
        if any(cell in dest_set for cell in cells):
            continue
        if any(not (0 <= r < GRID_H and 0 <= c < GRID_W) for r, c in cells):
            continue
        if any(int(grid[r, c]) != FLOOR_COLOR for r, c in cells):
            continue
        for r, c in border:
            grid[r, c] = WALL_COLOR
        return inside
    return set()


def _expected_output(base_grid: np.ndarray, dest_cells: list[tuple[int, int]], source_cells: list[tuple[int, int]], distractor_cells: list[tuple[int, int]]) -> np.ndarray:
    out = base_grid.copy()
    for r, c in source_cells:
        out[r, c] = TRACE_COLOR
    for r, c in distractor_cells:
        out[r, c] = UNREACHABLE_COLOR
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
    pocket_inside = _add_isolated_pocket(grid, dest_set)
    if not pocket_inside:
        return None

    reachable_floor = _bfs_reachable_floor(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {(r, c) for r in range(GRID_H) for c in range(GRID_W) if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set}
    unreachable_floor = all_floor - reachable_floor
    guaranteed_unreachable = pocket_inside & unreachable_floor
    if not guaranteed_unreachable:
        return None

    max_k = min(FILL_K_RANGE[1], len(reachable_floor), len(dest_cells), TOTAL_MAROON_RANGE[1] - 1)
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    max_distractors = min(DISTRACTOR_RANGE[1], len(unreachable_floor), TOTAL_MAROON_RANGE[1] - k)
    min_distractors = min(DISTRACTOR_RANGE[0], len(guaranteed_unreachable))
    if max_distractors < min_distractors:
        return None
    num_distractors = random.randint(min_distractors, max_distractors)
    if k + num_distractors < TOTAL_MAROON_RANGE[0]:
        need = TOTAL_MAROON_RANGE[0] - (k + num_distractors)
        if k + need <= max_k:
            k += need
        elif num_distractors + need <= max_distractors:
            num_distractors += need
        else:
            return None

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(guaranteed_unreachable), 1)
    remaining_unreachable = list(unreachable_floor - set(distractor_cells))
    extra_need = num_distractors - len(distractor_cells)
    if extra_need > len(remaining_unreachable):
        return None
    if extra_need > 0:
        distractor_cells.extend(random.sample(remaining_unreachable, extra_need))

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(grid, dest_cells, source_cells, distractor_cells)

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

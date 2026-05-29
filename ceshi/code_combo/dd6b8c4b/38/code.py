# Combo variant of dd6b8c4b (index 38)
# combo_spec: D17/2+D19/2
# Name: 边缘距离度量+红色像素作起点

import os
import random
import sys
from collections import deque

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import neighbors4


# Design analysis for the fused rule:
# - Original task used the cells just outside the 3x3 block as BFS sources and
#   also used BFS depth as the ranking score. This couples reachability and
#   ordering to the same corridor geometry.
# - D17/2 breaks that coupling: reachability now starts at the red center, and
#   the green ring must be traversable so the flood-fill can exit through any
#   wall opening. In a concrete example, a maroon one step outside the right
#   edge can still be reachable only after a long left-side detour through the
#   corridor system, because the center must first reach a gap and then loop
#   around.
# - D19/2 changes the sort key from path length to min Manhattan distance to the
#   eight edge cells of the 3x3 block. The same right-side maroon can be ranked
#   before a farther-left maroon even if its actual corridor path is longer.
# - This only affects the output when there are more than 9 reachable maroons;
#   otherwise every reachable maroon moves. The generator therefore guarantees
#   at least 10 reachable maroons and rejects layouts where the new top-9 set is
#   identical to the old top-9-by-BFS set.
# - Hidden assumptions audited in main():
#   1. exactly one red center exists -> generator fixes a unique center.
#   2. the 3x3 destination stays in-bounds -> center is fixed at (5,5).
#   3. green ring cells are passable for the new flood-fill -> generator never
#      overwrites the destination block.
#   4. a flood-fill can leave the block -> wall builder keeps 1-4 gaps.
#   5. ranking uses the nearest boundary cell, not a color-dependent trick ->
#      score is purely geometric and human-visible.
# - Visual observability: each accepted sample contains leftover reachable
#   maroons after the 3x3 block is filled, and those leftovers differ from the
#   old BFS-based choice. The user can compare "which reachable maroons stayed"
#   across examples and infer that path detours no longer define priority.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

WALL_PROB = 1.0
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.45
FRAME_GAP_REMOVE_PROB = 0.35
MIXED_EXTEND_PROB = 0.6

REACHABLE_TOTAL_RANGE = (10, 16)
DISTRACTOR_MAX = 4
TOTAL_MAROON_RANGE = (10, 16)
MIN_REACHABLE = 12
GEN_RETRIES = 500
SUBSET_RETRIES = 120


def _dest_cells():
    return [
        (r, c)
        for r in range(CENTER_R - 1, CENTER_R + 2)
        for c in range(CENTER_C - 1, CENTER_C + 2)
    ]


def _edge_cells():
    return [(r, c) for r, c in _dest_cells() if (r, c) != (CENTER_R, CENTER_C)]


def _center_reachable(grid: np.ndarray) -> set[tuple[int, int]]:
    passable = {FLOOR_COLOR, DEST_COLOR, RED_CENTER, MAROON}
    visited = {(CENTER_R, CENTER_C)}
    queue = deque([(CENTER_R, CENTER_C)])
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if int(grid[nr, nc]) in passable:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _old_style_distances(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    passable = {FLOOR_COLOR, MAROON}
    inf = 10**9
    dist = np.full((GRID_H, GRID_W), inf, dtype=int)
    queue = deque()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                    dist[nr, nc] = 0
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and dist[nr, nc] == inf:
                if int(grid[nr, nc]) in passable:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))
    return dist


def _edge_distance(r: int, c: int, edge_cells: list[tuple[int, int]]) -> int:
    return min(abs(r - er) + abs(c - ec) for er, ec in edge_cells)


def _reachable_maroons_sorted(grid: np.ndarray) -> list[tuple[int, int, int]]:
    reachable = _center_reachable(grid)
    edge_cells = _edge_cells()
    maroons = []
    for r, c in np.argwhere(grid == MAROON):
        rr, cc = int(r), int(c)
        if (rr, cc) in reachable:
            maroons.append((_edge_distance(rr, cc, edge_cells), rr, cc))
    maroons.sort()
    return maroons


def _top9_sets(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int]:
    dest_set = set(_dest_cells())
    reachable = _center_reachable(grid)
    reachable_floor = sorted(
        (r, c)
        for r, c in reachable
        if (r, c) not in dest_set and int(grid[r, c]) == FLOOR_COLOR
    )
    old_dist = _old_style_distances(grid, dest_set)
    edge_cells = _edge_cells()
    new_rank = sorted(
        (_edge_distance(r, c, edge_cells), r, c) for r, c in reachable_floor
    )
    old_rank = sorted(
        (int(old_dist[r, c]), r, c)
        for r, c in reachable_floor
        if int(old_dist[r, c]) < 10**9
    )
    if len(new_rank) < 9 or len(old_rank) < 9:
        return set(), set(), 0
    new_top9 = {(r, c) for _, r, c in new_rank[:9]}
    old_top9 = {(r, c) for _, r, c in old_rank[:9]}
    gap = len(new_top9.symmetric_difference(old_top9))
    return new_top9, old_top9, gap


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the nine edge-nearest reachable maroon pixels into the centered 3x3 block.

    Input structure:
    - The board is 11x11 with orange floor, optional magenta wall barriers, scattered maroon pixels,
      and a fixed 3x3 destination block made of a red center wrapped by green cells.
    - The red center is not just a marker: together with the surrounding green ring, it forms the
      traversable launch pad for the flood-fill.

    Rule:
    - Start a 4-connected flood-fill at the red center, allowing travel through red, green, orange,
      and maroon cells but never through magenta walls.
    - Among the maroon pixels reached by that flood-fill, rank them by the minimum Manhattan distance
      to any of the eight edge cells of the 3x3 block; break ties by row, then column.
    - Move the first nine ranked maroons (or all of them if fewer exist) into the destination block
      in row-major order, and erase only the moved sources back to orange floor.

    Key insight:
    Reachability still depends on corridor connectivity from the center, but priority no longer uses
    corridor length. A maroon that is geometrically close to one side of the block can outrank another
    maroon whose actual path from the center is shorter.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    cr, cc = map(int, centers[0])
    assert (cr, cc) == (CENTER_R, CENTER_C), "red center must stay at the fixed destination center"

    dest_cells = _dest_cells()
    maroons = _reachable_maroons_sorted(grid)
    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


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
            if random.random() < FRAME_CONNECT_PROB:
                dirs = _outward_dirs(r, c)
                if not dirs:
                    continue
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr += dr
                    nc += dc
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            n_remove = random.randint(1, max(1, len(outer_frame) // 4))
            for cell in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(cell)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, GRID_H - abs(r - CENTER_R) - 2)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _base_grid() -> np.ndarray:
    dest_cells = _dest_cells()
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    if random.random() < WALL_PROB:
        _build_walls(grid, set(dest_cells))
    return grid


def _pick_source_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], int] | tuple[None, int]:
    new_top9, old_top9, gap = _top9_sets(grid)
    if gap == 0:
        return None, 0

    source_set = set(new_top9 | old_top9)
    if not (REACHABLE_TOTAL_RANGE[0] <= len(source_set) <= REACHABLE_TOTAL_RANGE[1]):
        return None, gap

    reachable = _center_reachable(grid)
    dest_set = set(_dest_cells())
    candidates = [
        (r, c)
        for r, c in reachable
        if (r, c) not in dest_set and int(grid[r, c]) == FLOOR_COLOR and (r, c) not in source_set
    ]
    random.shuffle(candidates)

    target_total = random.randint(
        max(REACHABLE_TOTAL_RANGE[0], len(source_set)),
        REACHABLE_TOTAL_RANGE[1],
    )
    need = target_total - len(source_set)
    if need > len(candidates):
        return None, gap

    source_cells = sorted(source_set | set(candidates[:need]))

    # Adding lower-ranked extras must not destroy the intended behavioral gap.
    probe = grid.copy()
    for r, c in source_cells:
        probe[r, c] = MAROON
    new_rank = {(r, c) for _, r, c in _reachable_maroons_sorted(probe)[:9]}
    old_dist = _old_style_distances(probe, dest_set)
    old_rank = sorted(
        (int(old_dist[r, c]), int(r), int(c))
        for r, c in np.argwhere(probe == MAROON)
        if int(old_dist[int(r), int(c)]) < 10**9 and (int(r), int(c)) in _center_reachable(probe)
    )
    old_top_after = {(r, c) for _, r, c in old_rank[:9]}
    if new_rank == old_top_after:
        return None, gap
    return source_cells, len(new_rank.symmetric_difference(old_top_after))


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)
    grid = _base_grid()

    reachable = _center_reachable(grid)
    reachable_floor = {
        (r, c)
        for r, c in reachable
        if (r, c) not in dest_set and int(grid[r, c]) == FLOOR_COLOR
    }
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    source_cells = None
    gap = 0
    for _ in range(SUBSET_RETRIES):
        picked, gap = _pick_source_cells(grid)
        if picked is not None:
            source_cells = picked
            break
    if source_cells is None or len(source_cells) <= 9 or gap == 0:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = sorted(all_floor - reachable_floor - set(source_cells))
    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - len(source_cells))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0
    distractor_cells = random.sample(unreachable_floor, num_distractors) if num_distractors else []

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    reachable_maroons = _reachable_maroons_sorted(inp)
    if len(reachable_maroons) <= 9:
        return None

    result = main(inp)
    if np.count_nonzero(result == MAROON) < 9:
        return None

    # Reject degenerate samples that still behave like the original BFS metric.
    old_dist = _old_style_distances(inp, dest_set)
    old_rank = sorted(
        (int(old_dist[r, c]), int(r), int(c))
        for r, c in np.argwhere(inp == MAROON)
        if int(old_dist[int(r), int(c)]) < 10**9 and (int(r), int(c)) in _center_reachable(inp)
    )
    new_top9 = {(r, c) for _, r, c in reachable_maroons[:9]}
    old_top9 = {(r, c) for _, r, c in old_rank[:9]}
    if new_top9 == old_top9:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid fused sample after {GEN_RETRIES} attempts")

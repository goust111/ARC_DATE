# Combo variant of dd6b8c4b (index 46)
# combo_spec: D3/2+D15/1
# Name: 三阶段处理+欧几里得距离排序

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import neighbors4


# Design analysis
# ----------------
# a) Concrete behavior and trace:
#    - Original task: flood-fill from passable cells touching the 3x3 target,
#      keep only reachable maroons, sort them by BFS step distance, then place
#      the chosen maroons into the 3x3 block in row-major order.
#    - This combo keeps the same reachability test, but splits the logic into
#      three explicit stages:
#      1. Mark every reachable maroon with its BFS reachability information.
#      2. Re-sort those marked maroons by Euclidean distance to the red center.
#      3. Fill the destination block in row-major order from that Euclidean list.
#    - Small trace:
#        center = (5, 5)
#        A at (3, 4): Euclidean^2 = 5, but a wall detour makes BFS distance 8
#        B at (5, 0): Euclidean^2 = 25, but BFS distance 3
#      Original BFS sort prefers B before A.  This combo still marks both as
#      reachable in stage 1, but stage 2 prefers A before B because geometry,
#      not corridor length, controls the ranking.
#    - Because all moved pixels have the same color, the changed order matters
#      only when there are more than 9 reachable maroons.  Then the top-9 set
#      itself can differ, changing which scattered maroons disappear.
#
# b) Complexity calibration:
#    - The generator forces >9 reachable maroons and requires the Euclidean
#      top-9 set to differ from the BFS top-9 set.  Without that, the variant
#      would visually collapse back to the original task.
#    - The wall layouts stay in the original visual family: a 3x3 central block,
#      a magenta barrier ring with 1-2 openings, and irregular outward walls.
#      This keeps the scene readable while still allowing path detours.
#
# c) Hidden-assumption audit against the original solver:
#    - "there is exactly one red center": still required and generator enforces it.
#    - "destination is the fixed 3x3 around that center": unchanged.
#    - "only orange floor and maroon are passable": unchanged.
#    - "BFS distance also defines sort order": no longer true.  The new solver
#      stores BFS only for stage-1 reachability and switches to Euclidean order
#      in stage 2.
#    - "sorting and filling are one fused action": no longer true.  The solver
#      now exposes separate helpers so the processing pipeline is explicit.
#
# d) Placement legality:
#    - Generator places maroons only on floor cells, never inside the 3x3 block
#      and never on wall cells.
#    - Reachable and unreachable pools are disjoint by construction, so chosen
#      sources, leftover reachable maroons, and distractors never overlap.
#
# e) Visual observability:
#    - A human can see the distinction because some maroons that are geometrically
#      close to the center disappear even when a shorter corridor leads to a
#      farther maroon that stays behind.  The walls provide the visible reason
#      BFS and Euclidean distance disagree.
#    - The generator rejects layouts where the Euclidean and BFS top-9 sets are
#      identical, since those samples would not reveal the fused rule.
#
# f) Visual cleanliness:
#    - The task still uses only five semantic colors: floor, wall, green target,
#      red center, maroon tokens.  No extra marks are added.
#    - All reasoning remains attached to clearly separated roles: central block,
#      barrier network, scattered maroons.
#
# g) New solver assumptions and generator guarantees:
#    - Solver assumes there is at least one passable cell adjacent to the target.
#      Generator keeps 1-2 wall gaps so this always holds.
#    - Solver assumes enough reachable structure exists to run the three stages.
#      Generator explicitly checks for at least 10 reachable floor cells and for
#      a non-degenerate Euclidean-vs-BFS disagreement before accepting a sample.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

RING_GAP_RANGE = (1, 2)
WALL_STYLES = ("arms", "frame", "mixed")
ARM_EXTEND_PROB = 0.6
ARM_LENGTH_RANGE = (2, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.65
FRAME_GAP_REMOVE_PROB = 0.35
MIXED_EXTEND_PROB = 0.7

TOTAL_MAROON_RANGE = (10, 16)
DISTRACTOR_MAX = 4
GEN_RETRIES = 1200
INF = 10**9


def _dest_cells(center_r: int = CENTER_R, center_c: int = CENTER_C) -> list[tuple[int, int]]:
    return [(r, c) for r in range(center_r - 1, center_r + 2) for c in range(center_c - 1, center_c + 2)]


def _euclidean_key(rc: tuple[int, int], center_r: int = CENTER_R, center_c: int = CENTER_C) -> tuple[int, int, int]:
    r, c = rc
    return (r - center_r) ** 2 + (c - center_c) ** 2, r, c


def _reachable_distances(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    dist = np.full(grid.shape, INF, dtype=int)
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(grid[nr, nc]) in (FLOOR_COLOR, MAROON):
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    starts.append((nr, nc))
                    dist[nr, nc] = 0
    assert starts, "destination must have at least one adjacent passable cell"

    q = deque(starts)
    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(grid[nr, nc]) in (FLOOR_COLOR, MAROON):
                if dist[nr, nc] == INF:
                    dist[nr, nc] = d + 1
                    q.append((nr, nc))
    return dist, starts


def _mark_reachable_maroons(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    dist, _ = _reachable_distances(grid, dest_cells)
    marked = []
    for r, c in np.argwhere(grid == MAROON):
        if int(dist[r, c]) < INF:
            marked.append((int(dist[r, c]), int(r), int(c)))
    return marked


def _sort_marked_maroons(marked: list[tuple[int, int, int]], center_r: int, center_c: int) -> list[tuple[int, int, int]]:
    return sorted(marked, key=lambda item: _euclidean_key((item[1], item[2]), center_r, center_c))


def _fill_destination(grid: np.ndarray, ordered_marked: list[tuple[int, int, int]], dest_cells: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    k = min(len(dest_cells), len(ordered_marked))
    chosen = {(r, c) for _, r, c in ordered_marked[:k]}

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull reachable maroons into the central 3x3 block, but separate marking,
    Euclidean sorting, and filling into three distinct phases.

    Input structure:
    - The board is an 11x11 orange floor with a fixed 3x3 green target around a
      unique red center, plus magenta walls that create corridor detours.
    - Maroon pixels lie on floor cells outside the target block; some are flood-
      fill reachable from target-side openings and others may be stranded.

    Rule:
    - Phase 1 marks every maroon that is reachable through 4-connected floor or
      maroon cells starting from passable cells adjacent to the target block.
    - Phase 2 reorders only those marked maroons by Euclidean distance to the
      red center, using row/column order to break ties.
    - Phase 3 moves the first nine (or fewer) maroons from that Euclidean list:
      their old cells become floor, and the destination block is filled in
      row-major order.

    Key insight:
    - Reachability still depends on corridor connectivity, but priority no
      longer depends on corridor length.  A maroon that is geometrically close
      to the center can outrank a path-shorter maroon if walls force a detour.
    """

    grid = np.asarray(grid, dtype=int)
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"

    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells(cr, cc)
    marked = _mark_reachable_maroons(grid, dest_cells)   # phase 1: mark
    ordered = _sort_marked_maroons(marked, cr, cc)       # phase 2: sort
    return _fill_destination(grid, ordered, dest_cells)  # phase 3: fill


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
    length = random.randint(*ARM_LENGTH_RANGE)
    length = min(length, max_len)
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

    gap_count = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring, gap_count))
    wall_ring = [rc for rc in ring if rc not in gap_cells]
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
        outer_frame = [rc for rc in wall_cells if rc not in set(wall_ring)]
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
        side = random.choice(["north", "south", "west", "east"])
        if side in ("north", "south"):
            rr = 2 if side == "north" else GRID_H - 3
            gap_col = random.randint(2, GRID_W - 3)
            for cc in range(1, GRID_W - 1):
                if cc != gap_col:
                    wall_cells.add((rr, cc))
        else:
            cc = 2 if side == "west" else GRID_W - 3
            gap_row = random.randint(2, GRID_H - 3)
            for rr in range(1, GRID_H - 1):
                if rr != gap_row:
                    wall_cells.add((rr, cc))

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _base_grid() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]]]:
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    _build_walls(grid, dest_set)
    return grid, dest_cells, dest_set


def _reachable_floor_and_dist(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], np.ndarray]:
    dist, _ = _reachable_distances(grid, dest_cells)
    reachable_floor = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if (r, c) not in dest_set and int(grid[r, c]) == FLOOR_COLOR and int(dist[r, c]) < INF:
                reachable_floor.append((r, c))
    return reachable_floor, dist


def _find_source_plan(reachable_floor: list[tuple[int, int]], dist: np.ndarray) -> tuple[list[tuple[int, int]], dict[str, int]] | None:
    if len(reachable_floor) < 10:
        return None

    bfs_order = sorted(reachable_floor, key=lambda rc: (int(dist[rc]), rc[0], rc[1]))
    euc_order = sorted(reachable_floor, key=_euclidean_key)
    bfs_top = set(bfs_order[:9])
    euc_top = set(euc_order[:9])
    if bfs_top == euc_top:
        return None

    bfs_rank = {rc: i for i, rc in enumerate(bfs_order)}
    euc_rank = {rc: i for i, rc in enumerate(euc_order)}
    swapped_in = [rc for rc in euc_top if rc not in bfs_top]
    swapped_out = [rc for rc in bfs_top if rc not in euc_top]
    if not swapped_in or not swapped_out:
        return None

    best_in_gap = max(bfs_rank[rc] - euc_rank[rc] for rc in swapped_in)
    best_out_gap = max(euc_rank[rc] - bfs_rank[rc] for rc in swapped_out)
    if min(best_in_gap, best_out_gap) < 1:
        return None

    source_set = set(euc_top | bfs_top)
    if len(source_set) > TOTAL_MAROON_RANGE[1]:
        return None

    tail_pool = [rc for rc in reachable_floor if rc not in source_set and bfs_rank[rc] >= 9 and euc_rank[rc] >= 9]
    max_extra = min(TOTAL_MAROON_RANGE[1] - len(source_set), len(tail_pool))
    if max_extra > 0:
        n_extra = random.randint(0, min(3, max_extra))
        source_set.update(random.sample(tail_pool, n_extra))

    source_cells = sorted(source_set, key=_euclidean_key)
    if len(source_cells) < TOTAL_MAROON_RANGE[0]:
        return None

    metric = {
        "symdiff": len(bfs_top ^ euc_top),
        "best_in_gap": best_in_gap,
        "best_out_gap": best_out_gap,
    }
    return source_cells, metric


def _expected_output(inp: np.ndarray, source_cells: list[tuple[int, int]], dest_cells: list[tuple[int, int]]) -> np.ndarray:
    ordered = sorted([(0, r, c) for r, c in source_cells], key=lambda item: _euclidean_key((item[1], item[2])))
    out = inp.copy()
    chosen = {(r, c) for _, r, c in ordered[: len(dest_cells)]}
    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[: min(len(dest_cells), len(source_cells))]:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    grid, dest_cells, dest_set = _base_grid()
    reachable_floor, dist = _reachable_floor_and_dist(grid, dest_cells, dest_set)
    plan = _find_source_plan(reachable_floor, dist)
    if plan is None:
        return None
    source_cells, _metric = plan

    all_floor = {(r, c) for r in range(GRID_H) for c in range(GRID_W) if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set}
    unreachable_floor = sorted(all_floor - set(reachable_floor))
    max_distractors = min(DISTRACTOR_MAX, TOTAL_MAROON_RANGE[1] - len(source_cells), len(unreachable_floor))
    distractor_count = random.randint(0, max_distractors) if max_distractors > 0 else 0
    distractor_cells = random.sample(unreachable_floor, distractor_count) if distractor_count else []

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(inp, source_cells, dest_cells)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid non-degenerate input after {GEN_RETRIES} attempts")

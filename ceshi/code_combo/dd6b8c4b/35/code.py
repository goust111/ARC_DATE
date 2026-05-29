# Combo variant of dd6b8c4b (index 35)
# combo_spec: D3/1+D10/0
# Name: 列号优先消歧+空间位置排序

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import neighbors4
except ImportError:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "solvers"
        if candidate.is_dir():
            sys.path.append(str(candidate))
            break
    from arc_utils import neighbors4


# Design analysis for the combo D3/1 + D10/0 variant
# a) Concrete behavior and trace
#    - Base scene stays the same: 11x11 grid, fixed green 3x3 destination around one red center,
#      orange floor, optional magenta walls, and scattered maroon source pixels.
#    - Reachability is unchanged: only maroons connected to the destination through floor/maroon
#      cells may move.
#    - The overlap between D3/1 and D10/0 needs one explicit fusion choice. A literal D3/1
#      implementation would sort purely by (row,col), which would make D10/0 inert because
#      column order is already fully determined. To keep both dimensions active, this fused solver
#      uses a row-first scan: primary key = row (D3/1), then within the same row it prefers
#      smaller flood-fill distance, and if a row still contains several maroons at the same
#      distance it uses smaller column first (D10/0).
#    - Example trace for reachable maroons
#        A=(2,8,d=1), B=(4,7,d=5), C=(4,2,d=2), D=(4,9,d=2), E=(8,1,d=0)
#      original task:   sort by (d,r,c)   -> E, A, C, D, B
#      D3-only variant: sort by (r,c)     -> A, C, B, D, E
#      fused combo:     sort by (r,d,c)   -> A, C, D, B, E
#      Difference from D3-only appears inside row 4 because distance is consulted before column;
#      difference from the original remains global because rows dominate across different rows.
# b) Complexity calibration
#    - If reachable maroons are <= 9, all of them move and ranking becomes invisible. The
#      generator therefore uses 10-14 reachable sources so some maroons remain behind.
#    - To make the fused rule visible rather than collapsing to D3-only, the generator rejects
#      samples unless the selected 9 sources under (row,distance,column) differ from both
#      pure scan order (row,column) and the original order (distance,row,column).
#    - Distances come from the visible wall/corridor geometry only; no hidden metadata is used.
# c) Implicit-assumption audit on the original solver
#    - Unique red center detection still holds.
#    - Floor-color inference from the ring around the destination still needs at least one visible
#      non-maroon non-destination ring cell. The generator reserves one adjacent floor cell.
#    - BFS through {floor, maroon} still defines eligibility. What changes is only ranking.
#    - The original `maroons.sort()` encoded (distance,row,col). In the combo, ranking must become
#      an explicit helper so generation can compare fused, scan-only, and original behaviors.
# d) Placement legality
#    - Source maroons are sampled only from reachable floor cells outside the destination block.
#    - Distractors are sampled only from unreachable floor cells, so they cannot overlap walls,
#      sources, or destination cells.
#    - The wall builder only writes in-bounds cells outside the 3x3 target, and every candidate
#      grid is revalidated by running the new solver.
# e) Visual observability
#    - The ranking keys depend only on visible geometry: row position, corridor-derived distance,
#      and column position.
#    - With >9 reachable maroons, the output exposes the rule through which maroons remain behind.
#      In particular, some earlier-in-column maroons on the same row stay put because a nearer
#      maroon later in that row is selected first.
# f) Visual cleanliness
#    - The destination remains a stable landmark and the corridor walls still separate reachable
#      sources from unreachable distractors.
#    - No additional marker colors are introduced; variation is expressed only through which
#      maroons move versus remain.
# g) Change inferability
#    - A human can infer the combo from a few examples because the row-first sweep is obvious, but
#      rows containing multiple reachable maroons require a second observation: closer-in-corridor
#      maroons win within that row, and equal-distance ties within the row read left-to-right.
# h) Generator validation strategy
#    - For each candidate input, compute the chosen source set under fused order, D3-only scan
#      order, and the original distance order. Accept only if fused differs from both baselines and
#      the produced outputs differ as well.
# i) New assumptions introduced by the combo
#    - main() now assumes that the input can expose at least one row-level ranking conflict if the
#      generator wants the combo to be observable. generate_input() guarantees this by rejection.
#    - The fused order still assumes one global destination block and one distance map; the
#      generator preserves the original single-container geometry.
# 9) Audit of new solver assumptions vs generator guarantees
#    - `_infer_floor_color()` assumes a visible floor-colored ring neighbor. Guaranteed by keeping
#      one adjacent floor cell unoccupied.
#    - `_distance_map()` assumes at least one passable cell touches the destination. Guaranteed by
#      reserving that adjacent floor cell and rejecting layouts without adjacent floor.
#    - `_rank_key(..., "fused")` assumes distance values are defined for every candidate maroon.
#      Guaranteed because sources are chosen only from the reachable-floor BFS result.
#    - Destination writeback still assumes the 3x3 block fits in-bounds. Guaranteed by the fixed
#      center (5,5) on an 11x11 canvas.


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

REACHABLE_MAROON_RANGE = (10, 14)
DISTRACTOR_MAX = 6
TOTAL_MAROON_RANGE = (10, 16)
MIN_REACHABLE_FLOOR = REACHABLE_MAROON_RANGE[0] + 1
GEN_RETRIES = 500
SOURCE_SAMPLE_RETRIES = 120
INF = 10**9


def _destination_cells(center_r: int, center_c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(center_r - 1, center_r + 2):
        for c in range(center_c - 1, center_c + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _infer_floor_color(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> int:
    h, w = grid.shape
    dest_set = set(dest_cells)
    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set:
                ring_colors.append(int(grid[rr, cc]))

    candidates = [x for x in ring_colors if x not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert candidates, "ring around destination must show at least one visible floor-colored cell"
    vals, cnts = np.unique(candidates, return_counts=True)
    ring_major = int(vals[cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (RED_CENTER, DEST_COLOR, MAROON, ring_major)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    if floor is None:
        floor = ring_major
    return floor


def _distance_map(grid: np.ndarray, dest_cells: list[tuple[int, int]], passable: set[int]) -> np.ndarray:
    h, w = grid.shape
    dest_set = set(dest_cells)
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set and int(grid[rr, cc]) in passable:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    starts.append((rr, cc))
    assert starts, "there must be at least one passable cell adjacent to the destination block"

    dist = np.full((h, w), INF, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == INF:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _rank_key(r: int, c: int, d: int, mode: str) -> tuple[int, ...]:
    if mode == "fused":
        return (r, d, c)
    if mode == "scan":
        return (r, c)
    if mode == "original":
        return (d, r, c)
    if mode == "col_tie":
        return (d, c, r)
    raise ValueError(f"unknown rank mode: {mode}")


def _solve_with_rank(grid: np.ndarray, mode: str) -> tuple[np.ndarray, set[tuple[int, int]]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w)
    floor = _infer_floor_color(grid, dest_cells)
    dist = _distance_map(grid, dest_cells, {floor, MAROON})

    maroons = []
    for r, c in np.argwhere(grid == MAROON):
        d = int(dist[r, c])
        if d < INF:
            maroons.append((int(r), int(c), d))
    maroons.sort(key=lambda t: _rank_key(t[0], t[1], t[2], mode))

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for r, c, _d in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out, chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move reachable maroon pixels into the 3x3 destination around the red center, using a row-first fused ranking.

    Input structure:
    - The grid is always 11x11 with a fixed green 3x3 target around one red center, orange floor, optional magenta walls, and maroon pixels scattered on floor cells.
    - Some maroons are flood-fill reachable from the target through floor/maroon corridors; others may be sealed off behind walls as distractors.

    Rule:
    - Flood-fill outward from passable cells adjacent to the target, traversing only floor or maroon cells, and keep only reachable maroons as candidates.
    - Rank candidates primarily by row position; within the same row, prefer smaller flood-fill distance to the target; if a row still has multiple candidates at the same distance, use smaller column first.
    - Move up to nine ranked maroons into the destination in destination row-major order, and clear their source cells back to floor.

    Key insight:
    - Walls still determine eligibility, but the movement order is no longer purely distance-based and no longer purely scan-order either; you must combine visible row position with local corridor distance inside each row.
    """

    out, _chosen = _solve_with_rank(grid, "fused")
    return out


def _distance_ranked_output(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    return _solve_with_rank(grid, "original")


def _scan_ranked_output(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    return _solve_with_rank(grid, "scan")


def _bfs_reachable_floor(
    grid: np.ndarray, dest_set: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], np.ndarray, set[tuple[int, int]]]:
    visited: set[tuple[int, int]] = set()
    starts: set[tuple[int, int]] = set()
    dist = np.full((GRID_H, GRID_W), INF, dtype=int)
    q = deque()

    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    starts.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in dest_set:
                    visited.add((nr, nc))
                    dist[nr, nc] = int(dist[r, c]) + 1
                    q.append((nr, nc))
    return visited, dist, starts


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
    wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int
) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    nr, nc = r, c
    for _ in range(random.randint(1, max_len)):
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


def _selected_sources_by_mode(source_cells: list[tuple[int, int]], dist: np.ndarray, mode: str) -> set[tuple[int, int]]:
    ranked = sorted(source_cells, key=lambda rc: _rank_key(int(rc[0]), int(rc[1]), int(dist[rc[0], rc[1]]), mode))
    return set(ranked[:9])


def _fusion_witness_count(source_cells: list[tuple[int, int]], dist: np.ndarray) -> int:
    count = 0
    by_row: dict[int, list[tuple[int, int]]] = {}
    for r, c in source_cells:
        by_row.setdefault(int(r), []).append((int(r), int(c)))
    for row_cells in by_row.values():
        row_cells.sort(key=lambda rc: rc[1])
        for i in range(len(row_cells)):
            r1, c1 = row_cells[i]
            d1 = int(dist[r1, c1])
            for j in range(i + 1, len(row_cells)):
                r2, c2 = row_cells[j]
                d2 = int(dist[r2, c2])
                if d2 < d1:
                    count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor, dist, adjacent_floor = _bfs_reachable_floor(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE_FLOOR or not adjacent_floor:
        return None

    reserved_floor = random.choice(sorted(adjacent_floor))
    source_pool = sorted(reachable_floor - {reserved_floor})
    if len(source_pool) < REACHABLE_MAROON_RANGE[0]:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = sorted(all_floor - reachable_floor)

    max_sources = min(REACHABLE_MAROON_RANGE[1], len(source_pool), TOTAL_MAROON_RANGE[1])
    if max_sources < REACHABLE_MAROON_RANGE[0]:
        return None

    for _ in range(SOURCE_SAMPLE_RETRIES):
        num_sources = random.randint(REACHABLE_MAROON_RANGE[0], max_sources)
        source_cells = random.sample(source_pool, num_sources)

        if _fusion_witness_count(source_cells, dist) == 0:
            continue

        chosen_fused = _selected_sources_by_mode(source_cells, dist, "fused")
        chosen_scan = _selected_sources_by_mode(source_cells, dist, "scan")
        chosen_original = _selected_sources_by_mode(source_cells, dist, "original")

        if chosen_fused == chosen_scan or chosen_fused == chosen_original:
            continue

        max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - num_sources)
        num_distractors = random.randint(0, max_distractors) if max_distractors > 0 else 0
        distractor_cells = random.sample(unreachable_floor, num_distractors) if num_distractors else []

        inp = grid.copy()
        for r, c in source_cells:
            inp[r, c] = MAROON
        for r, c in distractor_cells:
            inp[r, c] = MAROON

        try:
            fused_out, fused_chosen = _solve_with_rank(inp, "fused")
            scan_out, scan_chosen = _scan_ranked_output(inp)
            original_out, original_chosen = _distance_ranked_output(inp)
        except Exception:
            continue

        if fused_chosen != chosen_fused or scan_chosen != chosen_scan or original_chosen != chosen_original:
            continue
        if fused_chosen == scan_chosen or fused_chosen == original_chosen:
            continue
        if np.array_equal(fused_out, scan_out) or np.array_equal(fused_out, original_out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

# Variant of dd6b8c4b
# Dimension: D3 — 收集对象的排序规则 (path)
# Change: sorting_by_distance——收集到的栗色像素按照到绿色方块的距离（flood-fill步数）从近到远排序... → 按空间位置排序（行优先扫描序）

import random
from collections import deque
from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import neighbors4
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import neighbors4


# Design analysis for the D3 variant
# a) Concrete behavior and trace:
#    - The scene stays the same: an 11x11 grid, a fixed 3x3 green destination around one red center,
#      orange floor, optional magenta walls, and scattered maroon tokens.
#    - Connectivity still comes from flood-fill through floor/maroon cells starting at passable cells
#      adjacent to the destination.
#    - The changed step is only the ranking of reachable maroons. In the original task, if reachable
#      maroons were at (1,8),(2,2),(8,1),(9,9) with distances 1,5,3,2, the chosen order was by
#      (distance,row,col): (1,8),(9,9),(8,1),(2,2). In this variant the chosen order is by
#      (row,col): (1,8),(2,2),(8,1),(9,9). The destination still fills in its own row-major order.
# b) Complexity calibration:
#    - If the number of reachable maroons is <= 9, every reachable maroon moves and the new ranking
#      becomes invisible. To make D3 observable, the generator intentionally creates 10-14 reachable
#      maroons, so only 9 move and some remain behind.
#    - The generator also rejects samples where distance-order and scan-order happen to pick the same
#      9 sources; otherwise the instance would collapse back to the original behavior.
# c) Implicit assumption audit on the original solver:
#    - Destination detection by the unique red center is still valid.
#    - Floor-color inference from the ring around the destination still assumes at least one adjacent
#      non-maroon non-destination cell remains visible. The generator therefore reserves at least one
#      floor opening next to the destination.
#    - BFS reachability through {floor, maroon} remains valid because D3 changes only ranking, not
#      eligibility.
#    - The original `maroons.sort()` encoded the old dimension directly. That line must change to sort
#      by coordinates only after reachability has been determined.
# d) Placement legality:
#    - Source maroons are sampled only from reachable floor cells outside the 3x3 destination.
#    - Distractors are sampled only from unreachable floor cells, so they never overlap walls,
#      destination cells, or source maroons.
#    - The reserved adjacent floor cell guarantees both in-bounds placement and solver compatibility.
# e) Visual observability:
#    - The changed ordering is based only on visible positions, never on hidden ids or generation-time
#      metadata.
#    - With >9 reachable maroons, the output shows a full 3x3 maroon block plus leftover maroons whose
#      positions differ from the old distance-based task, making the variant inferable from examples.
# f) Visual cleanliness:
#    - The destination region remains a simple fixed landmark.
#    - Walls still separate reachable from unreachable zones, so moved vs unmoved maroons are spatially
#      legible rather than mixed inside one cluttered area.
# g) Color inference:
#    - No new color mapping is introduced. The only colors stay red/green/orange/magenta/maroon.
#    - Sorting depends on row-major spatial order, which is directly readable from the grid.
# h) Generator validation strategy:
#    - Build a candidate input, run the new solver, run a private old-style distance-ranked solver,
#      and accept the input only if both succeed and produce different outputs.
# i) New main() assumptions and generator guarantees:
#    - main() assumes exactly one red center, at least one passable neighbor of the destination, and at
#      least one visible floor-colored ring cell for floor inference.
#    - generate_input() guarantees these conditions by construction and rejects candidates that do not.


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

REACHABLE_MAROON_RANGE = (10, 14)
DISTRACTOR_MAX = 6
TOTAL_MAROON_RANGE = (10, 16)
MIN_REACHABLE_FLOOR = REACHABLE_MAROON_RANGE[0] + 1
GEN_RETRIES = 400
SOURCE_SAMPLE_RETRIES = 80
INF = 10**9


def _destination_cells(center_r: int, center_c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
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

    cand_ring = [x for x in ring_colors if x not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert cand_ring, "ring around destination must contain at least one visible non-maroon passable color"
    vals, cnts = np.unique(cand_ring, return_counts=True)
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


def _solve_with_rank(grid: np.ndarray, rank_mode: str) -> tuple[np.ndarray, set[tuple[int, int]]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])
    dest_cells = _destination_cells(cr, cc, h, w)
    floor = _infer_floor_color(grid, dest_cells)
    dist = _distance_map(grid, dest_cells, {floor, MAROON})

    maroons = [(int(r), int(c), int(dist[r, c])) for r, c in np.argwhere(grid == MAROON) if int(dist[r, c]) < INF]
    if rank_mode == "scan":
        maroons.sort(key=lambda t: (t[0], t[1]))
    elif rank_mode == "distance":
        maroons.sort(key=lambda t: (t[2], t[0], t[1]))
    else:
        raise ValueError(f"unknown rank_mode: {rank_mode}")

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for r, c, _ in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out, chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move reachable maroon pixels into the 3×3 target around the unique red center, using scan order among reachable tokens.

    Input structure:
    - Orange floor surrounds a fixed green 3×3 destination with one red center, and magenta walls may carve corridors.
    - Maroon pixels sit on floor cells; some are connected to the destination through open corridors and some may be trapped elsewhere.

    Rule:
    - Flood-fill from passable cells touching the destination, traversing only floor or maroon cells to find the reachable maroon pixels.
    - Sort those reachable maroons by spatial position only: top-to-bottom, then left-to-right, and move up to nine of them into the destination in destination row-major order.
    - Clear the chosen source cells back to floor, while unreachable maroons remain where they are.

    Key insight:
    - Walls still decide which maroons are eligible, but once reachability is known the selection is no longer based on path length; it is based purely on visible scan order.
    """

    out, _chosen = _solve_with_rank(grid, "scan")
    return out


def _distance_ranked_output(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    return _solve_with_rank(grid, "distance")


def _bfs_reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], np.ndarray, set[tuple[int, int]]]:
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


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int) -> None:
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
    if mode == "scan":
        ranked = sorted(source_cells, key=lambda rc: (rc[0], rc[1]))
    elif mode == "distance":
        ranked = sorted(source_cells, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))
    else:
        raise ValueError(f"unknown mode: {mode}")
    return set(ranked[:9])


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

        chosen_scan = _selected_sources_by_mode(source_cells, dist, "scan")
        chosen_distance = _selected_sources_by_mode(source_cells, dist, "distance")
        if chosen_scan == chosen_distance:
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
            scan_out, scan_chosen = _solve_with_rank(inp, "scan")
            distance_out, distance_chosen = _distance_ranked_output(inp)
        except Exception:
            continue

        if scan_chosen == distance_chosen:
            continue
        if np.array_equal(scan_out, distance_out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

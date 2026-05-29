# Combo variant of dd6b8c4b (index 5)
# combo_spec: D3/1+D4/1
# Name: 空间位置排序+列优先填充

import random
from collections import deque
from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import neighbors4
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import neighbors4


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The canvas is still the original 11x11 scene: a fixed 3x3 destination centered on one red
#      pixel, green around it, orange floor, optional magenta wall barriers, and maroon tokens on
#      floor cells.
#    - Reachability is unchanged: only maroons connected to the destination through a 4-neighbor
#      flood-fill over floor+maroon cells are eligible.
#    - The fused change splits the original single trajectory into two separate orders:
#      1) source ranking order = row-major scan over reachable maroons, i.e. sort by (row, col);
#      2) destination fill order = column-major traversal of the 3x3 block, i.e.
#         [(4,4),(5,4),(6,4),(4,5),(5,5),(6,5),(4,6),(5,6),(6,6)] around the fixed center.
#    - Example trace on the same input:
#      reachable maroons = {(1,8),(2,2),(8,1),(9,9),...}
#      original task ranked by (distance,row,col) and filled row-major.
#      combo task ranks by (row,col): (1,8),(2,2),(8,1),(9,9),...
#      then places the moved maroons down the first column of the target before moving right.
#    - Because the task always moves k = min(9, reachable_count) maroons, there is an unavoidable
#      observability tension:
#      * if reachable_count > 9, the source ranking is visible because some reachable maroons stay
#        behind, but the destination is fully filled so row-major vs column-major becomes invisible;
#      * if reachable_count < 9, the column-major fill is visible, but every reachable maroon moves,
#        so source ranking is invisible.
#      The solver still implements both dimensions simultaneously; the generator therefore produces
#      two complementary families of examples so both dimensions appear across samples.
# b) Complexity calibration:
#    - The generator uses two explicit regimes:
#      * "sorting_visible": 10-14 reachable sources so scan-order selection differs from the
#        original distance-based selection.
#      * "fill_visible": 2-8 reachable sources so the partial 3x3 fill shows column-major order.
#    - This avoids collapsing everything into one ambiguous case while keeping the task readable:
#      human solvers only need to infer two binary choices, scan-vs-distance and row-vs-column.
# c) Implicit assumption audit:
#    - The unique red center still anchors the destination; that assumption remains valid.
#    - The original `dest_cells.sort()` encoded row-major fill implicitly. In the combo solver we
#      keep row-major destination cells only for adjacency bookkeeping and build a separate explicit
#      column-major `fill_cells`.
#    - The original `maroons.sort()` encoded distance ranking directly. That assumption is replaced
#      with sorting reachable maroons by coordinates only.
#    - Floor-color inference still needs at least one visible non-maroon ring-adjacent cell. The
#      generator reserves at least one adjacent floor cell so `main()` does not assert.
#    - BFS still assumes there is at least one passable neighbor next to the destination. The same
#      reserved floor cell guarantees that as well.
# d) Placement legality:
#    - Walls are drawn only outside the fixed 3x3 destination.
#    - Source maroons are sampled only from reachable floor cells outside the destination.
#    - Distractors are sampled only from unreachable floor cells, so they never overlap walls,
#      destination cells, or reachable sources.
#    - The 3x3 destination is centered at (5,5), so every destination/fill coordinate stays in bounds.
# e) Visual observability:
#    - Ranking depends only on visible positions, never on hidden ids or color values.
#    - Column-major fill is a directly visible geometric pattern in the target whenever k < 9.
#    - Scan-order source selection is visible through which reachable maroons remain on the floor
#      whenever reachable_count > 9.
# f) Visual cleanliness:
#    - The same simple landmark structure from the base task is preserved.
#    - Walls still provide clear corridor structure, so reachable and unreachable maroons remain
#      spatially interpretable rather than visually tangled.
# g) Change inferability:
#    - No extra colors or symbolic markers are introduced; the only changed cues are spatial order
#      among visible maroons and the fill trajectory inside the fixed 3x3 target.
# h) Generator validation strategy:
#    - Every accepted sample is checked by the combo solver.
#    - In sorting-visible mode, the accepted sample must differ from the original task and also
#      differ from the distance-ranked / column-filled hybrid.
#    - In fill-visible mode, the accepted sample must differ from the original task and also differ
#      from the scan-ranked / row-filled hybrid.
#    - Across repeated draws, both modes are sampled so the dataset-level behavior exercises both
#      fused dimensions even though a single sample cannot make both simultaneously visible.
# i) New main() assumptions and generator guarantees:
#    - `main()` assumes exactly one red center, at least one passable cell adjacent to the target,
#      and at least one visible non-maroon neighbor color around the target for floor inference.
#    - `generate_input()` guarantees these conditions by reserving one adjacent floor cell, checking
#      reachable floor counts before sampling, and rejecting any candidate that does not survive the
#      solver comparisons.


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

SORT_VISIBLE_REACHABLE_RANGE = (10, 14)
FILL_VISIBLE_REACHABLE_RANGE = (2, 8)
DISTRACTOR_MAX = 6
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE_FLOOR = SORT_VISIBLE_REACHABLE_RANGE[0] + 1
GEN_RETRIES = 500
SOURCE_SAMPLE_RETRIES = 100
INF = 10**9


def _destination_cells(center_r: int, center_c: int, h: int, w: int, order: str) -> list[tuple[int, int]]:
    rows = range(center_r - 1, center_r + 2)
    cols = range(center_c - 1, center_c + 2)
    cells: list[tuple[int, int]] = []
    if order == "row_major":
        for r in rows:
            for c in cols:
                if 0 <= r < h and 0 <= c < w:
                    cells.append((r, c))
    elif order == "col_major":
        for c in cols:
            for r in rows:
                if 0 <= r < h and 0 <= c < w:
                    cells.append((r, c))
    else:
        raise ValueError(f"unknown destination order: {order}")
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
    assert cand_ring, "ring around destination must contain at least one visible non-maroon color"
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


def _solve(
    grid: np.ndarray,
    rank_mode: str,
    fill_order: str,
) -> tuple[np.ndarray, set[tuple[int, int]], int, int]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w, "row_major")
    fill_cells = _destination_cells(cr, cc, h, w, fill_order)
    floor = _infer_floor_color(grid, dest_cells)
    dist = _distance_map(grid, dest_cells, {floor, MAROON})

    reachable = [(int(r), int(c), int(dist[r, c])) for r, c in np.argwhere(grid == MAROON) if int(dist[r, c]) < INF]
    if rank_mode == "scan":
        reachable.sort(key=lambda t: (t[0], t[1]))
    elif rank_mode == "distance":
        reachable.sort(key=lambda t: (t[2], t[0], t[1]))
    else:
        raise ValueError(f"unknown rank_mode: {rank_mode}")

    k = min(len(fill_cells), len(reachable))
    chosen = {(r, c) for r, c, _ in reachable[:k]}

    for r, c in chosen:
        out[r, c] = floor
    for r, c in fill_cells[:k]:
        out[r, c] = MAROON

    return out, chosen, k, len(reachable)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move reachable maroon pixels into the 3x3 target around the unique red center, ranking by scan position and filling by columns.

    Input structure:
    - The board contains orange floor, optional magenta walls, a fixed green 3x3 destination, and a single red center marker.
    - Maroon pixels lie on floor cells outside the destination; some are connected to the target through open corridors and others may be sealed off behind walls.

    Rule:
    - Flood-fill from passable cells adjacent to the destination through floor and maroon cells to determine which maroon pixels are reachable.
    - Ignore path distance when ordering those reachable maroons: sort them only by row first, then column, erase the first up to nine source cells, and place those maroons into the destination in column-major order from the top-left.
    - Destination cells beyond the moved count keep their original green/red colors, and unreachable maroons remain unchanged.

    Key insight:
    - The problem separates eligibility from placement: walls still decide which maroons can move, but once reachability is known the source order is a global scan order while the destination uses a different, column-first traversal.
    """

    out, _chosen, _k, _reachable_count = _solve(grid, "scan", "col_major")
    return out


def _original_output(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], int, int]:
    return _solve(grid, "distance", "row_major")


def _scan_row_output(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], int, int]:
    return _solve(grid, "scan", "row_major")


def _distance_col_output(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], int, int]:
    return _solve(grid, "distance", "col_major")


def _bfs_reachable_floor(
    grid: np.ndarray,
    dest_set: set[tuple[int, int]],
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


def _build_base_grid() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]]]:
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W, "row_major")
    dest_set = set(dest_cells)
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)
    return grid, dest_cells, dest_set


def _all_floor_cells(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }


def _random_distractors(
    unreachable_floor: list[tuple[int, int]],
    num_sources: int,
) -> list[tuple[int, int]] | None:
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - num_sources)
    min_distractors = max(0, TOTAL_MAROON_RANGE[0] - num_sources)
    if max_distractors < min_distractors:
        return None
    num_distractors = random.randint(min_distractors, max_distractors) if max_distractors > 0 else 0
    return random.sample(unreachable_floor, num_distractors) if num_distractors else []


def _assemble_input(
    grid: np.ndarray,
    source_cells: list[tuple[int, int]],
    distractor_cells: list[tuple[int, int]],
) -> np.ndarray:
    inp = grid.copy()
    for r, c in source_cells:
        inp[r, c] = MAROON
    for r, c in distractor_cells:
        inp[r, c] = MAROON
    return inp


def _gen_sorting_visible(
    grid: np.ndarray,
    reachable_floor: set[tuple[int, int]],
    unreachable_floor: list[tuple[int, int]],
    dist: np.ndarray,
    adjacent_floor: set[tuple[int, int]],
) -> np.ndarray | None:
    if len(reachable_floor) < MIN_REACHABLE_FLOOR or not adjacent_floor:
        return None

    reserved_floor = random.choice(sorted(adjacent_floor))
    source_pool = sorted(reachable_floor - {reserved_floor})
    min_sources = SORT_VISIBLE_REACHABLE_RANGE[0]
    max_sources = min(SORT_VISIBLE_REACHABLE_RANGE[1], len(source_pool), TOTAL_MAROON_RANGE[1])
    if max_sources < min_sources:
        return None

    for _ in range(SOURCE_SAMPLE_RETRIES):
        num_sources = random.randint(min_sources, max_sources)
        source_cells = random.sample(source_pool, num_sources)

        scan_top9 = set(sorted(source_cells, key=lambda rc: (rc[0], rc[1]))[:9])
        distance_top9 = set(sorted(source_cells, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))[:9])
        if scan_top9 == distance_top9:
            continue

        distractor_cells = _random_distractors(unreachable_floor, num_sources)
        if distractor_cells is None:
            continue
        inp = _assemble_input(grid, source_cells, distractor_cells)

        try:
            combo_out, combo_chosen, k, reachable_count = _solve(inp, "scan", "col_major")
            original_out, _, _, _ = _original_output(inp)
            distance_col_out, _, _, _ = _distance_col_output(inp)
        except Exception:
            continue

        if reachable_count <= 9 or k != 9:
            continue
        if combo_chosen == distance_top9:
            continue
        if np.array_equal(combo_out, original_out):
            continue
        if np.array_equal(combo_out, distance_col_out):
            continue
        return inp
    return None


def _gen_fill_visible(
    grid: np.ndarray,
    reachable_floor: set[tuple[int, int]],
    unreachable_floor: list[tuple[int, int]],
    adjacent_floor: set[tuple[int, int]],
) -> np.ndarray | None:
    if not adjacent_floor:
        return None

    reserved_floor = random.choice(sorted(adjacent_floor))
    source_pool = sorted(reachable_floor - {reserved_floor})
    min_sources = FILL_VISIBLE_REACHABLE_RANGE[0]
    max_sources = min(FILL_VISIBLE_REACHABLE_RANGE[1], len(source_pool), TOTAL_MAROON_RANGE[1])
    if max_sources < min_sources:
        return None

    for _ in range(SOURCE_SAMPLE_RETRIES):
        num_sources = random.randint(min_sources, max_sources)
        source_cells = random.sample(source_pool, num_sources)
        distractor_cells = _random_distractors(unreachable_floor, num_sources)
        if distractor_cells is None:
            continue
        inp = _assemble_input(grid, source_cells, distractor_cells)

        try:
            combo_out, combo_chosen, k, reachable_count = _solve(inp, "scan", "col_major")
            original_out, _, _, _ = _original_output(inp)
            scan_row_out, _, _, _ = _scan_row_output(inp)
        except Exception:
            continue

        if reachable_count != num_sources:
            continue
        if k >= 9:
            continue
        if len(combo_chosen) != num_sources:
            continue
        if np.array_equal(combo_out, original_out):
            continue
        if np.array_equal(combo_out, scan_row_out):
            continue
        return inp
    return None


def _gen_attempt() -> np.ndarray | None:
    grid, _dest_cells, dest_set = _build_base_grid()
    reachable_floor, dist, adjacent_floor = _bfs_reachable_floor(grid, dest_set)
    all_floor = _all_floor_cells(grid, dest_set)
    unreachable_floor = sorted(all_floor - reachable_floor)

    modes = ["sorting_visible", "fill_visible"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "sorting_visible":
            result = _gen_sorting_visible(grid, reachable_floor, unreachable_floor, dist, adjacent_floor)
        else:
            result = _gen_fill_visible(grid, reachable_floor, unreachable_floor, adjacent_floor)
        if result is not None:
            return result
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

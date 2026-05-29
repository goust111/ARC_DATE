# Combo variant of dd6b8c4b (index 1)
# combo_spec: D1/2+D19/0
# Name: 8连通可达性+切比雪夫距离

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4, neighbors8


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Original task starts flood-fill from the orthogonal ring around the 3x3 target and
#    also propagates orthogonally, so a full 12-cell wall ring blocks everything.
#    Here we keep that orthogonal ring walled, but leave the four diagonal corner cells
#    of the surrounding 5x5 square open. Example: with walls at (3,4..6), (4..6,3),
#    (4..6,7), (7,4..6), a maroon at (2,2) is unreachable in the original rule because
#    no orthogonal start exists; in the fused rule the corner (3,3) is an 8-neighbor of
#    the destination, so BFS can start there and reach (2,2) diagonally. Distances also
#    change: from start (3,3), cell (1,5) has Manhattan-style length 4 but Chebyshev
#    length 2, so diagonal-heavy tokens move ahead in the ranking.
# b) Complexity calibration:
#    The layout uses one stable structural motif: a full orthogonal wall ring plus open
#    diagonal corners. This keeps the visual cue repeated across samples while still
#    making the 8-neighbor rule necessary. Source tokens are sampled from a controlled
#    annulus near the target so the changed ranking is visible without clutter.
# c) Hidden assumptions audit:
#    - Original ring scan assumed 4-neighbor cells around the destination reveal floor
#      and wall colors; that still holds because the generator always places the wall ring
#      there.
#    - Original BFS assumed at least one orthogonally adjacent passable start; this is no
#      longer valid, so the solver must gather 8-neighbor starts instead.
#    - Sorting only matters if more than 9 reachable maroons exist; the generator now
#      always places exactly 10 reachable maroons and rejects samples where Chebyshev and
#      Manhattan would keep the same 9.
# d) Placement legality:
#    The destination stays the fixed 3x3 block at rows/cols 4..6. Walls are only written
#    on the 12-cell orthogonal ring. Maroons are sampled from floor cells outside both
#    the destination and the wall ring, so nothing overlaps or goes out of bounds.
# e) Visual observability:
#    A solver can see that the target is sealed off orthogonally yet still collects
#    diagonally positioned maroons, which directly exposes the 8-connectivity change.
#    Because only 9 of 10 maroons move, the leftover source position reveals the distance
#    ranking and differs from a Manhattan interpretation.
# f) Visual cleanliness:
#    The active geometry is concentrated in the 5x5-to-9x9 region around the center. The
#    wall ring, diagonal openings, and maroon pool stay visually separate, so the input
#    does not collapse into noisy full-board scatter.
# g) New solver assumptions vs generator guarantees:
#    main() now assumes there is at least one passable 8-neighbor around the destination
#    and that floor/wall are distinguishable from the orthogonal ring. generate_input()
#    guarantees this by forcing the 12 orthogonal ring cells to be walls and the four
#    diagonal corner cells to remain floor or maroon.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

SOURCE_COUNT = 10
MAX_SAMPLE_TRIES = 400
GEN_RETRIES = 300


def _dest_cells():
    return [
        (r, c)
        for r in range(CENTER_R - 1, CENTER_R + 2)
        for c in range(CENTER_C - 1, CENTER_C + 2)
    ]


def _orth_ring(dest_set):
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    return sorted(ring)


def _infer_floor(grid, dest_cells, dest_set):
    ring_colors = []
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and (rr, cc) not in dest_set:
                ring_colors.append(int(grid[rr, cc]))

    ring_non_special = [x for x in ring_colors if x not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert ring_non_special, "orthogonal ring must reveal at least one non-special color"
    vals, cnts = np.unique(ring_non_special, return_counts=True)
    ring_major = int(vals[cnts.argmax()])

    other = [int(x) for x in np.unique(grid) if int(x) not in (RED_CENTER, DEST_COLOR, MAROON, ring_major)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    return ring_major if floor is None else floor


def _collect_starts(grid, dest_cells, passable, neighbor_fn):
    h, w = grid.shape
    starts = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc in neighbor_fn(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    starts.append((rr, cc))
    return starts


def _distance_map(grid, passable, start_neighbor_fn, move_neighbor_fn):
    h, w = grid.shape
    dest_cells = _dest_cells()
    starts = _collect_starts(grid, dest_cells, passable, start_neighbor_fn)
    assert starts, "there must be at least one passable start around the destination"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for rr, cc in move_neighbor_fn(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == inf:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _rank_reachable_maroons(grid, dist):
    inf = 10**9
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [triple for triple in maroons if triple[0] < inf]
    maroons.sort()
    return maroons


def _solve_with_modes(grid, start_neighbor_fn, move_neighbor_fn):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                dest_cells.append((r, c))
    dest_cells.sort()
    dest_set = set(dest_cells)

    floor = _infer_floor(grid, dest_cells, dest_set)
    passable = {floor, MAROON}
    dist = _distance_map(grid, passable, start_neighbor_fn, move_neighbor_fn)
    ranked = _rank_reachable_maroons(grid, dist)

    k = min(len(dest_cells), len(ranked))
    chosen = {(r, c) for _, r, c in ranked[:k]}

    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out, ranked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the reachable maroon pixels into the 3x3 block centered on the unique red cell.

    Input structure:
    - Orange floor surrounds a fixed green 3x3 destination block with one red center.
    - Magenta walls can seal the destination orthogonally while maroon source pixels sit on floor cells outside it.

    Rule:
    - Start from every passable cell that touches the destination in the 8-neighborhood, then flood-fill through floor/maroon cells using 8-connectivity.
    - Rank reachable maroon pixels by the resulting 8-direction shortest-path distance (Chebyshev-style), breaking ties by row then column.
    - Move the closest ranked maroons into the destination in row-major order and erase their original positions back to floor.

    Key insight:
    - Diagonal contact now matters twice: it can make a source reachable even when all orthogonal openings are blocked, and it also counts as a single distance step during ranking.
    """

    out, _ranked = _solve_with_modes(grid, neighbors8, neighbors8)
    return out


def _combo_ranked(grid):
    passable = {FLOOR_COLOR, MAROON}
    dist = _distance_map(grid, passable, neighbors8, neighbors8)
    return _rank_reachable_maroons(grid, dist)


def _manhattan_ranked_with_diagonal_starts(grid):
    passable = {FLOOR_COLOR, MAROON}
    dist = _distance_map(grid, passable, neighbors8, neighbors4)
    return _rank_reachable_maroons(grid, dist)


def _build_base_grid():
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    for r, c in _orth_ring(dest_set):
        grid[r, c] = WALL_COLOR
    return grid


def _source_pool(grid):
    passable = {FLOOR_COLOR, MAROON}
    dist8 = _distance_map(grid, passable, neighbors8, neighbors8)
    pool = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if grid[r, c] != FLOOR_COLOR:
                continue
            d = int(dist8[r, c])
            if 1 <= d <= 4:
                pool.append((r, c))
    return pool


def _sample_sources(base_grid):
    pool = _source_pool(base_grid)
    assert len(pool) >= SOURCE_COUNT, "source pool must be large enough"
    for _ in range(MAX_SAMPLE_TRIES):
        chosen = random.sample(pool, SOURCE_COUNT)
        inp = base_grid.copy()
        for r, c in chosen:
            inp[r, c] = MAROON

        combo_ranked = _combo_ranked(inp)
        man_ranked = _manhattan_ranked_with_diagonal_starts(inp)
        if len(combo_ranked) != SOURCE_COUNT or len(man_ranked) != SOURCE_COUNT:
            continue

        combo_leftover = (combo_ranked[9][1], combo_ranked[9][2])
        man_leftover = (man_ranked[9][1], man_ranked[9][2])
        if combo_leftover == man_leftover:
            continue

        # Prefer samples where diagonal travel gives a noticeable advantage.
        dist8 = _distance_map(inp, {FLOOR_COLOR, MAROON}, neighbors8, neighbors8)
        dist4 = _distance_map(inp, {FLOOR_COLOR, MAROON}, neighbors8, neighbors4)
        max_gap = max(
            int(dist4[r, c]) - int(dist8[r, c])
            for r, c in chosen
        )
        if max_gap < 2:
            continue

        return inp
    return None


def _gen_attempt():
    base_grid = _build_base_grid()
    inp = _sample_sources(base_grid)
    if inp is None:
        return None

    try:
        result = main(inp)
    except Exception:
        return None

    # Ensure the fused behavior is active:
    # - exactly 10 reachable sources under the combo rule
    # - exactly 9 are moved into the destination
    # - the omitted source differs from the Manhattan-style ranking
    combo_ranked = _combo_ranked(inp)
    man_ranked = _manhattan_ranked_with_diagonal_starts(inp)
    if len(combo_ranked) != SOURCE_COUNT or len(man_ranked) != SOURCE_COUNT:
        return None
    if (combo_ranked[9][1], combo_ranked[9][2]) == (man_ranked[9][1], man_ranked[9][2]):
        return None
    if int((result == MAROON).sum()) != SOURCE_COUNT:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

# Combo variant of dd6b8c4b (index 52)
# combo_spec: D1/2+D3/2+D19/0
# Name: 8连通+切比雪夫距离+欧几里得距离排序

import random

import numpy as np

from arc_utils import neighbors4, neighbors8


# Design analysis for the fused rule:
# a) Concrete behavior:
#    - Original task expands from the 3x3 destination through 4-neighbor floor cells and
#      sorts reachable maroons by flood-fill distance.
#    - This variant changes the expansion geometry to 8-neighbors, so a floor cell that
#      touches the green block only at a corner is now a valid gateway. The path wave is
#      therefore an 8-step / Chebyshev-style BFS.
#    - Sorting is intentionally decoupled from that path wave: once maroons are known to
#      be reachable, the chosen nine are the ones with the smallest Euclidean distance to
#      the red center. Example: a token at (3,3) can be reachable through a diagonal
#      corner even when (3,4) and (4,3) are walls; among many reachable tokens, a
#      center-near token can beat a path-short but farther corner token.
# b) Complexity calibration:
#    - The grid size, colors, and single 3x3 destination remain unchanged to preserve the
#      original visual language.
#    - To make sorting observable, the generator always places more than nine reachable
#      maroons; otherwise "sorting" would be invisible because every reachable token would
#      move.
#    - To make 8-connectivity observable, the wall ring blocks all orthogonal contacts and
#      leaves only diagonal corner gateways.
# c) Implicit-assumption audit:
#    - The old code assumed 4-neighbor starts beside the block; that fails here, so start
#      cells must be collected with 8-neighbors.
#    - The old distance array doubled as both reachability and ranking. Here the 8-BFS
#      distance still defines reachability, but ranking must ignore it and use Euclidean
#      distance to the red center instead.
#    - Erasing moved maroons back to floor=7 remains valid because the spec keeps the same
#      fixed color roles.
# d) Placement legality:
#    - Walls, destination, source maroons, and sealed distractor pockets are placed on
#      disjoint cells inside the 11x11 bounds.
#    - Sealed pockets are 8-neighbor enclosed, so distractors inside them are genuinely
#      unreachable even under the new connectivity rule.
# e/f/g) Observability and cleanliness:
#    - The diagonal-only gateways are visible at the four destination corners.
#    - The output retains some outside maroons because only nine move; this exposes which
#      reachable sources were selected by Euclidean ranking.
#    - The generator rejects layouts unless at least one selected token is 8-reachable but
#      not 4-reachable, and unless Euclidean top-9 differs from Chebyshev-path top-9.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9
DEST_SIZE = 9

OPEN_DIAGONALS_RANGE = (1, 3)
REACHABLE_MAROON_RANGE = (10, 13)
SEALED_POCKET_RANGE = (1, 4)
DISTRACTOR_MAX = 3
GEN_RETRIES = 400

DIAGONAL_GATEWAYS = [(3, 3), (3, 7), (7, 3), (7, 7)]
POCKET_CENTERS = [(1, 1), (1, 9), (9, 1), (9, 9)]


def _in_bounds(r: int, c: int, h: int = GRID_H, w: int = GRID_W) -> bool:
    return 0 <= r < h and 0 <= c < w


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if _in_bounds(r, c, h, w):
                cells.append((r, c))
    cells.sort()
    return cells


def _passable_neighbors(use8: bool):
    return neighbors8 if use8 else neighbors4


def _flood_dist(grid: np.ndarray, dest_set: set[tuple[int, int]], use8: bool) -> np.ndarray:
    h, w = grid.shape
    neigh = _passable_neighbors(use8)
    dist = np.full((h, w), 10**9, dtype=int)
    starts: list[tuple[int, int]] = []
    seen = set()

    for r, c in dest_set:
        for nr, nc in neigh(r, c):
            if not _in_bounds(nr, nc, h, w) or (nr, nc) in dest_set:
                continue
            if int(grid[nr, nc]) in (FLOOR_COLOR, MAROON) and (nr, nc) not in seen:
                seen.add((nr, nc))
                starts.append((nr, nc))
                dist[nr, nc] = 0

    q = list(starts)
    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        for nr, nc in neigh(r, c):
            if not _in_bounds(nr, nc, h, w) or (nr, nc) in dest_set:
                continue
            if dist[nr, nc] != 10**9:
                continue
            if int(grid[nr, nc]) not in (FLOOR_COLOR, MAROON):
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    return dist


def _euclid2(r: int, c: int, cr: int, cc: int) -> int:
    return (r - cr) ** 2 + (c - cc) ** 2


def _reachable_maroons(
    grid: np.ndarray, dist: np.ndarray, cr: int, cc: int
) -> list[tuple[int, int, int]]:
    maroons: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid == MAROON):
        if int(dist[r, c]) < 10**9:
            maroons.append((_euclid2(int(r), int(c), cr, cc), int(r), int(c)))
    maroons.sort()
    return maroons


def _apply_move(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    chosen_sources: set[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in chosen_sources:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[: len(chosen_sources)]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the Euclidean-closest reachable maroon pixels into the 3x3 block around the red center.

    Input structure:
    - Orange floor holds scattered maroon tokens, magenta wall cells, and one fixed 3x3 target block whose center is the unique red pixel.
    - The ring directly touching the block can be fully walled orthogonally, leaving only diagonal corner contact points to the outside floor.

    Rule:
    - Start from every floor/maroon cell that touches the target block in 8-neighbor sense, then flood-fill through floor+maroon with 8-connectivity; this is the reachability test and its path metric is Chebyshev-style 8-step BFS.
    - Keep only maroon pixels reached by that wave, but rank them by Euclidean distance to the red center rather than by flood-fill distance; ties break by row then column.
    - Move the best k=min(9, reachable_count) maroons into the destination block in row-major order and erase those source cells back to orange floor.

    Key insight:
    - Diagonal contact controls eligibility, but straight-line nearness to the center controls selection once there are too many reachable tokens, so reachability geometry and ranking geometry must be separated.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    dest_set = set(dest_cells)
    dist8 = _flood_dist(grid, dest_set, use8=True)
    ranked = _reachable_maroons(grid, dist8, cr, cc)
    k = min(len(dest_cells), len(ranked))
    chosen = {(r, c) for _, r, c in ranked[:k]}
    return _apply_move(grid, dest_cells, chosen)


def _orth_ring(dest_cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    ring = set()
    dest_set = set(dest_cells)
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if _in_bounds(nr, nc) and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    return ring


def _reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]], use8: bool) -> set[tuple[int, int]]:
    dist = _flood_dist(grid, dest_set, use8=use8)
    cells = set()
    for r in range(GRID_H):
        for c in range(GRID_W):
            if grid[r, c] == FLOOR_COLOR and int(dist[r, c]) < 10**9:
                cells.add((r, c))
    return cells


def _build_layout() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]], np.ndarray, np.ndarray]:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    for r, c in _orth_ring(dest_cells):
        grid[r, c] = WALL_COLOR

    open_count = random.randint(*OPEN_DIAGONALS_RANGE)
    open_gateways = set(random.sample(DIAGONAL_GATEWAYS, open_count))
    for r, c in DIAGONAL_GATEWAYS:
        if (r, c) not in open_gateways:
            grid[r, c] = WALL_COLOR

    pocket_count = random.randint(*SEALED_POCKET_RANGE)
    for pr, pc in random.sample(POCKET_CENTERS, pocket_count):
        for r in range(pr - 1, pr + 2):
            for c in range(pc - 1, pc + 2):
                if not _in_bounds(r, c) or (r, c) in dest_set:
                    continue
                if (r, c) == (pr, pc):
                    grid[r, c] = FLOOR_COLOR
                else:
                    grid[r, c] = WALL_COLOR

    dist8 = _flood_dist(grid, dest_set, use8=True)
    dist4 = _flood_dist(grid, dest_set, use8=False)
    return grid, dest_cells, dest_set, dist8, dist4


def _rank_positions_by_euclid(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda rc: (_euclid2(rc[0], rc[1], CENTER_R, CENTER_C), rc[0], rc[1]))


def _rank_positions_by_path(cells: set[tuple[int, int]], dist: np.ndarray) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))


def _is_observable(
    source_cells: set[tuple[int, int]],
    dist8: np.ndarray,
    dist4: np.ndarray,
) -> bool:
    if len(source_cells) <= DEST_SIZE:
        return False

    euclid_rank = _rank_positions_by_euclid(source_cells)
    path_rank = _rank_positions_by_path(source_cells, dist8)
    euclid_top = set(euclid_rank[:DEST_SIZE])
    path_top = set(path_rank[:DEST_SIZE])
    if euclid_top == path_top:
        return False

    diagonal_only = {
        cell for cell in source_cells if int(dist8[cell[0], cell[1]]) < 10**9 and int(dist4[cell[0], cell[1]]) == 10**9
    }
    if not diagonal_only:
        return False

    if not (euclid_top & diagonal_only):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    base, dest_cells, dest_set, dist8, dist4 = _build_layout()
    reachable8 = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and int(dist8[r, c]) < 10**9
    }
    if len(reachable8) < REACHABLE_MAROON_RANGE[0]:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = sorted(all_floor - reachable8)
    if not unreachable_floor:
        return None

    max_reachable = min(REACHABLE_MAROON_RANGE[1], len(reachable8))
    if max_reachable < REACHABLE_MAROON_RANGE[0]:
        return None

    for _ in range(120):
        reachable_count = random.randint(REACHABLE_MAROON_RANGE[0], max_reachable)
        source_cells = set(random.sample(sorted(reachable8), reachable_count))
        if not _is_observable(source_cells, dist8, dist4):
            continue

        max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), 16 - reachable_count)
        distractor_count = random.randint(1, max_distractors) if max_distractors > 0 else 0
        distractor_cells = set(random.sample(unreachable_floor, distractor_count))

        inp = base.copy()
        for r, c in source_cells | distractor_cells:
            inp[r, c] = MAROON

        chosen = set(_rank_positions_by_euclid(source_cells)[:DEST_SIZE])
        expected = _apply_move(inp, dest_cells, chosen)

        try:
            result = main(inp)
        except Exception:
            continue
        if np.array_equal(result, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

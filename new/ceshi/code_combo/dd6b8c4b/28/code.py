# Combo variant of dd6b8c4b (index 28)
# combo_spec: D1/2+D2/1+D9/0
# Name: 8连通可达+墙壁通道口+规则框架

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4, neighbors8


# Design analysis for the fused variant
# 1. Concrete behavior:
#    - Original task: 4-connected flood-fill starts from floor cells orthogonally touching the
#      green 3x3 target block and moves through floor/maroon cells.
#    - Fused task: the magenta barrier is a strict 5x5 rectangle around the target, but one or
#      two 1-pixel side openings are placed near the rectangle's corners. Each opening is blocked
#      by a single wall pixel immediately outside it. In 4-connectivity that opening is a dead end;
#      in 8-connectivity the fill can step diagonally around the blocker and escape to the outside.
#    - Example: opening at (3,4), blocker at (2,4). From the target, 4-neighborhood reaches
#      (3,4) and stops. 8-neighborhood reaches (3,4), then diagonally slips to (2,3) or (2,5),
#      so the whole exterior becomes reachable.
# 2. Complexity calibration:
#    - The frame stays visually simple and repetitive: always the same centered rectangle, with
#      only 1-2 blocked mouths chosen from a small catalog of corner-adjacent positions.
#    - This keeps the fused behavior visible without adding too many unrelated wall shapes.
# 3. Hidden assumptions broken from the original solver:
#    - Starting the search from only 4-neighbors of the destination is no longer valid.
#    - Expanding with only neighbors4 is no longer valid because diagonal escape is the key signal.
#    - The old irregular-ring wall generator is no longer appropriate; a rectangular frame with
#      explicit mouths is required so the human can see why diagonal passage matters.
# 4. Placement legality:
#    - The target block is fixed at rows/cols 4..6 inside an 11x11 grid.
#    - The frame is the 5x5 perimeter at rows/cols 3..7, so blockers just outside mouths still
#      stay within bounds.
#    - Maroons are never placed on the frame, on blocker cells, or inside the target block.
# 5. Visual observability and neatness:
#    - Only the original five colors are used.
#    - The key cue is always local and inspectable: a neat rectangle, a missing side wall cell,
#      and a single magenta plug directly outside that mouth.
#    - Because sources are chosen from 8-only reachable cells, the fused output differs from the
#      old 4-connected reading on the same input instead of silently collapsing back to the base task.
# 6. New solver assumptions guaranteed by the generator:
#    - Exactly one red center exists.
#    - The destination remains the centered 3x3 block.
#    - Floor=7, wall=6, target ring=3, center=2, tokens=9.
#    - At least one blocked mouth exists, and at least four maroons lie in the 8-only reachable set.


GRID_H = 11
GRID_W = 11
CENTER_R = 5
CENTER_C = 5

FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9

FRAME_TOP = 3
FRAME_BOTTOM = 7
FRAME_LEFT = 3
FRAME_RIGHT = 7

GEN_RETRIES = 400
FILL_RANGE = (4, 9)
PORTAL_COUNT_RANGE = (1, 2)
SOURCE_PREFIX_EXTRA = (3, 8)
MIN_EXCLUSIVE_CELLS = 12


def _dest_cells() -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(CENTER_R - 1, CENTER_R + 2)
        for c in range(CENTER_C - 1, CENTER_C + 2)
    ]


def _frame_cells() -> set[tuple[int, int]]:
    cells = set()
    for c in range(FRAME_LEFT, FRAME_RIGHT + 1):
        cells.add((FRAME_TOP, c))
        cells.add((FRAME_BOTTOM, c))
    for r in range(FRAME_TOP, FRAME_BOTTOM + 1):
        cells.add((r, FRAME_LEFT))
        cells.add((r, FRAME_RIGHT))
    return cells


def _portal_catalog() -> list[dict]:
    return [
        {"name": "top_left", "opening": (3, 4), "blocker": (2, 4)},
        {"name": "top_right", "opening": (3, 6), "blocker": (2, 6)},
        {"name": "bottom_left", "opening": (7, 4), "blocker": (8, 4)},
        {"name": "bottom_right", "opening": (7, 6), "blocker": (8, 6)},
        {"name": "left_top", "opening": (4, 3), "blocker": (4, 2)},
        {"name": "left_bottom", "opening": (6, 3), "blocker": (6, 2)},
        {"name": "right_top", "opening": (4, 7), "blocker": (4, 8)},
        {"name": "right_bottom", "opening": (6, 7), "blocker": (6, 8)},
    ]


def _flood(grid: np.ndarray, dest_set: set[tuple[int, int]], passable: set[int], use8: bool):
    h, w = grid.shape
    neigh = neighbors8 if use8 else neighbors4
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque()
    seen_start = set()

    for r, c in dest_set:
        for nr, nc in neigh(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and (nr, nc) not in seen_start:
                if int(grid[nr, nc]) in passable:
                    seen_start.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for nr, nc in neigh(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) in passable and int(dist[nr, nc]) == inf:
                    dist[nr, nc] = d + 1
                    q.append((nr, nc))

    reached = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in dest_set and int(dist[r, c]) < inf
    }
    return reached, dist


def _solve_with_connectivity(grid: np.ndarray, use8: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    centers = np.argwhere(g == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = [
        (r, c)
        for r in range(cr - 1, cr + 2)
        for c in range(cc - 1, cc + 2)
        if 0 <= r < h and 0 <= c < w
    ]
    dest_cells.sort()
    dest_set = set(dest_cells)

    passable = {FLOOR_COLOR, MAROON}
    reachable, dist = _flood(g, dest_set, passable, use8=use8)
    assert reachable, "destination must touch at least one passable cell under the chosen connectivity"

    maroons = [
        (int(dist[r, c]), int(r), int(c))
        for r, c in np.argwhere(g == MAROON)
        if (int(r), int(c)) in reachable
    ]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the nearest 8-reachable maroon pixels into the 3x3 block around the red center.

    Input structure:
    - Orange floor surrounds a centered green 3x3 target block with a unique red center.
    - A strict magenta rectangular frame sits on the 5x5 perimeter around that block.
    - The frame has one or two 1-pixel side mouths near its corners, and each mouth is plugged
      by a single magenta wall pixel immediately outside the frame.
    - Maroon source pixels lie on floor cells outside the frame.

    Rule:
    - Start a flood-fill from the target block using 8-connectivity, treating orange floor and
      maroon pixels as passable and magenta as blocking.
    - Collect only maroon pixels reached by that 8-connected search.
    - Sort reachable maroons by distance, then row, then column, and copy the closest ones into
      the target block in row-major order while erasing their original locations back to floor.

    Key insight:
    - A mouth that looks blocked in the straight-ahead 4-connected sense can still leak under
      8-connectivity because the search may step diagonally around the single outside plug.
    """

    return _solve_with_connectivity(grid, use8=True)


def _build_base_grid():
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    frame = _frame_cells()
    for r, c in frame:
        grid[r, c] = WALL_COLOR
    return grid, dest_cells, dest_set, frame


def _apply_portals(grid: np.ndarray):
    portal_count = random.randint(*PORTAL_COUNT_RANGE)
    portals = random.sample(_portal_catalog(), portal_count)
    reserved = set()
    for portal in portals:
        orow, ocol = portal["opening"]
        brow, bcol = portal["blocker"]
        grid[orow, ocol] = FLOOR_COLOR
        grid[brow, bcol] = WALL_COLOR
        reserved.add((orow, ocol))
        reserved.add((brow, bcol))
    return portals, reserved


def _choose_sources(exclusive_sorted, dist8):
    max_k = min(FILL_RANGE[1], len(exclusive_sorted))
    if max_k < FILL_RANGE[0]:
        return None
    k = random.randint(FILL_RANGE[0], max_k)
    prefix = min(len(exclusive_sorted), k + random.randint(*SOURCE_PREFIX_EXTRA))
    pool = exclusive_sorted[:prefix]
    source_cells = random.sample(pool, k)
    source_cells.sort(key=lambda rc: (int(dist8[rc[0], rc[1]]), rc[0], rc[1]))
    return source_cells


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    dest_set = set(_dest_cells())
    reach8, dist8 = _flood(grid, dest_set, {FLOOR_COLOR, MAROON}, use8=True)
    reach4, _dist4 = _flood(grid, dest_set, {FLOOR_COLOR, MAROON}, use8=False)

    maroon_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == MAROON)}
    only8_maroons = maroon_cells & (reach8 - reach4)
    out8 = _solve_with_connectivity(grid, use8=True)
    out4 = _solve_with_connectivity(grid, use8=False)
    return {
        "reachable8": len(reach8),
        "reachable4": len(reach4),
        "exclusive8": len(reach8 - reach4),
        "only8_maroons": len(only8_maroons),
        "output_diff": int(np.count_nonzero(out8 != out4)),
        "nearest_only8_dist": min((int(dist8[r, c]) for r, c in only8_maroons), default=-1),
    }


def _gen_attempt():
    grid, _dest_cells_list, dest_set, _frame = _build_base_grid()
    _portals, reserved = _apply_portals(grid)

    reach8, dist8 = _flood(grid, dest_set, {FLOOR_COLOR}, use8=True)
    reach4, _dist4 = _flood(grid, dest_set, {FLOOR_COLOR}, use8=False)

    exclusive8 = sorted(
        [cell for cell in reach8 - reach4 if cell not in reserved and cell not in dest_set],
        key=lambda rc: (int(dist8[rc[0], rc[1]]), rc[0], rc[1]),
    )

    if len(exclusive8) < MIN_EXCLUSIVE_CELLS:
        return None

    source_cells = _choose_sources(exclusive8, dist8)
    if source_cells is None:
        return None

    inp = grid.copy()
    for r, c in source_cells:
        inp[r, c] = MAROON

    out8 = main(inp)
    out4 = _solve_with_connectivity(inp, use8=False)
    if np.array_equal(out8, out4):
        return None
    metrics = _behavior_metrics(inp)
    if metrics["only8_maroons"] < FILL_RANGE[0] or metrics["output_diff"] <= 0:
        return None

    moved8 = int((out8[4:7, 4:7] == MAROON).sum())
    if moved8 != len(source_cells):
        return None
    if int((inp == MAROON).sum()) != len(source_cells):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

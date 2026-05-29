# Combo variant of dd6b8c4b (index 18)
# combo_spec: D1/1+D15/0
# Name: 最短路径长度约束+逐步收集
#
# Design analysis for the fused rule:
# - Concrete behavior:
#   A canonical corridor sits beside the 3x3 destination block. The first maroon is within
#   3 floor-steps, the second and third are initially too far because only a detour reaches
#   them, and the fourth remains at distance 4 even after earlier maroons disappear. This
#   makes the shortest-path cap and the step-by-step recomputation both visible.
# - Complexity calibration:
#   The generator uses one dominant one-cell-wide chamber plus a few distractors. That keeps
#   the visual story legible: a short lane, a longer detour, and one leftover over-limit token.
# - Broken original assumptions:
#   In the base task maroon cells were globally passable, so deleting one could not shorten any
#   route. Here only orange floor is traversable; a maroon may be used only as the final target
#   cell. Therefore distances must be recomputed after every single move.
# - Placement legality:
#   The chamber is defined in local coordinates around the fixed center and then rotated /
#   mirrored. All transformed cells stay inside the 11x11 canvas, never overlap the destination,
#   and the ring openings are explicitly carved.
# - Visual observability:
#   Walls isolate one narrow chamber, so a human can see the front maroon disappear first,
#   then notice that deeper maroons become newly available one by one. The untouched fourth
#   maroon demonstrates the path-length threshold.
# - Generator guarantees:
#   Each accepted sample must satisfy:
#   1. at least one maroon is collected by the fused solver,
#   2. at least two collected maroons were not initially eligible under the threshold,
#   3. an iterative solver without the threshold would collect strictly more maroons.

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9
DEST_SIDE = 3
PATH_LIMIT = 3
GEN_RETRIES = 300
DISTRACTOR_RANGE = (1, 4)


# Canonical west-facing chamber.
CANONICAL_BOX = {(dr, dc) for dr in range(-3, 3) for dc in range(-5, -1)}
CANONICAL_FLOOR = {
    (0, -2),
    (-1, -2),
    (-2, -2),
    (-2, -3),
    (-2, -4),
    (-1, -4),
    (-1, -5),
}
CANONICAL_CHAIN = [
    (0, -3),
    (0, -4),
    (0, -5),
    (1, -5),
]


def _transform(dr: int, dc: int, rot: int, mirror: bool) -> tuple[int, int]:
    if mirror:
        dr = -dr
    for _ in range(rot % 4):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _to_abs(coords, rot: int, mirror: bool) -> set[tuple[int, int]]:
    out = set()
    for dr, dc in coords:
        tr, tc = _transform(int(dr), int(dc), rot, mirror)
        out.add((CENTER_R + tr, CENTER_C + tc))
    return out


def _chain_abs(rot: int, mirror: bool) -> list[tuple[int, int]]:
    out = []
    for dr, dc in CANONICAL_CHAIN:
        tr, tc = _transform(int(dr), int(dc), rot, mirror)
        out.append((CENTER_R + tr, CENTER_C + tc))
    return out


def _dest_cells(center_r: int, center_c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(center_r - 1, center_r + 2):
        for c in range(center_c - 1, center_c + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _ring_cells(dest_set: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    ring = set()
    for r, c in dest_set:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set:
                ring.add((rr, cc))
    return ring


def _floor_starts(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    starts = []
    seen = set()
    for r, c in dest_set:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set and int(grid[rr, cc]) == FLOOR_COLOR:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    starts.append((rr, cc))
    return starts


def _floor_distances(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    h, w = grid.shape
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque()
    for r, c in _floor_starts(grid, dest_set):
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == FLOOR_COLOR and dist[rr, cc] == inf:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _eligible_tokens(grid: np.ndarray, dest_set: set[tuple[int, int]], path_limit: int) -> list[tuple[int, int, int]]:
    dist = _floor_distances(grid, dest_set)
    h, w = grid.shape
    inf = 10**9
    cands = []
    for r, c in np.argwhere(grid == MAROON):
        r = int(r)
        c = int(c)
        if (r, c) in dest_set:
            continue
        best = inf
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == FLOOR_COLOR:
                best = min(best, int(dist[rr, cc]) + 1)
        if best <= path_limit:
            cands.append((int(best), r, c))
    cands.sort()
    return cands


def _simulate(
    grid: np.ndarray,
    *,
    path_limit: int,
    iterative: bool,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    out = np.array(grid, dtype=int).copy()
    h, w = out.shape
    centers = np.argwhere(out == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells(cr, cc, h, w)
    dest_set = set(dest_cells)

    history: list[tuple[int, int, int]] = []
    fill_idx = 0
    while fill_idx < len(dest_cells):
        cands = _eligible_tokens(out, dest_set, path_limit)
        if not cands:
            break
        chosen = cands[:1] if iterative else cands[: len(dest_cells) - fill_idx]
        for dist, r, c in chosen:
            out[r, c] = FLOOR_COLOR
            dr, dc = dest_cells[fill_idx]
            out[dr, dc] = MAROON
            history.append((int(dist), int(r), int(c)))
            fill_idx += 1
            if fill_idx >= len(dest_cells):
                break
        if not iterative:
            break
    return out, history


def _make_base_grid() -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W):
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    return grid


def _apply_chamber(grid: np.ndarray, rot: int, mirror: bool) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    dest_set = set(_dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W))
    ring = _ring_cells(dest_set, GRID_H, GRID_W)
    floor_cells = _to_abs(CANONICAL_FLOOR, rot, mirror)
    box_cells = _to_abs(CANONICAL_BOX, rot, mirror)
    chain_cells = _chain_abs(rot, mirror)
    openings = floor_cells & ring

    for r, c in ring - openings:
        grid[r, c] = WALL_COLOR

    carved = floor_cells | set(chain_cells)
    for r, c in box_cells:
        grid[r, c] = FLOOR_COLOR if (r, c) in carved else WALL_COLOR

    return chain_cells, box_cells | ring


def _reachable_floor(grid: np.ndarray) -> set[tuple[int, int]]:
    dest_set = set(_dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W))
    seen = set()
    q = deque(_floor_starts(grid, dest_set))
    for cell in q:
        seen.add(cell)
    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and int(grid[rr, cc]) == FLOOR_COLOR and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return seen


def _build_candidate() -> np.ndarray:
    grid = _make_base_grid()
    rot = random.randrange(4)
    mirror = bool(random.randrange(2))
    chain_cells, reserved = _apply_chamber(grid, rot, mirror)

    inp = grid.copy()
    for r, c in chain_cells:
        inp[r, c] = MAROON

    reachable_floor = _reachable_floor(grid)
    distractor_pool = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(inp[r, c]) == FLOOR_COLOR and (r, c) not in reachable_floor and (r, c) not in reserved
    ]
    random.shuffle(distractor_pool)
    distractor_count = min(len(distractor_pool), random.randint(*DISTRACTOR_RANGE))
    for r, c in distractor_pool[:distractor_count]:
        inp[r, c] = MAROON

    return inp


def _is_good_variant(inp: np.ndarray) -> bool:
    dest_set = set(_dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W))
    initial = _eligible_tokens(inp, dest_set, PATH_LIMIT)
    fused_out, fused_hist = _simulate(inp, path_limit=PATH_LIMIT, iterative=True)
    no_limit_out, no_limit_hist = _simulate(inp, path_limit=GRID_H * GRID_W, iterative=True)

    initial_set = {(r, c) for _d, r, c in initial}
    later_only = [(r, c) for _d, r, c in fused_hist if (r, c) not in initial_set]

    if not fused_hist:
        return False
    if len(later_only) < 2:
        return False
    if len(no_limit_hist) <= len(fused_hist):
        return False
    if not np.array_equal(main(inp), fused_out):
        return False
    if np.array_equal(no_limit_out, fused_out):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull maroon pixels into the 3x3 block around the unique red center, one shortest-path step at a time.

    Input structure:
    - The board is an 11x11 orange floor with a fixed green 3x3 destination around one red center.
    - Magenta walls form narrow corridors near the destination.
    - Maroon pixels sit on floor cells outside the block; some are near the opening, others are deeper in the corridor.

    Rule:
    - Repeatedly recompute which maroon pixel is reachable from the destination through orange floor cells.
    - A maroon qualifies only if its current shortest path length is at most 3, where the last step may enter the maroon but other maroons are not traversable.
    - On each iteration, move only the single nearest qualifying maroon into the next destination cell in row-major order, erase its source to floor, then recompute from the updated grid.
    - Stop when the block is full or no maroon remains within the path-length limit.

    Key insight:
    - Removing one maroon can open a shorter route to deeper maroons, so the reachable set is dynamic; however the 3-step cap still prevents some connected maroons from ever being collected.
    """

    out, _history = _simulate(np.asarray(grid, dtype=int), path_limit=PATH_LIMIT, iterative=True)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _build_candidate()
        try:
            if _is_good_variant(inp):
                return inp
        except Exception:
            pass
    raise RuntimeError(f"Failed to generate valid combo sample after {GEN_RETRIES} attempts")

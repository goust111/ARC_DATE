# Combo variant of dd6b8c4b (index 44)
# combo_spec: D2/0+D9/3
# Name: 颜色编码框架+多色墙壁分层阻断

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant
# -------------------------------------
# 1. Concrete fused behavior:
#    - The original task had a single wall color and an undirected flood-fill.
#    - Here the 3x3 green/red target is still surrounded by a frame, but the frame
#      now mixes two wall colors:
#      * magenta(6): fully solid wall
#      * blue(1): directional gate wall
#    - Blue cells are only placed on rays extending straight out from the red
#      center. They can be traversed only along that ray and only from the side
#      nearer the center toward the side farther from the center. Sideways exits
#      from a blue cell are forbidden.
#    - Example: if a blue chain goes upward from the target, a maroon sitting in a
#      one-cell side pocket touching that chain is visually adjacent to the chain
#      but is still unreachable, because the path would need a sideways move out of
#      blue. A maroon beyond the tip of the chain is reachable, because the motion
#      stays radial until it exits to orange floor.
#
# 2. Complexity calibration:
#    - Only two wall colors are used, with one repeated blue behavior everywhere.
#      This is enough to be visible without introducing too many mappings.
#    - Blue is restricted to the center row/column rays so the direction rule is
#      readable from geometry alone; we avoid arbitrary per-cell directions.
#
# 3. Broken original assumptions and solver changes:
#    - Original assumption: every non-wall non-target color next to the destination
#      is an ordinary passable start cell. Not true anymore: a blue neighbor can be
#      entered only from the inner side of its ray.
#    - Original assumption: passability is symmetric. Not true anymore: blue edges
#      are directed and do not allow reverse or sideways traversal.
#    - Original assumption: the wall ring still contains ordinary floor cells from
#      which floor color can be inferred. Not true anymore: the ring can be all
#      magenta/blue, so color inference from the ring is removed and the known
#      palette is used directly.
#
# 4. Placement legality and visual cleanliness:
#    - Blue chains stay on the center row/column, so they never overlap each other.
#    - Each side pocket is a single floor cell with its other neighbors turned into
#      magenta, guaranteeing it is a clean "false branch" and not an accidental
#      alternate corridor.
#    - Pockets are placed only beyond the first blue gate cell so they never touch
#      the 3x3 target block.
#
# 5. Non-degeneracy target:
#    - The generator always places distractor maroons inside blue-side pockets.
#      Those cells would be reachable if blue were treated as plain floor, but are
#      unreachable under the fused directed-wall rule. This guarantees the combo
#      behavior is not equivalent to the original task.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5

FLOOR_COLOR = 7
SOLID_WALL = 6
GATE_WALL = 1
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9

SIDES = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}

NUM_PORTS_RANGE = (2, 4)
CHAIN_LEN_RANGE = (2, 3)  # includes the ring-adjacent blue cell
FILL_K_RANGE = (2, 9)
TOTAL_MAROON_RANGE = (4, 16)
SOURCE_MIN_DIST = 3
GEN_RETRIES = 400


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _frame_ring(dest_set: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    return sorted(ring)


def _gate_dir(cr: int, cc: int, r: int, c: int) -> tuple[int, int]:
    dr = r - cr
    dc = c - cc
    if abs(dr) >= abs(dc):
        return (1 if dr > 0 else -1, 0)
    return (0, 1 if dc > 0 else -1)


def _can_start_into(grid: np.ndarray, cr: int, cc: int, src: tuple[int, int], dst: tuple[int, int]) -> bool:
    r, c = dst
    val = int(grid[r, c])
    if val in (SOLID_WALL, DEST_COLOR, RED_CENTER):
        return False
    if val == GATE_WALL:
        dr, dc = _gate_dir(cr, cc, r, c)
        return src == (r - dr, c - dc)
    return val in (FLOOR_COLOR, MAROON)


def _can_step(grid: np.ndarray, cr: int, cc: int, src: tuple[int, int], dst: tuple[int, int]) -> bool:
    h, w = grid.shape
    r, c = dst
    if not (0 <= r < h and 0 <= c < w):
        return False

    sr, sc = src
    src_val = int(grid[sr, sc])
    dst_val = int(grid[r, c])

    if dst_val in (SOLID_WALL, DEST_COLOR, RED_CENTER):
        return False

    if src_val == GATE_WALL:
        dr, dc = _gate_dir(cr, cc, sr, sc)
        return dst == (sr + dr, sc + dc) and dst_val not in (SOLID_WALL, DEST_COLOR, RED_CENTER)

    if dst_val == GATE_WALL:
        dr, dc = _gate_dir(cr, cc, r, c)
        return src == (r - dr, c - dc)

    return dst_val in (FLOOR_COLOR, MAROON)


def _directed_distances(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    cr: int,
    cc: int,
) -> np.ndarray:
    h, w = grid.shape
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque()
    seen = set()

    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                if (nr, nc) not in seen and _can_start_into(grid, cr, cc, (r, c), (nr, nc)):
                    seen.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))

    assert q, "there must be at least one legal entry cell adjacent to the destination block"

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if _can_step(grid, cr, cc, (r, c), (nr, nc)) and dist[nr, nc] == inf:
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    return dist


def _apply_rule(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    dest_set = set(dest_cells)
    dist = _directed_distances(grid, dest_cells, dest_set, cr, cc)

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [t for t in maroons if t[0] < 10**9]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = [(r, c) for _, r, c in maroons[:k]]

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out, chosen, dist


def _naive_output(grid: np.ndarray) -> np.ndarray:
    """Reference for degeneration checks: treat blue as ordinary floor."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells(cr, cc, h, w)
    dest_set = set(dest_cells)

    passable = {FLOOR_COLOR, GATE_WALL, MAROON}
    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque()
    seen = set()

    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [t for t in maroons if t[0] < inf]
    maroons.sort()
    k = min(len(dest_cells), len(maroons))
    chosen = [(r, c) for _, r, c in maroons[:k]]

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the nearest legally reachable maroon pixels into the 3x3 block around the unique red center.

    Input structure:
    - Orange floor carries scattered maroon tokens, while the target is the usual 3x3 green block with one red center.
    - Around that block sits a mixed-color wall frame: magenta cells are solid barriers, and blue cells are gate walls placed on straight rays away from the center.

    Rule:
    - Flood-fill begins from cells adjacent to the target, but traversal is now directional: magenta blocks everything, and blue can be used only along its ray from the center side toward the outside side.
    - Sideways turns out of a blue cell are illegal, so maroons in side pockets touching a blue ray are not reachable; reachable maroons are ranked by directed distance and then fill the target in row-major order.

    Key insight:
    - Blue is not an ordinary opening. It is a one-way straight-through gate, so reachability depends on wall color and travel direction together, not just geometric closeness.
    """

    out, _chosen, _dist = _apply_rule(grid)
    return out


def _build_base_grid() -> tuple[np.ndarray, list[tuple[int, int]], set[tuple[int, int]]]:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    return grid, dest_cells, dest_set


def _ring_cell_for_side(side: str) -> tuple[int, int]:
    dr, dc = SIDES[side]
    return CENTER_R + 2 * dr, CENTER_C + 2 * dc


def _sideways_dirs(dr: int, dc: int) -> list[tuple[int, int]]:
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _try_make_pocket(
    grid: np.ndarray,
    chain_cell: tuple[int, int],
    lateral: tuple[int, int],
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    r, c = chain_cell
    lr, lc = lateral
    pr, pc = r + lr, c + lc
    if not (0 <= pr < GRID_H and 0 <= pc < GRID_W):
        return None
    if (pr, pc) in reserved or int(grid[pr, pc]) != FLOOR_COLOR:
        return None

    walls = []
    for wr, wc in neighbors4(pr, pc):
        if (wr, wc) == chain_cell:
            continue
        if not (0 <= wr < GRID_H and 0 <= wc < GRID_W):
            return None
        val = int(grid[wr, wc])
        if val == GATE_WALL or val in (DEST_COLOR, RED_CENTER):
            return None
        walls.append((wr, wc))

    for wr, wc in walls:
        grid[wr, wc] = SOLID_WALL
        reserved.add((wr, wc))
    reserved.add((pr, pc))
    return pr, pc


def _add_ports_and_pockets(
    grid: np.ndarray,
    ring_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
) -> tuple[list[str], dict[str, list[tuple[int, int]]], list[tuple[int, int]]]:
    reserved = set(ring_cells) | set(dest_set)
    for r, c in ring_cells:
        grid[r, c] = SOLID_WALL

    num_ports = random.randint(*NUM_PORTS_RANGE)
    open_sides = random.sample(list(SIDES), num_ports)
    chains: dict[str, list[tuple[int, int]]] = {}
    pockets: list[tuple[int, int]] = []

    for side in open_sides:
        dr, dc = SIDES[side]
        rr, rc = _ring_cell_for_side(side)
        max_len = 0
        nr, nc = rr, rc
        while 0 <= nr < GRID_H and 0 <= nc < GRID_W:
            max_len += 1
            nr += dr
            nc += dc
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                max_len -= 1
                break
        max_len = min(max_len, CHAIN_LEN_RANGE[1])
        chain_len = random.randint(CHAIN_LEN_RANGE[0], max_len)

        chain = []
        r, c = rr, rc
        for _ in range(chain_len):
            chain.append((r, c))
            grid[r, c] = GATE_WALL
            reserved.add((r, c))
            r += dr
            c += dc
        chains[side] = chain

        chain_cell = chain[1]
        laterals = _sideways_dirs(dr, dc)
        random.shuffle(laterals)
        pocket = None
        for lateral in laterals:
            pocket = _try_make_pocket(grid, chain_cell, lateral, reserved)
            if pocket is not None:
                break
        assert pocket is not None, "failed to build a blue-side pocket"
        pockets.append(pocket)

    return open_sides, chains, pockets


def _reachable_floor_by_rule(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], np.ndarray]:
    dist = _directed_distances(grid, dest_cells, dest_set, CENTER_R, CENTER_C)
    reachable = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set and int(dist[r, c]) < 10**9
    }
    return reachable, dist


def _reachable_floor_naive(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    passable = {FLOOR_COLOR, GATE_WALL, MAROON}
    inf = 10**9
    dist = np.full((GRID_H, GRID_W), inf, dtype=int)
    q = deque()
    seen = set()

    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    dist[nr, nc] = 0
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    return {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set and int(dist[r, c]) < inf
    }


def _gen_attempt() -> np.ndarray | None:
    grid, dest_cells, dest_set = _build_base_grid()
    ring_cells = _frame_ring(dest_set, GRID_H, GRID_W)
    _open_sides, _chains, pockets = _add_ports_and_pockets(grid, ring_cells, dest_set)

    reachable, dist = _reachable_floor_by_rule(grid, dest_cells, dest_set)
    naive_reachable = _reachable_floor_naive(grid, dest_cells, dest_set)
    sensitive_cells = [cell for cell in pockets if cell in naive_reachable and cell not in reachable]
    if not sensitive_cells:
        return None

    eligible_sources = [cell for cell in reachable if int(dist[cell]) >= SOURCE_MIN_DIST]
    if len(eligible_sources) < FILL_K_RANGE[0]:
        return None

    max_k = min(FILL_K_RANGE[1], len(eligible_sources))
    k = random.randint(FILL_K_RANGE[0], max_k)

    source_cells = random.sample(eligible_sources, k)
    distractor_cells = list(sensitive_cells)

    remaining_unreachable = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set and (r, c) not in reachable and (r, c) not in distractor_cells
    ]
    extra_budget = TOTAL_MAROON_RANGE[1] - (k + len(distractor_cells))
    if remaining_unreachable and extra_budget > 0:
        extra = random.randint(0, min(len(remaining_unreachable), extra_budget, 3))
        distractor_cells.extend(random.sample(remaining_unreachable, extra))

    if k + len(distractor_cells) < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - (k + len(distractor_cells))
        remaining_sources = [cell for cell in eligible_sources if cell not in source_cells]
        if len(remaining_sources) < needed:
            return None
        source_cells.extend(random.sample(remaining_sources, needed))
        k += needed

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    fused_out = main(inp)
    naive_out = _naive_output(inp)
    if np.array_equal(fused_out, naive_out):
        return None

    sensitive_tokens = sum(1 for cell in distractor_cells if cell in naive_reachable and cell not in reachable)
    if sensitive_tokens == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate valid fused input after {GEN_RETRIES} attempts")

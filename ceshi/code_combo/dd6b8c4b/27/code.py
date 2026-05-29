# Combo variant of dd6b8c4b (index 27)
# combo_spec: D13/0+D17/1
# Name: 红色像素编码收集上限+恰好填满

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the fused variant:
# - The original task used the red cell only as the geometric center of the 3x3 destination block.
#   Here the red cell moves inside that 3x3 block and becomes a visible counter: its row-major
#   position encodes the collection limit k.
# - We interpret the destination capacity as a row-major prefix of the 3x3 block. If the red cell
#   is the k-th cell in row-major order, then the fillable container is exactly the first k cells.
#   This makes D17/1 human-visible: the red pixel marks the last slot that may be filled.
# - D13/0 is enforced by generation: the number of reachable maroon sources is exactly k, so the
#   encoded limit and the reachable supply match perfectly. The solver still computes reachability;
#   it simply never sees underflow/overflow on valid inputs.
# - This removes a key hidden assumption in the original solver: the red pixel is no longer the
#   center of the block. The solver must first locate the unique 3x3 red/green block, then decode
#   k from the red cell's position inside that bbox.
# - Visual observability: across samples, the red marker appears at different positions in the
#   block, and the output always fills the row-major prefix ending at that marker. Walls and
#   distractor maroons remain to preserve the original flood-fill reachability cue.


def _find_dest_block(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    rg = np.argwhere((grid == 2) | (grid == 3))
    assert len(rg) == 9, "grid must contain exactly one 3x3 red/green destination block"
    r0, c0 = rg.min(axis=0)
    r1, c1 = rg.max(axis=0)
    bbox = (int(r0), int(c0), int(r1), int(c1))
    assert bbox[2] - bbox[0] == 2 and bbox[3] - bbox[1] == 2, "destination bbox must be 3x3"

    dest_cells = [(r, c) for r in range(bbox[0], bbox[2] + 1) for c in range(bbox[1], bbox[3] + 1)]
    assert all(int(grid[r, c]) in (2, 3) for r, c in dest_cells), "destination bbox must contain only red/green cells"

    reds = np.argwhere(grid == 2)
    assert len(reds) == 1, "grid must have exactly one red marker"
    red_pos = (int(reds[0, 0]), int(reds[0, 1]))
    assert red_pos in dest_cells, "red marker must lie inside the destination block"
    return bbox, dest_cells, set(dest_cells), red_pos


def _infer_floor(grid: np.ndarray) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (2, 3, 9)]
    assert colors, "grid must contain at least one non-destination background color"
    return max(colors, key=lambda c: int((grid == c).sum()))


def _reachable_maroons(grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]], floor: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    passable = {floor, 9}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set and int(grid[rr, cc]) in passable:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    starts.append((rr, cc))
    assert starts, "destination block must have at least one adjacent passable entry cell"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == inf:
                dist[rr, cc] = int(dist[r, c]) + 1
                q.append((rr, cc))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9) if int(dist[r, c]) < inf]
    maroons.sort()
    return maroons


def _encoded_capacity(dest_cells: list[tuple[int, int]], red_pos: tuple[int, int]) -> int:
    return int(dest_cells.index(red_pos) + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every reachable maroon source into the row-major prefix of a marked 3x3 block.

    Input structure:
    - bg = orange floor, optional magenta walls, scattered maroon (9) pixels.
    - There is one 3x3 green destination block containing a single red marker (2); the red marker
      can be anywhere inside that 3x3, not necessarily at the geometric center.

    Rule:
    - Flood-fill from cells adjacent to the destination through floor/maroon cells to find reachable
      maroon sources.
    - Decode the collection limit from the red marker's row-major position inside the 3x3 block.
      If the red marker is the k-th cell, fill exactly the first k destination cells in row-major order.
    - Erase the collected source maroons back to floor. On valid instances, the reachable maroon count
      is guaranteed to equal the encoded limit, so the destination prefix is filled exactly to the red cell.

    Key insight: The red pixel no longer just locates the block; it encodes how many reachable maroons
    should be collected. Walls still matter because only flood-fill-reachable maroons participate.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    _bbox, dest_cells, dest_set, red_pos = _find_dest_block(grid)
    floor = _infer_floor(grid)
    reachable = _reachable_maroons(grid, dest_cells, dest_set, floor)
    limit = _encoded_capacity(dest_cells, red_pos)

    # Valid generated inputs satisfy len(reachable) == limit; keep a tolerant fallback for safety.
    chosen = reachable[:limit]

    for _d, r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[: len(chosen)]:
        out[r, c] = 9

    return out


# --------------- generator (c) ---------------

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

FILL_K_RANGE = (2, 9)
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 2
GEN_RETRIES = 400


def _dest_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(CENTER_R - 1, CENTER_R + 2) for c in range(CENTER_C - 1, CENTER_C + 2)]


def _bfs_reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited = set()
    queue = deque()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and int(grid[nr, nc]) == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and int(grid[nr, nc]) == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


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
            dirs = _outward_dirs(r, c)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr += dr
                    nc += dc
        outer = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer:
            for rc in random.sample(outer, random.randint(1, max(1, len(outer) // 4))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, GRID_H - abs(r - CENTER_R) - 2)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells()
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    k = random.randint(*FILL_K_RANGE)
    red_pos = dest_cells[k - 1]
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[red_pos] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable_floor(grid, dest_set)
    if len(reachable_floor) < max(MIN_REACHABLE, k):
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0
    total = k + num_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        need = TOTAL_MAROON_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + need:
            num_distractors += need
        else:
            return None
    if k + num_distractors > TOTAL_MAROON_RANGE[1]:
        num_distractors = max(0, TOTAL_MAROON_RANGE[1] - k)

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = grid.copy()
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON

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
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

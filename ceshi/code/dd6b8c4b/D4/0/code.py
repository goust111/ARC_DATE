# Variant of dd6b8c4b
# Dimension: D4 — 填充目标区域的方式 (path)
# Change: row_major_fill——按行优先顺序从绿色方块内部左上角开始，逐个填入收集到的栗色像素... → 从右下角开始反向行优先填充
#
# Design analysis for this variant
# --------------------------------
# a) Concrete behavior and trace
# - The global scene is unchanged: an 11x11 orange floor contains a fixed 3x3
#   green destination block centered on one red pixel, plus optional magenta
#   walls and scattered maroon tokens.
# - The original task sorted reachable maroon tokens by BFS distance and then
#   copied the closest k tokens into destination cells in row-major order:
#   [(4,4),(4,5),(4,6),(5,4),(5,5),(5,6),(6,4),(6,5),(6,6)].
# - This variant keeps candidate selection and ranking identical, but mirrors
#   the target path. The fill order becomes reverse row-major:
#   [(6,6),(6,5),(6,4),(5,6),(5,5),(5,4),(4,6),(4,5),(4,4)].
# - Small trace with k=4:
#   original output fills {(4,4),(4,5),(4,6),(5,4)};
#   variant output fills {(6,6),(6,5),(6,4),(5,6)}.
#   The same four source tokens move, but the occupied destination corner flips.
# b) Complexity calibration
# - Only D4 changes. Reachability, walls, ranking and erasure remain the same,
#   so the task is still "collect nearest reachable maroons through corridors".
# - To keep the D4 change visible, the generator forbids k=9. A full 3x3 fill
#   would make the mirrored path indistinguishable from the original task.
# - The generator still varies k across 2..8 and retains optional distractors
#   and multiple wall styles, so the puzzle does not collapse to a trivial
#   corner-copy rule.
# c) Old implicit assumptions that must be reviewed
# - `dest_cells.sort()` in the original solver was harmless because it defined
#   both the geometry set and the fill path. In this variant only the geometry
#   should remain row-major; the write path must be decoupled and reversed.
# - `for r, c in dest_cells[:k]: out[r, c] = 9` encoded the old D4 directly.
#   Leaving that line unchanged would silently revert the task.
# - All code before candidate placement depends only on destination membership,
#   not on fill direction, so it stays valid.
# d) Placement legality
# - The destination block is still fixed at rows/cols 4..6, so reverse filling
#   never goes out of bounds.
# - Source maroons remain on floor cells outside the destination and walls.
# - Distractors remain on unreachable floor cells, preventing overlap and
#   guaranteeing the solver can erase only moved sources.
# e) Visual observability
# - Human solvers can detect the change from 2-4 samples because partial fills
#   now accumulate from the lower-right corner instead of the upper-left.
# - This uses only visible spatial structure; no hidden color-id ordering or
#   invisible metadata affects the transformation.
# f) Visual cleanliness
# - No new markers are added. The task still has one central block, clear wall
#   barriers, and scattered tokens, so the mirrored fill remains easy to read.
# g) Color-change inference
# - Colors are unchanged. Maroon tokens move, floor is restored, and untouched
#   green/red destination cells remain as before.
# h) Variant metric used after implementation
# - For generated samples, compare the new solver output against a hypothetical
#   old row-major fill on the same ranked tokens. Every accepted sample should
#   differ because k is restricted to at most 8.
# i) New solver assumptions to guarantee in the generator
# - The solver now assumes there is at least one passable neighbor next to the
#   destination and at least k reachable maroon placements can be selected.
#   The generator preserves these by validating every sample with `main()`.

from pathlib import Path
import sys
import random
from collections import deque

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import neighbors4


def _dest_cells_for_center(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _reverse_row_major_fill_order(dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(dest_cells, reverse=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the 3×3 block around the unique red center.

    Input structure:
    - The canvas is mostly orange floor, sometimes split by magenta wall barriers, with scattered maroon tokens.
    - A single red pixel marks the center of a fixed 3×3 destination block whose other cells are green.

    Rule:
    - Maroon tokens reachable from the destination through floor/maroon cells are ranked by BFS distance, with row/col tie-breaks.
    - The closest reachable tokens are removed from their source positions and copied into the destination block.
    - Unlike the original task, destination filling starts at the lower-right cell and proceeds in reverse row-major order.

    Key insight: The hard part is still the flood-fill reachability through wall-made corridors, but once the valid maroons are chosen, the destination path is a mirrored fill that grows from bottom-right toward top-left.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells_for_center(cr, cc, h, w)
    fill_order = _reverse_row_major_fill_order(dest_cells)
    dest_set = set(dest_cells)

    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, "ring around destination must contain passable (non-maroon, non-red, non-wall) cells to infer floor color"
    vals, counts = np.unique(cand_ring, return_counts=True)
    ring_blocker = int(vals[counts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, ring_blocker)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    if floor is None:
        floor = ring_blocker

    passable = {floor, 9}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                dist[rr, cc2] = d + 1
                q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [triple for triple in maroons if triple[0] < inf]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in fill_order[:k]:
        out[r, c] = 9

    return out


# --------------- generator (c) ---------------

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

# Keep fills partial so the mirrored destination path is always visible.
FILL_K_RANGE = (2, 8)
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 2
GEN_RETRIES = 300


def _dest_cells() -> list[tuple[int, int]]:
    return _dest_cells_for_center(CENTER_R, CENTER_C, GRID_H, GRID_W)


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """BFS from floor cells adjacent to dest through floor cells."""
    visited = set()
    queue = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in dest_set:
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
    length = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(length):
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


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells()
    fill_order = _reverse_row_major_fill_order(dest_cells)
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    max_k = min(FILL_K_RANGE[1], len(reachable_floor))
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0

    total = k + num_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        elif max_k >= k + needed:
            k += needed
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
    for r, c in fill_order[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

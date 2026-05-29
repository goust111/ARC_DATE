# Combo variant of dd6b8c4b (index 40)
# combo_spec: D6/3+D8/1
# Name: 空心框架容器+只替换绿色

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4

# Design analysis for combo D6/3 + D8/1
# a) Concrete trace:
#    - Original task ranked reachable maroons the same way as before, but then wrote them into all 9
#      cells of the 3x3 destination in row-major order:
#      [(4,4),(4,5),(4,6),(5,4),(5,5),(5,6),(6,4),(6,5),(6,6)].
#    - This combo keeps the same center marker and the same BFS selection, but the destination is now
#      a hollow frame and only green cells may be replaced. The writable order becomes:
#      [(4,4),(4,5),(4,6),(5,4),(5,6),(6,4),(6,5),(6,6)].
#    - Example with k=6 reachable maroons:
#         original -> the 5th token overwrites the red center at (5,5)
#         combo    -> the red center stays red, and the 5th token goes to (5,6)
#      This is the visible interaction between "hollow frame" and "replace green only".
# b) Complexity calibration:
#    - The frame stays a fixed 3x3 object around one unique red pixel, so the target is easy to find.
#    - The meaningful change is not extra object variety but a changed writable subset plus changed
#      capacity. Generator samples k in [5, 8] so the skip-center behavior is repeatedly visible.
# c) Implicit-assumption audit against the original solver:
#    - "dest_cells are all fillable" is false now; only cells that are green in the input are fill slots.
#    - "k can reach 9" is false now; max writable capacity is 8.
#    - "writing may overwrite red" is false now; non-green cells inside the 3x3 block are preserved.
#    - BFS connectivity assumptions still hold because walls remain outside the same central 3x3 block
#      and flood-fill still starts from passable neighbors around that block.
# d) Placement legality:
#    - Generator never places walls or maroons inside the 3x3 destination block.
#    - Reachable sources are sampled from floor cells in the BFS component touching the destination.
#    - Distractors are sampled only from unreachable floor cells, so source and distractor sets are disjoint.
# e) Visual observability:
#    - With k >= 5, several examples necessarily show the red center surviving while later border cells
#      still fill, which differentiates this combo from the original row-major overwrite behavior.
# f) Visual cleanliness:
#    - The palette stays the original five colors and the only dense region is the single center frame.
#      Walls remain outside the frame, so the border-vs-center distinction stays legible.
# g) Inferability:
#    - The ordering cue is visible and human-readable: fill the green frame cells in row-major order,
#      skipping the red center because it is not green.


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

FILL_K_RANGE = (5, 8)
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (5, 16)
MIN_REACHABLE = 5
GEN_RETRIES = 300


def _destination_cells(center_r: int, center_c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(center_r - 1, center_r + 2):
        for c in range(center_c - 1, center_c + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _green_fill_cells(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(r, c) for r, c in dest_cells if int(grid[r, c]) == DEST_COLOR]


def _reachable_maroons(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    dest_set = set(dest_cells)
    passable = {FLOOR_COLOR, MAROON}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    starts.append((nr, nc))
    assert starts, "there must be at least one passable cell adjacent to the destination"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        base = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = base + 1
                q.append((nr, nc))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [triple for triple in maroons if triple[0] < inf]
    maroons.sort()
    return maroons


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the nearest reachable maroon pixels into the green frame around the unique red center.

    Input structure:
    - The board uses orange floor, optional magenta walls, scattered maroon tokens, and one red center.
    - The 3x3 area around that red center is a hollow green frame: eight green border cells plus the red middle.

    Rule:
    - Flood-fill from passable cells touching the 3x3 destination block, traveling through orange or maroon cells.
    - Rank reachable maroon tokens by BFS distance, then by row and column.
    - Erase the chosen sources back to orange and write them into the destination's green cells in row-major order.
      Non-green cells inside the 3x3 block are not overwritten.

    Key insight: The reachability test is unchanged, but the write mask is no longer the full 3x3 square.
    The red center is a permanent anchor, so only the green border participates in the fill.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _destination_cells(cr, cc, h, w)
    fill_cells = _green_fill_cells(grid, dest_cells)
    assert fill_cells, "destination block must contain green border cells"

    reachable = _reachable_maroons(grid, dest_cells)
    k = min(len(fill_cells), len(reachable))
    chosen = {(r, c) for _, r, c in reachable[:k]}

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR

    for r, c in fill_cells[:k]:
        out[r, c] = MAROON

    return out


def _bfs_reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
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
    dest_cells = _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable_floor(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    fill_cells = _green_fill_cells(grid, dest_cells)
    max_k = min(FILL_K_RANGE[1], len(fill_cells), len(reachable_floor))
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
    for r, c in fill_cells[:k]:
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

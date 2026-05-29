# Variant of dd6b8c4b
# Dimension: D6 — 目标容器（绿色方块） (dst)
# Change: 绿色3x3方块（内含红色中心像素）——作为收集容器，内部空间被收集到的栗色像素填充... → 可变大小的目标容器

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# a) Concrete behavior:
#    - The old task had a fixed 3x3 destination centered on the unique red cell.
#    - This variant replaces that with a solid square made of {red, green} cells.
#      The square side length is variable per puzzle, chosen from {4, 5}.
#      To keep even sizes identifiable, the unique red pixel is moved to the
#      square's top-left corner and the remaining square cells are green.
#    - Input still contains orange floor, optional magenta walls, and scattered
#      maroon tokens. Output still moves the closest reachable maroon tokens
#      into the destination in row-major order, but now the fill cap is the
#      square area: 16 for 4x4 and 25 for 5x5.
#    - Small trace:
#      old: 3x3 block -> cap 9 -> 12 reachable maroons still fill only 9 cells.
#      new: 4x4 block -> cap 16 -> the same 12 reachable maroons fill 12 cells.
#      new: 4x4 block + 19 reachable maroons -> only the closest 16 move, 3 stay.
#
# b) Complexity calibration:
#    - Sizes are restricted to two repeated, learnable cases: 4x4 and 5x5.
#    - Overflow cases are intentionally sampled, so humans can see that extra
#      reachable maroons remain outside once the square is full.
#    - Walls/corridors remain from the original task so reachability still
#      matters; the change is not reduced to raw counting.
#
# c) Old implicit assumptions that break:
#    - "red pixel is the center" no longer holds for even sizes.
#    - "destination is exactly cr-1:cr+1, cc-1:cc+1" no longer holds.
#    - "capacity is always 9" no longer holds.
#    - The solver therefore must detect the full {2,3} square first, then use
#      its recovered side length to define destination cells and capacity.
#
# d) Placement legality:
#    - The destination square is kept at a fixed top-left anchor so both 4x4 and
#      5x5 fit inside the 11x11 canvas with room for wall rings.
#    - Walls are only written outside the destination square.
#    - Maroon source and distractor cells are sampled only from floor cells.
#
# e) Visual observability:
#    - The variable square outline is visible immediately from the green area.
#    - The top-left red anchor indicates where row-major filling begins.
#    - Overflow samples make the size-dependent cap observable: a 4x4 square can
#      fill completely while nearby reachable maroons remain on the floor.
#
# f) Visual cleanliness:
#    - Only one structured object occupies the center region: the solid square.
#    - Walls remain outside the square; maroons remain on floor cells, so layers
#      are visually separated.
#
# g) Color inference:
#    - No hidden color-id ordering is used. Ranking depends only on BFS distance
#      and row-major spatial order, both visually grounded.
#
# h) Generator non-degeneration target:
#    - The generator must produce both size branches and a healthy share of
#      overflow cases (reachable maroons > capacity) so the D6 change is active.
#
# i) New solver assumptions and how generation enforces them:
#    - Assumption: all red/green cells belong to one solid square.
#      Generator: writes exactly one square of {2,3}.
#    - Assumption: the unique red cell is the square's top-left anchor.
#      Generator: places red exactly at that square corner.
#    - Assumption: at least one passable neighbor touches the square.
#      Generator: wall ring always leaves 1-4 gaps and rejects dead layouts.
# ---------------------------------------------------------------------------


GRID_H, GRID_W = 11, 11
DEST_TOP, DEST_LEFT = 3, 3
DEST_SIZE_CHOICES = (4, 5)

FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_ANCHOR = 3, 2
MAROON = 9

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ("arms", "frame", "mixed")

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_MARGIN_RANGE = (2, 3)
FRAME_CONNECT_PROB = 0.45
FRAME_GAP_REMOVE_PROB = 0.35
MIXED_EXTEND_PROB = 0.6

DISTRACTOR_MAX = 10
MIN_SOURCE_COUNT = 4
EXTRA_SOURCE_MARGIN = 8
OVERFLOW_PROB = 0.55
MIN_REACHABLE_FLOOR = {4: 10, 5: 14}
GEN_RETRIES = 400


def _destination_cells(size: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(DEST_TOP, DEST_TOP + size)
        for c in range(DEST_LEFT, DEST_LEFT + size)
    ]


def _find_destination(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(np.isin(grid, [RED_ANCHOR, DEST_COLOR]))
    assert len(coords) > 0, "grid must contain a red/green destination square"

    red_coords = np.argwhere(grid == RED_ANCHOR)
    assert len(red_coords) == 1, "grid must contain exactly one red anchor"
    rr, rc = map(int, red_coords[0])

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height == width, "destination must be a square"
    assert grid[r0, c0] == RED_ANCHOR, "red anchor must mark the top-left of the square"

    dest_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for r, c in dest_cells:
        assert int(grid[r, c]) in (RED_ANCHOR, DEST_COLOR), "destination square must be solid"
    assert (rr, rc) == (r0, c0), "red anchor must be the destination's top-left cell"
    return dest_cells


def _floor_color(grid: np.ndarray) -> int:
    colors = [int(x) for x in np.unique(grid) if int(x) not in (RED_ANCHOR, DEST_COLOR, MAROON)]
    assert colors, "grid must contain at least one non-destination background color"
    return max(colors, key=lambda color: int((grid == color).sum()))


def _reachable_distances(grid: np.ndarray, dest_cells: list[tuple[int, int]], passable: set[int]):
    h, w = grid.shape
    dest_set = set(dest_cells)

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    starts.append((nr, nc))
    assert starts, "there must be at least one passable cell touching the destination square"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into a variable-size square container marked by a red corner.

    Input structure:
    - Orange floor contains optional magenta walls and scattered maroon (9) pixels.
    - A solid red/green square is the destination container: its unique red (2) cell is the top-left anchor, and the square side length may vary.

    Rule:
    - Flood-fill outward from passable cells touching the square, traversing only floor and maroon cells.
    - Sort reachable maroon pixels by BFS distance, then by row and column, and move as many as fit into the square in row-major order.
    - Moved maroons are erased back to floor; any extra reachable or unreachable maroons remain where they are.

    Key insight:
    - The red pixel is no longer the center of a fixed 3x3 target. The solver must first recover the full red/green square, because its area determines how many maroons can be collected.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    dest_cells = _find_destination(grid)
    floor = _floor_color(grid)
    passable = {floor, MAROON}

    dist = _reachable_distances(grid, dest_cells, passable)
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [item for item in maroons if item[0] < 10**9]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def _outward_dirs(r: int, c: int, size: int) -> list[tuple[int, int]]:
    center_r = DEST_TOP + (size - 1) / 2.0
    center_c = DEST_LEFT + (size - 1) / 2.0
    dirs = []
    if r < center_r:
        dirs.append((-1, 0))
    if r > center_r:
        dirs.append((1, 0))
    if c < center_c:
        dirs.append((0, -1))
    if c > center_c:
        dirs.append((0, 1))
    return dirs


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], size: int, r: int, c: int, max_len: int):
    dirs = _outward_dirs(r, c, size)
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


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]], size: int):
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
                _extend_arm(wall_cells, dest_set, size, r, c, ARM_LENGTH_RANGE[1])
    elif style == "frame":
        margin = random.randint(*FRAME_MARGIN_RANGE)
        r0 = DEST_TOP - margin
        c0 = DEST_LEFT - margin
        r1 = DEST_TOP + size - 1 + margin
        c1 = DEST_LEFT + size - 1 + margin
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in dest_set:
                    if r in (r0, r1) or c in (c0, c1):
                        wall_cells.add((r, c))
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c, size)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr += dr
                    nc += dc
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            n_remove = random.randint(1, max(1, len(outer_frame) // 4))
            for cell in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(cell)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                _extend_arm(wall_cells, dest_set, size, r, c, ARM_LENGTH_RANGE[1])

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]):
    visited: set[tuple[int, int]] = set()
    dist: dict[tuple[int, int], int] = {}
    q = deque()

    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    dist[(nr, nc)] = 0
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited and (nr, nc) not in dest_set:
                if grid[nr, nc] == FLOOR_COLOR:
                    visited.add((nr, nc))
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))
    return visited, dist


def _choose_source_count(size: int, reachable_count: int) -> int | None:
    capacity = size * size
    max_sources = min(reachable_count, capacity + EXTRA_SOURCE_MARGIN)
    if max_sources < MIN_SOURCE_COUNT:
        return None

    overflow_possible = max_sources > capacity
    if overflow_possible and random.random() < OVERFLOW_PROB:
        low = capacity + 1
        high = max_sources
    else:
        high = min(capacity, max_sources)
        low = MIN_SOURCE_COUNT
    if low > high:
        return None
    return random.randint(low, high)


def _gen_attempt():
    size = random.choice(DEST_SIZE_CHOICES)
    dest_cells = _destination_cells(size)
    dest_set = set(dest_cells)
    capacity = len(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[DEST_TOP, DEST_LEFT] = RED_ANCHOR

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set, size)

    reachable_floor, dist = _reachable_floor(base, dest_set)
    if len(reachable_floor) < MIN_REACHABLE_FLOOR[size]:
        return None

    source_count = _choose_source_count(size, len(reachable_floor))
    if source_count is None:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    source_cells = random.sample(list(reachable_floor), source_count)
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor))
    distractor_count = random.randint(0, max_distractors) if max_distractors else 0
    distractor_cells = random.sample(list(unreachable_floor), distractor_count)

    ranked_sources = sorted((dist[(r, c)], r, c) for r, c in source_cells)
    moved_sources = {(r, c) for _, r, c in ranked_sources[:capacity]}
    leftover_sources = set(source_cells) - moved_sources

    inp = base.copy()
    for r, c in source_cells:
        inp[r, c] = MAROON
    for r, c in distractor_cells:
        inp[r, c] = MAROON

    out = base.copy()
    for r, c in leftover_sources:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    for r, c in dest_cells[: min(capacity, source_count)]:
        out[r, c] = MAROON

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

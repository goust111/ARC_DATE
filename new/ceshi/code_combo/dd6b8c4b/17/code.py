# Combo variant of dd6b8c4b (index 17)
# combo_spec: D6/0+D13/1
# Name: 溢出处理+可变大小容器

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the fused D6/0 + D13/1 variant
#
# a) Concrete behavior and trace
#    - Original task:
#      a fixed 3x3 container around a red center collects the closest reachable
#      maroon cells, and anything beyond the 3x3 capacity simply stays outside.
#    - Fused task:
#      the destination is now a solid red/green square of variable size 4x4 or
#      5x5, with the unique red cell at the top-left corner so even sizes remain
#      identifiable. The square area is the first capacity.
#      If reachable maroons exceed that capacity, the spill does not stay put:
#      the extra reachable maroons continue into the one-cell-thick outer ring
#      around the square, also in row-major order.
#    - Small trace:
#      size 4 -> inner capacity 16, outer ring capacity 20.
#      If 13 reachable maroons exist, output fills the first 13 square cells.
#      If 19 reachable maroons exist, output fills all 16 square cells and then
#      the first 3 cells of the surrounding ring.
#      This differs from both the original task and plain D6/0, where those 3
#      extra maroons would remain at their source locations.
#
# b) Complexity calibration
#    - Container size is restricted to two repeated cases {4, 5}; this keeps the
#      new capacity inference learnable.
#    - Overflow is restricted to one extra layer only: the immediate outer ring.
#      That is visually obvious but still simple enough to infer from 2-4 pairs.
#    - Walls/corridors are preserved, so the solver still needs reachability and
#      nearest-first ordering; the task does not collapse into counting alone.
#
# c) Old implicit assumptions that break
#    - "the red cell is the center of a 3x3 block" is invalid.
#    - "destination cells are exactly cr-1:cr+1, cc-1:cc+1" is invalid.
#    - "capacity is always 9" is invalid.
#    - "overflow maroons are untouched" is invalid.
#    - Any solver line that hard-codes a 3x3 neighborhood, assumes the red cell
#      is central, or stops writing after the inner block must be replaced.
#
# d) Placement legality
#    - The square is anchored at a fixed top-left position so both 4x4 and 5x5
#      fit on the 11x11 board.
#    - The overflow ring is reserved as empty floor in the input.
#    - Walls are generated outside the square-plus-ring zone, so overflow writes
#      never collide with walls and source/distractor maroons never start inside
#      a target cell.
#
# e) Visual observability
#    - The square size is directly visible from the red/green object.
#    - Overflow is directly visible because the maroon fill expands from the
#      square into a second surrounding layer, not just "some maroons remain".
#    - Non-overflow and overflow samples can be contrasted by eye: one stops at
#      the square interior, the other continues onto the outer ring.
#
# f) Visual cleanliness
#    - The central region has two clean, nested zones:
#      inner square = primary container, outer ring = overflow container.
#    - Walls stay one layer farther out, so the three roles remain separable:
#      target, overflow band, and connectivity barriers.
#
# g) Change inferability
#    - No hidden color ordering is used. The only ordering signals are BFS
#      distance and row-major position, both spatially visible.
#    - The overflow rule is structural rather than color-remapping based: once
#      the square is full, the same maroon payload continues into the ring.
#
# h) Generator anti-degeneration target
#    - The generator must produce both size branches and a substantial share of
#      true overflow cases, otherwise the combo would degenerate into D6/0 only.
#    - The generator therefore explicitly samples source counts on both sides of
#      the inner capacity and rejects samples where the overflow layer is absent
#      too often.
#
# i) New solver assumptions and generator support
#    - Assumption: all red/green cells form one solid square.
#      Generator: writes exactly one square of {2, 3}.
#    - Assumption: the red cell is the square's top-left anchor.
#      Generator: places red only at that corner.
#    - Assumption: the outer ring around the square is available as overflow
#      space and is not preoccupied by walls.
#      Generator: reserves that ring as floor and moves the wall barrier outward.
#    - Assumption: at least one passable path leaves the square through the wall
#      barrier.
#      Generator: always keeps 1-4 wall gaps and rejects dead layouts.
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
ARM_LENGTH_RANGE = (1, 4)
FRAME_MARGIN_RANGE = (1, 2)
FRAME_CONNECT_PROB = 0.45
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

MIN_SOURCE_COUNT = {4: 6, 5: 8}
MAX_OVERFLOW_TOKENS = {4: 8, 5: 10}
OVERFLOW_PROB = 0.6
DISTRACTOR_MAX = 8
MIN_REACHABLE_SOURCE_CELLS = {4: 10, 5: 14}
GEN_RETRIES = 500
INF = 10**9


def _destination_cells(size: int, top: int = DEST_TOP, left: int = DEST_LEFT) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(top, top + size)
        for c in range(left, left + size)
    ]


def _overflow_cells(size: int, top: int = DEST_TOP, left: int = DEST_LEFT) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r0 = top - 1
    c0 = left - 1
    r1 = top + size
    c1 = left + size
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r0 < r < r1 and c0 < c < c1:
                continue
            cells.append((r, c))
    return cells


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
    assert (rr, rc) == (r0, c0), "red anchor must be the square's top-left cell"

    dest_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for r, c in dest_cells:
        assert int(grid[r, c]) in (RED_ANCHOR, DEST_COLOR), "destination must be solid"
    return dest_cells


def _floor_color(grid: np.ndarray) -> int:
    colors = [int(x) for x in np.unique(grid) if int(x) not in (RED_ANCHOR, DEST_COLOR, MAROON)]
    assert colors, "grid must contain at least one non-destination background color"
    return max(colors, key=lambda color: int((grid == color).sum()))


def _reachable_distances(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    passable: set[int],
) -> np.ndarray:
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

    dist = np.full((h, w), INF, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == INF:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into a variable-size square, then spill overflow into its surrounding ring.

    Input structure:
    - Orange floor contains optional magenta walls and scattered maroon (9) pixels.
    - A solid red/green square is the destination container: its unique red (2) cell is the top-left anchor, and the square side length varies between examples.

    Rule:
    - Flood-fill outward from passable cells touching the square, traversing only floor and maroon cells.
    - Sort reachable maroon pixels by BFS distance, then by row and column.
    - Fill the square in row-major order first; if reachable maroons still remain, continue filling the one-cell-thick outer ring around that square in row-major order. Moved maroons are erased back to floor.

    Key insight:
    - The solver must recover the full red/green square before doing any counting. Its side length determines both the inner capacity and the shape of the overflow ring, so overflow is now a second structured destination rather than "leave extras outside".
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    dest_cells = _find_destination(grid)
    size = int(round(len(dest_cells) ** 0.5))
    top = min(r for r, _ in dest_cells)
    left = min(c for _, c in dest_cells)
    overflow_cells = _overflow_cells(size, top, left)
    targets = dest_cells + overflow_cells

    floor = _floor_color(grid)
    passable = {floor, MAROON}

    dist = _reachable_distances(grid, dest_cells, passable)
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [item for item in maroons if item[0] < INF]
    maroons.sort()

    moved = maroons[: min(len(targets), len(maroons))]
    for _, r, c in moved:
        out[r, c] = floor
    for (r, c), _src in zip(targets, moved):
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


def _extend_arm(
    wall_cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    size: int,
    r: int,
    c: int,
    max_len: int,
) -> None:
    dirs = _outward_dirs(r, c, size)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    nr, nc = r, c
    for _ in range(random.randint(1, max_len)):
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in blocked:
            wall_cells.add((nr, nc))
        else:
            break


def _wall_frontier(blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    ring = set()
    for r, c in blocked:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in blocked:
                ring.add((nr, nc))
    return sorted(ring)


def _build_walls(grid: np.ndarray, blocked: set[tuple[int, int]], size: int) -> None:
    ring = _wall_frontier(blocked)
    num_gaps = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring, min(num_gaps, len(ring))))
    wall_ring = [cell for cell in ring if cell not in gap_cells]
    wall_cells = set(wall_ring)

    style = random.choice(WALL_STYLES)
    if style == "arms":
        for r, c in wall_ring:
            if random.random() < ARM_EXTEND_PROB:
                _extend_arm(wall_cells, blocked, size, r, c, ARM_LENGTH_RANGE[1])
    elif style == "frame":
        outer_top = DEST_TOP - 1
        outer_left = DEST_LEFT - 1
        margin = random.randint(*FRAME_MARGIN_RANGE)
        r0 = outer_top - margin
        c0 = outer_left - margin
        r1 = DEST_TOP + size + margin
        c1 = DEST_LEFT + size + margin
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in blocked:
                    if r in (r0, r1) or c in (c0, c1):
                        wall_cells.add((r, c))
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c, size)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in blocked:
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
                _extend_arm(wall_cells, blocked, size, r, c, ARM_LENGTH_RANGE[1])

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
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
    inner_cap = size * size
    max_sources = min(reachable_count, inner_cap + MAX_OVERFLOW_TOKENS[size])
    if max_sources < MIN_SOURCE_COUNT[size]:
        return None

    overflow_possible = max_sources > inner_cap
    if overflow_possible and random.random() < OVERFLOW_PROB:
        low = inner_cap + 1
        high = max_sources
    else:
        low = MIN_SOURCE_COUNT[size]
        high = min(inner_cap, max_sources)
    if low > high:
        return None
    return random.randint(low, high)


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(DEST_SIZE_CHOICES)
    dest_cells = _destination_cells(size)
    overflow_cells = _overflow_cells(size)
    dest_set = set(dest_cells)
    target_set = dest_set | set(overflow_cells)
    blocked_for_walls = set(target_set)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[DEST_TOP, DEST_LEFT] = RED_ANCHOR

    if random.random() < WALL_PROB:
        _build_walls(base, blocked_for_walls, size)

    reachable_floor, dist = _reachable_floor(base, dest_set)
    source_candidates = reachable_floor - set(overflow_cells)
    if len(source_candidates) < MIN_REACHABLE_SOURCE_CELLS[size]:
        return None

    source_count = _choose_source_count(size, len(source_candidates))
    if source_count is None:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in target_set
    }
    unreachable_floor = all_floor - source_candidates

    source_cells = random.sample(list(source_candidates), source_count)
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor))
    distractor_count = random.randint(0, max_distractors) if max_distractors else 0
    distractor_cells = random.sample(list(unreachable_floor), distractor_count)

    targets = dest_cells + overflow_cells
    ranked_sources = sorted((dist[(r, c)], r, c) for r, c in source_cells)
    moved_count = min(len(targets), len(ranked_sources))
    moved_sources = {(r, c) for _, r, c in ranked_sources[:moved_count]}
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
    for r, c in targets[:moved_count]:
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

# Combo variant of dd6b8c4b (index 36)
# combo_spec: D6/0+D13/2
# Name: 不足时容器缩小+可变大小容器

import math
import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the fused D6/0 + D13/2 variant
#
# a) Concrete behavior:
#    - The original task used a fixed 3x3 destination around a red center.
#    - D6/0 first changes that destination into a solid red/green square whose
#      side length is visible in the input and varies between 4 and 5.
#      The unique red cell is the square's top-left anchor so even sizes remain
#      unambiguous.
#    - D13/2 then changes what happens when too few reachable maroons exist:
#      the solver still ranks reachable maroons by BFS distance, but if fewer
#      than the full container capacity can move, the output container shrinks
#      to the smallest anchored square that can hold the moved count.
#    - Trace example:
#      input square = 5x5, reachable maroons = 6.
#      Old D6-only logic: place 6 maroons into the first 6 cells of the full
#      5x5 square and leave the remaining 19 square cells green.
#      Fused logic: 6 maroons need only a 3x3 square, so the output keeps only
#      the top-left 3x3 part of the original container, fills its first 6 cells
#      with maroon, leaves the remaining 3 cells green, and turns the removed
#      outer 16 container cells back into floor.
#
# b) Complexity calibration:
#    - Container sizes are restricted to two repeated cases: 4x4 and 5x5.
#    - Shrink outcomes are restricted to a few repeated side lengths, mainly
#      3x3 or 4x4, so the rule is visible across a small number of examples.
#    - Some full-capacity / overflow cases are still generated so the solver
#      cannot reduce the task to "always shrink".
#
# c) Old implicit assumptions that break:
#    - "red marks the center" is invalid; it now marks the top-left anchor.
#    - "destination is always 3x3" is invalid; the full {2,3} square must be
#      recovered first.
#    - "moved maroons fill exactly the visible input container" is invalid;
#      the output container size now depends on the reachable count.
#    - "all destination cells survive as destination-colored cells" is invalid;
#      cells outside the final shrunken square revert to floor.
#
# d) Placement legality:
#    - The variable square is anchored at a fixed top-left position so both 4x4
#      and 5x5 fit well inside the 11x11 canvas.
#    - Walls are only placed outside the square.
#    - Maroon sources and distractors are sampled only from floor cells, never
#      from the destination square or walls.
#
# e) Visual observability:
#    - The input clearly shows the original square size via the solid green area
#      and red top-left anchor.
#    - The output clearly shows the shrink because the container's outer band
#      disappears back to orange floor rather than merely staying green.
#    - Examples with different reachable counts make the minimal-square rule
#      inferable from visible counts and shapes, not from hidden ids.
#
# f) Visual cleanliness:
#    - The center region contains only one structured object: one anchored
#      square container.
#    - Walls remain external; maroons remain on floor in the input, so objects
#      stay visually separated.
#
# g) Change inferability:
#    - No rule depends on color ordering or numeric ids.
#    - The only count-based step is human-visible: count the reachable maroons,
#      then infer the smallest square that could hold them.
#
# h) Generator non-degeneration target:
#    - The generator intentionally favors samples where final_side < input_side,
#      because otherwise the combo would collapse back toward the plain D6/0
#      behavior on many instances.
#    - It still emits some no-shrink cases so the size cap remains part of the
#      concept instead of every puzzle being a pure shrinking puzzle.
#
# i) New solver assumptions and how generation enforces them:
#    - Assumption: all red/green cells form one solid square.
#      Generator: writes exactly one anchored square of {2,3}.
#    - Assumption: the red cell is the square's top-left anchor.
#      Generator: places red only at that corner.
#    - Assumption: at least one passable cell touches the square.
#      Generator: wall ring always leaves 1-4 gaps and rejects dead layouts.
#    - Assumption: there is at least one reachable maroon in accepted samples,
#      so the shrink rule is meaningful.
#      Generator: chooses at least 5 reachable source maroons.
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
MIN_MOVED = 5
EXTRA_SOURCE_MARGIN = 8
SHRINK_MODE_PROB = 0.8
OVERFLOW_PROB = 0.55
MIN_REACHABLE_FLOOR = {4: 12, 5: 18}
GEN_RETRIES = 500


def _square_cells(top: int, left: int, size: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(top, top + size)
        for c in range(left, left + size)
    ]


def _find_destination(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]]]:
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
    assert (rr, rc) == (r0, c0), "red anchor must be the top-left corner of the square"

    dest_cells = _square_cells(r0, c0, height)
    for r, c in dest_cells:
        assert int(grid[r, c]) in (RED_ANCHOR, DEST_COLOR), "destination square must be solid"
    return r0, c0, height, dest_cells


def _floor_color(grid: np.ndarray) -> int:
    colors = [int(x) for x in np.unique(grid) if int(x) not in (RED_ANCHOR, DEST_COLOR, MAROON)]
    assert colors, "grid must contain at least one non-destination background color"
    return max(colors, key=lambda color: int((grid == color).sum()))


def _reachable_distances(grid: np.ndarray, dest_cells: list[tuple[int, int]], passable: set[int]) -> np.ndarray:
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


def _minimal_square_side(count: int) -> int:
    count = max(1, int(count))
    side = math.isqrt(count)
    if side * side < count:
        side += 1
    return side


def _paint_final_container(
    out: np.ndarray,
    floor: int,
    original_dest_cells: list[tuple[int, int]],
    top: int,
    left: int,
    final_side: int,
    moved_count: int,
) -> None:
    for r, c in original_dest_cells:
        out[r, c] = floor

    final_cells = _square_cells(top, left, final_side)
    for r, c in final_cells:
        out[r, c] = DEST_COLOR
    out[top, left] = RED_ANCHOR
    for r, c in final_cells[:moved_count]:
        out[r, c] = MAROON


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into a variable square container, then shrink that container when the moved count is too small.

    Input structure:
    - Orange floor contains optional magenta walls and scattered maroon (9) pixels.
    - A single solid red/green square is the destination container: its unique red (2) cell marks the square's top-left corner, and the input square side length may be 4 or 5.

    Rule:
    - Flood-fill from passable cells touching the square, traversing only floor and maroon cells, then rank reachable maroons by BFS distance with row-major tie-breaking.
    - Move as many ranked maroons as possible, capped by the input square area. If that moved count is smaller than the input capacity, rebuild the destination as the smallest anchored square that can hold those moved maroons; otherwise keep the original square size.
    - Clear the moved source maroons back to floor, fill the final square in row-major order with the moved maroons, and leave any unused cells inside that final square as green/red container cells.

    Key insight:
    - The visible input square is only an upper bound on capacity. The solver must first detect the whole anchored square, then count how many maroons are actually reachable, because that count determines whether the output stays full-size or collapses to a smaller square.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    top, left, size, dest_cells = _find_destination(grid)
    floor = _floor_color(grid)
    passable = {floor, MAROON}

    dist = _reachable_distances(grid, dest_cells, passable)
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [item for item in maroons if item[0] < 10**9]
    maroons.sort()

    capacity = len(dest_cells)
    moved_count = min(capacity, len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:moved_count]}

    for r, c in chosen:
        out[r, c] = floor

    final_side = size if moved_count >= capacity else _minimal_square_side(moved_count)
    _paint_final_container(out, floor, dest_cells, top, left, final_side, moved_count)
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
    dest_set: set[tuple[int, int]],
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
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]], size: int) -> None:
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
    capacity = size * size
    max_sources = min(reachable_count, capacity + EXTRA_SOURCE_MARGIN)
    if max_sources < MIN_MOVED:
        return None

    shrink_sides = []
    for final_side in range(3, size):
        low = max(MIN_MOVED, (final_side - 1) ** 2 + 1)
        high = min(final_side * final_side, reachable_count)
        if low <= high:
            shrink_sides.append((final_side, low, high))

    if shrink_sides and random.random() < SHRINK_MODE_PROB:
        _, low, high = random.choice(shrink_sides)
        return random.randint(low, high)

    if reachable_count >= capacity:
        if max_sources > capacity and random.random() < OVERFLOW_PROB:
            return random.randint(capacity + 1, max_sources)
        return capacity

    return random.randint(MIN_MOVED, reachable_count)


def _expected_output(
    base: np.ndarray,
    source_cells: list[tuple[int, int]],
    distractor_cells: list[tuple[int, int]],
    top: int,
    left: int,
    size: int,
    dist: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, int]:
    dest_cells = _square_cells(top, left, size)
    inp = base.copy()
    for r, c in source_cells:
        inp[r, c] = MAROON
    for r, c in distractor_cells:
        inp[r, c] = MAROON

    out = inp.copy()
    ordered_sources = sorted((dist[(r, c)], r, c) for r, c in source_cells)
    capacity = len(dest_cells)
    moved_count = min(capacity, len(ordered_sources))
    for _, r, c in ordered_sources[:moved_count]:
        out[r, c] = FLOOR_COLOR

    final_side = size if moved_count >= capacity else _minimal_square_side(moved_count)
    _paint_final_container(out, FLOOR_COLOR, dest_cells, top, left, final_side, moved_count)
    return inp, out, final_side


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(DEST_SIZE_CHOICES)
    dest_cells = _square_cells(DEST_TOP, DEST_LEFT, size)
    dest_set = set(dest_cells)

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

    inp, out, final_side = _expected_output(
        base,
        source_cells,
        distractor_cells,
        DEST_TOP,
        DEST_LEFT,
        size,
        dist,
    )

    # Reject too many degenerate 5x5 -> 5x5 underfull layouts; shrink-heavy
    # sampling makes the combo visually distinct from plain D6/0.
    if final_side == size and source_count < size * size and random.random() < 0.75:
        return None

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

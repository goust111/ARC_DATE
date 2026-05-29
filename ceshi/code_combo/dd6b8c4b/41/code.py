# Combo variant of dd6b8c4b (index 41)
# combo_spec: D6/0+D15/3
# Name: 双向收集（方块向外扩展）+可变大小容器

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# ---------------------------------------------------------------------------
# Design analysis for combo D6/0 + D15/3
#
# a) Concrete behavior:
#    - The fixed 3x3 destination becomes a variable solid square of {2,3} with
#      side length 4 or 5; the unique red pixel is the square's top-left anchor.
#    - The collection is no longer "one flood-fill, then pack into the original
#      square". The square grows outward while collecting: a 4x4 square expands
#      by 1 ring, a 5x5 square expands by 2 rings. This ties the D15 growth
#      process directly to the D6 size choice.
#    - Reachability still uses 4-neighbor BFS through floor/maroon cells, but a
#      maroon is moved only if it lies within the number of BFS layers unlocked
#      by the square's growth. The final container is the expanded square.
#    - Small trace:
#      old: fixed 3x3, all reachable maroons compete for 9 cells.
#      new 4x4: only distance<=1 maroons are moved, and output becomes 6x6.
#      new 5x5: distance<=2 maroons are moved, and output becomes 9x9.
#
# b) Complexity calibration:
#    - Only two size branches are used: 4 and 5. Their expansion counts are
#      likewise only 1 and 2. This keeps the rule compact and repeated.
#    - Generators force at least one moved maroon on the outermost unlocked BFS
#      layer and at least one farther reachable maroon that remains, so the
#      "growth depth" is visible instead of collapsing back to the original rule.
#
# c) Implicit assumptions that break from the original solver:
#    - The red pixel is no longer the center of a 3x3 block.
#    - The destination is no longer a fixed-size neighborhood around red.
#    - A single global BFS is no longer sufficient; the solver must interpret
#      only the prefix of BFS layers allowed by the square's growth budget.
#    - The output target is no longer the input square itself; it is a larger
#      square obtained by expanding the original container.
#
# d) Placement legality:
#    - The destination square is anchored so that both size branches and their
#      corresponding expanded outputs fit inside the 11x11 canvas.
#    - Maroon source cells are sampled only from floor cells reachable by BFS.
#    - Distractor maroons are sampled outside the final expanded square so they
#      are not accidentally overwritten by the growing container.
#
# e) Visual observability:
#    - The input visibly shows whether the container is 4x4 or 5x5.
#    - The output visibly shows a larger green/maroon square, revealing that the
#      container itself expanded rather than merely refilling its original area.
#    - Farther reachable maroons remain on the floor, so the growth cutoff is
#      inferable from a few examples.
#
# f) Visual cleanliness:
#    - The central object stays a single solid square in both input and output.
#    - Walls and leftover maroons remain outside the final container, avoiding
#      ambiguous overlaps inside the filled destination.
#
# g) Change inferability:
#    - No hidden color ordering or numeric color semantics are used.
#    - The only controlling signal is the visible square size; it determines how
#      far the collector grows and therefore which BFS layers are harvested.
#
# h) Generator non-degeneration target:
#    - Samples must include both size branches across runs.
#    - Each accepted grid must have at least one moved maroon on the last growth
#      layer and at least one still-reachable maroon beyond that layer.
#
# i) New solver assumptions and generator guarantees:
#    - Assumption: all red/green cells form one solid square.
#      Generator: writes exactly one such square.
#    - Assumption: the red pixel is the square's top-left anchor.
#      Generator: places the unique red cell at that corner only.
#    - Assumption: the square's side length is one of the supported branches.
#      Generator: uses only size 4 or 5.
#    - Assumption: leftover maroons outside the moved set are outside the final
#      expanded square.
#      Generator: samples distractors only from cells outside that final square.
# ---------------------------------------------------------------------------


GRID_H, GRID_W = 11, 11
DEST_TOP, DEST_LEFT = 2, 2
DEST_SIZE_TO_ROUNDS = {4: 1, 5: 2}
DEST_SIZE_CHOICES = tuple(DEST_SIZE_TO_ROUNDS)

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

SOURCE_COUNT_RANGE = {4: (7, 15), 5: (10, 20)}
REACHABLE_DISTRACTOR_RANGE = (1, 4)
UNREACHABLE_DISTRACTOR_MAX = 5
MIN_REACHABLE_FLOOR = {4: 18, 5: 24}
GEN_RETRIES = 500
INF = 10**9


def _destination_cells(size: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(DEST_TOP, DEST_TOP + size)
        for c in range(DEST_LEFT, DEST_LEFT + size)
    ]


def _expanded_square_from_bbox(r0: int, c0: int, size: int, rounds: int) -> list[tuple[int, int]]:
    top = r0 - rounds
    left = c0 - rounds
    bottom = r0 + size - 1 + rounds
    right = c0 + size - 1 + rounds
    assert 0 <= top <= bottom < GRID_H
    assert 0 <= left <= right < GRID_W
    return [(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)]


def _expanded_square(dest_cells: list[tuple[int, int]], rounds: int) -> list[tuple[int, int]]:
    rows = [r for r, _ in dest_cells]
    cols = [c for _, c in dest_cells]
    r0, c0 = min(rows), min(cols)
    size = max(rows) - r0 + 1
    return _expanded_square_from_bbox(r0, c0, size, rounds)


def _find_destination(grid: np.ndarray) -> tuple[list[tuple[int, int]], int]:
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
    assert height == width, "destination must be square"
    size = height
    assert size in DEST_SIZE_TO_ROUNDS, "unsupported destination size"
    assert (rr, rc) == (r0, c0), "red anchor must mark the square's top-left corner"

    dest_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for r, c in dest_cells:
        assert int(grid[r, c]) in (RED_ANCHOR, DEST_COLOR), "destination square must be solid"
    return dest_cells, size


def _floor_color(grid: np.ndarray) -> int:
    colors = [int(x) for x in np.unique(grid) if int(x) not in (RED_ANCHOR, DEST_COLOR, MAROON)]
    assert colors, "grid must contain at least one background color"
    return max(colors, key=lambda color: int((grid == color).sum()))


def _reachable_distances(grid: np.ndarray, dest_cells: list[tuple[int, int]], passable: set[int]) -> np.ndarray:
    h, w = grid.shape
    dest_set = set(dest_cells)

    dist = np.full((h, w), INF, dtype=int)
    q = deque()

    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if dist[nr, nc] == INF:
                    dist[nr, nc] = 1
                    q.append((nr, nc))

    assert q, "there must be at least one passable cell touching the destination square"

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == INF:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a variable-size square container outward and move only the maroons reached within its growth depth.

    Input structure:
    - Orange floor contains optional magenta walls and scattered maroon (9) pixels.
    - The destination is a solid red/green square whose unique red (2) cell marks the square's top-left corner; its side length is either 4 or 5.

    Rule:
    - Flood-fill through floor and maroon cells from the square's outside edge, measuring 4-neighbor BFS distance.
    - The square expands by a size-dependent number of rings: 4x4 grows by 1 ring, 5x5 grows by 2 rings. Move only the maroons within those unlocked BFS layers.
    - Erase moved maroons to floor, replace the expanded square by green, then write the collected maroons into that expanded square in row-major order.

    Key insight:
    - The visible square size controls both the final container geometry and the maximum collection depth. The solver must recover the whole square first, then use its size to decide how far the collector is allowed to grow.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    dest_cells, size = _find_destination(grid)
    rounds = DEST_SIZE_TO_ROUNDS[size]
    final_cells = _expanded_square(dest_cells, rounds)

    floor = _floor_color(grid)
    passable = {floor, MAROON}
    dist = _reachable_distances(grid, dest_cells, passable)

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    chosen = [(d, r, c) for d, r, c in maroons if d <= rounds]
    chosen.sort()

    for _, r, c in chosen:
        out[r, c] = floor

    for r, c in final_cells:
        out[r, c] = DEST_COLOR
    for r, c in final_cells[: len(chosen)]:
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
                    dist[(nr, nc)] = 1
                    q.append((nr, nc))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR:
                    visited.add((nr, nc))
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))

    return visited, dist


def _expected_output(
    base: np.ndarray,
    dest_cells: list[tuple[int, int]],
    rounds: int,
    source_cells: list[tuple[int, int]],
    distractor_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = base.copy()
    for r, c in distractor_cells:
        out[r, c] = MAROON

    final_cells = _expanded_square(dest_cells, rounds)
    for r, c in final_cells:
        out[r, c] = DEST_COLOR
    for r, c in final_cells[: len(source_cells)]:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(DEST_SIZE_CHOICES)
    rounds = DEST_SIZE_TO_ROUNDS[size]
    dest_cells = _destination_cells(size)
    dest_set = set(dest_cells)
    final_cells = _expanded_square(dest_cells, rounds)
    final_set = set(final_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[DEST_TOP, DEST_LEFT] = RED_ANCHOR

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set, size)

    reachable_floor, dist = _reachable_floor(base, dest_set)
    if len(reachable_floor) < MIN_REACHABLE_FLOOR[size]:
        return None

    eligible = [cell for cell in reachable_floor if dist[cell] <= rounds]
    outer_layer = [cell for cell in eligible if dist[cell] == rounds]
    far_reachable = [cell for cell in reachable_floor if dist[cell] > rounds and cell not in final_set]
    if not outer_layer or not far_reachable:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = [cell for cell in all_floor - reachable_floor if cell not in final_set]

    low, high = SOURCE_COUNT_RANGE[size]
    high = min(high, len(eligible))
    if high < low:
        return None
    source_count = random.randint(low, high)

    chosen_outer = random.choice(outer_layer)
    remaining_pool = [cell for cell in eligible if cell != chosen_outer]
    source_cells = [chosen_outer]
    source_cells.extend(random.sample(remaining_pool, source_count - 1))

    reach_low = REACHABLE_DISTRACTOR_RANGE[0]
    reach_high = min(REACHABLE_DISTRACTOR_RANGE[1], len(far_reachable))
    if reach_high < reach_low:
        return None
    reachable_distractor_count = random.randint(reach_low, reach_high)
    reachable_distractors = random.sample(far_reachable, reachable_distractor_count)

    unreachable_distractor_count = 0
    if unreachable_floor:
        unreachable_distractor_count = random.randint(0, min(UNREACHABLE_DISTRACTOR_MAX, len(unreachable_floor)))
    unreachable_distractors = random.sample(unreachable_floor, unreachable_distractor_count)
    distractor_cells = reachable_distractors + unreachable_distractors

    inp = base.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(base, dest_cells, rounds, source_cells, distractor_cells)

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

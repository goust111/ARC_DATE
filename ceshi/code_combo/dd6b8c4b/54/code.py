# Combo variant of dd6b8c4b (index 54)
# combo_spec: D4/2+D6/0+D13/1
# Name: 可变容器+螺旋填充+溢出处理

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# ---------------------------------------------------------------------------
# Design analysis for combo D4/2 + D6/0 + D13/1
#
# a) Concrete behavior and trace
#    - Destination:
#      The old 3x3 "green ring + red center" becomes one solid red/green square.
#      The square size varies per puzzle: 4x4 or 5x5. To keep both sizes legible,
#      the unique red cell marks the top-left anchor and every other square cell is
#      green. This removes the old "red is the center" cue.
#    - Internal fill path:
#      Inside that square, placement is no longer row-major. The moved maroons follow
#      one fixed clockwise inward spiral starting at the anchor corner.
#    - Overflow:
#      If reachable maroons exceed the square capacity, the next maroons do not stay
#      on the floor. They spill onto the one-cell-thick ring immediately outside the
#      square, using a fixed clockwise perimeter order that starts at the spill ring's
#      own top-left corner. That outer ring is the visible "overflow channel".
#    - Small trace:
#      old task with 12 reachable maroons:
#        fixed 3x3 -> only 9 move, filled row-major, 3 remain outside.
#      this combo with a 4x4 square and 19 reachable maroons:
#        first 16 move into the 4x4 spiral, next 3 move onto the top edge of the
#        surrounding spill ring, and only the still-later maroons would remain outside.
#      this combo with a 5x5 square and 11 reachable maroons:
#        only 11 move, all stay inside the square, but their partial pattern bends
#        around the spiral instead of forming a row-major prefix rectangle.
#
# b) Complexity calibration
#    - The variant uses only two square sizes, {4, 5}, so humans see repeated size
#      evidence instead of a large family of unrelated containers.
#    - The spiral orientation is fixed: clockwise, starting from the square's
#      top-left. There is no hidden orientation branch.
#    - Overflow uses only one extra layer: the immediate outer ring. This is enough
#      to make overflow visible without turning the puzzle into an arbitrary multi-
#      layer packing task.
#    - Generation mixes two regimes:
#      partial non-overflow cases, where the spiral is visible inside the square,
#      and overflow cases, where the spill ring becomes active.
#
# c) Implicit-assumption audit against the original main()
#    - "red pixel is the center of a 3x3 block" is invalid. The solver must recover
#      the whole red/green square from its bounding box and red corner anchor.
#    - "destination cells are cr-1:cr+1, cc-1:cc+1" is invalid. Destination size is
#      data-dependent.
#    - "capacity is always 9" is invalid. Capacity is size^2.
#    - "sorted(dest_cells) is the write order" is invalid. The write order is now the
#      inward spiral of the recovered square.
#    - "overflow maroons just stay put" is invalid. The solver must erase not only the
#      maroons moved inside the square, but also the extra selected maroons moved onto
#      the outside spill ring.
#    - The old ring-color heuristic around the destination is no longer a good place
#      to infer floor color, because the immediate ring is now deliberately reserved
#      for overflow output. The new solver instead infers floor as the dominant
#      non-destination, non-maroon color in the whole grid.
#
# d) Placement legality
#    - The square is anchored at a fixed top-left position so both 4x4 and 5x5 fit
#      comfortably inside the 11x11 grid together with a spill ring and a wall ring.
#    - The spill ring is kept free of walls in the input, so overflow has guaranteed,
#      non-overlapping landing cells.
#    - Walls, when present, are moved one layer farther out than the spill ring. This
#      preserves the original reachability logic while keeping the overflow channel
#      visually clean.
#    - Source maroons and distractors are sampled only from floor cells, never from
#      destination cells, spill cells, or wall cells.
#
# e) Visual observability
#    - Container size is directly visible from the red/green square footprint.
#    - In partial cases the spiral is readable because generation avoids trivial
#      counts where spiral and row-major are identical or nearly identical.
#    - In overflow cases the outside ring lights up immediately around the square,
#      making the "more than capacity" branch visible without hidden counting tricks.
#    - Ranking still depends only on BFS distance, then row, then column, which are
#      human-visible spatial properties.
#
# f) Visual cleanliness
#    - The central region now has three clean layers:
#      square container, empty spill ring, optional farther wall ring.
#      This separation prevents overflow tokens from landing on top of barrier cells.
#    - The generator also keeps source maroons off the spill ring in the input, so the
#      output-only outer ring remains a reliable signal of overflow.
#
# g) Change inferability and color reasoning
#    - No new color remapping is introduced. Colors keep their structural roles:
#      floor, wall, green square, red anchor, maroon tokens.
#    - The variant does not rely on color-id ordering or any invisible metadata.
#      The only ordering comes from visible geometry: BFS distance and fixed paths.
#
# h) Generator anti-degeneration target
#    - Non-overflow samples are forced to be partial fills (at least 6 maroons and at
#      least two empty square cells left), so the inner spiral differs from row-major.
#    - Overflow samples are forced whenever reachable space allows, and the number of
#      extra maroons is capped to a small visible band so the spill ring activates
#      regularly without cluttering the board.
#
# i) New solver assumptions and how generation enforces them
#    - Assumption: all red/green cells form one solid square.
#      Enforcement: generator writes exactly one anchored square of {2, 3}.
#    - Assumption: the unique red cell is the square's top-left corner.
#      Enforcement: generator places red exactly there.
#    - Assumption: at least one passable cell touches the square.
#      Enforcement: the spill ring is floor, and wall layouts with no reachable floor
#      are rejected.
#    - Assumption: overflow landing cells exist and are unambiguous.
#      Enforcement: the spill ring is reserved from other structures in the input.
#    - Assumption: generated source counts stay within the supported one-ring overflow
#      budget.
#      Enforcement: source count is capped below capacity + spill-ring length, and in
#      practice much lower for readability.
# ---------------------------------------------------------------------------


GRID_H, GRID_W = 11, 11
DEST_TOP, DEST_LEFT = 3, 3
DEST_SIZE_CHOICES = (4, 5)

FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_ANCHOR = 3, 2
MAROON = 9

WALL_PROB = 0.75
BARRIER_GAP_RANGE = (1, 4)
WALL_STYLES = ("arms", "frame", "mixed")

ARM_EXTEND_PROB = 0.45
ARM_LENGTH_RANGE = (1, 3)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.65

DISTRACTOR_MAX = 6
OVERFLOW_PROB = 0.65
MAX_OVERFLOW_COUNT = 6
PARTIAL_EMPTY_MARGIN = 2
MIN_SOURCE_COUNT = {4: 6, 5: 8}
MIN_REACHABLE_SOURCE_CELLS = {4: 18, 5: 24}
GEN_RETRIES = 500


def _perimeter_path(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    """Clockwise perimeter path starting at the top-left corner."""
    if height <= 0 or width <= 0:
        return []
    bottom = top + height - 1
    right = left + width - 1
    cells: list[tuple[int, int]] = []

    for c in range(left, right + 1):
        cells.append((top, c))
    for r in range(top + 1, bottom + 1):
        cells.append((r, right))
    if bottom > top:
        for c in range(right - 1, left - 1, -1):
            cells.append((bottom, c))
    if right > left:
        for r in range(bottom - 1, top, -1):
            cells.append((r, left))

    return [(r, c) for r, c in cells if 0 <= r < GRID_H and 0 <= c < GRID_W]


def _square_spiral_path(top: int, left: int, size: int) -> list[tuple[int, int]]:
    """Clockwise inward spiral for a square, starting at its top-left corner."""
    path: list[tuple[int, int]] = []
    r0, c0 = top, left
    r1, c1 = top + size - 1, left + size - 1

    while r0 <= r1 and c0 <= c1:
        for c in range(c0, c1 + 1):
            path.append((r0, c))
        for r in range(r0 + 1, r1 + 1):
            path.append((r, c1))
        if r0 < r1:
            for c in range(c1 - 1, c0 - 1, -1):
                path.append((r1, c))
        if c0 < c1:
            for r in range(r1 - 1, r0, -1):
                path.append((r, c0))
        r0 += 1
        c0 += 1
        r1 -= 1
        c1 -= 1

    return path


def _destination_cells(size: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(DEST_TOP, DEST_TOP + size)
        for c in range(DEST_LEFT, DEST_LEFT + size)
    ]


def _spill_path(top: int, left: int, size: int) -> list[tuple[int, int]]:
    return _perimeter_path(top - 1, left - 1, size + 2, size + 2)


def _barrier_ring(top: int, left: int, size: int) -> list[tuple[int, int]]:
    return _perimeter_path(top - 2, left - 2, size + 4, size + 4)


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
    assert (rr, rc) == (r0, c0), "red anchor must mark the square's top-left corner"

    dest_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    assert len(coords) == len(dest_cells), "destination square must be solid"
    for r, c in dest_cells:
        assert int(grid[r, c]) in (RED_ANCHOR, DEST_COLOR), "destination square must be solid"

    return r0, c0, height, dest_cells


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
    seen: set[tuple[int, int]] = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    starts.append((nr, nc))
    assert starts, "there must be at least one passable cell touching the destination square"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q: deque[tuple[int, int]] = deque(starts)
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
    Move the closest reachable maroon pixels into a variable-size square, and spill any excess onto the square's outer ring.

    Input structure:
    - Orange floor cells may be partitioned by magenta walls, with maroon tokens scattered on reachable and unreachable floor.
    - The target is one solid red/green square: the unique red cell is its top-left anchor, and the square side length varies.

    Rule:
    - Flood-fill from passable cells touching the square, traversing only floor and maroon cells, then rank reachable maroons by BFS distance, row, and column.
    - Fill the square first along a clockwise inward spiral. If more ranked maroons remain after the square is full, continue by placing them on the one-cell-thick ring immediately outside the square in clockwise perimeter order.
    - Every moved maroon is erased from its source cell back to floor; unreachable maroons and any maroons beyond the supported spill ring remain in place.

    Key insight:
    - The solver must recover both the square size and the two-stage write geometry. The square size sets the internal capacity, while overflow activates a second visible path just outside the square instead of leaving extra reachable maroons on the floor.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()

    top, left, size, dest_cells = _find_destination(grid)
    fill_path = _square_spiral_path(top, left, size)
    spill_order = _spill_path(top, left, size)
    floor = _floor_color(grid)
    passable = {floor, MAROON}

    dist = _reachable_distances(grid, dest_cells, passable)
    reachable_maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    reachable_maroons = [item for item in reachable_maroons if item[0] < 10**9]
    reachable_maroons.sort()

    inner_count = min(len(fill_path), len(reachable_maroons))
    overflow_count = min(len(spill_order), max(0, len(reachable_maroons) - inner_count))
    moved_total = inner_count + overflow_count
    chosen = {(r, c) for _, r, c in reachable_maroons[:moved_total]}

    for r, c in chosen:
        out[r, c] = floor
    for r, c in fill_path[:inner_count]:
        out[r, c] = MAROON
    for r, c in spill_order[:overflow_count]:
        out[r, c] = MAROON

    return out


def _outward_dirs(r: int, c: int, top: int, left: int, size: int) -> list[tuple[int, int]]:
    center_r = top + (size - 1) / 2.0
    center_c = left + (size - 1) / 2.0
    dirs: list[tuple[int, int]] = []
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
    reserved: set[tuple[int, int]],
    top: int,
    left: int,
    size: int,
    r: int,
    c: int,
    max_len: int,
) -> None:
    dirs = _outward_dirs(r, c, top, left, size)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    nr, nc = r, c
    for _ in range(random.randint(1, max_len)):
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in reserved:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    top: int,
    left: int,
    size: int,
) -> None:
    barrier = _barrier_ring(top, left, size)
    num_gaps = min(random.randint(*BARRIER_GAP_RANGE), max(1, len(barrier) - 1))
    gap_cells = set(random.sample(barrier, num_gaps))
    wall_ring = [cell for cell in barrier if cell not in gap_cells]
    wall_cells = set(wall_ring)

    style = random.choice(WALL_STYLES)
    if style == "arms":
        for r, c in wall_ring:
            if random.random() < ARM_EXTEND_PROB:
                _extend_arm(wall_cells, reserved, top, left, size, r, c, ARM_LENGTH_RANGE[1])
    elif style == "frame":
        outer = _perimeter_path(top - 3, left - 3, size + 6, size + 6)
        wall_cells.update(cell for cell in outer if cell not in reserved)
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c, top, left, size)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in reserved:
                    wall_cells.add((nr, nc))
                    nr += dr
                    nc += dc
        outer_only = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_only:
            n_remove = random.randint(1, max(1, len(outer_only) // 5))
            for cell in random.sample(outer_only, min(n_remove, len(outer_only))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(cell)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                _extend_arm(wall_cells, reserved, top, left, size, r, c, ARM_LENGTH_RANGE[1])

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _reachable_floor_and_dist(
    grid: np.ndarray,
    dest_set: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    visited: set[tuple[int, int]] = set()
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int]] = deque()

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
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR:
                    visited.add((nr, nc))
                    dist[(nr, nc)] = dist[(r, c)] + 1
                    q.append((nr, nc))

    return visited, dist


def _choose_source_count(size: int, reachable_count: int) -> int | None:
    capacity = size * size
    min_count = MIN_SOURCE_COUNT[size]
    if reachable_count < min_count:
        return None

    overflow_high = min(reachable_count, capacity + MAX_OVERFLOW_COUNT)
    wants_overflow = overflow_high >= capacity + 1 and random.random() < OVERFLOW_PROB
    if wants_overflow:
        return random.randint(capacity + 1, overflow_high)

    partial_high = min(reachable_count, capacity - PARTIAL_EMPTY_MARGIN)
    if partial_high < min_count:
        partial_high = min(reachable_count, capacity)
    if partial_high < min_count:
        return None
    return random.randint(min_count, partial_high)


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(DEST_SIZE_CHOICES)
    top, left = DEST_TOP, DEST_LEFT
    dest_cells = _destination_cells(size)
    dest_set = set(dest_cells)
    fill_path = _square_spiral_path(top, left, size)
    spill_order = _spill_path(top, left, size)
    spill_set = set(spill_order)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[top, left] = RED_ANCHOR

    reserved = dest_set | spill_set
    if random.random() < WALL_PROB:
        _build_walls(base, reserved, top, left, size)

    reachable_floor, dist = _reachable_floor_and_dist(base, dest_set)
    source_pool = list(reachable_floor - spill_set)
    if len(source_pool) < MIN_REACHABLE_SOURCE_CELLS[size]:
        return None

    source_count = _choose_source_count(size, len(source_pool))
    if source_count is None:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in dest_set and (r, c) not in spill_set
    }
    unreachable_floor = list(all_floor - reachable_floor)

    source_cells = random.sample(source_pool, source_count)
    distractor_count = random.randint(0, min(DISTRACTOR_MAX, len(unreachable_floor)))
    distractor_cells = random.sample(unreachable_floor, distractor_count)

    ranked_sources = sorted((dist[(r, c)], r, c) for r, c in source_cells)
    inner_count = min(len(fill_path), source_count)
    overflow_count = min(len(spill_order), max(0, source_count - inner_count))
    moved_total = inner_count + overflow_count
    moved_sources = {(r, c) for _, r, c in ranked_sources[:moved_total]}
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
    for r, c in fill_path[:inner_count]:
        out[r, c] = MAROON
    for r, c in spill_order[:overflow_count]:
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

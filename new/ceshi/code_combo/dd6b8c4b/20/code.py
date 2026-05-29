# Combo variant of dd6b8c4b (index 20)
# combo_spec: D4/2+D6/2
# Name: 非正方形容器+螺旋填充

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis
# a) Concrete behavior and trace:
#    - The red cell is no longer the center of an implied 3x3 square. It is an
#      anchor inside one connected red+green container, chosen from L/T shapes.
#    - The fill path is no longer row-major on container cells. Instead, take the
#      container's bounding box, walk that box in a clockwise outer-to-inner spiral,
#      and keep only the visited positions that actually belong to the container.
#    - Example: for L_dr = {(5,5),(5,6),(5,7),(6,5),(7,5)}, old irregular row-major
#      order is [(5,5),(5,6),(5,7),(6,5),(7,5)], while fused order becomes
#      [(5,5),(5,6),(5,7),(7,5),(6,5)]. With k=4, the old output leaves (7,5) green
#      and fills (6,5); the fused output does the opposite.
# b) Complexity calibration:
#    - Only L/T templates are used. They always contain holes inside a 3x3 bounding
#      box, so "spiral with skips" is visible and does not collapse into a plain
#      rectangle traversal.
#    - The generator fixes k=4. Since every template has 5 cells, one cell remains
#      unfilled; that makes the write order observable. Allowing k=5 would hide the
#      path because a fully filled container looks identical under many orders.
#    - Templates whose first four spiral cells match row-major are excluded; this
#      avoids silent degeneration back to the D6-only behavior.
# c) Implicit assumption audit:
#    - The original 3x3 loop around the red center is invalid. The solver must first
#      recover the full red+green connected component.
#    - Sorting dest_cells is no longer the write path; it is only a row-major
#      baseline. The actual write order comes from the bounding-box spiral.
#    - Full-square assumptions in wall placement and fill capacity are removed.
#      Wall logic is parameterized by dest_set, and fill capacity comes from shape size.
# d) Placement legality:
#    - Every template includes the fixed anchor (5,5) and fits inside the 11x11 grid.
#    - Walls are written only outside dest_set. Source and distractor maroons are
#      chosen only from floor cells, so there is no overlap with walls or destination.
# e) Visual observability:
#    - A solver can see the irregular container directly from the red+green component.
#      The output differs from row-major in a visible 4th placement, not via hidden
#      color ids or metadata.
#    - Connectivity remains governed by visible corridors and wall gaps, so the task
#      still teaches "reachable maroons only" before the write-path twist.
# f) Visual cleanliness:
#    - Colors keep their original roles, there is one destination component, and the
#      container stays near the center. The scene remains sparse enough to read.
# g) Change inferability:
#    - No extra color mapping is introduced. The only new inference is geometric:
#      infer the irregular container, then infer the spiral-with-skips traversal.
# h) New solver assumptions and generator guarantees:
#    - main() assumes exactly one red anchor and exactly one connected red+green
#      container containing it; generate_input() constructs that directly.
#    - main() assumes floor is the dominant non-special color and that at least one
#      passable cell touches the container; the generator preserves both conditions.
# i) Stage-by-stage main() audit:
#    - _destination_component() assumes the intended destination is the full 4-connected
#      {2,3} component through the anchor; the generator creates exactly one such component.
#    - _spiral_bbox_order() assumes the destination has a compact bounding box whose spiral
#      traversal is meaningful; the generator uses only small L/T templates with 3x3 boxes.
#    - floor inference by majority assumes wall cells never outnumber floor cells; on an
#      11x11 board with a 5-cell destination and sparse walls, generation keeps floor dominant.
#    - BFS start collection assumes some floor cell touches the destination; wall generation
#      leaves 1-4 ring gaps and rejects layouts with too few reachable cells.
#    - Final writeback assumes partial fills are common enough to expose the path; generation
#      fixes k=4 for 5-cell containers so the omitted cell always reveals the traversal rule.


GRID_H, GRID_W = 11, 11
ANCHOR_R, ANCHOR_C = 5, 5
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

FIXED_FILL_COUNT = 4
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = FIXED_FILL_COUNT
GEN_RETRIES = 400

DEST_TEMPLATES = {
    "L_dr": [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)],
    "L_ur": [(0, 0), (-1, 0), (-2, 0), (0, 1), (0, 2)],
    "L_ul": [(0, 0), (-1, 0), (-2, 0), (0, -1), (0, -2)],
    "T_up": [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
    "T_down": [(1, -1), (1, 0), (1, 1), (0, 0), (-1, 0)],
    "T_left": [(-1, -1), (0, -1), (1, -1), (0, 0), (0, 1)],
    "T_right": [(-1, 1), (0, 1), (1, 1), (0, 0), (0, -1)],
}


def _destination_component(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return the connected red+green destination component containing the unique red anchor."""
    h, w = grid.shape
    anchors = np.argwhere(grid == RED_CENTER)
    assert len(anchors) == 1, "grid must have exactly one red (2) anchor pixel"
    sr, sc = map(int, anchors[0])

    dest_set = {(sr, sc)}
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in (RED_CENTER, DEST_COLOR):
                dest_set.add((nr, nc))
                queue.append((nr, nc))

    assert len(dest_set) >= 5, "destination container must be a multi-cell red+green shape"
    return sorted(dest_set)


def _row_major_order(dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(dest_cells)


def _spiral_bbox_order(dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Traverse the destination bounding box in spiral order and skip cells outside the shape."""
    dest_set = set(dest_cells)
    rows = [r for r, _ in dest_cells]
    cols = [c for _, c in dest_cells]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)

    order: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def maybe_add(r: int, c: int) -> None:
        if (r, c) in dest_set and (r, c) not in seen:
            seen.add((r, c))
            order.append((r, c))

    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            maybe_add(top, c)
        for r in range(top + 1, bottom + 1):
            maybe_add(r, right)
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                maybe_add(bottom, c)
        if left < right:
            for r in range(bottom - 1, top, -1):
                maybe_add(r, left)
        top += 1
        bottom -= 1
        left += 1
        right -= 1

    assert len(order) == len(dest_cells), "spiral order must cover every destination cell exactly once"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the irregular destination shape using a spiral-with-skips fill.

    Input structure:
    - Orange floor cells may be split by magenta walls, and maroon tokens sit on some floor cells.
    - One red anchor belongs to a connected green container, which is an L or T shape rather than a full 3x3 square.

    Rule:
    - Recover the full red+green container, then flood-fill outward through floor/maroon cells touching it to find reachable maroon candidates.
    - Rank reachable maroons by BFS distance, then by row and column, erase the closest ones from their source cells, and place them into the container.
    - The placement path is a clockwise spiral over the container's bounding box, skipping any bounding-box positions that are not part of the container.

    Key insight:
    - Two geometric inferences interact: walls still decide which maroons are eligible, but the destination is an irregular shape whose fill order is defined by a spiral on its enclosing rectangle rather than by row-major order on the shape itself.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    dest_cells = _destination_component(grid)
    dest_set = set(dest_cells)
    fill_order = _spiral_bbox_order(dest_cells)

    non_special = [int(x) for x in np.unique(grid) if int(x) not in (RED_CENTER, DEST_COLOR, MAROON)]
    assert non_special, "grid must contain at least one non-destination background color"
    floor = max(non_special, key=lambda color: int((grid == color).sum()))
    passable = {floor, MAROON}

    starts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and (nr, nc) not in seen:
                seen.add((nr, nc))
                starts.append((nr, nc))
    assert starts, "there must be at least one passable cell adjacent to the destination shape"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    queue = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(queue):
        r, c = queue[qi]
        qi += 1
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [triple for triple in maroons if triple[0] < inf]
    maroons.sort()

    k = min(len(fill_order), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in fill_order[:k]:
        out[r, c] = MAROON

    return out


def _make_dest_cells(shape_name: str) -> list[tuple[int, int]]:
    cells = [(ANCHOR_R + dr, ANCHOR_C + dc) for dr, dc in DEST_TEMPLATES[shape_name]]
    assert all(0 <= r < GRID_H and 0 <= c < GRID_W for r, c in cells)
    return sorted(cells)


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """BFS from floor cells adjacent to the destination through floor cells only."""
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and int(grid[nr, nc]) == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and (nr, nc) not in visited:
                if int(grid[nr, nc]) == FLOOR_COLOR:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    """Return directions that tend to extend walls away from the fixed anchor."""
    dirs = []
    if r < ANCHOR_R:
        dirs.append((-1, 0))
    if r > ANCHOR_R:
        dirs.append((1, 0))
    if c < ANCHOR_C:
        dirs.append((0, -1))
    if c > ANCHOR_C:
        dirs.append((0, 1))
    if not dirs:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return dirs


def _extend_arm(
    wall_cells: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    r: int,
    c: int,
    max_len: int,
) -> None:
    dr, dc = random.choice(_outward_dirs(r, c))
    nr, nc = r, c
    for _ in range(random.randint(1, max_len)):
        nr, nc = nr + dr, nc + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    """Build a partial wall ring around the irregular destination, plus optional outward structure."""
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)
    assert ring, "destination must have an exterior ring"

    num_gaps = random.randint(RING_GAP_RANGE[0], min(RING_GAP_RANGE[1], len(ring)))
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
        for r in range(ANCHOR_R - dist2, ANCHOR_R + dist2 + 1):
            for c in range(ANCHOR_C - dist2, ANCHOR_C + dist2 + 1):
                if 0 <= r < GRID_H and 0 <= c < GRID_W and (r, c) not in dest_set:
                    if r in (ANCHOR_R - dist2, ANCHOR_R + dist2) or c in (ANCHOR_C - dist2, ANCHOR_C + dist2):
                        wall_cells.add((r, c))
        for r, c in wall_ring:
            if random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(_outward_dirs(r, c))
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((nr, nc))
                    nr, nc = nr + dr, nc + dc
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            n_remove = random.randint(1, max(1, len(outer_frame) // 4))
            for cell in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(cell)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, max(abs(r - ANCHOR_R), abs(c - ANCHOR_C)) + 1)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _expected_output(
    base_grid: np.ndarray,
    fill_order: list[tuple[int, int]],
    k: int,
    distractor_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = base_grid.copy()
    for r, c in fill_order[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    shape_name = random.choice(list(DEST_TEMPLATES))
    dest_cells = _make_dest_cells(shape_name)
    dest_set = set(dest_cells)
    fill_order = _spiral_bbox_order(dest_cells)
    row_order = _row_major_order(dest_cells)

    if fill_order[:FIXED_FILL_COUNT] == row_order[:FIXED_FILL_COUNT]:
        return None

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[ANCHOR_R, ANCHOR_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    k = FIXED_FILL_COUNT

    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - k)
    num_distractors = random.randint(0, max_distractors) if max_distractors > 0 else 0
    if k + num_distractors < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - (k + num_distractors)
        if num_distractors + needed <= max_distractors:
            num_distractors += needed
        else:
            return None

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors) if num_distractors else []

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(grid, fill_order, k, distractor_cells)

    try:
        if np.array_equal(main(inp), out):
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

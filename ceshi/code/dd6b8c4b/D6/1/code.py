# Variant of dd6b8c4b
# Dimension: D6 — 目标容器（绿色方块） (dst)
# Change: 绿色3x3方块（内含红色中心像素）——作为收集容器，内部空间被收集到的栗色像素填充... → 多个目标容器

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


"""
design_analysis:
  source_task: none
  variant_core:
    preserved:
      - 11x11 fixed canvas
      - orange floor, optional magenta walls, maroon movable tokens
      - each target still uses a red center inside a green 3x3 block
      - reachability is still decided by 4-connected flood-fill through floor/maroon
      - each container still fills in row-major order using its closest assigned maroons
    changed:
      - target container count changes from exactly one to exactly two
      - each red center defines its own 3x3 destination block
      - a maroon reachable from both containers is assigned to the nearer container
      - tie-break for equal distance uses container-center row-major order

  concrete_layout:
    containers:
      count: exactly 2
      shape: two non-overlapping 3x3 blocks
      colors: green ring (3) with red center (2)
      centers: chosen from corner anchors [(2,2), (2,8), (8,2), (8,8)]
      reason: this keeps the two objects visually obvious and leaves enough floor between them
    walls:
      style: optional local wall rings around each block with 2-4 corridor gaps and short outward arms
      reason: preserves the original flood-fill logic without overcrowding the 11x11 grid
    tokens:
      sources: maroon cells placed on floor cells owned by one of the two containers
      distractors: optional maroon cells unreachable from both containers

  behavior_trace:
    original_single_container:
      - compute one BFS distance map from the unique destination block
      - sort all reachable maroons by that one distance
      - move the first k into the single 3x3 block
    new_multi_container:
      - compute one BFS distance map per destination block
      - for each maroon, compare the two distances and assign the maroon to the nearer block
      - inside each block, sort only that block's assigned maroons and fill its own 3x3 cells
    concrete_example:
      - if a central maroon can reach both blocks, the old solver would have no notion of competition
      - the new solver must decide which block owns it before any filling happens
      - therefore a single global ranking is no longer sufficient; ownership and per-block ranking are both required

  complexity_calibration:
    choices:
      - fixed container count = 2 instead of 2-4
      - fixed container size = 3x3 as in the seed task
      - fixed assignment rule = nearest BFS distance, ties by center row-major
    why:
      - two containers are enough to make the variant visually distinct
      - more containers on 11x11 would make wall corridors and fills cluttered
      - a single deterministic ownership rule avoids ambiguous training examples

  implicit_assumption_audit_against_original_main:
    - original `assert len(centers) == 1` depends on uniqueness of the red center; invalid now
    - original single `dest_cells` list assumes one collection area; invalid now
    - original single BFS distance map assumes every reachable maroon belongs to the same target; invalid now
    - original `chosen = maroons[:k]` assumes one global quota; invalid now because each block needs its own quota
    - original erase/fill loop is safe only because there is one destination; now erasing and filling must happen per owner block

  placement_legality:
    guarantees:
      - the two 3x3 blocks never overlap and always stay in-bounds
      - wall cells are never written on top of destination cells
      - chosen maroon sources and distractors are sampled from floor cells only
      - every accepted grid has at least one passable opening next to each destination

  visual_observability:
    enforced_signals:
      - every sample contains two red centers and two green 3x3 blocks
      - every accepted sample contains at least one floor cell reachable from both containers
      - at least one moved maroon comes from such an overlap region, so multi-target ownership is exercised
    note:
      - no color-id-based hidden ordering is used; all sorting keys are geometric (distance, row, col)

  neatness_and_color_reasoning:
    - colors are unchanged from the seed task, so the new idea is isolated to destination multiplicity
    - corner placement keeps the central area readable instead of stacking markers in one crowded region

  new_solver_assumptions_to_verify:
    - every container has at least one adjacent passable cell
    - non-special colors contain at least the floor color, with floor being the most common such color
    - destination blocks do not overlap
    generator_support:
      - local wall generation always leaves 2-4 ring gaps
      - the floor occupies most non-special cells
      - centers are chosen from well-separated anchors
"""


GRID_H, GRID_W = 11, 11
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

CENTER_OPTIONS = [(2, 2), (2, 8), (8, 2), (8, 8)]
NUM_CONTAINERS = 2

WALL_PROB = 0.8
RING_GAP_RANGE = (2, 4)
ARM_EXTEND_PROB = 0.35
ARM_LENGTH_RANGE = (1, 3)

FILL_RANGE_PER_CONTAINER = (2, 6)
DISTRACTOR_MAX = 4
MIN_SHARED_REACHABLE = 1
GEN_RETRIES = 400
INF = 10**9


def _dest_cells(center, h=GRID_H, w=GRID_W):
    cr, cc = center
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _infer_floor(grid: np.ndarray) -> int:
    specials = {RED_CENTER, DEST_COLOR, MAROON}
    colors = [int(c) for c in np.unique(grid) if int(c) not in specials]
    assert colors, "grid must contain a non-special background color"
    return max(colors, key=lambda c: int((grid == c).sum()))


def _bfs_distance(grid: np.ndarray, dest_set: set[tuple[int, int]], passable: set[int]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), INF, dtype=int)
    q = deque()

    for r, c in sorted(dest_set):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if dist[nr, nc] == INF:
                    dist[nr, nc] = 0
                    q.append((nr, nc))

    assert q, "each destination block needs at least one adjacent passable opening"

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == INF:
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    return dist


def _assign_owner(r: int, c: int, centers, dists):
    best = None
    for idx, center in enumerate(centers):
        d = int(dists[idx][r, c])
        if d >= INF:
            continue
        key = (d, center[0], center[1], idx)
        if best is None or key < best[0]:
            best = (key, idx)
    return None if best is None else best[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route reachable maroon pixels into multiple 3×3 destination blocks around red centers.

    Input structure:
    - bg = orange floor, optional magenta walls, scattered maroon (9) pixels.
    - There are two green 3×3 destination blocks, each marked by its own red (2) center.

    Rule:
    - For each destination block, compute reachability through floor/maroon cells, stopping at walls.
    - Every reachable maroon is assigned to the nearest destination block; equal distances go to the earlier center in row-major order.
    - Inside each block, assigned maroons are ranked by distance to that block and fill the block in row-major order; moved maroons are erased back to floor.

    Key insight: The task is no longer a single flood-fill. You must compare multiple BFS frontiers, decide ownership in overlapping reachable zones, and only then perform the row-major refill inside each separate container.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == RED_CENTER))
    assert len(centers) >= 2, "grid must contain multiple red centers for the multi-container variant"

    containers = []
    for center in centers:
        dest_cells = _dest_cells(center, h, w)
        containers.append((center, dest_cells, set(dest_cells)))

    floor = _infer_floor(grid)
    passable = {floor, MAROON}
    dists = [_bfs_distance(grid, dest_set, passable) for _, _, dest_set in containers]

    owned_maroons = [[] for _ in containers]
    for r, c in np.argwhere(grid == MAROON):
        r, c = int(r), int(c)
        owner = _assign_owner(r, c, centers, dists)
        if owner is None:
            continue
        owned_maroons[owner].append((int(dists[owner][r, c]), r, c))

    for idx, (_, dest_cells, _) in enumerate(containers):
        owned_maroons[idx].sort()
        chosen = owned_maroons[idx][: len(dest_cells)]
        for _, r, c in chosen:
            out[r, c] = floor
        for (r, c), _src in zip(dest_cells, chosen):
            out[r, c] = MAROON

    return out


def _reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited = set()
    queue = deque()
    for r, c in sorted(dest_set):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited and grid[nr, nc] == FLOOR_COLOR:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited


def _outward_dirs(center, r, c):
    cr, cc = center
    dirs = []
    if r < cr:
        dirs.append((-1, 0))
    if r > cr:
        dirs.append((1, 0))
    if c < cc:
        dirs.append((0, -1))
    if c > cc:
        dirs.append((0, 1))
    return dirs


def _build_local_walls(grid: np.ndarray, center, all_dest_set: set[tuple[int, int]]) -> None:
    dest_set = set(_dest_cells(center))
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)

    num_gaps = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring, num_gaps))
    wall_cells = {cell for cell in ring if cell not in gap_cells}

    for r, c in list(wall_cells):
        dirs = _outward_dirs(center, r, c)
        if dirs and random.random() < ARM_EXTEND_PROB:
            dr, dc = random.choice(dirs)
            nr, nc = r, c
            for _ in range(random.randint(*ARM_LENGTH_RANGE)):
                nr, nc = nr + dr, nc + dc
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in all_dest_set:
                    wall_cells.add((nr, nc))
                else:
                    break

    for r, c in wall_cells:
        if (r, c) not in all_dest_set:
            grid[r, c] = WALL_COLOR


def _distance_maps_for_generation(grid: np.ndarray, centers):
    passable = {FLOOR_COLOR}
    dists = []
    for center in centers:
        dists.append(_bfs_distance(grid, set(_dest_cells(center)), passable))
    return dists


def _gen_attempt():
    centers = sorted(random.sample(CENTER_OPTIONS, NUM_CONTAINERS))
    all_dest_cells = set()
    for center in centers:
        all_dest_cells.update(_dest_cells(center))

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for center in centers:
        for r, c in _dest_cells(center):
            base[r, c] = DEST_COLOR
        base[center] = RED_CENTER

    if random.random() < WALL_PROB:
        for center in centers:
            _build_local_walls(base, center, all_dest_cells)

    try:
        dists = _distance_maps_for_generation(base, centers)
    except AssertionError:
        return None

    owner_buckets = [[] for _ in centers]
    contested_buckets = [[] for _ in centers]
    unreachable = []

    for r in range(GRID_H):
        for c in range(GRID_W):
            if base[r, c] != FLOOR_COLOR:
                continue
            reachable = []
            for idx, center in enumerate(centers):
                d = int(dists[idx][r, c])
                if d < INF:
                    reachable.append((d, center[0], center[1], idx))
            if not reachable:
                unreachable.append((r, c))
                continue
            reachable.sort()
            owner = reachable[0][3]
            item = (reachable[0][0], r, c)
            owner_buckets[owner].append(item)
            if len(reachable) > 1:
                contested_buckets[owner].append(item)

    if any(len(bucket) < FILL_RANGE_PER_CONTAINER[0] for bucket in owner_buckets):
        return None

    shared_total = sum(len(bucket) for bucket in contested_buckets)
    if shared_total < MIN_SHARED_REACHABLE:
        return None

    fill_counts = []
    for bucket in owner_buckets:
        hi = min(FILL_RANGE_PER_CONTAINER[1], len(bucket))
        if hi < FILL_RANGE_PER_CONTAINER[0]:
            return None
        fill_counts.append(random.randint(FILL_RANGE_PER_CONTAINER[0], hi))

    forced_owner_choices = [idx for idx, bucket in enumerate(contested_buckets) if bucket]
    if not forced_owner_choices:
        return None
    forced_owner = random.choice(forced_owner_choices)

    selected_by_owner = [[] for _ in centers]
    forced_item = random.choice(contested_buckets[forced_owner])
    selected_by_owner[forced_owner].append(forced_item)

    used_cells = {(forced_item[1], forced_item[2])}
    for idx, bucket in enumerate(owner_buckets):
        need = fill_counts[idx] - len(selected_by_owner[idx])
        available = [item for item in bucket if (item[1], item[2]) not in used_cells]
        if len(available) < need:
            return None
        chosen = random.sample(available, need)
        selected_by_owner[idx].extend(chosen)
        used_cells.update((r, c) for _, r, c in chosen)

    num_distractors = random.randint(0, min(DISTRACTOR_MAX, len(unreachable)))
    distractors = random.sample(unreachable, num_distractors)

    inp = base.copy()
    out = base.copy()

    for idx, center in enumerate(centers):
        selected_by_owner[idx].sort()
        for _, r, c in selected_by_owner[idx]:
            inp[r, c] = MAROON
        dest_cells = _dest_cells(center)
        for (r, c), _src in zip(dest_cells, selected_by_owner[idx]):
            out[r, c] = MAROON

    for r, c in distractors:
        inp[r, c] = MAROON
        out[r, c] = MAROON

    try:
        result = main(inp)
    except Exception:
        return None
    if np.array_equal(result, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

# Combo variant of dd6b8c4b (index 10)
# combo_spec: D6/1+D12/1
# Name: 多个目标容器+多对多映射

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


"""
design_analysis:
  source_task: dd6b8c4b
  fused_variant:
    preserved:
      - fixed 11x11 canvas
      - floor/wall/maroon color roles stay the same
      - each destination is still a green 3x3 block with a red center
      - reachability still uses 4-connected BFS through floor+maroon cells
      - each destination still fills in row-major order with its owned maroons
    changed:
      - there are multiple destination blocks instead of exactly one
      - the mapping is many:many: many maroons may feed many blocks, and a block
        competes with the others for shared maroons
      - ownership is decided globally by nearest BFS distance, with center
        row-major order as the deterministic tie-break

  concrete_layout:
    containers:
      count: exactly 3
      centers: chosen from the four corner anchors [(2,2), (2,8), (8,2), (8,8)]
      reason:
        - three containers are visually stronger than two for "many:many"
        - corner anchors keep all 3x3 blocks disjoint and readable on 11x11
    walls:
      style: local wall rings with 2-4 gaps per container, plus short outer arms
      reason:
        - preserves the original flood-fill flavor
        - keeps the shared central floor open, so overlap/competition is visible
    tokens:
      sources: sampled only from floor cells and assigned by nearest-container BFS
      distractors: optional unreachable maroons that remain in place

  behavior_trace:
    original:
      - one red center defines one 3x3 container
      - one BFS distance map ranks all reachable maroons
      - the first k maroons fill that single container
    fused:
      - three red centers define three separate 3x3 containers
      - compute one BFS distance map per container
      - every maroon compares all reachable containers and is owned by the nearest
      - each container fills only from its owned maroons
    small_example:
      - suppose a maroon lies in the central corridor and is reachable from the
        top-left and top-right containers
      - if it is one step closer to the top-left container, it counts only for
        that container even though both can reach it
      - therefore the top-right container may visibly fill fewer cells than an
        "independent per-container" interpretation would predict
      - this is the key visible effect of D12/1; ownership changes counts, not
        just which source cell disappears

  complexity_calibration:
    choices:
      - fixed container count = 3
      - fixed container size = original 3x3
      - fixed ownership rule = nearest BFS distance, then center row-major
    why:
      - three containers make the many:many competition obvious
      - more than three would clutter the 11x11 board and obscure corridors
      - a single deterministic owner rule avoids ambiguous training pairs

  implicit_assumption_audit_against_original_main:
    - original `assert len(centers) == 1` is invalid; now multiple red centers are required
    - original single `dest_cells` / single `dest_set` logic no longer works
    - original single BFS distance map is invalid because ownership must compare
      distances to multiple containers
    - original global `maroons[:k]` slice is invalid because each container has
      its own quota based on owned sources
    - original erase/fill loops were safe for one container only; now they must
      be repeated per container after ownership is fixed

  fused_observability_guard:
    issue:
      - merely selecting contested maroons is not enough, because the output only
        shows filled destination cells and erased sources
    enforcement:
      - the generator rejects cases unless at least one container can see more
        input maroons than it actually owns after competition
      - this guarantees the visible fill counts differ from a degenerate
        "every container uses every reachable maroon independently" reading

  placement_legality:
    guarantees:
      - chosen anchors keep 3x3 destination blocks in-bounds and non-overlapping
      - wall cells are never written on destination cells
      - maroons are sampled only from floor cells
      - every destination keeps at least one adjacent floor opening

  visual_observability:
    enforced_signals:
      - every sample contains three red centers / green blocks
      - at least two containers receive contested maroons
      - at least one container loses visible fill capacity because some reachable
        maroons are owned by another container
    note:
      - all ordering uses visible geometry only: BFS distance, row, col, center order

  neatness_and_color_reasoning:
    - colors are unchanged from the seed task, so the new difficulty comes only
      from multi-container competition
    - corner placement avoids stacking multiple markers in one crowded region

  new_solver_assumptions_to_verify:
    - all red centers define valid 3x3 destination blocks
    - floor is the most common non-special color
    - each destination has at least one adjacent passable cell
    - destination blocks do not overlap
    generator_support:
      - centers come from separated corner anchors
      - wall rings always leave 2-4 openings with at least one inward gap
      - accepted samples are filtered by explicit competition metrics
"""


GRID_H, GRID_W = 11, 11
MID_R, MID_C = GRID_H // 2, GRID_W // 2

FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

CENTER_OPTIONS = [(2, 2), (2, 8), (8, 2), (8, 8)]
NUM_CONTAINERS = 3

RING_GAP_RANGE = (2, 4)
ARM_EXTEND_PROB = 0.25
ARM_LENGTH_RANGE = (1, 2)

FILL_RANGE_PER_CONTAINER = (2, 4)
DISTRACTOR_MAX = 4
MIN_CONTESTED_OWNERS = 2
MIN_COMPETITION_GAP_SUM = 2
GEN_RETRIES = 600
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
    assert colors, "grid must contain at least one non-special background color"
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

    assert q, "each destination block needs at least one adjacent passable cell"

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


def _ring_cells(dest_set: set[tuple[int, int]]):
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    return sorted(ring)


def _away_from_grid_center_dirs(r: int, c: int):
    dirs = []
    if r < MID_R:
        dirs.append((-1, 0))
    if r > MID_R:
        dirs.append((1, 0))
    if c < MID_C:
        dirs.append((0, -1))
    if c > MID_C:
        dirs.append((0, 1))
    return dirs


def _build_local_walls(grid: np.ndarray, center, all_dest_set: set[tuple[int, int]]) -> None:
    dest_set = set(_dest_cells(center))
    ring = _ring_cells(dest_set)

    inward_rank = sorted(
        ring,
        key=lambda rc: (abs(rc[0] - MID_R) + abs(rc[1] - MID_C), rc[0], rc[1]),
    )
    inward_pool = inward_rank[: min(4, len(inward_rank))]
    gap_cells = {random.choice(inward_pool)}

    total_gaps = random.randint(*RING_GAP_RANGE)
    remaining_ring = [cell for cell in ring if cell not in gap_cells]
    extra_gaps = total_gaps - len(gap_cells)
    if extra_gaps > 0:
        gap_cells.update(random.sample(remaining_ring, extra_gaps))

    wall_cells = {cell for cell in ring if cell not in gap_cells}

    center_dist = abs(center[0] - MID_R) + abs(center[1] - MID_C)
    for r, c in list(wall_cells):
        ring_dist = abs(r - MID_R) + abs(c - MID_C)
        if ring_dist <= center_dist:
            continue
        if random.random() >= ARM_EXTEND_PROB:
            continue
        dirs = _away_from_grid_center_dirs(r, c)
        if not dirs:
            continue
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
    dists = []
    passable = {FLOOR_COLOR}
    for center in centers:
        dists.append(_bfs_distance(grid, set(_dest_cells(center)), passable))
    return dists


def _cell_reachability(centers, dists, r: int, c: int):
    reachable = []
    for idx, center in enumerate(centers):
        d = int(dists[idx][r, c])
        if d < INF:
            reachable.append((d, center[0], center[1], idx))
    reachable.sort()
    return reachable


def _classify_floor_cells(base: np.ndarray, centers, dists):
    owner_buckets = [[] for _ in centers]
    unique_buckets = [[] for _ in centers]
    contested_buckets = [[] for _ in centers]
    unreachable = []
    reach_sets = {}

    for r in range(GRID_H):
        for c in range(GRID_W):
            if base[r, c] != FLOOR_COLOR:
                continue
            reachable = _cell_reachability(centers, dists, r, c)
            if not reachable:
                unreachable.append((r, c))
                continue

            owner = reachable[0][3]
            item = (reachable[0][0], r, c)
            reach_sets[(r, c)] = tuple(x[3] for x in reachable)
            owner_buckets[owner].append(item)
            if len(reachable) == 1:
                unique_buckets[owner].append(item)
            else:
                contested_buckets[owner].append(item)

    return owner_buckets, unique_buckets, contested_buckets, unreachable, reach_sets


def _competition_metrics(selected_by_owner, reach_sets, num_containers: int):
    selected_cells = []
    contested_selected = [0] * num_containers
    owned_counts = [len(bucket) for bucket in selected_by_owner]

    for idx, bucket in enumerate(selected_by_owner):
        for _, r, c in bucket:
            selected_cells.append((r, c))
            if len(reach_sets[(r, c)]) > 1:
                contested_selected[idx] += 1

    visible_counts = [0] * num_containers
    for r, c in selected_cells:
        for idx in reach_sets[(r, c)]:
            visible_counts[idx] += 1

    gaps = [visible_counts[idx] - owned_counts[idx] for idx in range(num_containers)]
    return contested_selected, visible_counts, gaps


def analyze_input(grid: np.ndarray) -> dict:
    grid = np.asarray(grid, dtype=int)
    centers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == RED_CENTER))
    floor = _infer_floor(grid)
    passable = {floor, MAROON}
    dists = [_bfs_distance(grid, set(_dest_cells(center, *grid.shape)), passable) for center in centers]

    owned_counts = [0] * len(centers)
    independent_counts = [0] * len(centers)
    contested_owned_counts = [0] * len(centers)

    for r, c in np.argwhere(grid == MAROON):
        r, c = int(r), int(c)
        reachable = _cell_reachability(centers, dists, r, c)
        if not reachable:
            continue
        owner = reachable[0][3]
        owned_counts[owner] += 1
        if len(reachable) > 1:
            contested_owned_counts[owner] += 1
        for _, _, _, idx in reachable:
            independent_counts[idx] += 1

    competition_gap = [independent_counts[i] - owned_counts[i] for i in range(len(centers))]
    return {
        "num_containers": len(centers),
        "owned_counts": owned_counts,
        "independent_counts": independent_counts,
        "contested_owned_counts": contested_owned_counts,
        "competition_gap": competition_gap,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route reachable maroon pixels into several 3x3 destination blocks around red centers.

    Input structure:
    - The board is an 11x11 floor-and-wall maze with scattered maroon (9) pixels.
    - There are multiple green 3x3 destination blocks, each marked by its own red (2) center.

    Rule:
    - For every destination block, compute BFS reachability through floor/maroon cells, stopping at walls.
    - Each reachable maroon is owned by the nearest destination block; equal distances break by center row-major order.
    - Every block sorts only its owned maroons by distance and writes them into its own 3x3 cells in row-major order, erasing the moved sources back to floor.

    Key insight:
    - This is a competitive multi-target collection task, not independent per-block filling.
    - Shared corridors create maroons that more than one block can reach, so you must compare several BFS frontiers first and only then refill each block from its assigned pool.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == RED_CENTER))
    assert len(centers) >= 2, "grid must contain multiple red centers"

    containers = []
    occupied = set()
    for center in centers:
        dest_cells = _dest_cells(center, h, w)
        dest_set = set(dest_cells)
        assert not occupied.intersection(dest_set), "destination blocks must not overlap"
        occupied.update(dest_set)
        containers.append((center, dest_cells, dest_set))

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


def _gen_attempt():
    centers = sorted(random.sample(CENTER_OPTIONS, NUM_CONTAINERS))
    all_dest_set = set()
    for center in centers:
        all_dest_set.update(_dest_cells(center))

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for center in centers:
        for r, c in _dest_cells(center):
            base[r, c] = DEST_COLOR
        base[center] = RED_CENTER

    for center in centers:
        _build_local_walls(base, center, all_dest_set)

    try:
        dists = _distance_maps_for_generation(base, centers)
    except AssertionError:
        return None

    owner_buckets, _unique_buckets, contested_buckets, unreachable, reach_sets = _classify_floor_cells(base, centers, dists)

    if any(len(bucket) < FILL_RANGE_PER_CONTAINER[0] for bucket in owner_buckets):
        return None

    contested_owners = [idx for idx, bucket in enumerate(contested_buckets) if bucket]
    if len(contested_owners) < MIN_CONTESTED_OWNERS:
        return None

    fill_counts = []
    for bucket in owner_buckets:
        hi = min(FILL_RANGE_PER_CONTAINER[1], len(bucket))
        if hi < FILL_RANGE_PER_CONTAINER[0]:
            return None
        fill_counts.append(random.randint(FILL_RANGE_PER_CONTAINER[0], hi))

    selected_by_owner = [[] for _ in centers]
    used_cells = set()

    forced_owners = random.sample(contested_owners, MIN_CONTESTED_OWNERS)
    for idx in forced_owners:
        item = random.choice(contested_buckets[idx])
        selected_by_owner[idx].append(item)
        used_cells.add((item[1], item[2]))

    for idx, bucket in enumerate(owner_buckets):
        need = fill_counts[idx] - len(selected_by_owner[idx])
        if need < 0:
            return None

        contested_available = [item for item in contested_buckets[idx] if (item[1], item[2]) not in used_cells]
        if need > 0 and contested_available and random.random() < 0.6:
            item = random.choice(contested_available)
            selected_by_owner[idx].append(item)
            used_cells.add((item[1], item[2]))
            need -= 1

        available = [item for item in bucket if (item[1], item[2]) not in used_cells]
        if len(available) < need:
            return None
        chosen = random.sample(available, need)
        selected_by_owner[idx].extend(chosen)
        used_cells.update((r, c) for _, r, c in chosen)

    contested_selected, visible_counts, gaps = _competition_metrics(selected_by_owner, reach_sets, len(centers))
    if sum(1 for x in contested_selected if x > 0) < MIN_CONTESTED_OWNERS:
        return None
    if max(gaps) < 1 or sum(gaps) < MIN_COMPETITION_GAP_SUM:
        return None

    distractor_count = random.randint(0, min(DISTRACTOR_MAX, len(unreachable)))
    distractors = random.sample(unreachable, distractor_count)

    inp = base.copy()
    out = base.copy()

    for idx, center in enumerate(centers):
        selected_by_owner[idx].sort()
        for _, r, c in selected_by_owner[idx]:
            inp[r, c] = MAROON
        for (r, c), _src in zip(_dest_cells(center), selected_by_owner[idx]):
            out[r, c] = MAROON

    for r, c in distractors:
        inp[r, c] = MAROON
        out[r, c] = MAROON

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, out):
        return None

    metrics = analyze_input(inp)
    if sum(1 for x in metrics["contested_owned_counts"] if x > 0) < MIN_CONTESTED_OWNERS:
        return None
    if max(metrics["competition_gap"]) < 1 or sum(metrics["competition_gap"]) < MIN_COMPETITION_GAP_SUM:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

# Combo variant of dd6b8c4b (index 31)
# combo_spec: D6/1+D12/1+D14/2
# Name: 多容器+多对多映射+框架方块分离

from __future__ import annotations

import random
import sys
from collections import deque

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import neighbors4


"""
design_analysis:
  source_task: dd6b8c4b
  combo_spec: D6/1 + D12/1 + D14/2
  combo_name: 多容器+多对多映射+框架方块分离

  variant_core:
    preserved:
      - canvas stays 11x11
      - colors stay orange floor(7), magenta frame/wall(6), green destination ring(3), red center(2), maroon tokens(9)
      - movement is still based on 4-connected flood-fill through floor/maroon cells
      - each destination block still fills in row-major order with the closest assigned maroons
    changed:
      - there are now two destination containers instead of one
      - maroons are assigned many:many: each token is claimed by the nearest reachable container
      - the magenta obstacle is no longer adjacent to either block; it is a detached central frame
      - examples must show both private ownership regions and a shared competitive region

  concrete_layout:
    containers:
      count: exactly 2
      shape: two non-overlapping 3x3 green blocks with red centers
      centers: one diagonal pair chosen from [(1,1),(9,9)] or [(1,9),(9,1)]
      reason: corner anchors keep the blocks visually separated and leave a clean middle area
    frame:
      shape: one detached 3x3 magenta square frame at rows 4..6 and cols 4..6
      separation: every destination cell is at least Manhattan distance 4 from the frame
      role: it is an independent obstacle that bends shortest paths and encloses an unreachable interior cell
    token_regions:
      - each container gets at least one selected maroon from a region that is strictly nearer to that container
      - at least one selected maroon comes from an overlap region reachable from both containers
      - optional distractor maroons may appear inside the detached frame, unreachable from both containers

  behavior_trace:
    original_single_container:
      - find the unique red center and its 3x3 block
      - flood-fill from cells adjacent to that one block
      - globally rank reachable maroons and pour them into one container
    fused_combo:
      - find two red centers and build two 3x3 destination blocks
      - run one BFS distance map per block, with the detached central frame acting as a blocker
      - for each maroon, compare both distance maps and assign the token to the nearer block
      - each block independently sorts its own assigned maroons and fills its own cells in row-major order
    concrete_example:
      - with centers at (1,1) and (9,9), a maroon near the upper-left corridor belongs only to the first block
      - a maroon near the lower-right corridor belongs only to the second block
      - a maroon near the middle lanes around the detached frame can be reached from both, so ownership must be decided before filling
      - this differs from the seed task because no single global ordering is sufficient anymore

  complexity_calibration:
    choices:
      - use exactly 2 containers, not 3-4
      - use exactly 1 detached frame, always in the middle
      - use one deterministic ownership rule: nearest BFS distance, ties by center row-major order
    why:
      - two containers are enough to make many:many competition visible without overcrowding 11x11
      - a fixed detached frame makes the D14 change obvious across a few examples
      - deterministic ownership keeps examples learnable from small ARC training sets

  implicit_assumption_audit_against_original_main:
    - original `assert len(centers) == 1` is invalid because the combo requires multiple red centers
    - original single `dest_cells` / `dest_set` object is invalid because each block now has its own fill order
    - original floor inference from the immediate ring around the block is fragile because the frame is no longer adjacent there
    - original single BFS distance map is invalid because many:many assignment needs one map per container
    - original global `maroons[:k]` selection is invalid because each container has its own quota and ownership list
    - original erase/fill loop assumed one output container; now erasing and filling must be done per owner

  placement_legality:
    guarantees:
      - the chosen diagonal centers keep both 3x3 blocks fully in-bounds and non-overlapping
      - the central frame never overlaps a destination block and is never edge-adjacent to one
      - source maroons are sampled only from floor cells outside blocks and outside the frame boundary
      - optional distractors are sampled only from unreachable interior frame cells
      - every accepted layout is validated by the solver before being returned

  visual_observability:
    enforced_signals:
      - every sample visibly contains two green/red containers
      - every sample visibly contains a detached magenta frame in the middle rather than a tight ring around a block
      - every accepted sample has maroons owned by both containers
      - every accepted sample includes at least one overlap maroon reachable from both containers
    no_hidden_features:
      - ordering uses only BFS distance and row-major geometry
      - no color-id ordering or invisible metadata is used

  neatness_and_inference:
    - corners for the two containers and the center for the frame produce a clear three-object composition
    - the color palette is unchanged from the seed task, so the new behavior is isolated to structure and ownership
    - forcing one private maroon per container plus one contested maroon keeps the rule readable from a few examples

  new_solver_assumptions_to_verify:
    - exactly two non-overlapping 3x3 blocks exist, each marked by a red center
    - the floor is the most common non-special color
    - magenta cells are hard blockers for both BFS maps
    - each block has at least one adjacent passable cell
    generator_support:
      - centers are sampled only from the two diagonal corner pairs
      - the central detached frame is fixed and separated from both blocks
      - accepted samples require strict-nearest ownership for both blocks and overlap ownership overall
      - solver validation is the final acceptance gate
"""


GRID_H, GRID_W = 11, 11
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9
INF = 10**9

CENTER_PAIRS = [
    [(1, 1), (9, 9)],
    [(1, 9), (9, 1)],
]
FRAME_BOX = (4, 4, 6, 6)  # inclusive bbox of the detached 3x3 frame

FILL_RANGE_PER_CONTAINER = (2, 5)
DISTRACTOR_PROB = 0.6
GEN_RETRIES = 500


def _dest_cells(center: tuple[int, int], h: int = GRID_H, w: int = GRID_W) -> list[tuple[int, int]]:
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


def _frame_boundary_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                cells.append((r, c))
    cells.sort()
    return cells


def _frame_interior_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            cells.append((r, c))
    cells.sort()
    return cells


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

    assert q, "each detached destination block needs at least one adjacent passable cell"

    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == INF:
                dist[nr, nc] = d + 1
                q.append((nr, nc))

    return dist


def _assign_owner(r: int, c: int, centers, dists) -> int | None:
    best = None
    for idx, center in enumerate(centers):
        d = int(dists[idx][r, c])
        if d >= INF:
            continue
        key = (d, center[0], center[1], idx)
        if best is None or key < best[0]:
            best = (key, idx)
    return None if best is None else best[1]


def _measure_input_features(grid: np.ndarray) -> dict[str, int]:
    centers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == RED_CENTER))
    floor = _infer_floor(grid)
    passable = {floor, MAROON}
    containers = [(center, _dest_cells(center), set(_dest_cells(center))) for center in centers]
    dists = [_bfs_distance(grid, dest_set, passable) for _, _, dest_set in containers]

    overlap_maroons = 0
    strict_owner_maroons = [0 for _ in containers]
    for r, c in np.argwhere(grid == MAROON):
        r, c = int(r), int(c)
        reaches = []
        for idx in range(len(containers)):
            d = int(dists[idx][r, c])
            if d < INF:
                reaches.append((d, idx))
        if len(reaches) > 1:
            overlap_maroons += 1
        if reaches:
            reaches.sort()
            owner = reaches[0][1]
            if len(reaches) == 1 or reaches[0][0] < reaches[1][0]:
                strict_owner_maroons[owner] += 1

    frame_gap = INF
    frame_cells = _frame_boundary_cells(FRAME_BOX)
    for _, dest_cells, _ in containers:
        for r0, c0 in dest_cells:
            for r1, c1 in frame_cells:
                frame_gap = min(frame_gap, abs(r0 - r1) + abs(c0 - c1))

    return {
        "num_centers": len(centers),
        "overlap_maroons": overlap_maroons,
        "strict0": strict_owner_maroons[0] if strict_owner_maroons else 0,
        "strict1": strict_owner_maroons[1] if len(strict_owner_maroons) > 1 else 0,
        "frame_gap": int(frame_gap),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull maroon pixels into two separated 3x3 containers, each centered on a red marker.

    Input structure:
    - The background is orange floor with scattered maroon source pixels and a detached magenta frame in the middle.
    - Two green 3x3 destination blocks appear in opposite corners, each identified by its own red center.

    Rule:
    - For each destination block, run a flood-fill distance search through floor/maroon cells, with magenta cells acting as blockers.
    - Each reachable maroon is assigned to the nearest block; equal distances are broken by the row-major order of the red centers.
    - Within each block, its assigned maroons are sorted by distance then row-major position, erased from their source cells, and written into that block in row-major order.

    Key insight:
    - The detached frame changes path lengths without enclosing either destination, so the task is not local to one block anymore.
    - You must combine two BFS maps with an ownership step: first decide which block owns each maroon, then fill the two containers independently.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = sorted((int(r), int(c)) for r, c in np.argwhere(grid == RED_CENTER))
    assert len(centers) == 2, "this combo expects exactly two red centers"

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


def _distance_maps_for_generation(grid: np.ndarray, centers):
    passable = {FLOOR_COLOR}
    return [_bfs_distance(grid, set(_dest_cells(center)), passable) for center in centers]


def _base_grid(centers) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for center in centers:
        for r, c in _dest_cells(center):
            grid[r, c] = DEST_COLOR
        grid[center] = RED_CENTER
    for r, c in _frame_boundary_cells(FRAME_BOX):
        grid[r, c] = WALL_COLOR
    return grid


def _classify_floor_cells(grid: np.ndarray, centers):
    dists = _distance_maps_for_generation(grid, centers)
    owner_buckets = [[] for _ in centers]
    strict_owner_buckets = [[] for _ in centers]
    overlap_buckets = [[] for _ in centers]
    unreachable = []

    for r in range(GRID_H):
        for c in range(GRID_W):
            if grid[r, c] != FLOOR_COLOR:
                continue
            reaches = []
            for idx, center in enumerate(centers):
                d = int(dists[idx][r, c])
                if d < INF:
                    reaches.append((d, center[0], center[1], idx))
            if not reaches:
                unreachable.append((r, c))
                continue
            reaches.sort()
            owner = reaches[0][3]
            item = (reaches[0][0], r, c)
            owner_buckets[owner].append(item)
            if len(reaches) == 1 or reaches[0][0] < reaches[1][0]:
                strict_owner_buckets[owner].append(item)
            else:
                overlap_buckets[owner].append(item)

    return dists, owner_buckets, strict_owner_buckets, overlap_buckets, unreachable


def _pick_sources(owner_buckets, strict_owner_buckets, overlap_buckets):
    fill_counts = []
    for bucket in owner_buckets:
        hi = min(FILL_RANGE_PER_CONTAINER[1], len(bucket))
        if hi < FILL_RANGE_PER_CONTAINER[0]:
            return None
        fill_counts.append(random.randint(FILL_RANGE_PER_CONTAINER[0], hi))

    if any(not bucket for bucket in strict_owner_buckets):
        return None

    forced_owner_choices = [idx for idx, bucket in enumerate(overlap_buckets) if bucket]
    if not forced_owner_choices:
        return None

    selected_by_owner = [[] for _ in owner_buckets]
    used = set()

    overlap_owner = random.choice(forced_owner_choices)
    overlap_item = random.choice(overlap_buckets[overlap_owner])
    selected_by_owner[overlap_owner].append(overlap_item)
    used.add((overlap_item[1], overlap_item[2]))

    for idx, bucket in enumerate(strict_owner_buckets):
        strict_choices = [item for item in bucket if (item[1], item[2]) not in used]
        if not strict_choices:
            return None
        item = random.choice(strict_choices)
        selected_by_owner[idx].append(item)
        used.add((item[1], item[2]))

    for idx, bucket in enumerate(owner_buckets):
        need = fill_counts[idx] - len(selected_by_owner[idx])
        available = [item for item in bucket if (item[1], item[2]) not in used]
        if len(available) < need:
            return None
        extra = random.sample(available, need)
        selected_by_owner[idx].extend(extra)
        used.update((r, c) for _, r, c in extra)

    return selected_by_owner


def _expected_output(base: np.ndarray, centers, selected_by_owner, distractors):
    out = base.copy()
    for idx, center in enumerate(centers):
        dest_cells = _dest_cells(center)
        ordered_sources = sorted(selected_by_owner[idx])
        for (r, c), _src in zip(dest_cells, ordered_sources):
            out[r, c] = MAROON
    for r, c in distractors:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    centers = sorted(random.choice(CENTER_PAIRS))
    base = _base_grid(centers)

    try:
        _dists, owner_buckets, strict_owner_buckets, overlap_buckets, unreachable = _classify_floor_cells(base, centers)
    except AssertionError:
        return None

    selected_by_owner = _pick_sources(owner_buckets, strict_owner_buckets, overlap_buckets)
    if selected_by_owner is None:
        return None

    interior_cells = [cell for cell in _frame_interior_cells(FRAME_BOX) if cell in unreachable]
    distractors = []
    if interior_cells and random.random() < DISTRACTOR_PROB:
        distractors = [random.choice(interior_cells)]

    inp = base.copy()
    for owner_sources in selected_by_owner:
        for _, r, c in owner_sources:
            inp[r, c] = MAROON
    for r, c in distractors:
        inp[r, c] = MAROON

    out = _expected_output(base, centers, selected_by_owner, distractors)

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, out):
        return None

    features = _measure_input_features(inp)
    if features["overlap_maroons"] < 1:
        return None
    if min(features["strict0"], features["strict1"]) < 1:
        return None
    if features["frame_gap"] <= 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

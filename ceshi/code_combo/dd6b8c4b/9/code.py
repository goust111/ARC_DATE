# Combo variant of dd6b8c4b (index 9)
# combo_spec: D5/1+D8/2
# Name: 多色像素+距离梯度填充色

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - Keep the original 11x11 scene, fixed 3x3 destination centered on the unique red pixel,
#      orange floor, and optional magenta wall corridors.
#    - Replace the single maroon source class with multicolor single-pixel sources. This combo
#      uses only blue(1) and azure(8) as source colors so the input visibly contains multiple
#      movable colors without becoming noisy.
#    - The collectible set is still defined by wall-restricted BFS reachability. Example:
#      if reachable tokens sorted by (distance,row,col) are
#      [(0,3,5,color=1), (2,6,5,color=8), (5,7,8,color=1)],
#      then the original task would write [9,9,9], the D5-only variant would write [1,8,1],
#      but this combo writes the distance gradient [9,5,4]. Source color matters only for
#      identifying movable tokens and for showing that multiple token colors can be reachable;
#      once moved, the destination color is determined solely by distance.
# b) Complexity calibration:
#    - Source colors are limited to two recurring colors {1,8}. This is enough to make the
#      D5 change obvious while keeping the scene legible.
#    - Distance recoloring is limited to three fixed tiers:
#      dist 0-1 -> maroon(9), dist 2-3 -> gray(5), dist 4+ -> yellow(4).
#      A small repeated palette makes the heat-map rule inferable from a few examples.
# c) Original implicit assumptions that break:
#    - `passable = {floor, 9}` no longer works because movable tokens are not color-9 only.
#    - `np.argwhere(grid == 9)` no longer finds all candidates; every non-structural token color
#      must be collected.
#    - Writing the chosen token's own color into the destination is wrong for this combo because
#      D8/2 overrides token identity with a distance-dependent gradient.
#    - The original floor-color inference relied on maroon being the only non-structural color.
#      Here the structural colors are fixed by construction, so the solver uses constants.
# d) Placement legality:
#    - Generator samples all sources and distractors from floor cells outside the destination.
#    - Reachable sources come only from the BFS-reachable floor component; distractors come only
#      from unreachable floor cells. This prevents overlap, wall placement conflicts, and
#      out-of-bounds placement.
# e) Visual observability:
#    - Each accepted sample has both source colors among the reachable tokens and at least two
#      distinct distance tiers among the selected tokens, so both halves of the combo are visible.
#    - Gradient color depends on visible geometric distance through corridors, not on hidden color
#      ids or arbitrary per-example mappings.
# f) Visual cleanliness:
#    - Structural colors stay unchanged. Only two input token colors and three output gradient
#      colors are used, all confined to sparse floor pixels and the 3x3 destination block.
# g) Change inferability:
#    - Source colors and fill colors are intentionally disjoint: input uses {1,8}, output fill
#      uses {9,5,4}. This makes it obvious that tokens are being collected by location, but their
#      original colors are not preserved in the destination.
# h) New solver assumptions and generator guarantees:
#    - Assumption: all movable tokens use colors outside {2,3,6,7}. Generator enforces this.
#    - Assumption: at least one passable exterior cell touches the destination. Wall generation
#      always leaves 1-4 ring gaps, and validation rejects any broken layout.
#    - Assumption: reachable tokens span at least two gradient tiers. Generator seeds source
#      placement from multiple distance buckets and rejects degenerate single-tier cases.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2

SOURCE_PALETTE = (1, 8)
GRADIENT_COLORS = (9, 5, 4)

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

FILL_K_RANGE = (3, 9)
DISTRACTOR_MAX = 10
TOTAL_TOKEN_RANGE = (4, 16)
MIN_REACHABLE = 4
GEN_RETRIES = 400
INF = 10**9


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _distance_bucket(distance: int) -> int:
    if distance <= 1:
        return 0
    if distance <= 3:
        return 1
    return 2


def _gradient_color(distance: int) -> int:
    return GRADIENT_COLORS[_distance_bucket(distance)]


def _distance_map(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    passable: set[int],
) -> np.ndarray:
    h, w = grid.shape
    dest_set = set(dest_cells)
    dist = np.full((h, w), INF, dtype=int)
    queue = deque()

    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set:
                if int(grid[rr, cc]) in passable and dist[rr, cc] == INF:
                    dist[rr, cc] = 0
                    queue.append((rr, cc))

    assert queue, "there must be at least one passable cell adjacent to the destination block"

    while queue:
        r, c = queue.popleft()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == INF:
                if int(grid[rr, cc]) in passable:
                    dist[rr, cc] = int(dist[r, c]) + 1
                    queue.append((rr, cc))

    return dist


def _sorted_reachable_tokens(
    grid: np.ndarray,
    dist: np.ndarray,
    token_colors: set[int],
) -> list[tuple[int, int, int, int]]:
    tokens = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in token_colors and int(dist[r, c]) < INF:
                tokens.append((int(dist[r, c]), int(r), int(c), color))
    tokens.sort()
    return tokens


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable colored pixels into the 3x3 block around the unique red center, then recolor them by distance.

    Input structure:
    - Orange floor and optional magenta walls surround a fixed green 3x3 destination whose center is a single red pixel.
    - Blue and azure single-pixel source tokens are scattered on floor cells outside that block.

    Rule:
    - Flood-fill outward from passable cells touching the destination, moving through floor or any source token color.
    - Collect every reachable source token, sort them by distance to the destination and then by row/column, and place the first ones into the destination in row-major order.
    - Erase the moved source tokens back to floor, but fill each destination cell with a gradient color based on that token's BFS distance rather than its original source color.

    Key insight: The source colors still tell you which pixels are collectible and which corridor cells stay passable, but the destination ignores those source colors and instead encodes near/mid/far reachability as a small distance heat map.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    token_colors = {
        int(color)
        for color in np.unique(grid)
        if int(color) not in {FLOOR_COLOR, WALL_COLOR, DEST_COLOR, RED_CENTER}
    }
    assert token_colors, "grid must contain at least one non-structural token color"

    passable = {FLOOR_COLOR, *token_colors}
    dist = _distance_map(grid, dest_cells, passable)
    tokens = _sorted_reachable_tokens(grid, dist, token_colors)

    k = min(len(dest_cells), len(tokens))
    chosen = tokens[:k]

    for _, r, c, _ in chosen:
        out[r, c] = FLOOR_COLOR

    for (r, c), (distance, _, _, _) in zip(dest_cells[:k], chosen):
        out[r, c] = _gradient_color(distance)

    return out


def _reachable_floor_and_dist(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], np.ndarray]:
    dist = _distance_map(grid, dest_cells, {FLOOR_COLOR})
    dest_set = set(dest_cells)
    reachable = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if (r, c) not in dest_set and grid[r, c] == FLOOR_COLOR and int(dist[r, c]) < INF
    }
    return reachable, dist


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    dirs = []
    if r < CENTER_R:
        dirs.append((-1, 0))
    if r > CENTER_R:
        dirs.append((1, 0))
    if c < CENTER_C:
        dirs.append((0, -1))
    if c > CENTER_C:
        dirs.append((0, 1))
    return dirs


def _extend_arm(
    wall_cells: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    r: int,
    c: int,
    max_len: int,
) -> None:
    dirs = _outward_dirs(r, c)
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


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)

    gap_cells = set(random.sample(ring, random.randint(*RING_GAP_RANGE)))
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
                    nr += dr
                    nc += dc
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


def _assign_source_colors(count: int) -> list[int]:
    colors = [SOURCE_PALETTE[i % len(SOURCE_PALETTE)] for i in range(count)]
    random.shuffle(colors)
    return colors


def _sample_reachable_sources(
    reachable_floor: set[tuple[int, int]],
    dist: np.ndarray,
    k: int,
) -> list[tuple[int, int]]:
    bucket_cells: dict[int, list[tuple[int, int]]] = {0: [], 1: [], 2: []}
    for r, c in reachable_floor:
        bucket_cells[_distance_bucket(int(dist[r, c]))].append((r, c))

    nonempty = [bucket for bucket, cells in bucket_cells.items() if cells]
    if len(nonempty) < 2:
        return []

    target_bucket_count = 3 if len(nonempty) >= 3 and k >= 3 and random.random() < 0.7 else 2
    target_bucket_count = min(target_bucket_count, len(nonempty), k)
    active_buckets = random.sample(nonempty, target_bucket_count)

    chosen = []
    used = set()
    for bucket in active_buckets:
        cell = random.choice(bucket_cells[bucket])
        chosen.append(cell)
        used.add(cell)

    remaining = [cell for cell in reachable_floor if cell not in used]
    if len(chosen) + len(remaining) < k:
        return []

    chosen.extend(random.sample(remaining, k - len(chosen)))
    return chosen


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set)

    reachable_floor, dist = _reachable_floor_and_dist(base, dest_cells)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor

    max_k = min(FILL_K_RANGE[1], len(reachable_floor))
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    source_cells = _sample_reachable_sources(reachable_floor, dist, k)
    if len(source_cells) != k:
        return None

    ordered_distances = sorted(int(dist[r, c]) for r, c in source_cells)
    if len({_gradient_color(distance) for distance in ordered_distances}) < 2:
        return None

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0

    total = k + num_distractors
    if total < TOTAL_TOKEN_RANGE[0]:
        needed = TOTAL_TOKEN_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        else:
            return None
    if k + num_distractors > TOTAL_TOKEN_RANGE[1]:
        num_distractors = max(0, TOTAL_TOKEN_RANGE[1] - k)

    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    source_colors = _assign_source_colors(k)
    distractor_colors = [random.choice(SOURCE_PALETTE) for _ in range(num_distractors)]

    if len(set(source_colors)) < 2:
        return None

    inp = base.copy()
    for (r, c), color in zip(source_cells, source_colors):
        inp[r, c] = color
    for (r, c), color in zip(distractor_cells, distractor_colors):
        inp[r, c] = color

    ordered_sources = sorted(
        (int(dist[r, c]), int(r), int(c), color)
        for (r, c), color in zip(source_cells, source_colors)
    )

    out = base.copy()
    for (r, c), color in zip(distractor_cells, distractor_colors):
        out[r, c] = color
    for (r, c), (distance, _, _, _) in zip(dest_cells[:k], ordered_sources):
        out[r, c] = _gradient_color(distance)

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

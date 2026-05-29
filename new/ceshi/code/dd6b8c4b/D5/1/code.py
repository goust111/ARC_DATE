# Variant of dd6b8c4b
# Dimension: D5 — 源对象（被收集的对象） (src)
# Change: 栗色（maroon/9）散布像素——散落在网格各处的单个栗色像素，作为被收集的目标... → 多种颜色的待收集像素

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior:
#    - Keep the original 11x11 scene, fixed 3x3 destination at the red center, orange floor,
#      optional magenta walls, and single-pixel source tokens scattered on floor cells.
#    - The only semantic change is that source tokens are no longer all maroon. Each puzzle now
#      uses 2-3 colors chosen from a small palette such as blue/yellow/azure/maroon.
#    - Solver behavior stays "reachable via floor flood-fill -> sort by (distance,row,col) ->
#      place into destination row-major", but each destination cell receives the chosen token's
#      own color instead of a uniform maroon.
#    - Example trace: if reachable tokens sorted by the original rule are
#      [(d=0,r=3,c=5,color=1), (d=1,r=6,c=5,color=4), (d=1,r=7,c=5,color=9)],
#      then the first three destination cells become [1,4,9]. The old task would have written
#      [9,9,9] instead. This is the exact locus of change.
# b) Complexity calibration:
#    - Limit token colors to a small visible palette {1,4,8,9}; each sample activates only 2-3
#      of them. This is enough to make the variant obvious while avoiding an unstructured color soup.
#    - Tokens remain single pixels, so the original reasoning burden (connectivity + ordering) is
#      preserved; only color tracking is added.
# c) Original implicit assumptions that break under D5:
#    - `passable = {floor, 9}` assumed 9 was the only movable source color; now every token color
#      must be passable.
#    - `np.argwhere(grid == 9)` assumed candidate detection was color-specific; now candidates are
#      all non-structural token colors.
#    - `out[r, c] = 9` for destination fill assumed all moved tokens share one color; now fill must
#      use the chosen tokens' individual colors in sorted order.
#    - Floor inference in the original solver relied on 9 being the only non-structural color.
#      For this controlled variant we keep the original structural colors fixed, so the solver can
#      use explicit constants and avoid confusing token colors with terrain.
# d) Placement legality:
#    - Generator samples source and distractor cells only from floor cells outside the destination.
#    - Reachable sources are sampled from the BFS-reachable floor component; distractors come from
#      unreachable floor cells, so no overlap or out-of-bounds cases can occur.
#    - Validation still uses output-first checking: build the expected output, run `main(input)`,
#      and accept only exact matches.
# e) Visual observability:
#    - Every accepted sample has at least two distinct reachable token colors, so the destination
#      visibly contains multiple colors after the move.
#    - Ordering never depends on color id, only on visible geometry (distance then row-major ties).
# f) Visual cleanliness:
#    - Reuse the original sparse single-pixel layout and small color set. Structural colors remain
#      unchanged, so the new information is isolated to the moved tokens.
# g) Color inference:
#    - There is no hidden recoloring map. A token keeps its color when moved, which is directly
#      inferable across a few examples.
# h) New solver assumptions and generator guarantees:
#    - Assumption: all movable tokens use colors outside {2,3,6,7}. Generator enforces this.
#    - Assumption: at least one floor/token cell touches the destination from outside. Generator
#      rejects layouts with no BFS start.
#    - Assumption: all reachable tokens should be moved because the generator places exactly `k`
#      reachable tokens and `k <= 9`. Generator enforces this and validates by exact solve-check.
# i) Reference-task calibration:
#    - The source task `b5ca7ac4` preserves each square's interior color pattern while moving the
#      square by a structural rule. This variant mirrors that complexity level: motion is still
#      governed by geometry, but color content must be preserved instead of collapsed to one color.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
SOURCE_PALETTE = (1, 4, 8, 9)

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

FILL_K_RANGE = (2, 9)
DISTRACTOR_MAX = 10
TOTAL_TOKEN_RANGE = (4, 16)
MIN_REACHABLE = 2
GEN_RETRIES = 300
INF = 10**9


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


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
    Move the closest reachable colored pixels into the 3×3 block around the unique red center.

    Input structure:
    - Orange floor and optional magenta walls surround a fixed green 3×3 destination with one red center.
    - Single-pixel source tokens of several colors are scattered on floor cells outside that block.

    Rule:
    - Flood-fill outward from floor/token cells touching the destination, moving only through floor or token cells.
    - Collect every reachable colored token, sort them by distance to the destination and then by row/column, and write their colors into the destination in row-major order.
    - Erase the moved source tokens back to floor, while unreachable colored tokens stay where they are.

    Key insight: The geometry is still controlled by wall-restricted reachability, but now the solver must preserve the color sequence of the selected tokens instead of collapsing everything to one color.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    token_colors = {
        int(color) for color in np.unique(grid)
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

    for (r, c), (_, _, _, color) in zip(dest_cells[:k], chosen):
        out[r, c] = color

    return out


def _bfs_reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], np.ndarray]:
    dist = _distance_map(grid, sorted(dest_set), {FLOOR_COLOR})
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


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int) -> None:
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


def _assign_colors(count: int, active_colors: list[int]) -> list[int]:
    colors = list(active_colors)
    while len(colors) < count:
        colors.append(random.choice(active_colors))
    random.shuffle(colors)
    return colors


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set)

    reachable_floor, dist = _bfs_reachable_floor(base, dest_set)
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

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0

    total = k + num_distractors
    if total < TOTAL_TOKEN_RANGE[0]:
        needed = TOTAL_TOKEN_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        elif max_k >= k + needed:
            k += needed
        else:
            return None
    if k + num_distractors > TOTAL_TOKEN_RANGE[1]:
        num_distractors = max(0, TOTAL_TOKEN_RANGE[1] - k)

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    active_count = random.randint(2, min(3, k, len(SOURCE_PALETTE)))
    active_colors = random.sample(list(SOURCE_PALETTE), active_count)
    source_colors = _assign_colors(k, active_colors)
    if len(set(source_colors)) < 2:
        return None
    distractor_colors = [random.choice(active_colors) for _ in range(num_distractors)]

    inp = base.copy()
    for (r, c), color in zip(source_cells, source_colors):
        inp[r, c] = color
    for (r, c), color in zip(distractor_cells, distractor_colors):
        inp[r, c] = color

    ordered_sources = sorted(
        [(int(dist[r, c]), int(r), int(c), color) for (r, c), color in zip(source_cells, source_colors)]
    )

    out = base.copy()
    for (r, c), color in zip(distractor_cells, distractor_colors):
        out[r, c] = color
    for (r, c), (_, _, _, color) in zip(dest_cells[:k], ordered_sources):
        out[r, c] = color

    try:
        result = main(inp)
        if np.array_equal(result, out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

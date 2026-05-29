# Combo variant of dd6b8c4b (index 30)
# combo_spec: D3/3+D4/1+D5/1
# Name: 多色像素+颜色分组排序+列优先填充

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Preserve the original 11x11 stage: orange floor, optional magenta wall barriers, and a
#      fixed 3x3 green target centered on the lone red pixel at (5,5).
#    - Replace the single maroon source class with 2-3 token colors drawn from a small fixed
#      palette {1,4,8,9}. Tokens are still single pixels on floor cells outside the target.
#    - The new pipeline is:
#        reachable colored tokens
#        -> sort by color groups first, then BFS distance inside each color group
#        -> write the resulting color sequence into the target in column-major order.
#    - Trace example with reachable tokens:
#        distance order      = [(d0,blue), (d1,maroon), (d2,blue), (d3,yellow)]
#        grouped order       = [(blue,d0), (blue,d2), (yellow,d3), (maroon,d1)]
#        column-major fill   = [(4,4),(5,4),(6,4),(4,5),...]
#      Original task would have written a uniform maroon row-major prefix; this combo instead
#      preserves multiple colors, reorders them by color blocks, and lays those blocks down by
#      columns rather than rows.
# b) Complexity calibration:
#    - Restrict token colors to a small repeated palette so human solvers can infer a stable color
#      grouping rule from 2-4 examples instead of facing arbitrary one-off hues.
#    - Cap fills at 8 cells so column-major stays visibly different from row-major; require at
#      least 4 moved tokens so both color grouping and fill order usually leave a strong signal.
#    - Generator rejects samples where color-group sorting would accidentally match pure distance
#      sorting, preventing regression to a weaker D5-only or D4-only puzzle.
# c) Implicit assumption audit against the original solver:
#    - `passable = {floor, 9}` breaks because movable tokens are no longer all maroon; now every
#      non-structural token color must be traversable.
#    - `np.argwhere(grid == 9)` breaks because candidate detection cannot be color-specific.
#    - Writing `9` into the destination breaks because moved tokens must keep their own colors.
#    - `dest_cells.sort()` encoded row-major output order; adjacency logic can keep a canonical
#      destination list, but fill order must now be explicitly column-major.
#    - Pure `(dist,row,col)` ranking is no longer sufficient; the key becomes `(color,dist,row,col)`.
#    - Original floor inference from "the only other color is maroon" is unsafe with multi-color
#      tokens, so this controlled combo keeps the structural colors fixed and uses explicit constants.
# d) Placement legality:
#    - Sources and distractors are sampled only from floor cells outside the 3x3 target.
#    - Reachable sources come only from the floor component adjacent to the destination through wall
#      openings; distractors come only from BFS-unreachable floor cells, so they cannot be moved.
#    - No placed object overlaps a wall or another token, and the target coordinates stay in bounds.
# e) Visual observability:
#    - Every accepted sample contains at least two reachable token colors and is filtered so the
#      grouped order differs from the distance-only order on the same input.
#    - Because output uses column-major fill with k<=8, the target silhouette also differs from the
#      original row-major task. A solver must notice both the color blocking and the downward-first
#      fill trajectory.
# f) Visual cleanliness:
#    - Structural colors remain unchanged and token colors are sparse single pixels, so the added
#      complexity is localized to the moved sequence rather than cluttering the whole board.
#    - Using only 2-3 active token colors per puzzle avoids a noisy "rainbow" destination.
# g) Color-rule inferability:
#    - The grouping order is a fixed palette order over a tiny reusable set, not a per-instance
#      remapping. Repeated examples let a human observe that equal-colored tokens cluster together
#      in the target before the next color starts, while each token still preserves its own color.
# h) New solver assumptions and generator guarantees:
#    - Assumption: all movable colors are outside {2,3,6,7}. Generator enforces this via the palette.
#    - Assumption: there is at least one passable floor/token neighbor touching the target. Generator
#      rejects wall layouts with no reachable starting cell.
#    - Assumption: samples should exhibit both combo-specific effects. Generator validates by exact
#      solve-check and also rejects cases where grouped-column output matches grouped-row or
#      distance-column alternatives.
# i) Post-change audit target:
#    - The only ordering dimensions are visible token color groups and visible geometric distance.
#      No hidden metadata, black-cell counting, or generation-only labels are used by the solver.


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

FILL_K_RANGE = (4, 8)
DISTRACTOR_MAX = 10
TOTAL_TOKEN_RANGE = (4, 16)
MIN_REACHABLE = 4
GEN_RETRIES = 400
INF = 10**9


def _dest_cells(center_r: int, center_c: int, h: int, w: int, order: str) -> list[tuple[int, int]]:
    rows = range(center_r - 1, center_r + 2)
    cols = range(center_c - 1, center_c + 2)
    cells: list[tuple[int, int]] = []
    if order == "row_major":
        for r in rows:
            for c in cols:
                if 0 <= r < h and 0 <= c < w:
                    cells.append((r, c))
    elif order == "col_major":
        for c in cols:
            for r in rows:
                if 0 <= r < h and 0 <= c < w:
                    cells.append((r, c))
    else:
        raise ValueError(f"unknown order: {order}")
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


def _token_colors(grid: np.ndarray) -> set[int]:
    return {
        int(color)
        for color in np.unique(grid)
        if int(color) not in {FLOOR_COLOR, WALL_COLOR, DEST_COLOR, RED_CENTER}
    }


def _sorted_reachable_tokens(
    grid: np.ndarray,
    dist: np.ndarray,
    token_colors: set[int],
    sort_mode: str,
) -> list[tuple[int, int, int, int]]:
    tokens: list[tuple[int, int, int, int]] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color in token_colors and int(dist[r, c]) < INF:
                tokens.append((color, int(dist[r, c]), int(r), int(c)))

    if sort_mode == "distance":
        tokens.sort(key=lambda t: (t[1], t[2], t[3], t[0]))
    elif sort_mode == "color_then_distance":
        tokens.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    else:
        raise ValueError(f"unknown sort mode: {sort_mode}")
    return tokens


def _solve_with_orders(grid: np.ndarray, sort_mode: str, fill_order: str) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w, "row_major")
    fill_cells = _dest_cells(cr, cc, h, w, fill_order)
    token_colors = _token_colors(grid)
    assert token_colors, "grid must contain at least one non-structural token color"

    passable = {FLOOR_COLOR, *token_colors}
    dist = _distance_map(grid, dest_cells, passable)
    tokens = _sorted_reachable_tokens(grid, dist, token_colors, sort_mode)

    k = min(len(fill_cells), len(tokens))
    chosen = tokens[:k]

    for _, _, r, c in chosen:
        out[r, c] = FLOOR_COLOR

    for (r, c), (color, _, _, _) in zip(fill_cells[:k], chosen):
        out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the reachable colored pixels into the 3x3 block around the unique red center, grouping them by color before placing them down by columns.

    Input structure:
    - Orange floor and optional magenta walls surround a fixed green 3x3 destination whose center cell is red.
    - Outside that block, scattered single-pixel tokens can use several colors from a small palette instead of all sharing one source color.

    Rule:
    - Flood-fill from passable floor/token cells touching the destination, traversing only floor or token cells, so walls still decide which tokens are reachable.
    - Collect all reachable tokens, sort them first by token color and then by BFS distance to the destination with row/column tie-breaks, erase the chosen source pixels back to floor, and copy the resulting color sequence into the destination in column-major order from the top-left.
    - Unreachable colored tokens remain where they are, and any destination cells beyond the moved sequence keep their original green/red colors.

    Key insight:
    - This variant is not just "multi-color tokens": the colors are also the first ordering key, so farther tokens of an earlier color can be moved before nearer tokens of a later color, and the destination fills downward by columns rather than across rows.
    """

    return _solve_with_orders(grid, "color_then_distance", "col_major")


def _bfs_reachable_floor(
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
    dirs: list[tuple[int, int]] = []
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


def _group_counts(total: int, group_count: int) -> list[int]:
    counts = [1] * group_count
    remaining = total - group_count
    while remaining > 0:
        idx = random.randrange(group_count)
        counts[idx] += 1
        remaining -= 1
    return counts


def _assign_source_colors(
    source_cells: list[tuple[int, int]],
    dist: np.ndarray,
) -> tuple[list[int], list[int]] | None:
    ordered_cells = sorted(source_cells, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))
    k = len(ordered_cells)
    max_groups = min(3, k, len(SOURCE_PALETTE))
    if max_groups < 2:
        return None

    active_count = random.randint(2, max_groups)
    active_colors = sorted(random.sample(list(SOURCE_PALETTE), active_count))
    counts = _group_counts(k, active_count)

    reversed_blocks: list[int] = []
    for color, count in zip(reversed(active_colors), reversed(counts)):
        reversed_blocks.extend([color] * count)

    cell_to_color = {
        cell: color for cell, color in zip(ordered_cells, reversed_blocks)
    }
    source_colors = [cell_to_color[cell] for cell in source_cells]
    return source_colors, active_colors


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W, "row_major")
    fill_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W, "col_major")
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(base, dest_set)

    reachable_floor, dist = _bfs_reachable_floor(base, dest_cells)
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
    if k + num_distractors > TOTAL_TOKEN_RANGE[1]:
        num_distractors = max(0, TOTAL_TOKEN_RANGE[1] - k)

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    assigned = _assign_source_colors(source_cells, dist)
    if assigned is None:
        return None
    source_colors, active_colors = assigned
    if len(set(source_colors)) < 2:
        return None
    distractor_colors = [random.choice(active_colors) for _ in range(num_distractors)]

    inp = base.copy()
    for (r, c), color in zip(source_cells, source_colors):
        inp[r, c] = color
    for (r, c), color in zip(distractor_cells, distractor_colors):
        inp[r, c] = color

    expected = base.copy()
    for (r, c), color in zip(distractor_cells, distractor_colors):
        expected[r, c] = color

    source_records = [
        (color, int(dist[r, c]), int(r), int(c))
        for (r, c), color in zip(source_cells, source_colors)
    ]
    grouped = sorted(source_records, key=lambda t: (t[0], t[1], t[2], t[3]))
    by_distance = sorted(source_records, key=lambda t: (t[1], t[2], t[3], t[0]))

    if grouped == by_distance:
        return None

    for (r, c), (color, _, _, _) in zip(fill_cells[:k], grouped):
        expected[r, c] = color

    try:
        result = main(inp)
        grouped_row = _solve_with_orders(inp, "color_then_distance", "row_major")
        distance_col = _solve_with_orders(inp, "distance", "col_major")
        if (
            np.array_equal(result, expected)
            and not np.array_equal(result, grouped_row)
            and not np.array_equal(result, distance_col)
        ):
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

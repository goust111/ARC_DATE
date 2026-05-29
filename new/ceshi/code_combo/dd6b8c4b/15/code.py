# Combo variant of dd6b8c4b (index 15)
# combo_spec: D4/2+D10/1
# Name: 顺时针角度消歧+螺旋填充
#
# Design analysis
# combo_variant_spec:
#   invariant_structure:
#     grid: always 11x11
#     colors: floor=7, wall=6 or absent, destination ring=3, center marker=2, movable tokens=9
#     destination: fixed 3x3 block centered at (5,5)
#     connectivity: candidate maroons are still determined only by 4-connected flood-fill through floor/maroon
#   fused_changes:
#     source_ranking:
#       old: sort reachable maroons by (bfs_distance, row, col)
#       new: sort by (bfs_distance, clockwise_angle_from_center, row, col)
#       angle_origin: straight up from the red center is angle 0
#       angle_direction: increase clockwise
#     destination_fill:
#       old: write the chosen maroons into the 3x3 block in row-major order
#       new: write them along a fixed clockwise spiral:
#         - (cr-1, cc-1)
#         - (cr-1, cc)
#         - (cr-1, cc+1)
#         - (cr, cc+1)
#         - (cr+1, cc+1)
#         - (cr+1, cc)
#         - (cr+1, cc-1)
#         - (cr, cc-1)
#         - (cr, cc)
#   behavior_trace:
#     spiral_witness:
#       setup: 6 reachable maroons, all of them move
#       original: destination takes the row-major prefix of 6 cells
#       combo: destination takes the spiral prefix of 6 cells
#       visible_difference: the 4th-6th placements bend around the right/bottom edge instead of filling the middle row left-to-right
#     angle_witness:
#       setup: 10 reachable maroons on the same bfs shell around the destination
#       original: 9 chosen maroons are the row-major earliest shell cells
#       combo: 9 chosen maroons are the clockwise-earliest shell cells from the top
#       visible_difference: the leftover outside maroon moves to a different side of the ring, even though the destination ends up full
#   complexity_calibration:
#     fixed_orientation:
#       - use one global angle convention (north, then clockwise)
#       - use one global spiral orientation (clockwise, outer ring first)
#       - avoid per-instance rotations that would need extra examples to infer
#     observability_constraint:
#       - in this task family, angle tie-break is only visible when reachable_count > 9
#       - spiral fill is only visible when reachable_count is between 4 and 8
#       - therefore one board cannot visibly exhibit both effects at once without changing an extra rule
#       - the solver still implements both rules on every board, and the generator alternates between two non-degenerate witness regimes so the sample set exposes both halves of the combo
#   implicit_assumption_audit:
#     preserved:
#       - there is exactly one red center pixel
#       - the 3x3 destination is still centered on that red pixel
#       - floor inference still relies on the colors touching the destination ring
#       - walls remain hard blockers because bfs only walks on floor/maroon
#     changed_code_paths:
#       - ranking code can no longer rely on plain tuple sort (dist, row, col); it must compute a geometric angle key
#       - destination writes can no longer reuse sorted(dest_cells); they must use the spiral path
#       - generator can no longer assume "exactly k reachable maroons" is enough to expose every change
#     new_guardrails:
#       - exact angle ties are broken by (row, col) only as a deterministic fallback
#       - angle-visible boards must guarantee a same-distance shell with at least 10 maroons and different row-vs-angle top-9 subsets
#       - spiral-visible boards restrict moved count to 4..8 so the new fill path is actually distinguishable from row-major
#   placement_legality:
#     - source maroons are placed only on reachable floor cells, never inside the destination or on walls
#     - distractor maroons are placed only on unreachable floor cells
#     - all shell searches are computed from the same bfs metric used by the solver, so generation never depends on hidden geometry
#   visual_readability:
#     - the only extra geometry introduced is angular ordering around the already-salient red center
#     - the clockwise shell prefix outside and the clockwise spiral prefix inside both reuse the same center, so the rotational cue is coherent rather than cluttered
#   color_reasoning:
#     - no new colors and no color remapping are introduced
#     - every ordering decision is based on visible position relative to the red center, never on numeric color ids
#   solver_generator_contract:
#     - main() assumes at least one passable cell touches the destination and that the ring around the destination contains enough non-destination color evidence to infer the floor
#     - generate_input() guarantees those assumptions and rejects any candidate where the combo would be observationally identical to the original rule

from collections import deque
import math
import random

import numpy as np

from arc_utils import neighbors4


def _dest_cells_row_major(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _dest_cells_spiral(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (0, 0),
    ]
    return [
        (cr + dr, cc + dc)
        for dr, dc in offsets
        if 0 <= cr + dr < h and 0 <= cc + dc < w
    ]


def _infer_floor_color(grid: np.ndarray, dest_cells: list[tuple[int, int]]) -> int:
    dest_set = set(dest_cells)
    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and (rr, cc) not in dest_set:
                ring_colors.append(int(grid[rr, cc]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, "ring around destination must contain passable cells to infer floor color"
    vals, cnts = np.unique(cand_ring, return_counts=True)
    dominant_ring = int(vals[cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, dominant_ring)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    return dominant_ring if floor is None else floor


def _distance_map(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    passable: set[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = grid.shape
    starts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable:
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    starts.append((rr, cc))
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    queue = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while queue:
        r, c = queue.popleft()
        d = int(dist[r, c])
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == inf:
                dist[rr, cc] = d + 1
                queue.append((rr, cc))

    return dist, starts


def _clockwise_angle(r: int, c: int, cr: int, cc: int) -> float:
    return math.atan2(c - cc, cr - r) % (2 * math.pi)


def _rank_reachable_maroons(
    grid: np.ndarray,
    dist: np.ndarray,
    cr: int,
    cc: int,
) -> list[tuple[int, float, int, int]]:
    inf = 10**9
    maroons = [
        (int(dist[r, c]), _clockwise_angle(int(r), int(c), cr, cc), int(r), int(c))
        for r, c in np.argwhere(grid == 9)
        if int(dist[r, c]) < inf
    ]
    maroons.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return maroons


def _row_rank_reachable_maroons(dist: np.ndarray, maroons: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    return sorted((int(dist[r, c]), int(r), int(c)) for r, c in maroons)


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    centers = np.argwhere(grid == 2)
    assert len(centers) == 1
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells_row_major(cr, cc, h, w)
    fill_order = _dest_cells_spiral(cr, cc, h, w)
    floor = _infer_floor_color(grid, dest_cells)
    dist, _ = _distance_map(grid, dest_cells, {floor, 9})

    reachable_maroons = [(int(r), int(c)) for r, c in np.argwhere((grid == 9) & (dist < 10**9))]
    moved = min(len(fill_order), len(reachable_maroons))
    angle_ranked = _rank_reachable_maroons(grid, dist, cr, cc)
    row_ranked = _row_rank_reachable_maroons(dist, reachable_maroons)

    angle_chosen = {(r, c) for _, _, r, c in angle_ranked[:moved]}
    row_chosen = {(r, c) for _, r, c in row_ranked[:moved]}

    row_fill = _dest_cells_row_major(cr, cc, h, w)
    spiral_active = 1 if 4 <= moved <= 8 and set(fill_order[:moved]) != set(row_fill[:moved]) else 0
    angle_active = 1 if len(reachable_maroons) > len(fill_order) and angle_chosen != row_chosen else 0
    return {
        "moved": moved,
        "reachable": len(reachable_maroons),
        "spiral_active": spiral_active,
        "angle_active": angle_active,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move reachable maroon pixels into the 3×3 block around the unique red center using clockwise tie-breaks and a spiral fill.

    Input structure:
    - Orange floor cells form the passable background, magenta walls may block parts of it, and maroon tokens sit on some floor cells.
    - A unique red center is embedded in a green 3×3 destination block.

    Rule:
    - Flood-fill from passable cells touching the destination, treating floor and maroon as traversable; only reachable maroons are eligible.
    - Sort eligible maroons by BFS distance, then break same-distance ties by clockwise angle around the red center starting from straight up.
    - Move the first nine-or-fewer maroons: erase their source cells back to floor and write them into the destination along a fixed clockwise outer-to-inner spiral.

    Key insight:
    - The same center controls both decisions: outside the block it orders tied maroons by rotation, and inside the block it lays the moved maroons down in the matching clockwise spiral.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells_row_major(cr, cc, h, w)
    fill_order = _dest_cells_spiral(cr, cc, h, w)
    floor = _infer_floor_color(grid, dest_cells)
    dist, _ = _distance_map(grid, dest_cells, {floor, 9})

    maroons = _rank_reachable_maroons(grid, dist, cr, cc)
    k = min(len(fill_order), len(maroons))
    chosen = {(r, c) for _, _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in fill_order[:k]:
        out[r, c] = 9

    return out


# --------------- generator (c) ---------------

# ---- CFG: tunable generation parameters ----
GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

SPIRAL_WALL_PROB = 0.70
ANGLE_WALL_PROB = 0.20
RING_GAP_RANGE = (2, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

SPIRAL_K_RANGE = (4, 8)
ANGLE_EXTRA_DISTRACTOR_MAX = 6
SPIRAL_DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 4
GEN_RETRIES = 600
# ---- end CFG ----


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if grid[nr, nc] == FLOOR_COLOR and (nr, nc) not in dest_set:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _distance_over_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> np.ndarray:
    inf = 10**9
    dist = np.full((GRID_H, GRID_W), inf, dtype=int)
    queue: deque[tuple[int, int]] = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                if dist[nr, nc] == inf:
                    dist[nr, nc] = 0
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        d = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                if dist[nr, nc] == inf:
                    dist[nr, nc] = d + 1
                    queue.append((nr, nc))
    return dist


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
        nr, nc = nr + dr, nc + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    ring = set()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
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
                    nr, nc = nr + dr, nc + dc
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


def _pick_spiral_sources(reachable_floor: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int] | None:
    max_k = min(SPIRAL_K_RANGE[1], len(reachable_floor))
    if max_k < SPIRAL_K_RANGE[0]:
        return None
    k = random.randint(SPIRAL_K_RANGE[0], max_k)
    return random.sample(list(reachable_floor), k), k


def _pick_angle_sources(
    reachable_floor: set[tuple[int, int]],
    dist: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    shells: dict[int, list[tuple[int, int]]] = {}
    for r, c in reachable_floor:
        d = int(dist[r, c])
        if d < 10**9:
            shells.setdefault(d, []).append((r, c))

    shell_items = list(shells.items())
    random.shuffle(shell_items)
    for _, cells in shell_items:
        if len(cells) < 10:
            continue
        angle_sorted = sorted(cells, key=lambda rc: (_clockwise_angle(rc[0], rc[1], CENTER_R, CENTER_C), rc[0], rc[1]))
        row_sorted = sorted(cells)
        chosen = angle_sorted[:9]
        row_prefix = row_sorted[:9]
        candidates = [rc for rc in row_prefix if rc not in chosen]
        if not candidates:
            continue
        extra = random.choice(candidates)
        return chosen + [extra], chosen
    return None


def _expected_output(
    base_grid: np.ndarray,
    chosen_sources: list[tuple[int, int]],
    all_sources: list[tuple[int, int]],
    distractor_cells: list[tuple[int, int]],
    moved: int,
    fill_order: list[tuple[int, int]],
) -> np.ndarray:
    out = base_grid.copy()
    for r, c in all_sources[len(chosen_sources):]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    for r, c in fill_order[:moved]:
        out[r, c] = MAROON
    return out


def _original_solver_output(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    centers = np.argwhere(grid == 2)
    assert len(centers) == 1
    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells_row_major(cr, cc, h, w)
    floor = _infer_floor_color(grid, dest_cells)
    dist, _ = _distance_map(grid, dest_cells, {floor, 9})
    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9) if int(dist[r, c]) < 10**9]
    maroons.sort()
    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}
    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = 9
    return out


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells_row_major(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    fill_order = _dest_cells_spiral(CENTER_R, CENTER_C, GRID_H, GRID_W)
    mode = random.choice(["spiral", "angle"])

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    wall_prob = SPIRAL_WALL_PROB if mode == "spiral" else ANGLE_WALL_PROB
    if random.random() < wall_prob:
        _build_walls(grid, dest_set)

    reachable_floor = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None
    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor
    dist = _distance_over_floor(grid, dest_set)

    if mode == "spiral":
        picked = _pick_spiral_sources(reachable_floor)
        if picked is None:
            return None
        source_cells, moved = picked
        chosen_sources = list(source_cells)
        max_dist = min(SPIRAL_DISTRACTOR_MAX, len(unreachable_floor))
        num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0
    else:
        picked = _pick_angle_sources(reachable_floor, dist)
        if picked is None:
            return None
        source_cells, chosen_sources = picked
        moved = 9
        max_dist = min(ANGLE_EXTRA_DISTRACTOR_MAX, len(unreachable_floor))
        num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0

    if len(source_cells) + num_distractors > TOTAL_MAROON_RANGE[1]:
        num_distractors = max(0, TOTAL_MAROON_RANGE[1] - len(source_cells))

    total = len(source_cells) + num_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        else:
            return None

    distractor_cells = random.sample(list(unreachable_floor), num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    expected = _expected_output(grid, chosen_sources, source_cells, distractor_cells, moved, fill_order)
    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None

    if np.array_equal(result, _original_solver_output(inp)):
        return None

    metrics = _combo_metrics(inp)
    if mode == "spiral" and metrics["spiral_active"] != 1:
        return None
    if mode == "angle" and metrics["angle_active"] != 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

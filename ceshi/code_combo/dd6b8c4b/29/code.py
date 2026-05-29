# Combo variant of dd6b8c4b (index 29)
# combo_spec: D3/0+D4/0+D8/0
# Name: 反向距离排序+反向填充+跳过红色中心
#
# Design analysis
# a) Concrete behavior and trace:
#    - The scene remains the same 11x11 layout: a fixed green 3x3 target block
#      centered on a unique red cell, orange floor, optional magenta walls, and
#      scattered maroon tokens outside the block.
#    - The original task flood-filled from passable cells touching the block,
#      ranked reachable maroons by increasing BFS distance, and wrote them into
#      the destination in row-major order, potentially overwriting the red center.
#    - This combo changes three linked stages at once:
#      1. Candidate maroons are ranked from far to near.
#      2. Destination writing starts at the lower-right corner and proceeds in
#         reverse row-major order.
#      3. The red center is a fixed hole and is never overwritten.
#    - Small trace with reachable token distances [1, 2, 4, 5, 6, 7] and k=5:
#      original picks [1, 2, 4, 5, 6] and fills
#      [(4,4),(4,5),(4,6),(5,4),(5,5)];
#      combo picks [7, 6, 5, 4, 2] and fills
#      [(6,6),(6,5),(6,4),(5,6),(5,4)].
#      The moved sources differ, the occupied corner flips, and the center stays red.
# b) Complexity calibration:
#    - D3 and D4 cannot both be maximally visible in the same sample after D8 is
#      added. With identical maroon tokens, D3 only changes the output when more
#      than 8 reachable tokens compete for the 8 writable slots; but once all 8
#      non-center cells are filled, D4 becomes visually saturated.
#    - Conversely, to make D4 visible the fill must stay partial (5..7 tokens),
#      but then all reachable tokens are moved and D3 becomes invisible because
#      source selection no longer differs.
#    - The generator therefore alternates between two valid sample families:
#      1. `selection_visible`: 9..11 reachable tokens, so D3+D8 are visible.
#      2. `fill_visible`: 5..7 reachable tokens, so D4+D8 are visible.
#    - Across a small ARC training set this is still learnable, because all
#      examples share the same solver rule while exposing different observable
#      consequences of that rule.
# c) Implicit-assumption audit:
#    - Destination geometry and BFS reachability are still valid, but the old
#      assumption "sorted dest_cells is also the write path" no longer holds.
#      Geometry and fill order must be separated.
#    - The old `k = min(len(dest_cells), len(maroons))` also breaks because the
#      center is no longer a writable slot. The writable capacity is now 8.
#    - Erasing chosen sources back to floor is still correct, but only the far
#      set should be erased; the extra near maroons must survive in place.
#    - Floor inference still relies on walls dominating the ring when present.
#      The generator preserves that original hidden assumption by keeping only
#      1..4 ring gaps, so walls outnumber floor cells around the destination.
# d) Placement legality:
#    - Source and distractor maroons are sampled only from floor cells outside
#      the destination, so they never overlap the fixed block or the walls.
#    - Wall growth stops at bounds and never enters the 3x3 destination.
#    - Reverse fill uses a precomputed list of the eight non-center cells, so it
#      cannot step out of bounds or accidentally write onto the red center.
# e) Visual observability:
#    - A human can infer the combo from a few samples because three visible
#      effects co-occur: far maroons disappear while near ones remain, the
#      destination grows from bottom-right toward top-left, and the red center
#      stays untouched even once the fill reaches the middle row.
#    - No invisible metadata is used: ranking depends only on flood-fill steps,
#      and the generator avoids layouts where the combo output would match the
#      original or a simpler partial variant.
# f) Visual cleanliness:
#    - The puzzle still uses one compact target block and one token color. The
#      added complexity is purely structural, not extra ornamentation.
#    - Keeping k in 5..7 makes the block visibly patterned without saturating it,
#      so the preserved red center remains easy to notice.
# g) Change inferability:
#    - No new color mapping is introduced. Colors retain their original roles:
#      maroon moves, floor is restored, green stays as empty target cells, and
#      the red center is the single fixed exception.
# h) Degeneracy metric enforced after implementation:
#    - Each accepted sample must satisfy `center_saved = 1` and
#      `combo_vs_original = 1`.
#    - In addition, `selection_visible` samples require `selection_gap > 0`,
#      while `fill_visible` samples require `fill_gap > 0`.
#    - Post-generation evaluation checks the distribution of those metrics over
#      many samples, confirming that both sample families actually occur.
# i) New main() assumptions guaranteed by generate_input():
#    - There must be at least one passable opening next to the destination.
#    - For `selection_visible`, reachable floor cells must span at least two BFS
#      distance layers with at least 8 farther cells and at least 1 nearer cell.
#    - For `fill_visible`, there must be 5..7 reachable cells so the reverse
#      partial fill and the skipped center are both exposed.
#    - Every candidate sample is validated by `main()` before being returned.

from pathlib import Path
import sys
import random

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import neighbors4


def _destination_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _reverse_fill_order_skip_center(
    dest_cells: list[tuple[int, int]], center: tuple[int, int]
) -> list[tuple[int, int]]:
    return [cell for cell in sorted(dest_cells, reverse=True) if cell != center]


def _row_major_fill_order_skip_center(
    dest_cells: list[tuple[int, int]], center: tuple[int, int]
) -> list[tuple[int, int]]:
    return [cell for cell in dest_cells if cell != center]


def _reverse_fill_order_with_center(dest_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(dest_cells, reverse=True)


def _infer_floor(
    grid: np.ndarray, dest_cells: list[tuple[int, int]], dest_set: set[tuple[int, int]]
) -> int:
    ring_colors: list[int] = []
    h, w = grid.shape
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, "ring around destination must contain passable cells to infer floor color"
    vals, cnts = np.unique(cand_ring, return_counts=True)
    ring_blocker = int(vals[cnts.argmax()])
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, ring_blocker)]
    floor = other[0] if len(other) == 1 else (max(other, key=lambda c: int((grid == c).sum())) if other else None)
    return ring_blocker if floor is None else floor


def _distance_map(
    grid: np.ndarray,
    dest_cells: list[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    passable: set[int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h, w = grid.shape
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                if int(grid[rr, cc2]) in passable and (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for rr, cc2 in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                if int(grid[rr, cc2]) in passable and dist[rr, cc2] == inf:
                    dist[rr, cc2] = d + 1
                    q.append((rr, cc2))
    return dist, starts


def _analyze_grid(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], int, list[tuple[int, int, int]]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) center pixel"
    cr, cc = map(int, centers[0])
    center = (cr, cc)

    dest_cells = _destination_cells(cr, cc, h, w)
    fill_order = _reverse_fill_order_skip_center(dest_cells, center)
    dest_set = set(dest_cells)
    floor = _infer_floor(grid, dest_cells, dest_set)
    passable = {floor, 9}
    dist, starts = _distance_map(grid, dest_cells, dest_set, passable)
    assert starts, "there must be at least one passable cell adjacent to the destination block to begin BFS"

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [triple for triple in maroons if triple[0] < 10**9]
    return dest_cells, fill_order, center, floor, maroons


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the farthest reachable maroon pixels into the destination block while leaving the red center untouched.

    Input structure:
    - The board is an orange floor with optional magenta wall barriers and scattered maroon source pixels.
    - A single red cell sits at the center of a green 3x3 target block.

    Rule:
    - Flood-fill outward from passable cells touching the target, walking only through floor or maroon cells.
    - Reachable maroon pixels are ranked from farthest to nearest by BFS distance, with row/column tie-breaks.
    - The chosen maroons are erased from their source positions and copied into the target in reverse row-major order starting from the lower-right, but the red center cell is always skipped.

    Key insight:
    - The puzzle now depends on three visible constraints at once: walls still define reachability, distance preference is inverted, and the destination path mirrors toward the top-left while treating the center as a fixed hole.
    """

    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    dest_cells, fill_order, _center, floor, maroons = _analyze_grid(grid)

    maroons.sort(key=lambda t: (-t[0], t[1], t[2]))
    k = min(len(fill_order), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

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

WALL_PROB = 0.75
RING_GAP_RANGE = (1, 4)
WALL_STYLES = ["arms", "frame", "mixed"]

ARM_EXTEND_PROB = 0.5
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.4
FRAME_GAP_REMOVE_PROB = 0.3
MIXED_EXTEND_PROB = 0.6

PARTIAL_FILL_RANGE = (5, 7)
SELECTION_MOVED_COUNT = 8
SELECTION_LEFTOVER_RANGE = (1, 3)
DISTRACTOR_MAX = 8
TOTAL_MAROON_RANGE = (6, 16)
MIN_REACHABLE = 5
GEN_RETRIES = 500
MODE_SEQUENCE = ("selection_visible", "fill_visible")
_NEXT_MODE_INDEX = 0


def _dest_cells() -> list[tuple[int, int]]:
    return _destination_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)


def _bfs_reachable(
    grid: np.ndarray, dest_set: set[tuple[int, int]]
) -> tuple[set[tuple[int, int]], np.ndarray]:
    dest_cells = sorted(dest_set)
    dist, starts = _distance_map(grid, dest_cells, dest_set, {FLOOR_COLOR})
    if not starts:
        return set(), dist
    reachable = {(r, c) for r in range(GRID_H) for c in range(GRID_W) if dist[r, c] < 10**9}
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
    wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int
) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    length = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(length):
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


def _choose_selection_visible_sources(
    reachable_floor: set[tuple[int, int]], dist: np.ndarray
) -> dict[str, object] | None:
    by_distance: dict[int, list[tuple[int, int]]] = {}
    for r, c in reachable_floor:
        d = int(dist[r, c])
        by_distance.setdefault(d, []).append((r, c))

    layers = sorted(by_distance)
    if len(layers) < 2:
        return None

    candidates = []
    for idx in range(len(layers) - 1):
        near_pool = [cell for d in layers[: idx + 1] for cell in by_distance[d]]
        far_pool = [cell for d in layers[idx + 1 :] for cell in by_distance[d]]
        if len(near_pool) >= SELECTION_LEFTOVER_RANGE[0] and len(far_pool) >= SELECTION_MOVED_COUNT:
            candidates.append((near_pool, far_pool))
    if not candidates:
        return None

    near_pool, far_pool = random.choice(candidates)
    max_leftover = min(SELECTION_LEFTOVER_RANGE[1], len(near_pool), TOTAL_MAROON_RANGE[1] - SELECTION_MOVED_COUNT)
    if max_leftover < SELECTION_LEFTOVER_RANGE[0]:
        return None
    leftover = random.randint(SELECTION_LEFTOVER_RANGE[0], max_leftover)

    far_cells = random.sample(far_pool, SELECTION_MOVED_COUNT)
    near_cells = random.sample(near_pool, leftover)
    source_cells = near_cells + far_cells

    ordered = sorted((int(dist[r, c]), r, c) for r, c in source_cells)
    nearest = {(r, c) for _, r, c in ordered[:SELECTION_MOVED_COUNT]}
    farthest = {
        (r, c)
        for _, r, c in sorted(ordered, key=lambda t: (-t[0], t[1], t[2]))[:SELECTION_MOVED_COUNT]
    }
    if nearest == farthest:
        return None

    return {
        "mode": "selection_visible",
        "source_cells": source_cells,
        "chosen": farthest,
    }


def _choose_fill_visible_sources(reachable_floor: set[tuple[int, int]]) -> dict[str, object] | None:
    max_k = min(PARTIAL_FILL_RANGE[1], len(reachable_floor))
    if max_k < PARTIAL_FILL_RANGE[0]:
        return None
    k = random.randint(PARTIAL_FILL_RANGE[0], max_k)
    source_cells = random.sample(list(reachable_floor), k)
    chosen = set(source_cells)
    return {
        "mode": "fill_visible",
        "source_cells": source_cells,
        "chosen": chosen,
    }


def _expected_output(
    base_grid: np.ndarray,
    fill_order: list[tuple[int, int]],
    source_cells: list[tuple[int, int]],
    chosen_far: set[tuple[int, int]],
    distractor_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = base_grid.copy()
    survivors = set(source_cells) - chosen_far
    for r, c in survivors:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    for r, c in fill_order[: len(chosen_far)]:
        out[r, c] = MAROON
    return out


def _combo_metrics(inp: np.ndarray) -> dict[str, int]:
    dest_cells, fill_order, center, floor, maroons = _analyze_grid(inp)
    k = min(len(fill_order), len(maroons))
    ordered_far = sorted(maroons, key=lambda t: (-t[0], t[1], t[2]))
    ordered_near = sorted(maroons)
    farthest = {(r, c) for _, r, c in ordered_far[:k]}
    nearest = {(r, c) for _, r, c in ordered_near[:k]}

    reverse_skip = set(fill_order[:k])
    row_skip = set(_row_major_fill_order_skip_center(dest_cells, center)[:k])
    reverse_with_center = set(_reverse_fill_order_with_center(dest_cells)[:k])

    def _render(chosen_sources: set[tuple[int, int]], target_cells: set[tuple[int, int]]) -> np.ndarray:
        out = inp.copy()
        for r, c in chosen_sources:
            out[r, c] = floor
        for r, c in target_cells:
            out[r, c] = MAROON
        return out

    combo_out = _render(farthest, reverse_skip)
    original_out = _render(nearest, set(dest_cells[:k]))
    return {
        "k": k,
        "selection_gap": len(farthest.symmetric_difference(nearest)),
        "fill_gap": len(reverse_skip.symmetric_difference(row_skip)),
        "center_saved": int(center not in reverse_skip and center in reverse_with_center),
        "combo_vs_original": int(not np.array_equal(combo_out, original_out)),
        "reachable_count": len(maroons),
    }


def _gen_attempt(mode: str) -> np.ndarray | None:
    dest_cells = _dest_cells()
    center = (CENTER_R, CENTER_C)
    fill_order = _reverse_fill_order_skip_center(dest_cells, center)
    dest_set = set(dest_cells)

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    if random.random() < WALL_PROB:
        _build_walls(grid, dest_set)

    reachable_floor, dist = _bfs_reachable(grid, dest_set)
    if len(reachable_floor) < MIN_REACHABLE:
        return None

    if mode == "selection_visible":
        pick = _choose_selection_visible_sources(reachable_floor, dist)
    else:
        pick = _choose_fill_visible_sources(reachable_floor)
    if pick is None:
        return None
    source_cells = pick["source_cells"]
    chosen = pick["chosen"]

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if grid[r, c] == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = list(all_floor - reachable_floor - set(source_cells))

    base_total = len(source_cells)
    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_RANGE[1] - base_total)
    min_distractors = max(0, TOTAL_MAROON_RANGE[0] - base_total)
    if min_distractors > max_distractors:
        return None
    num_distractors = random.randint(min_distractors, max_distractors)
    distractor_cells = random.sample(unreachable_floor, num_distractors)

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(grid, fill_order, source_cells, chosen, distractor_cells)

    try:
        result = main(inp)
        if not np.array_equal(result, out):
            return None

        metrics = _combo_metrics(inp)
        if metrics["center_saved"] == 0:
            return None
        if metrics["combo_vs_original"] == 0:
            return None
        if mode == "selection_visible":
            if metrics["reachable_count"] <= len(fill_order):
                return None
            if metrics["selection_gap"] == 0:
                return None
        else:
            if not (PARTIAL_FILL_RANGE[0] <= metrics["k"] <= PARTIAL_FILL_RANGE[1]):
                return None
            if metrics["fill_gap"] == 0:
                return None
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    global _NEXT_MODE_INDEX
    preferred = MODE_SEQUENCE[_NEXT_MODE_INDEX % len(MODE_SEQUENCE)]
    _NEXT_MODE_INDEX += 1
    mode_order = [preferred] + [mode for mode in MODE_SEQUENCE if mode != preferred]
    for mode in mode_order:
        for _ in range(GEN_RETRIES):
            result = _gen_attempt(mode)
            if result is not None:
                return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

# Combo variant of dd6b8c4b (index 32)
# combo_spec: D7/0+D11/0+D18/1
# Name: 原位痕迹+不可达变色+裁剪输出

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import neighbors4


# Design analysis for the fused variant
# a) Concrete behavior on a small trace:
#    - Original rule: a reachable maroon near one corridor gap is moved into the 3x3 target and its old
#      cell turns back into orange; an unreachable maroon behind a wall stays maroon; the whole 11x11 grid
#      is returned.
#    - Fused rule: that same reachable maroon still moves into the target, but its source cell becomes
#      gray(5) to mark "collected". A blocked maroon in a sealed corner becomes blue(1) to mark "rejected".
#      After all marks are painted, only the 5x5 rectangle covering the 3x3 target plus its surrounding
#      ring/corners is returned.
#    - Difference: the solver must distinguish three local signs in the cropped view:
#      maroon inside the target = packed token, gray = consumed reachable source, blue = unreachable source.
#
# b) Complexity calibration:
#    - Colors are kept to a small, repeatable palette: orange floor, magenta wall, green/red target,
#      maroon tokens, gray traces, blue rejects.
#    - The crop is always the 5x5 bbox around the target, so the new difficulty comes from local evidence,
#      not from varying crop sizes.
#    - Generator forces both a reachable local source and an unreachable local distractor, so the fused
#      behaviors are visible instead of degenerating into a pure crop.
#
# c) Hidden-assumption audit against the original solver:
#    - Original code inferred floor/wall colors from local statistics. This fused version uses the fixed
#      ARC palette from the task spec, because the added gray/blue output markers would break those
#      heuristics if they were naively reused.
#    - BFS still assumes at least one passable ring gap next to the target; generator guarantees this.
#    - Cropping assumes the red center is not near the border; generator keeps the center fixed at (5,5),
#      so the 5x5 crop is always in-bounds.
#    - Gray traces are only meaningful for moved sources, so generator places exactly k reachable maroons:
#      every reachable input token is consumed, making the trace semantics unambiguous.
#
# d) Placement legality:
#    - Walls, destination cells, reachable sources, and unreachable distractors are sampled from disjoint
#      coordinate sets. Corner pockets are isolated by added wall cells before maroons are placed.
#
# e/f/g) Visual observability and tidiness:
#    - The 5x5 crop can show magenta walls, orange floor, green/red target residue, gray traces, blue
#      rejects, and maroon packed tokens, but the layout is still structured: blue is forced into sealed
#      crop corners, gray is forced onto reachable crop cells, and maroon fill stays inside the 3x3 target.
#    - Because these markers are separated by role and region, a human can compare a few examples and infer
#      which local pattern means "reachable and used" versus "blocked and left out".


GRID_H = 11
GRID_W = 11
CENTER_R = 5
CENTER_C = 5

FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9
TRACE_COLOR = 5
REJECT_COLOR = 1

FILL_K_RANGE = (2, 7)
DISTRACTOR_RANGE = (1, 3)
GEN_RETRIES = 400
ARM_EXTEND_PROB = 0.45
ARM_LENGTH_RANGE = (1, 3)


def _dest_cells(cr: int, cc: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((int(r), int(c)))
    cells.sort()
    return cells


def _crop_bounds(dest_cells: list[tuple[int, int]], h: int, w: int) -> tuple[int, int, int, int]:
    rows = [r for r, _ in dest_cells]
    cols = [c for _, c in dest_cells]
    r0 = max(0, min(rows) - 1)
    c0 = max(0, min(cols) - 1)
    r1 = min(h - 1, max(rows) + 1)
    c1 = min(w - 1, max(cols) + 1)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_cells(dest_set: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                ring.add((int(nr), int(nc)))
    return sorted(ring)


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


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    length = random.randint(*ARM_LENGTH_RANGE)
    nr, nc = r, c
    for _ in range(length):
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((int(nr), int(nc)))
        else:
            break


def _corner_pocket(name: str) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]:
    if name == "nw":
        pocket = (CENTER_R - 2, CENTER_C - 2)
        incident = {(CENTER_R - 2, CENTER_C - 1), (CENTER_R - 1, CENTER_C - 2)}
        support = incident | {(CENTER_R - 3, CENTER_C - 2), (CENTER_R - 2, CENTER_C - 3)}
    elif name == "ne":
        pocket = (CENTER_R - 2, CENTER_C + 2)
        incident = {(CENTER_R - 2, CENTER_C + 1), (CENTER_R - 1, CENTER_C + 2)}
        support = incident | {(CENTER_R - 3, CENTER_C + 2), (CENTER_R - 2, CENTER_C + 3)}
    elif name == "sw":
        pocket = (CENTER_R + 2, CENTER_C - 2)
        incident = {(CENTER_R + 2, CENTER_C - 1), (CENTER_R + 1, CENTER_C - 2)}
        support = incident | {(CENTER_R + 3, CENTER_C - 2), (CENTER_R + 2, CENTER_C - 3)}
    else:
        pocket = (CENTER_R + 2, CENTER_C + 2)
        incident = {(CENTER_R + 2, CENTER_C + 1), (CENTER_R + 1, CENTER_C + 2)}
        support = incident | {(CENTER_R + 3, CENTER_C + 2), (CENTER_R + 2, CENTER_C + 3)}
    return pocket, support, incident


def _reachable_floor(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((int(nr), int(nc)))
                    queue.append((int(nr), int(nc)))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((int(nr), int(nc)))
                    queue.append((int(nr), int(nc)))
    return visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack reachable maroon tokens into the central 3x3 target, leave gray pickup traces, repaint blocked maroons blue, then crop to the target neighborhood.

    Input structure:
    - An 11x11 orange floor with a fixed 3x3 destination marker: green around one red center.
    - Magenta wall segments create corridor gaps and sealed pockets around that target, while maroon tokens sit on floor cells outside the 3x3 block.

    Rule:
    - Flood-fill from passable cells touching the target through orange/maroon cells, then rank reachable maroons by BFS distance with row-major tie-breaks.
    - Move the closest reachable maroons into the 3x3 target in row-major order, turn each consumed source gray, and recolor every unreachable maroon blue.
    - After painting all marks on the full board, return only the 5x5 rectangle covering the target plus its immediate ring and corners.

    Key insight:
    - The crop hides the distant corridors, so the important evidence is local: maroon inside the 3x3 block means "selected", gray means "reachable and consumed", and blue means "blocked out by walls".
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc, h, w)
    dest_set = set(dest_cells)
    crop_r0, crop_c0, crop_r1, crop_c1 = _crop_bounds(dest_cells, h, w)

    passable = {FLOOR_COLOR, MAROON}
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in passable:
                if (nr, nc) not in seen:
                    seen.add((int(nr), int(nc)))
                    starts.append((int(nr), int(nc)))
    assert starts, "at least one corridor gap next to the target is required"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    queue: deque[tuple[int, int]] = deque()
    for r, c in starts:
        dist[r, c] = 0
        queue.append((int(r), int(c)))

    while queue:
        r, c = queue.popleft()
        base = int(dist[r, c])
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = base + 1
                queue.append((int(nr), int(nc)))

    reachable = []
    unreachable = []
    for r, c in np.argwhere(grid == MAROON):
        rr, cc2 = int(r), int(c)
        if int(dist[rr, cc2]) < inf:
            reachable.append((int(dist[rr, cc2]), rr, cc2))
        else:
            unreachable.append((rr, cc2))
    reachable.sort()

    k = min(len(dest_cells), len(reachable))
    chosen = {(r, c) for _, r, c in reachable[:k]}

    for r, c in chosen:
        out[r, c] = TRACE_COLOR
    for r, c in unreachable:
        out[r, c] = REJECT_COLOR
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    crop_r0, crop_c0, crop_r1, crop_c1 = _crop_bounds(dest_cells, GRID_H, GRID_W)
    crop_cells = {
        (r, c)
        for r in range(crop_r0, crop_r1 + 1)
        for c in range(crop_c0, crop_c1 + 1)
        if (r, c) not in dest_set
    }

    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    wall_cells: set[tuple[int, int]] = set()
    blocked_ring: set[tuple[int, int]] = set()
    pocket_cells: list[tuple[int, int]] = []

    pocket_names = random.sample(["nw", "ne", "sw", "se"], random.randint(1, 2))
    for name in pocket_names:
        pocket, support, incident = _corner_pocket(name)
        pocket_cells.append(pocket)
        wall_cells.update(support)
        blocked_ring.update(incident)

    ring = set(_ring_cells(dest_set, GRID_H, GRID_W))
    gap_candidates = sorted(ring - blocked_ring)
    if not gap_candidates:
        return None
    gap_count = random.randint(1, min(4, len(gap_candidates)))
    gap_cells = set(random.sample(gap_candidates, gap_count))
    wall_ring = ring - gap_cells
    wall_cells.update(wall_ring)

    for r, c in sorted(wall_ring):
        if random.random() < ARM_EXTEND_PROB:
            _extend_arm(wall_cells, dest_set, r, c)

    for r, c in wall_cells:
        if (r, c) not in dest_set:
            grid[r, c] = WALL_COLOR

    reachable_floor = _reachable_floor(grid, dest_set)
    if len(reachable_floor) < FILL_K_RANGE[0]:
        return None

    all_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == FLOOR_COLOR and (r, c) not in dest_set
    }
    unreachable_floor = all_floor - reachable_floor
    local_reachable = sorted(reachable_floor & crop_cells)
    local_unreachable = sorted(set(pocket_cells) & unreachable_floor & crop_cells)

    # New solver assumptions enforced by the generator:
    # - there is at least one reachable crop cell, so the cropped output visibly contains a gray trace;
    # - there is at least one unreachable crop pocket, so the cropped output visibly contains a blue reject.
    if not local_reachable or not local_unreachable:
        return None

    max_k = min(FILL_K_RANGE[1], len(reachable_floor))
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    chosen_sources: list[tuple[int, int]] = [random.choice(local_reachable)]
    remaining_reachable = sorted(reachable_floor - set(chosen_sources))
    extra_local = sorted((reachable_floor & crop_cells) - set(chosen_sources))
    if len(chosen_sources) < k and extra_local and random.random() < 0.7:
        chosen_sources.append(random.choice(extra_local))
        remaining_reachable = sorted(reachable_floor - set(chosen_sources))
    if len(remaining_reachable) < k - len(chosen_sources):
        return None
    chosen_sources.extend(random.sample(remaining_reachable, k - len(chosen_sources)))

    max_distractors = min(DISTRACTOR_RANGE[1], len(unreachable_floor))
    if max_distractors < DISTRACTOR_RANGE[0]:
        return None
    num_distractors = random.randint(DISTRACTOR_RANGE[0], max_distractors)
    distractors: list[tuple[int, int]] = [random.choice(local_unreachable)]
    remaining_unreachable = sorted(unreachable_floor - set(distractors))
    if len(remaining_unreachable) < num_distractors - len(distractors):
        return None
    distractors.extend(random.sample(remaining_unreachable, num_distractors - len(distractors)))

    inp = grid.copy()
    for r, c in chosen_sources + distractors:
        inp[r, c] = MAROON

    try:
        out = main(inp)
    except Exception:
        return None

    if out.shape != (5, 5):
        return None
    if TRACE_COLOR not in out or REJECT_COLOR not in out:
        return None
    if np.count_nonzero(out == MAROON) < k:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

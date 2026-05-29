# Variant of dd6b8c4b
# Dimension: D6 — 目标容器（绿色方块） (dst)
# Change: 绿色3x3方块（内含红色中心像素）——作为收集容器，内部空间被收集到的栗色像素填充... → 非正方形容器（L形/矩形）

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the D6 variant
# a) Concrete behavior and trace:
#    - The destination is no longer "the 3x3 square around the red center".
#      Instead, the red pixel is an anchor cell that belongs to a connected green
#      container made from a small catalog of shapes: L, T, or non-square rectangle.
#    - The solver must first recover the whole red+green connected component, then
#      fill only those cells in row-major order. Cells inside the component's
#      bounding box but outside the component are skipped.
#    - Example trace:
#      Original: red at (5,5) implied the 9 cells rows 4..6, cols 4..6. If k=5,
#      row-major filling writes the first five positions of that full square.
#      Variant: an L-shape might occupy {(5,5),(6,5),(7,5),(5,6),(5,7)}. If k=4,
#      row-major filling writes (5,5),(5,6),(5,7),(6,5), and does not write to
#      (6,6) or (7,6) because those cells are not part of the container.
# b) Complexity calibration:
#    - The change is visually obvious because many samples have holes in the
#      container bounding box, so the output is no longer consistent with a filled
#      3x3 square.
#    - The shape family is intentionally small: L, T, and 2x3 / 3x2 rectangles.
#      This keeps the rule learnable from a few examples while still varying the
#      destination enough to matter.
# c) Implicit assumptions from the original main() that break here:
#    - "for r in cr-1..cr+1 / c in cc-1..cc+1" assumed the container was always a
#      centered square. That is invalid now and is replaced by BFS over the red+green
#      component containing the unique red cell.
#    - "len(dest_cells)==9" was baked into the generator fill range. The new
#      generator derives fill capacity from the chosen template size instead.
#    - The rest of the logic (wall ring, reachable-source BFS, distance ordering,
#      erase-then-fill conservation) does not inherently depend on squareness and
#      can stay conceptually unchanged once dest_set is generalized.
# d) Placement legality:
#    - All destination templates include the red anchor at a fixed safe location
#      (5,5), so every template fits inside the 11x11 canvas.
#    - Walls are only placed outside dest_set. Maroon sources/distractors are only
#      placed on floor cells, never on walls or inside the container.
#    - Validation still uses main(input)==expected_output, so any illegal layout is
#      rejected during generation.
# e) Visual observability:
#    - L and T containers create skipped positions inside their bounding boxes, so
#      the new row-major-with-skips behavior is visible in 2-3 examples.
#    - Rectangles remain non-square to preserve the "not 3x3" signal even when no
#      skip is needed.
# f) Visual cleanliness:
#    - There is exactly one red+green destination component, a single floor color,
#      and an optional wall color. No overlapping markers are introduced.
# g) Color-change inferability:
#    - Colors keep their original roles. The only changed factor is which cells
#      count as the destination container. No hidden color-dependent ordering is used.
# h) New solver assumptions and generator guarantees:
#    - Assumption: exactly one red pixel exists, and the intended destination is the
#      connected component of colors {red, green} containing it.
#    - Assumption: at least one passable floor/maroon cell touches the container so
#      BFS can start.
#    - generate_input() guarantees both conditions by construction and rejects any
#      failed attempt.


def _destination_component(grid: np.ndarray) -> list[tuple[int, int]]:
    """Return the connected red+green destination component that contains the unique red anchor."""
    h, w = grid.shape
    centers = np.argwhere(grid == 2)
    assert len(centers) == 1, "grid must have exactly one red (2) anchor pixel"
    sr, sc = map(int, centers[0])

    dest_set = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set and int(grid[nr, nc]) in (2, 3):
                dest_set.add((nr, nc))
                q.append((nr, nc))

    assert len(dest_set) >= 5, "destination container must be a multi-cell red+green shape"
    return sorted(dest_set)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the irregular green container attached to the unique red anchor.

    Input structure:
    - The board still uses orange floor, optional magenta walls, and scattered maroon tokens.
    - A connected component made of green cells plus one red anchor cell is the destination container; it may be an L, a T, or a non-square rectangle.

    Rule:
    - Maroon pixels that can reach the container through floor/maroon cells are candidates, with walls blocking traversal exactly as before.
    - Rank reachable maroons by BFS distance to the container, then place the closest ones into the container's own cells in row-major order, skipping positions outside the shape.
    - Any moved maroon source is erased back to floor.

    Key insight:
    - The hard part is no longer finding a fixed 3x3 block from the red cell; you must first recover the whole red+green shape and then use that exact shape as the fill path while respecting wall-limited reachability.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    dest_cells = _destination_component(grid)
    dest_set = set(dest_cells)

    ring_colors: list[int] = []
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dest_set:
                ring_colors.append(int(grid[rr, cc]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    assert cand_ring, "ring around destination must contain non-destination cells"
    non_special = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9)]
    assert non_special, "grid must contain at least one non-destination background color"
    floor = max(non_special, key=lambda c: int((grid == c).sum()))

    passable = {floor, 9}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and (rr, cc) not in seen:
                seen.add((rr, cc))
                starts.append((rr, cc))
    assert starts, "there must be at least one passable cell adjacent to the destination shape"

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
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == inf:
                dist[rr, cc] = d + 1
                q.append((rr, cc))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [t for t in maroons if t[0] < inf]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in dest_cells[:k]:
        out[r, c] = 9

    return out


# --------------- generator (c) ---------------

# ---- CFG: tunable generation parameters ----
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

FILL_K_MIN = 2
DISTRACTOR_MAX = 10
TOTAL_MAROON_RANGE = (4, 16)
MIN_REACHABLE = 2
GEN_RETRIES = 400

DEST_TEMPLATES = {
    "L_dr": [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2)],
    "L_dl": [(0, 0), (1, 0), (2, 0), (0, -1), (0, -2)],
    "L_ur": [(0, 0), (-1, 0), (-2, 0), (0, 1), (0, 2)],
    "L_ul": [(0, 0), (-1, 0), (-2, 0), (0, -1), (0, -2)],
    "T_up": [(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)],
    "T_down": [(1, -1), (1, 0), (1, 1), (0, 0), (-1, 0)],
    "T_left": [(-1, -1), (0, -1), (1, -1), (0, 0), (0, 1)],
    "T_right": [(-1, 1), (0, 1), (1, 1), (0, 0), (0, -1)],
    "Rect_2x3": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    "Rect_3x2": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
}


def _make_dest_cells(shape_name: str) -> list[tuple[int, int]]:
    cells = [(ANCHOR_R + dr, ANCHOR_C + dc) for dr, dc in DEST_TEMPLATES[shape_name]]
    assert all(0 <= r < GRID_H and 0 <= c < GRID_W for r, c in cells)
    return sorted(cells)


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """BFS from floor cells adjacent to dest through floor cells."""
    visited = set()
    queue = deque()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) == FLOOR_COLOR and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited and (nr, nc) not in dest_set:
                if int(grid[nr, nc]) == FLOOR_COLOR:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def _outward_dirs(r: int, c: int) -> list[tuple[int, int]]:
    """Return directions pointing away from the fixed red anchor."""
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


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int) -> None:
    """Extend a wall arm outward from (r,c)."""
    dr, dc = random.choice(_outward_dirs(r, c))
    nr, nc = r, c
    for _ in range(random.randint(1, max_len)):
        nr, nc = nr + dr, nc + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((nr, nc))
        else:
            break


def _build_walls(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> None:
    """Build wall barriers around the irregular destination shape."""
    ring = set()
    for dr, dc in dest_set:
        for nr, nc in neighbors4(dr, dc):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    ring = sorted(ring)
    assert ring

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
            for rc in random.sample(outer_frame, min(n_remove, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                max_len = max(1, max(abs(r - ANCHOR_R), abs(c - ANCHOR_C)) + 1)
                _extend_arm(wall_cells, dest_set, r, c, max_len)

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _expected_output(base_grid: np.ndarray, dest_cells: list[tuple[int, int]], source_cells: list[tuple[int, int]], distractor_cells: list[tuple[int, int]]) -> np.ndarray:
    out = base_grid.copy()
    k = len(source_cells)
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    shape_name = random.choice(list(DEST_TEMPLATES))
    dest_cells = _make_dest_cells(shape_name)
    dest_set = set(dest_cells)

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

    max_k = min(len(dest_cells), len(reachable_floor))
    if max_k < FILL_K_MIN:
        return None
    k = random.randint(FILL_K_MIN, max_k)

    max_dist = min(DISTRACTOR_MAX, len(unreachable_floor))
    num_distractors = random.randint(0, max_dist) if max_dist > 0 else 0

    total = k + num_distractors
    if total < TOTAL_MAROON_RANGE[0]:
        needed = TOTAL_MAROON_RANGE[0] - total
        if len(unreachable_floor) >= num_distractors + needed:
            num_distractors += needed
        elif max_k >= k + needed:
            k += needed
        else:
            return None
    if k + num_distractors > TOTAL_MAROON_RANGE[1]:
        num_distractors = max(0, TOTAL_MAROON_RANGE[1] - k)

    source_cells = random.sample(list(reachable_floor), k)
    distractor_cells = random.sample(list(unreachable_floor), num_distractors) if num_distractors else []

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = _expected_output(grid, dest_cells, source_cells, distractor_cells)

    try:
        result = main(inp)
        if np.array_equal(result, out):
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

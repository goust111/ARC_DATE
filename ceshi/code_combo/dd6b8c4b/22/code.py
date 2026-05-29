# Combo variant of dd6b8c4b (index 22)
# combo_spec: D5/2+D14/1
# Name: 边缘像素作为源+方块位于角落

import random
from collections import Counter, deque

import numpy as np

from arc_utils import neighbors4

# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task used a centered 3x3 destination and maroon tokens scattered anywhere on floor.
#    Here the red-centered 3x3 block moves to one of four corner-biased centers: (3,3), (3,7), (7,3), (7,7).
#    Maroon sources are no longer interior speckles: every movable or distracting maroon lies on the outer border.
#    Example: with the block near the top-left, a ring of magenta walls plus long barriers on the bottom/right
#    makes top/left border cells easy to reach while bottom/right border cells can be sealed off. BFS still decides
#    which maroons qualify, but now the candidate pool itself is visibly "edge only" and strongly asymmetric.
# b) Complexity calibration:
#    The target keeps only four corner-biased placements instead of arbitrary offsets, so humans can notice the
#    positional change from a few examples. Border-only maroons are a strong visual cue, and the wall layout uses
#    repeated side-aligned barriers rather than noisy free-form walls.
# c) Implicit assumptions reviewed:
#    - main() can no longer assume the center is fixed at (5,5); it must locate the unique red cell.
#    - Collectible maroons are filtered to border cells only; interior maroons, if present, are ignored as sources.
#    - Distance ranking is still row/col visible, not dependent on hidden color ids or metadata.
#    - BFS still assumes at least one passable opening next to the block; generator guarantees near-side gaps.
# d) Placement legality:
#    The chosen centers keep the full 5x5 wall frame in-bounds. Walls, destination cells, and border maroons occupy
#    disjoint cell sets, and all source/distractor maroons are sampled only from floor border cells.
# e) Visual observability:
#    In 2-4 examples, a solver can see that maroons only appear on the boundary and that the block sits off-center.
#    Which border maroons move is explained by visible corridors/openings rather than invisible bookkeeping.
# f) Visual cleanliness:
#    Each role has its own region: the 3x3 target, a magenta wall frame, and edge-only maroons. This avoids the
#    clutter of mixed interior tokens and keeps the asymmetric reachability legible.
# g) Inferrability of change:
#    No new color remapping is introduced. The only semantic change is positional: "border maroons reachable through
#    visible floor corridors move into the corner-biased 3x3 block."

GRID_H, GRID_W = 11, 11
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
MAROON = 9

CORNER_CENTERS = [(3, 3), (3, 7), (7, 3), (7, 7)]
FILL_K_RANGE = (2, 9)
TOTAL_MAROON_RANGE = (4, 16)
GEN_RETRIES = 400
EXTRA_FAR_GAP_PROB = 0.35
NEAR_ARM_PROB = 0.35


def _is_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _dest_cells(cr: int, cc: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(cr - 1, cr + 2) for c in range(cc - 1, cc + 2)]


def _frame_ring(cr: int, cc: int) -> list[tuple[int, int]]:
    ring = []
    for r in range(cr - 2, cr + 3):
        for c in range(cc - 2, cc + 3):
            if max(abs(r - cr), abs(c - cc)) == 2:
                ring.append((r, c))
    return sorted(ring)


def _near_sides(cr: int, cc: int) -> tuple[str, str]:
    return ("top" if cr < GRID_H // 2 else "bottom", "left" if cc < GRID_W // 2 else "right")


def _side_tags(cr: int, cc: int, r: int, c: int) -> set[str]:
    tags: set[str] = set()
    if r == cr - 2:
        tags.add("top")
    if r == cr + 2:
        tags.add("bottom")
    if c == cc - 2:
        tags.add("left")
    if c == cc + 2:
        tags.add("right")
    return tags


def _outward_dirs(cr: int, cc: int, r: int, c: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    if r < cr:
        dirs.append((-1, 0))
    if r > cr:
        dirs.append((1, 0))
    if c < cc:
        dirs.append((0, -1))
    if c > cc:
        dirs.append((0, 1))
    return dirs


def _extend_to_border(wall_cells: set[tuple[int, int]], cr: int, cc: int, r: int, c: int, side: str) -> None:
    dr, dc = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]
    nr, nc = r, c
    while 0 <= nr < GRID_H and 0 <= nc < GRID_W:
        wall_cells.add((nr, nc))
        nr += dr
        nc += dc


def _build_walls(grid: np.ndarray, cr: int, cc: int, dest_set: set[tuple[int, int]]) -> None:
    near_v, near_h = _near_sides(cr, cc)
    far_v = "bottom" if near_v == "top" else "top"
    far_h = "right" if near_h == "left" else "left"

    ring = _frame_ring(cr, cc)
    near_pool = [cell for cell in ring if _side_tags(cr, cc, *cell) & {near_v, near_h}]
    far_pool = [cell for cell in ring if _side_tags(cr, cc, *cell) & {far_v, far_h}]

    num_near_gaps = random.randint(1, 3)
    gap_cells = set(random.sample(near_pool, min(num_near_gaps, len(near_pool))))
    if far_pool and random.random() < EXTRA_FAR_GAP_PROB:
        gap_cells.add(random.choice(far_pool))

    wall_cells = {cell for cell in ring if cell not in gap_cells}

    # Long far-side barriers are the main source of asymmetric edge reachability.
    for r, c in list(wall_cells):
        tags = _side_tags(cr, cc, r, c)
        if far_v in tags:
            _extend_to_border(wall_cells, cr, cc, r, c, far_v)
        if far_h in tags:
            _extend_to_border(wall_cells, cr, cc, r, c, far_h)

    # A few short near-side arms keep layouts varied without destroying the main opening.
    for r, c in list(wall_cells):
        tags = _side_tags(cr, cc, r, c)
        if tags & {near_v, near_h} and random.random() < NEAR_ARM_PROB:
            dirs = _outward_dirs(cr, cc, r, c)
            if dirs:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                steps = random.randint(1, 2)
                for _ in range(steps):
                    if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                        wall_cells.add((nr, nc))
                        nr += dr
                        nc += dc
                    else:
                        break

    for r, c in wall_cells:
        if (r, c) not in dest_set:
            grid[r, c] = WALL_COLOR


def _bfs_reachable(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited:
                if (nr, nc) not in dest_set and grid[nr, nc] == FLOOR_COLOR:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the nearest reachable border maroons into the off-center 3x3 block around the red pixel.

    Input structure:
    - Orange floor and magenta walls form corridors around a green 3x3 block whose red center is shifted toward a corner.
    - Maroon source pixels appear only on the outer border of the grid; some border maroons may be cut off by walls.

    Rule:
    - Find the unique red cell, take its surrounding 3x3 as the destination, then flood-fill outward through floor and maroon cells.
    - Only border maroons reached by that flood-fill are candidates; sort them by BFS distance, then row, then column.
    - Move the closest candidates into the destination in row-major order and erase the moved border sources back to floor.

    Key insight:
    - The asymmetry comes from two places at once: candidate maroons live only on the border, and the target block sits closer to
      some borders than others. Visible wall barriers decide which border sides can actually contribute maroons.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells(cr, cc)
    dest_set = set(dest_cells)

    floor_candidates = [int(x) for x in grid.flatten() if int(x) not in (RED_CENTER, DEST_COLOR, WALL_COLOR, MAROON)]
    assert floor_candidates, "grid must contain a passable floor color"
    floor = Counter(floor_candidates).most_common(1)[0][0]
    passable = {floor, MAROON}

    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and (nr, nc) not in seen:
                seen.add((nr, nc))
                starts.append((nr, nc))
    assert starts, "destination block must have at least one passable opening"

    inf = 10**9
    dist = np.full((h, w), inf, dtype=int)
    q = deque(starts)
    for r, c in starts:
        dist[r, c] = 0

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == inf:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    maroons = []
    for r, c in np.argwhere(grid == MAROON):
        r = int(r)
        c = int(c)
        if _is_border(r, c, h, w) and int(dist[r, c]) < inf:
            maroons.append((int(dist[r, c]), r, c))
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def _gen_attempt() -> np.ndarray | None:
    cr, cc = random.choice(CORNER_CENTERS)
    dest_cells = _dest_cells(cr, cc)
    dest_set = set(dest_cells)

    base = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in dest_cells:
        base[r, c] = DEST_COLOR
    base[cr, cc] = RED_CENTER

    _build_walls(base, cr, cc, dest_set)

    reachable_floor = _bfs_reachable(base, dest_set)
    border_floor = {
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if base[r, c] == FLOOR_COLOR and _is_border(r, c, GRID_H, GRID_W)
    }
    reachable_edge = sorted(border_floor & reachable_floor)
    unreachable_edge = sorted(border_floor - reachable_floor)

    max_k = min(FILL_K_RANGE[1], len(reachable_edge))
    if max_k < FILL_K_RANGE[0]:
        return None
    k = random.randint(FILL_K_RANGE[0], max_k)

    total_low = max(TOTAL_MAROON_RANGE[0], k)
    total_high = min(TOTAL_MAROON_RANGE[1], k + len(unreachable_edge))
    if total_high < total_low:
        return None
    total = random.randint(total_low, total_high)
    num_distractors = total - k

    source_cells = random.sample(reachable_edge, k)
    distractor_cells = random.sample(unreachable_edge, num_distractors)

    inp = base.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = base.copy()
    for r, c in source_cells:
        out[r, c] = FLOOR_COLOR
    for r, c in distractor_cells:
        out[r, c] = MAROON
    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    try:
        result = main(inp)
    except Exception:
        return None

    if not np.array_equal(result, out):
        return None

    # Fusion guardrails: ensure this did not collapse back to the original centered/scattered regime.
    if (cr, cc) == (5, 5):
        return None
    if any(not _is_border(r, c, GRID_H, GRID_W) for r, c in source_cells + distractor_cells):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

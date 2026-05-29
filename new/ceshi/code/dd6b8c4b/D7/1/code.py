# Variant of dd6b8c4b
# Dimension: D7 — 收集后原位置的变换 (trajectory)
# Change: 栗色像素被收集后从原位置消失（变为橙色背景），同时出现在方块内部新位置... → 收集路径上留下轨迹

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the D7 variant
# a) Concrete behavior:
#    The 3x3 destination block stays unchanged as the anchor object: green ring with a red center.
#    Reachable maroon tokens are still gathered by flood-fill reachability and distance, but now the
#    orange floor cells on the chosen route from each collected token back to the destination are
#    recolored teal (8) in the output. The maroon source cell itself still disappears to orange.
#    Example: if a source token sits two corridor steps away from a destination opening, the original
#    task only erased that token and filled one more destination cell; this variant also paints those
#    two corridor floor cells teal, so the transport path is visible.
# b) Complexity calibration:
#    Open floor regions would create many equally short shortest paths, making the trail choice hard
#    to infer from 2-4 ARC examples. To keep the new D7 effect visible and unambiguous, the generator
#    always builds wall-bounded corridor trees, so each reachable token has a unique route to the
#    block. This changes only the geometry needed to expose D7, not the ranking/filling rule itself.
# c) Old implicit assumptions reviewed:
#    - Original `out[r, c] = floor` on each chosen source assumed the source was the only changed cell
#      outside the destination. That is no longer enough; now we must also recolor every floor cell on
#      the chosen path.
#    - Original BFS stored only distances because reachability alone was sufficient. For trajectory we
#      also need predecessors, so BFS must remember a parent for each discovered passable cell.
#    - Original generator tolerated wide open reachable areas because only source selection mattered.
#      In this variant wide open regions are harmful, because multiple shortest paths would make the
#      displayed trail depend on arbitrary BFS tie-breaking. The new generator therefore guarantees a
#      tree-shaped reachable region.
# d) Placement legality:
#    The destination remains the fixed 3x3 block centered at (5, 5). Corridor cells are carved only
#    inside the 11x11 bounds, never into the destination, and new branch cells may touch the existing
#    corridor tree at exactly one 4-neighbor so no loops are created. Distractor maroons are placed in
#    isolated one-cell floor pockets that are not 4-adjacent to the reachable corridor.
# e) Visual observability:
#    The new color 8 is used only for transport trails. Because reachable tokens sit on leaf corridors
#    with depth at least 2, the output almost always contains multi-cell teal traces; the difference
#    from the original erase-only behavior is therefore visible at a glance.
# f) Visual cleanliness:
#    Trails occupy narrow corridors outside the destination block and do not overlap the destination or
#    walls. Multiple trails may share a trunk near the destination, which is visually coherent and
#    still reads as a transport network rather than clutter.
# g) Color inference:
#    No hidden color-id ordering is used. The only added color semantics is structural: teal marks the
#    route on former orange floor cells. This is supported by consistent position and shape cues.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
TRAIL_COLOR, MAROON = 8, 9

NUM_GAPS_RANGE = (1, 2)
SOURCE_COUNT_RANGE = (2, 6)
SOURCE_BRANCH_LEN_RANGE = (2, 5)
EXTRA_BRANCH_COUNT_RANGE = (0, 2)
EXTRA_BRANCH_LEN_RANGE = (1, 4)
DISTRACTOR_MAX = 7
TOTAL_MAROON_RANGE = (4, 13)
GEN_RETRIES = 400


def _dest_cells_around(center_r: int, center_c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(center_r - 1, center_r + 2):
        for c in range(center_c - 1, center_c + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((r, c))
    cells.sort()
    return cells


def _ring_cells(dest_set: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dest_set:
                ring.add((nr, nc))
    return sorted(ring)


def _infer_floor_color(grid: np.ndarray, dest_set: set[tuple[int, int]]) -> int:
    counts: dict[int, int] = {}
    for r, c in _ring_cells(dest_set, *grid.shape):
        value = int(grid[r, c])
        if value not in (DEST_COLOR, RED_CENTER, MAROON):
            counts[value] = counts.get(value, 0) + 1
    assert counts, "ring around destination must contain at least one non-destination color"
    if len(counts) == 1:
        return next(iter(counts))
    return min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _bfs_from_destination(
    grid: np.ndarray, dest_cells: list[tuple[int, int]], floor_color: int
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    h, w = grid.shape
    passable = {floor_color, MAROON}
    dist = np.full((h, w), 10**9, dtype=int)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {}

    starts = set()
    for r, c in dest_cells:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable:
                starts.add((nr, nc))

    starts = sorted(starts)
    assert starts, "there must be a passable opening adjacent to the destination block"

    queue = deque(starts)
    for r, c in starts:
        dist[r, c] = 0
        parent[(r, c)] = None

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == 10**9:
                dist[nr, nc] = dist[r, c] + 1
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return dist, parent


def _trail_cells_for_sources(
    grid: np.ndarray,
    floor_color: int,
    parent: dict[tuple[int, int], tuple[int, int] | None],
    sources: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    trail = set()
    for source in sources:
        cur = parent.get(source)
        while cur is not None:
            if int(grid[cur]) == floor_color:
                trail.add(cur)
            cur = parent.get(cur)
    return trail


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull reachable maroon pixels into the central 3×3 block and leave teal transport trails behind.

    Input structure:
    - The canvas uses orange floor, magenta walls, scattered maroon tokens, and a fixed green 3×3 target with one red center.
    - Floor openings beside the target connect into corridors; only maroons reachable through those corridors participate.

    Rule:
    - Flood-fill outward from passable cells touching the target, treating orange floor and maroon tokens as passable.
    - Sort reachable maroons by flood-fill distance, then by row and column; the closest ones fill the target in row-major order.
    - Each chosen source maroon disappears back to orange, and every orange floor cell on its discovered route to the target is recolored teal.

    Key insight:
    - The destination fill still depends on BFS reachability and distance, but the output now exposes the actual transport routes, so recovering predecessor paths is necessary rather than just counting candidates.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"

    cr, cc = map(int, centers[0])
    dest_cells = _dest_cells_around(cr, cc, h, w)
    dest_set = set(dest_cells)
    floor_color = _infer_floor_color(grid, dest_set)
    dist, parent = _bfs_from_destination(grid, dest_cells, floor_color)

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    maroons = [item for item in maroons if item[0] < 10**9]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = [(r, c) for _, r, c in maroons[:k]]

    for r, c in chosen:
        out[r, c] = floor_color

    for r, c in _trail_cells_for_sources(grid, floor_color, parent, chosen):
        out[r, c] = TRAIL_COLOR

    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def _branch_candidates(
    cur: tuple[int, int],
    occupied: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    blocked_ring: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    options = []
    for nr, nc in neighbors4(*cur):
        nxt = (nr, nc)
        if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
            continue
        if nxt in occupied or nxt in dest_set or nxt in blocked_ring:
            continue

        occupied_neighbors = 0
        touches_cur = False
        for ar, ac in neighbors4(nr, nc):
            if (ar, ac) in occupied:
                occupied_neighbors += 1
                if (ar, ac) == cur:
                    touches_cur = True
        if occupied_neighbors != 1 or not touches_cur:
            continue
        options.append(nxt)

    options.sort(
        key=lambda rc: (
            abs(rc[0] - CENTER_R) + abs(rc[1] - CENTER_C),
            abs(rc[0] - cur[0]) + abs(rc[1] - cur[1]),
            -rc[0],
            -rc[1],
        ),
        reverse=True,
    )
    return options


def _grow_branch(
    network: set[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int] | None],
    depth: dict[tuple[int, int], int],
    reserved_sources: set[tuple[int, int]],
    dest_set: set[tuple[int, int]],
    blocked_ring: set[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    attachments = [cell for cell in network if cell not in reserved_sources]
    random.shuffle(attachments)
    attachments.sort(key=lambda cell: depth[cell])

    for attach in attachments:
        for _ in range(12):
            target_len = random.randint(min_len, max_len)
            used = set(network)
            path: list[tuple[int, int]] = []
            cur = attach

            while len(path) < target_len:
                options = _branch_candidates(cur, used, dest_set, blocked_ring)
                if not options:
                    break
                top = options[: min(3, len(options))]
                nxt = random.choice(top)
                path.append(nxt)
                used.add(nxt)
                cur = nxt

            if len(path) < min_len:
                continue

            prev = attach
            for cell in path:
                network.add(cell)
                parent[cell] = prev
                depth[cell] = depth[prev] + 1
                prev = cell
            return path[-1]
    return None


def _trail_from_tree(
    parent: dict[tuple[int, int], tuple[int, int] | None], sources: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    trail = set()
    for source in sources:
        cur = parent[source]
        while cur is not None:
            trail.add(cur)
            cur = parent[cur]
    return trail


def _gen_attempt() -> np.ndarray | None:
    dest_cells = _dest_cells_around(CENTER_R, CENTER_C, GRID_H, GRID_W)
    dest_set = set(dest_cells)
    ring = _ring_cells(dest_set, GRID_H, GRID_W)

    grid = np.full((GRID_H, GRID_W), WALL_COLOR, dtype=int)
    for r, c in dest_cells:
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER

    num_gaps = random.randint(*NUM_GAPS_RANGE)
    gap_cells = set(random.sample(ring, num_gaps))
    blocked_ring = set(ring) - gap_cells

    network = set(gap_cells)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {cell: None for cell in gap_cells}
    depth: dict[tuple[int, int], int] = {cell: 0 for cell in gap_cells}

    source_count = random.randint(*SOURCE_COUNT_RANGE)
    source_cells: list[tuple[int, int]] = []
    reserved_sources: set[tuple[int, int]] = set()

    for _ in range(source_count):
        endpoint = _grow_branch(
            network,
            parent,
            depth,
            reserved_sources,
            dest_set,
            blocked_ring,
            SOURCE_BRANCH_LEN_RANGE[0],
            SOURCE_BRANCH_LEN_RANGE[1],
        )
        if endpoint is None:
            return None
        source_cells.append(endpoint)
        reserved_sources.add(endpoint)

    extra_branch_count = random.randint(*EXTRA_BRANCH_COUNT_RANGE)
    for _ in range(extra_branch_count):
        _grow_branch(
            network,
            parent,
            depth,
            reserved_sources,
            dest_set,
            blocked_ring,
            EXTRA_BRANCH_LEN_RANGE[0],
            EXTRA_BRANCH_LEN_RANGE[1],
        )

    for r, c in network:
        grid[r, c] = FLOOR_COLOR

    min_distractors = max(0, TOTAL_MAROON_RANGE[0] - len(source_cells))
    max_distractors = min(DISTRACTOR_MAX, TOTAL_MAROON_RANGE[1] - len(source_cells))
    if max_distractors < min_distractors:
        return None

    distractor_spots = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            cell = (r, c)
            if cell in dest_set or cell in network:
                continue
            if any((nr, nc) in network or (nr, nc) in dest_set for nr, nc in neighbors4(r, c)):
                continue
            distractor_spots.append(cell)

    if len(distractor_spots) < min_distractors:
        return None

    distractor_count = random.randint(min_distractors, min(max_distractors, len(distractor_spots)))
    distractor_cells = random.sample(distractor_spots, distractor_count)
    for r, c in distractor_cells:
        grid[r, c] = FLOOR_COLOR

    inp = grid.copy()
    for r, c in source_cells + distractor_cells:
        inp[r, c] = MAROON

    out = grid.copy()
    for r, c in dest_cells[: len(source_cells)]:
        out[r, c] = MAROON
    for r, c in distractor_cells:
        out[r, c] = MAROON
    for r, c in _trail_from_tree(parent, source_cells):
        out[r, c] = TRAIL_COLOR

    # Enforce visible D7 behavior: at least one trail cell must appear.
    if not np.any(out == TRAIL_COLOR):
        return None

    try:
        result = main(inp)
    except Exception:
        return None
    if np.array_equal(result, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

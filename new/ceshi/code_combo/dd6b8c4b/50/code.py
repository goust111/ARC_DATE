# Combo variant of dd6b8c4b (index 50)
# combo_spec: D7/1+D11/2
# Name: 扩大网格显示路径+收集路径轨迹

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for the combo variant
# a) Concrete behavior:
#    The input keeps the original 11x11 language: a fixed green 3x3 block with a red center, orange
#    floor, magenta walls, and maroon tokens outside the block. The original task only erased chosen
#    reachable maroons and refilled the destination block. Here the same BFS ranking still chooses the
#    collected tokens, but the output also stretches space around the 3x3 block by a width equal to
#    the farthest reachable maroon distance. A shortest path from each collected source to the block is
#    then drawn in teal. Example: if a chosen token is three BFS steps from a top opening, the original
#    output changed only its source cell and one destination cell; the combo output additionally inserts
#    a three-cell moat around the block and paints the token's corridor plus the new moat-crossing
#    bridge in teal.
# b) Complexity calibration:
#    If the reachable region were a wide open room, multiple shortest paths would make the displayed
#    trail arbitrary and hard to infer from 2-4 examples. The generator therefore builds a tree-shaped
#    corridor network rooted at 1-2 ring openings, so every reachable token has a unique BFS parent
#    chain. The expansion width has only one degree of freedom, the maximum reachable-token distance,
#    which keeps the D11 effect visible without introducing too many unrelated cases.
# c) Old implicit assumptions reviewed:
#    - The original `out = grid.copy()` assumed output size never changes. The combo solver must instead
#      allocate a larger floor canvas and map every original cell across inserted bands.
#    - The original BFS kept only distances. D7 requires parent pointers so the exact transport routes
#      can be reconstructed.
#    - The original erase step changed only the chosen source cells. After fusion, a chosen route also
#      colors intermediate floor cells plus the extra cells created by the inserted bands.
#    - The original row/column adjacency around the destination stayed unit-length. After D11, any path
#      crossing from the outer world into the destination neighborhood spans a longer straight segment,
#      so the solver must draw full line segments rather than recolor isolated cells.
# d) Placement legality:
#    The generator never places maroons inside the destination block or on walls. Corridor branches stay
#    inside the 11x11 bounds, never overlap one another except at their intended tree attachments, and
#    isolated distractor pockets are kept 4-disconnected from the reachable network. Because expansion
#    only inserts rows/columns around the destination block, mapped output coordinates also stay inside
#    the enlarged canvas by construction.
# e) Visual observability:
#    The new teal color appears only on transport routes, and the output shape itself changes from 11x11
#    to a larger square. Since every chosen route must enter the block through one of the inserted bands,
#    the extra border is actively used rather than empty padding, making both D7 and D11 visible at a
#    glance.
# f) Visual cleanliness:
#    The expanded bands are floor-colored except where a chosen path crosses them, so the output reads
#    as a clean stretched map instead of a cluttered overlay. Shared trunks near the destination remain
#    coherent because all overlapping chosen routes use the same teal color.
# g) Change inferability:
#    No hidden ordering by color id is introduced. Humans can infer the behavior from structure alone:
#    reachable maroons still fill the destination in distance order, the farthest reachable maroon sets
#    the amount of outward stretching, and teal consistently marks the transport corridor from source to
#    destination.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
TRAIL_COLOR, MAROON = 8, 9

NUM_GAPS_RANGE = (1, 2)
SOURCE_COUNT_RANGE = (3, 8)
SOURCE_BRANCH_LEN_RANGE = (2, 5)
EXTRA_BRANCH_COUNT_RANGE = (0, 3)
EXTRA_BRANCH_LEN_RANGE = (1, 4)
DISTRACTOR_MAX = 7
TOTAL_MAROON_RANGE = (4, 14)
GEN_RETRIES = 500
INF = 10**9


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
    dist = np.full((h, w), INF, dtype=int)
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
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in passable and dist[nr, nc] == INF:
                dist[nr, nc] = dist[r, c] + 1
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return dist, parent


def _mapped_coord(r: int, c: int, center_r: int, center_c: int, pad: int) -> tuple[int, int]:
    top = center_r - 1
    bottom = center_r + 1
    left = center_c - 1
    right = center_c + 1

    rr = int(r)
    cc = int(c)
    if rr >= top:
        rr += pad
    if rr > bottom:
        rr += pad
    if cc >= left:
        cc += pad
    if cc > right:
        cc += pad
    return rr, cc


def _adjacent_dest_cell(start: tuple[int, int], center_r: int, center_c: int) -> tuple[int, int]:
    r, c = start
    if r < center_r - 1:
        return center_r - 1, c
    if r > center_r + 1:
        return center_r + 1, c
    if c < center_c - 1:
        return r, center_c - 1
    if c > center_c + 1:
        return r, center_c + 1
    raise AssertionError("start cell must be 4-adjacent to the destination block")


def _paint_segment(
    out: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    center_r: int,
    center_c: int,
    pad: int,
    color: int,
    *,
    include_a: bool,
    include_b: bool,
) -> None:
    ar, ac = _mapped_coord(a[0], a[1], center_r, center_c, pad)
    br, bc = _mapped_coord(b[0], b[1], center_r, center_c, pad)
    if ar == br:
        step = 1 if bc >= ac else -1
        c0 = ac if include_a else ac + step
        c1 = bc if include_b else bc - step
        if (step == 1 and c0 > c1) or (step == -1 and c0 < c1):
            return
        for cc in range(c0, c1 + step, step):
            out[ar, cc] = color
    else:
        step = 1 if br >= ar else -1
        r0 = ar if include_a else ar + step
        r1 = br if include_b else br - step
        if (step == 1 and r0 > r1) or (step == -1 and r0 < r1):
            return
        for rr in range(r0, r1 + step, step):
            out[rr, ac] = color


def _paint_trail(
    out: np.ndarray,
    parent: dict[tuple[int, int], tuple[int, int] | None],
    source: tuple[int, int],
    center_r: int,
    center_c: int,
    pad: int,
) -> None:
    cur = source
    prev = parent.get(cur)
    while prev is not None:
        _paint_segment(
            out, cur, prev, center_r, center_c, pad, TRAIL_COLOR, include_a=False, include_b=True
        )
        cur = prev
        prev = parent.get(cur)

    dest = _adjacent_dest_cell(cur, center_r, center_c)
    _paint_segment(
        out,
        cur,
        dest,
        center_r,
        center_c,
        pad,
        TRAIL_COLOR,
        include_a=(cur != source),
        include_b=False,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull reachable maroon pixels into the central 3×3 block, stretch space around that block, and show teal transport routes.

    Input structure:
    - The input still uses the original 11×11 vocabulary: orange floor, magenta walls, scattered maroon tokens, and a fixed green 3×3 destination with one red center.
    - Openings beside the destination lead into floor corridors; only maroons connected through those corridors are reachable.

    Rule:
    - Flood-fill from passable cells adjacent to the destination, treating floor and maroon as passable, then sort reachable maroons by BFS distance and by row/column.
    - Let the farthest reachable maroon distance be the expansion width; output a larger canvas formed by inserting that many floor rows and columns around the destination block.
    - Move the closest reachable maroons into the destination in row-major order, erase their source cells, and paint every floor cell on each chosen BFS route teal, including the stretched bridge through the inserted bands.

    Key insight:
    - The same connectivity-and-distance rule decides which maroons are collected, but solving the combo now requires two extra inferences: reconstructing predecessor paths for the trail, and remapping the whole scene into a larger coordinate system whose added width depends on the farthest reachable maroon.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape

    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must have exactly one red center pixel"
    cr, cc = map(int, centers[0])

    dest_cells = _dest_cells_around(cr, cc, h, w)
    dest_set = set(dest_cells)
    floor_color = _infer_floor_color(grid, dest_set)
    dist, parent = _bfs_from_destination(grid, dest_cells, floor_color)

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == MAROON)]
    reachable = [item for item in maroons if item[0] < INF]
    reachable.sort()
    assert reachable, "there must be at least one reachable maroon token"

    pad = max(d for d, _, _ in reachable)
    out = np.full((h + 2 * pad, w + 2 * pad), floor_color, dtype=int)

    for r in range(h):
        for c in range(w):
            rr, cc2 = _mapped_coord(r, c, cr, cc, pad)
            out[rr, cc2] = int(grid[r, c])

    k = min(len(dest_cells), len(reachable))
    chosen = [(r, c) for _, r, c in reachable[:k]]
    for r, c in chosen:
        rr, cc2 = _mapped_coord(r, c, cr, cc, pad)
        out[rr, cc2] = floor_color

    for source in chosen:
        _paint_trail(out, parent, source, cr, cc, pad)

    mapped_dest = [_mapped_coord(r, c, cr, cc, pad) for r, c in dest_cells]
    for rr, cc2 in mapped_dest[:k]:
        out[rr, cc2] = MAROON

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
                nxt = random.choice(options[: min(3, len(options))])
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


def _trail_cells_from_tree(
    parent: dict[tuple[int, int], tuple[int, int] | None], sources: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    trail = set()
    for source in sources:
        cur = source
        prev = parent.get(cur)
        while prev is not None:
            trail.add(prev)
            cur = prev
            prev = parent.get(cur)
    return trail


def _expected_pad_for_input(inp: np.ndarray) -> int:
    dest_cells = _dest_cells_around(CENTER_R, CENTER_C, GRID_H, GRID_W)
    floor_color = _infer_floor_color(inp, set(dest_cells))
    dist, _parent = _bfs_from_destination(inp, dest_cells, floor_color)
    reachable = [int(dist[r, c]) for r, c in np.argwhere(inp == MAROON) if int(dist[r, c]) < INF]
    assert reachable
    return max(reachable)


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

    expected_pad = _expected_pad_for_input(inp)
    if expected_pad < 2:
        return None

    trail_cells = _trail_cells_from_tree(parent, source_cells)
    if not trail_cells:
        return None

    result = main(inp)
    if result.shape != (GRID_H + 2 * expected_pad, GRID_W + 2 * expected_pad):
        return None

    extra_band = np.ones(result.shape, dtype=bool)
    for r in range(GRID_H):
        for c in range(GRID_W):
            rr, cc = _mapped_coord(r, c, CENTER_R, CENTER_C, expected_pad)
            extra_band[rr, cc] = False
    if not np.any(result[extra_band] == TRAIL_COLOR):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")

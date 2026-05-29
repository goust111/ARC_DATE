# Combo variant of dd6b8c4b (index 12)
# combo_spec: D7/1+D16/2
# Name: 收集路径轨迹+背景色变化显示路径

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import neighbors4


# Design analysis for combo D7/1 + D16/2
# a) Concrete behavior:
#    The original task only moved reachable maroon tokens into the fixed 3x3 destination and erased
#    the chosen sources. The fused variant still keeps that ranking/filling rule, but the output now
#    exposes two nested movement signals:
#    - every reachable floor cell touched by the destination-side BFS becomes yellow (4), showing the
#      explored background region;
#    - within that region, the actual predecessor chains that carry chosen maroons back to the target
#      become teal (8), showing the transport trajectories.
#    Small example: if one source sits at the end of a three-cell corridor and a second side branch is
#    empty, the original output changes only the source and the target. The fused output paints the
#    empty side branch yellow and paints the used corridor teal, so the solver must separate explored
#    background from source-carrying paths.
# b) Complexity calibration:
#    If the reachable area were an open floor blob, shortest-path tie breaking would make the teal
#    route arbitrary and the yellow layer would swallow too much of the board. The generator therefore
#    uses wall-bounded tree corridors with 1-2 openings near the target. This keeps paths unique,
#    makes both colors visually legible, and still preserves the original BFS-distance reasoning.
# c) Old implicit assumptions reviewed:
#    - Original source erasure assumed the only output-side change outside the target was turning
#      chosen maroons back into floor. That no longer holds; reachable background must also change.
#    - Original BFS needed only distances. The fused rule needs parents as well, because the teal
#      subset is defined by actual source-to-target routes inside the larger yellow reachable region.
#    - Original generator allowed all reachable floor to be source-bearing. That would collapse yellow
#      and teal into the same shape, making D16/2 inert. The new generator forces at least one extra
#      reachable branch with no maroon source, so some explored cells remain yellow-only.
# d) Placement legality:
#    The 3x3 destination stays centered at (5,5) inside the fixed 11x11 grid. Corridors are grown one
#    cell at a time, each new cell touching exactly one existing corridor cell, so branches never
#    overlap, never leave bounds, and never enter blocked ring cells or the destination. Distractor
#    maroons are placed on isolated floor pockets outside the corridor network.
# e) Visual observability:
#    Humans can infer the rule from a few examples because the output shows a stable layering pattern:
#    untouched walls stay magenta, unreachable pockets stay orange/maroon, all reachable background
#    shifts to yellow, and used transport chains stand out in teal inside that yellow region.
# f) Visual cleanliness:
#    Yellow and teal are confined to narrow corridors outside the destination. Teal overrides yellow
#    only on the subset of cells that actually belong to chosen source routes, so the two signals are
#    readable rather than noisy.
# g) Color inference:
#    Only two new output-only roles are introduced, both tied to obvious structure instead of numeric
#    color ids: yellow means "reachable background visited by the search" and teal means "route that
#    carries collected maroons". The generator enforces both roles in every accepted sample.


GRID_H, GRID_W = 11, 11
CENTER_R, CENTER_C = 5, 5
FLOOR_COLOR, WALL_COLOR = 7, 6
DEST_COLOR, RED_CENTER = 3, 2
VISITED_COLOR, TRAIL_COLOR, MAROON = 4, 8, 9

NUM_GAPS_RANGE = (1, 2)
SOURCE_COUNT_RANGE = (2, 6)
SOURCE_BRANCH_LEN_RANGE = (2, 5)
EXTRA_BRANCH_COUNT_RANGE = (1, 3)
EXTRA_BRANCH_LEN_RANGE = (2, 4)
DISTRACTOR_MAX = 6
TOTAL_MAROON_RANGE = (4, 12)
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


def _reachable_floor_cells(grid: np.ndarray, floor_color: int, dist: np.ndarray) -> set[tuple[int, int]]:
    cells = set()
    for r, c in np.argwhere(grid == floor_color):
        rr, cc = int(r), int(c)
        if int(dist[rr, cc]) < INF:
            cells.add((rr, cc))
    return cells


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


def _render_output(grid: np.ndarray) -> np.ndarray:
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
    maroons = [item for item in maroons if item[0] < INF]
    maroons.sort()

    k = min(len(dest_cells), len(maroons))
    chosen = [(r, c) for _, r, c in maroons[:k]]
    reachable_floor = _reachable_floor_cells(grid, floor_color, dist)
    trail_cells = _trail_cells_for_sources(grid, floor_color, parent, chosen)

    for r, c in reachable_floor:
        out[r, c] = VISITED_COLOR

    for r, c in chosen:
        out[r, c] = floor_color

    for r, c in trail_cells:
        out[r, c] = TRAIL_COLOR

    for r, c in dest_cells[:k]:
        out[r, c] = MAROON

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gather reachable maroon pixels into the central 3x3 block and visualize both the explored region and the transport routes.

    Input structure:
    - The board contains orange floor, magenta walls, scattered maroon tokens, and a fixed green 3x3 destination with one red center.
    - Passable openings beside the destination lead into corridor-like floor regions; isolated floor pockets may hold unreachable distractor maroons.

    Rule:
    - Flood-fill from passable cells touching the destination, treating orange floor and maroon tokens as passable.
    - Rank reachable maroons by flood-fill distance, then by row and column; the closest ones fill the destination in row-major order.
    - In the output, every reachable floor cell becomes yellow to show the explored background, while the actual floor cells on each chosen source-to-destination parent chain are recolored teal; chosen source maroons themselves disappear back to orange.

    Key insight:
    - The fill order is still controlled by reachability and distance, but the fused variant exposes two nested BFS products: the full reachable background region and the narrower subset of floor cells that lie on the transport routes of the collected maroons.
    """

    return _render_output(grid)


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
        for _ in range(16):
            target_len = random.randint(min_len, max_len)
            used = set(network)
            path: list[tuple[int, int]] = []
            cur = attach

            while len(path) < target_len:
                options = _branch_candidates(cur, used, dest_set, blocked_ring)
                if not options:
                    break
                top = options[: min(4, len(options))]
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

    extra_branch_target = random.randint(*EXTRA_BRANCH_COUNT_RANGE)
    extra_branch_ends: list[tuple[int, int]] = []
    for _ in range(extra_branch_target):
        endpoint = _grow_branch(
            network,
            parent,
            depth,
            reserved_sources,
            dest_set,
            blocked_ring,
            EXTRA_BRANCH_LEN_RANGE[0],
            EXTRA_BRANCH_LEN_RANGE[1],
        )
        if endpoint is not None:
            extra_branch_ends.append(endpoint)

    if not extra_branch_ends:
        return None

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

    out = _render_output(inp)

    trail_cells = _trail_from_tree(parent, source_cells)
    visited_only = set(network) - trail_cells - set(source_cells)
    if not trail_cells or not visited_only:
        return None
    if not np.any(out == TRAIL_COLOR) or not np.any(out == VISITED_COLOR):
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
    raise RuntimeError("Failed to generate valid input after repeated retries")

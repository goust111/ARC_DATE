# Combo variant of dd6b8c4b (index 6)
# combo_spec: D3/2+D17/2
# Name: 欧几里得距离排序+红色像素作flood-fill起点

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import neighbors4


# Design analysis for combo D3/2 + D17/2
# ---------------------------------------
# a) Concretization and behavior trace
# - The visible scene stays close to the base task: an 11x11 board, a central 3x3 destination block
#   with a red center and green ring, orange floor, magenta walls, and many maroon tokens outside.
# - The fusion changes two different stages:
#   1. Reachability now starts at the RED center and is allowed to walk through the GREEN ring.
#   2. Among reachable maroons, the chosen movers are ordered by Euclidean distance to the block center,
#      not by flood-fill path length.
# - Concrete example:
#   suppose a corridor opens on the right and another on the top. A token at (4,8) can be reached from the
#   center quickly through the right opening but is Euclidean-far from (5,5). Another token at (3,5) sits
#   visually near the center but only after a long detour through the top opening and a bent corridor.
#   Base behavior: the quick-path token would be preferred by BFS distance.
#   Fused behavior here: both tokens are reachable from the red center, but (3,5) is preferred because its
#   straight-line distance to the center is smaller.
# - Important difference from the original generator: if there are <= 9 reachable maroons, every reachable
#   token moves and the sorting rule becomes invisible. Therefore this generator deliberately uses 10-14
#   reachable maroons so the 9-cell capacity forces a visible keep-vs-move selection.
#
# b) Complexity calibration
# - Keep one destination block and the original five-color vocabulary to avoid extra decoding burden.
# - Use 10-14 reachable maroons and at most 0-3 unreachable distractors. This is enough to expose the
#   Euclidean-vs-BFS selection difference without turning the board into noise.
# - Walls are always present and usually provide multiple openings or long arms so path length and geometric
#   closeness can disagree. The generator rejects layouts where the fusion selection matches the original
#   BFS-from-outside selection.
#
# c) Implicit assumption audit against the original main()
# - Original assumption: BFS starts from passable cells adjacent to the destination exterior.
#   Broken by D17/2. The new solver must start at the unique red center and include green ring cells in the
#   passable set, otherwise the center would be sealed off from the outside.
# - Original assumption: the same BFS distance used both for reachability and sorting.
#   Broken by D3/2. The new solver must compute reachability by flood-fill but rank selected maroons by
#   Euclidean distance to the center instead.
# - Original assumption hidden in the old generator: reachable maroon count can be 2-9.
#   Unsafe now because sorting would not affect the output when all reachable tokens fit into the 3x3 block.
#   The new generator guarantees more reachable maroons than capacity.
# - Original floor-color inference is unnecessary here because the spec fixes colors. Using explicit colors
#   removes brittle heuristics that were tied to the old ring analysis.
#
# d) Placement legality
# - The destination block is fixed at rows 4-6 and cols 4-6; source maroons are placed only on floor cells
#   outside that block.
# - Walls are drawn only outside the destination block, so the green ring and red center remain intact.
# - Reachable sources come from center-reachable floor cells; distractors come from unreachable floor cells.
#   This prevents overlap, out-of-bounds placement, and illegal wall/destination collisions.
#
# e) Visual observability
# - Because only 9 of 10-14 reachable maroons move, some reachable maroons remain outside in the output.
#   Humans can compare which nearby/farther tokens disappear and infer that "closest by path" is not the
#   deciding rule anymore.
# - Multiple wall openings make the red-center flood-fill idea plausible from the examples: the whole green
#   ring acts like a traversable hub rather than a blocked target object.
#
# f) Cleanliness
# - The wall templates are restricted to ring/frame/arm patterns around the center, so the scene still reads
#   as one central container plus corridors instead of unrelated clutter.
# - Distractors are limited and optional. Most of the visual signal comes from the reachable maroons around
#   the wall openings, not from dense scatter everywhere.
#
# g) New assumptions introduced by the fused main()
# - There must be exactly one red center and its surrounding 3x3 block must lie inside the board.
# - The green ring must stay passable; generator never replaces it with walls or maroons in the input.
# - To make D3/2 observable, generator enforces a reachable source set where Euclidean top-9 differs from
#   the original BFS-distance top-9. It also prefers cases where center-start BFS and old outside-start BFS
#   disagree, so the D17/2 start change is present in the sample statistics as well.


GRID_H = 11
GRID_W = 11
FLOOR_COLOR = 7
WALL_COLOR = 6
DEST_COLOR = 3
RED_CENTER = 2
MAROON = 9

CENTER_R = 5
CENTER_C = 5

RING_GAP_RANGE = (2, 4)
WALL_STYLES = ("arms", "frame", "mixed")
ARM_EXTEND_PROB = 0.55
ARM_LENGTH_RANGE = (1, 5)
FRAME_DIST_RANGE = (3, 4)
FRAME_CONNECT_PROB = 0.45
FRAME_GAP_REMOVE_PROB = 0.35
MIXED_EXTEND_PROB = 0.65

REACHABLE_SOURCE_RANGE = (10, 14)
DISTRACTOR_MAX = 3
TOTAL_MAROON_MAX = 16
MIN_REACHABLE_FLOOR = 12
GEN_RETRIES = 500
INF = 10**9


def _find_center(grid: np.ndarray) -> tuple[int, int]:
    centers = np.argwhere(grid == RED_CENTER)
    assert len(centers) == 1, "grid must contain exactly one red center"
    return int(centers[0, 0]), int(centers[0, 1])


def _dest_cells_for(center: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = center
    h, w = map(int, shape)
    cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                cells.append((int(r), int(c)))
    cells.sort()
    return cells


def _bfs(grid: np.ndarray, starts: list[tuple[int, int]], passable: set[int]) -> np.ndarray:
    h, w = map(int, grid.shape)
    dist = np.full((h, w), INF, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and int(grid[r, c]) in passable and dist[r, c] == INF:
            dist[r, c] = 0
            q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        d = int(dist[r, c])
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in passable and dist[rr, cc] == INF:
                dist[rr, cc] = d + 1
                q.append((int(rr), int(cc)))
    return dist


def _euclid2(center: tuple[int, int], cell: tuple[int, int]) -> int:
    cr, cc = center
    r, c = cell
    return int((r - cr) ** 2 + (c - cc) ** 2)


def _old_like_starts(grid: np.ndarray) -> list[tuple[int, int]]:
    center = _find_center(grid)
    dest_cells = _dest_cells_for(center, grid.shape)
    dest_set = set(dest_cells)
    starts: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in dest_cells:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if (rr, cc) not in dest_set and int(grid[rr, cc]) in {FLOOR_COLOR, MAROON} and (rr, cc) not in seen:
                    seen.add((int(rr), int(cc)))
                    starts.append((int(rr), int(cc)))
    return starts


def _reachable_maroons_from_center(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray, list[tuple[int, int]]]:
    center = _find_center(grid)
    dist = _bfs(grid, [center], {FLOOR_COLOR, DEST_COLOR, RED_CENTER, MAROON})
    maroons = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == MAROON)
        if int(dist[r, c]) < INF
    ]
    maroons.sort()
    return center, dist, maroons


def _fusion_selected(grid: np.ndarray) -> list[tuple[int, int]]:
    center, _dist, maroons = _reachable_maroons_from_center(grid)
    ranked = sorted(maroons, key=lambda rc: (_euclid2(center, rc), rc[0], rc[1]))
    return ranked[: min(9, len(ranked))]


def _center_bfs_selected(grid: np.ndarray) -> list[tuple[int, int]]:
    _center, dist, maroons = _reachable_maroons_from_center(grid)
    ranked = sorted(maroons, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))
    return ranked[: min(9, len(ranked))]


def _original_like_selected(grid: np.ndarray) -> list[tuple[int, int]]:
    starts = _old_like_starts(grid)
    assert starts, "grid must have at least one exterior floor neighbor next to the destination"
    dist = _bfs(grid, starts, {FLOOR_COLOR, MAROON})
    maroons = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == MAROON)
        if int(dist[r, c]) < INF
    ]
    ranked = sorted(maroons, key=lambda rc: (int(dist[rc[0], rc[1]]), rc[0], rc[1]))
    return ranked[: min(9, len(ranked))]


def _pairwise_inversions(items: list[tuple[int, int]], key_a, key_b) -> int:
    inv = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ai = key_a(items[i])
            aj = key_a(items[j])
            bi = key_b(items[i])
            bj = key_b(items[j])
            if ai == aj or bi == bj:
                continue
            if (ai < aj and bi > bj) or (ai > aj and bi < bj):
                inv += 1
    return int(inv)


def behavior_stats(grid: np.ndarray) -> dict[str, int]:
    center, dist_center, reachable = _reachable_maroons_from_center(grid)
    fusion = _fusion_selected(grid)
    original = _original_like_selected(grid)
    center_bfs = _center_bfs_selected(grid)
    starts = _old_like_starts(grid)
    dist_old = _bfs(grid, starts, {FLOOR_COLOR, MAROON})
    return {
        "reachable_count": int(len(reachable)),
        "fusion_vs_original_symdiff": int(len(set(fusion) ^ set(original))),
        "center_vs_original_symdiff": int(len(set(center_bfs) ^ set(original))),
        "euclid_vs_original_inversions": _pairwise_inversions(
            reachable,
            key_a=lambda rc: (_euclid2(center, rc), rc[0], rc[1]),
            key_b=lambda rc: (int(dist_old[rc[0], rc[1]]), rc[0], rc[1]),
        ),
        "center_vs_original_inversions": _pairwise_inversions(
            reachable,
            key_a=lambda rc: (int(dist_center[rc[0], rc[1]]), rc[0], rc[1]),
            key_b=lambda rc: (int(dist_old[rc[0], rc[1]]), rc[0], rc[1]),
        ),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the selected reachable maroons into the centered 3x3 block.

    Input structure:
    - The board contains orange floor, magenta walls, a fixed green 3x3 target ring, and a unique red center.
    - Maroon tokens are scattered on floor cells outside the block; some may be cut off by wall corridors.

    Rule:
    - Start flood-fill from the red center itself and allow the search to walk through the green ring, floor,
      and maroon cells.
    - Keep only the reachable maroons, then rank them by Euclidean distance to the block center, breaking ties
      by row then column.
    - Move up to nine ranked maroons into the destination block in row-major order and erase only those sources.

    Key insight:
    - Reachability still depends on corridor connectivity, but selection is no longer based on path length.
      A token can win because it is geometrically closer to the center even if the corridor route to it is longer.
    """

    g = np.asarray(grid, dtype=int)
    out = g.copy()

    center = _find_center(g)
    dest_cells = _dest_cells_for(center, g.shape)
    chosen = _fusion_selected(g)

    for r, c in chosen:
        out[r, c] = FLOOR_COLOR

    for r, c in dest_cells[: len(chosen)]:
        out[r, c] = MAROON

    return out


def _dest_cells_fixed() -> list[tuple[int, int]]:
    return _dest_cells_for((CENTER_R, CENTER_C), (GRID_H, GRID_W))


def _layout_base() -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), FLOOR_COLOR, dtype=int)
    for r, c in _dest_cells_fixed():
        grid[r, c] = DEST_COLOR
    grid[CENTER_R, CENTER_C] = RED_CENTER
    return grid


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


def _extend_arm(wall_cells: set[tuple[int, int]], dest_set: set[tuple[int, int]], r: int, c: int, max_len: int) -> None:
    dirs = _outward_dirs(r, c)
    if not dirs:
        return
    dr, dc = random.choice(dirs)
    steps = random.randint(1, max_len)
    nr, nc = r, c
    for _ in range(steps):
        nr += dr
        nc += dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
            wall_cells.add((int(nr), int(nc)))
        else:
            break


def _build_walls(grid: np.ndarray) -> None:
    dest_set = set(_dest_cells_fixed())
    ring = set()
    for r, c in dest_set:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                ring.add((int(nr), int(nc)))
    ring_list = sorted(ring)

    gap_count = random.randint(*RING_GAP_RANGE)
    gap_cells = set(random.sample(ring_list, gap_count))
    wall_ring = [cell for cell in ring_list if cell not in gap_cells]
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
                        wall_cells.add((int(r), int(c)))
        for r, c in wall_ring:
            dirs = _outward_dirs(r, c)
            if dirs and random.random() < FRAME_CONNECT_PROB:
                dr, dc = random.choice(dirs)
                nr, nc = r + dr, c + dc
                while 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in dest_set:
                    wall_cells.add((int(nr), int(nc)))
                    nr += dr
                    nc += dc
        outer_frame = [cell for cell in wall_cells if cell not in set(wall_ring)]
        if outer_frame:
            remove_n = random.randint(1, max(1, len(outer_frame) // 4))
            for rc in random.sample(outer_frame, min(remove_n, len(outer_frame))):
                if random.random() < FRAME_GAP_REMOVE_PROB:
                    wall_cells.discard(rc)
    else:
        for r, c in wall_ring:
            if random.random() < MIXED_EXTEND_PROB:
                _extend_arm(wall_cells, dest_set, r, c, max(1, GRID_H - abs(r - CENTER_R) - 2))

    for r, c in wall_cells:
        grid[r, c] = WALL_COLOR


def _reachable_floor_center(layout: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    dist = _bfs(layout, [(CENTER_R, CENTER_C)], {FLOOR_COLOR, DEST_COLOR, RED_CENTER})
    cells = [
        (int(r), int(c))
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(layout[r, c]) == FLOOR_COLOR and int(dist[r, c]) < INF
    ]
    return dist, cells


def _old_like_floor_dist(layout: np.ndarray) -> np.ndarray | None:
    starts = _old_like_starts(layout)
    if not starts:
        return None
    return _bfs(layout, starts, {FLOOR_COLOR})


def _select_sources(layout: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    dist_center, reachable_floor = _reachable_floor_center(layout)
    if len(reachable_floor) < MIN_REACHABLE_FLOOR:
        return None
    dist_old = _old_like_floor_dist(layout)
    if dist_old is None:
        return None

    center = (CENTER_R, CENTER_C)
    e_sorted = sorted(reachable_floor, key=lambda rc: (_euclid2(center, rc), rc[0], rc[1]))
    o_sorted = sorted(reachable_floor, key=lambda rc: (int(dist_old[rc[0], rc[1]]), rc[0], rc[1]))
    c_sorted = sorted(reachable_floor, key=lambda rc: (int(dist_center[rc[0], rc[1]]), rc[0], rc[1]))

    candidates: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]] = []
    max_a = min(REACHABLE_SOURCE_RANGE[1], len(e_sorted))
    max_b = min(REACHABLE_SOURCE_RANGE[1], len(o_sorted))
    for a in range(9, max_a + 1):
        for b in range(9, max_b + 1):
            source_set = sorted(set(e_sorted[:a]) | set(o_sorted[:b]))
            if not (REACHABLE_SOURCE_RANGE[0] <= len(source_set) <= REACHABLE_SOURCE_RANGE[1]):
                continue
            fusion_top = sorted(source_set, key=lambda rc: (_euclid2(center, rc), rc[0], rc[1]))[:9]
            old_top = sorted(source_set, key=lambda rc: (int(dist_old[rc[0], rc[1]]), rc[0], rc[1]))[:9]
            if set(fusion_top) == set(old_top):
                continue
            center_top = sorted(source_set, key=lambda rc: (int(dist_center[rc[0], rc[1]]), rc[0], rc[1]))[:9]
            inv_eo = _pairwise_inversions(
                source_set,
                key_a=lambda rc: (_euclid2(center, rc), rc[0], rc[1]),
                key_b=lambda rc: (int(dist_old[rc[0], rc[1]]), rc[0], rc[1]),
            )
            inv_co = _pairwise_inversions(
                source_set,
                key_a=lambda rc: (int(dist_center[rc[0], rc[1]]), rc[0], rc[1]),
                key_b=lambda rc: (int(dist_old[rc[0], rc[1]]), rc[0], rc[1]),
            )
            score = int(len(set(fusion_top) ^ set(old_top)))
            center_delta = int(len(set(center_top) ^ set(old_top)))
            candidates.append((source_set, fusion_top, score, inv_eo + inv_co * 2 + center_delta * 3))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[2], item[3], len(item[0])))
    source_set, fusion_top, _score, _bonus = candidates[-1]
    return source_set, fusion_top


def _build_expected(layout: np.ndarray, sources: list[tuple[int, int]], chosen: list[tuple[int, int]], distractors: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = layout.copy()
    out = layout.copy()
    chosen_set = set(chosen)
    source_set = set(sources)
    retained = source_set - chosen_set

    for r, c in source_set | set(distractors):
        inp[r, c] = MAROON

    for r, c in retained | set(distractors):
        out[r, c] = MAROON
    for r, c in _dest_cells_fixed():
        out[r, c] = MAROON

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    layout = _layout_base()
    _build_walls(layout)

    selected = _select_sources(layout)
    if selected is None:
        return None
    sources, chosen = selected

    _dist_center, reachable_floor = _reachable_floor_center(layout)
    reachable_set = set(reachable_floor)
    all_floor = {
        (int(r), int(c))
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(layout[r, c]) == FLOOR_COLOR
    }
    unreachable_floor = sorted(all_floor - reachable_set)

    max_distractors = min(DISTRACTOR_MAX, len(unreachable_floor), TOTAL_MAROON_MAX - len(sources))
    distractor_count = random.randint(0, max_distractors) if max_distractors > 0 else 0
    distractors = random.sample(unreachable_floor, distractor_count) if distractor_count > 0 else []

    inp, expected = _build_expected(layout, sources, chosen, distractors)

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None

    stats = behavior_stats(inp)
    if stats["reachable_count"] < REACHABLE_SOURCE_RANGE[0]:
        return None
    if stats["fusion_vs_original_symdiff"] == 0:
        return None
    if stats["center_vs_original_inversions"] == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate valid input for dd6b8c4b combo D3/2+D17/2 after {GEN_RETRIES} retries")

# Variant of 332f06d7
# Dimension: D10 — 阻塞场景处理 (meta)
# Change: 不可达时（stuck）方块停在BFS可达的最远位置（距起点最远的可行放置点），目标方块保持不变... → 停在最近阻塞点前

from __future__ import annotations

from collections import Counter, defaultdict, deque
import heapq
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior:
#    The scene stays close to the original ARC task: a blue winding channel on a
#    green background, with one black n×n mover square and one red n×n target
#    square of the same size. The red square still sits on the border side of
#    the channel and the black square still starts just inside another border.
#    The key new stuck behavior is only visible if the reachable region before
#    the choke is not a simple chain. Therefore stuck inputs here deliberately
#    contain a branch: one branch heads toward the red square and hits a narrow
#    gate; another branch stays reachable and extends farther away.
#    Example trace in placement space:
#      start -> junction -> trunk -> gate-prefix -> [blocked gate] -> goal
#                       \
#                        -> long side branch leaf
#    Old logic picked the side-branch leaf because it maximized BFS distance
#    inside the reachable component. New logic must ignore that lure and stop at
#    gate-prefix, i.e. the last fully feasible n×n placement before the first
#    narrowing on the route toward the goal.
# b) Complexity calibration:
#    The branch is only added in stuck cases and only one choke is carved. This
#    keeps the rule readable while still making the D10 inversion observable.
#    Reachable cases stay close to the base task. The branch is long enough that
#    the "farthest reachable" and "nearest bottleneck" outputs differ on the
#    same input, otherwise the sample is rejected.
# c) Old implicit assumptions that must change:
#    - The original solver identified mover/dest by color ordering. That is not
#      visually inferable, so the new solver instead uses geometry: the border-
#      touching square is the destination and the inset square is the mover.
#    - The original stuck fallback scanned the whole BFS map for max distance.
#      That only works when "stop at farthest reachable" is the rule. Here we
#      instead need a route-to-goal notion, so the solver computes a least-
#      damage path toward the destination in placement space and stops one step
#      before the first placement whose n×n footprint leaks outside the river.
# d) Placement legality:
#    Generator geometry is built on a coarse 4×4 template and then rendered into
#    pixels using separated anchor offsets with gap >= n. The rendered river is a
#    single connected component. The destination is extended to exactly one grid
#    edge; the mover stays inset by one pixel. The choke removes only part of one
#    connector column/row, so the river remains pixel-connected while the n×n
#    square can no longer pass.
# e) Visual observability:
#    Humans can see that one blue branch continues farther but does not lead to
#    the red square, while the red-directed branch pinches shut earlier. Across
#    a few samples, the moved black square consistently stops at that first pinch
#    point rather than at the most distant reachable leaf.
# f) Visual cleanliness:
#    The template is a sparse tree rather than a dense maze. Objects do not
#    overlap; there is only one mover and one destination. The choke is a clean
#    one-column/one-row narrowing, so the blocking location is visible.
# g) Color reasoning:
#    No rule depends on invisible color ids. Colors remain stable in role:
#    black moves, red marks the goal, blue is channel, green is background.


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

_CANON_MAIN = [(3, 2), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 2)]
_CANON_BRANCH = [(1, 1), (1, 0), (2, 0), (3, 0)]
_CANON_CHOKE_EDGE_INDEX = 5  # edge main[5] -> main[6]


def _find_role_squares(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        n = int(bbox_h(b))
        r0 = int(b[0])
        c0 = int(b[1])
        touch = r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W
        edge_dist = min(r0, c0, H - (r0 + n), W - (c0 + n))
        squares.append(
            {
                "col": col,
                "r0": r0,
                "c0": c0,
                "n": n,
                "touch": touch,
                "edge_dist": edge_dist,
            }
        )

    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    edge_touching = [sq for sq in cand if bool(sq["touch"])]
    inset = [sq for sq in cand if not bool(sq["touch"])]
    assert edge_touching and inset

    dest = min(edge_touching, key=lambda sq: (sq["edge_dist"], sq["r0"], sq["c0"]))
    mover = min(inset, key=lambda sq: (sq["edge_dist"], sq["r0"], sq["c0"]))
    return mover, dest


def _compute_feasible_and_defect(trav: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0

    padded = np.pad(trav.astype(int), ((1, 0), (1, 0)), constant_values=0)
    sat = padded.cumsum(axis=0).cumsum(axis=1)

    feasible = np.zeros((h2, w2), dtype=bool)
    defect = np.zeros((h2, w2), dtype=int)
    area = n * n
    for r in range(h2):
        r1 = r + n
        for c in range(w2):
            c1 = c + n
            filled = int(sat[r1, c1] - sat[r, c1] - sat[r1, c] + sat[r, c])
            defect[r, c] = area - filled
            feasible[r, c] = filled == area
    return feasible, defect


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _least_damage_path(defect: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    h2, w2 = defect.shape
    sr, sc = start
    gr, gc = goal

    inf = 10**9
    best_blocked = np.full((h2, w2), inf, dtype=int)
    best_defect = np.full((h2, w2), inf, dtype=int)
    best_steps = np.full((h2, w2), inf, dtype=int)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}

    pq: list[tuple[int, int, int, int, int]] = [(0, 0, 0, sr, sc)]
    best_blocked[sr, sc] = 0
    best_defect[sr, sc] = 0
    best_steps[sr, sc] = 0

    while pq:
        blocked, damage, steps, r, c = heapq.heappop(pq)
        if (
            blocked != int(best_blocked[r, c])
            or damage != int(best_defect[r, c])
            or steps != int(best_steps[r, c])
        ):
            continue
        if (r, c) == (gr, gc):
            break
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h2 and 0 <= cc < w2):
                continue
            step_defect = int(defect[rr, cc])
            cand = (blocked + int(step_defect > 0), damage + step_defect, steps + 1)
            prev = (int(best_blocked[rr, cc]), int(best_defect[rr, cc]), int(best_steps[rr, cc]))
            if cand < prev:
                best_blocked[rr, cc], best_defect[rr, cc], best_steps[rr, cc] = cand
                parent[(rr, cc)] = (r, c)
                heapq.heappush(pq, (cand[0], cand[1], cand[2], rr, cc))

    cur = (gr, gc)
    assert cur in parent
    path = [cur]
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _choose_target(
    feasible: np.ndarray,
    defect: np.ndarray,
    sr: int,
    sc: int,
    dr: int,
    dc: int,
) -> tuple[int, int]:
    dist = _bfs_placement(feasible, sr, sc)
    if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        return dr, dc

    path = _least_damage_path(defect, (sr, sc), (dr, dc))
    prev = path[0]
    for node in path[1:]:
        if int(defect[node]) > 0:
            return prev
        prev = node

    best = -1
    tr, tc = sr, sc
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def _old_farthest_reachable(feasible: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    dist = _bfs_placement(feasible, sr, sc)
    best = -1
    tr, tc = sr, sc
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a solid square through a winding blue channel toward a same-sized edge goal, but stop at the first bottleneck if the path pinches shut.

    Input structure:
    - The corner-dominant background surrounds one connected blue waterway.
    - Two same-sized solid squares lie on that waterway: an inset mover square and a border-touching destination square.

    Rule:
    - The mover may slide only to top-left placements whose full n×n footprint stays inside the waterway or the two marker squares.
    - If the destination placement is reachable, the mover finishes on the destination square.
    - If the route toward the destination becomes too narrow, the mover stops at the last feasible placement immediately before the first blocked placement on the best route toward that destination.
    - The mover's original footprint turns back into river.

    Key insight: In blocked scenes, the right fallback is not "go to the farthest reachable branch"; it is "follow the route toward the goal and stop one step before the first n×n placement that would spill outside the channel."
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _find_role_squares(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible, defect = _compute_feasible_and_defect(trav, n)
    assert bool(feasible[sr, sc])

    tr, tc = _choose_target(feasible, defect, sr, sc, dr, dc)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _sample_offsets(G: int, n: int, count: int) -> list[int] | None:
    low = 1
    high = G - n - 1
    slack = high - low - (count - 1) * n
    if slack < 0:
        return None
    extras = [0] * (count - 1)
    for _ in range(slack):
        extras[random.randrange(count - 1)] += 1
    random.shuffle(extras)
    offsets = [low]
    for extra in extras:
        offsets.append(offsets[-1] + n + extra)
    return offsets


def _transform_cell(cell: tuple[int, int], rot: int, flip: bool) -> tuple[int, int]:
    r, c = cell
    if flip:
        c = 3 - c
    for _ in range(rot % 4):
        r, c = c, 3 - r
    return r, c


def _transform_path(path: list[tuple[int, int]], rot: int, flip: bool) -> list[tuple[int, int]]:
    return [_transform_cell(cell, rot, flip) for cell in path]


def _edge_list(path: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return list(zip(path, path[1:]))


def _render_tree(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> dict[tuple[int, int], tuple[int, int]]:
    pix: dict[tuple[int, int], tuple[int, int]] = {}
    for cr, cc in cells:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pix[(cr, cc)] = (pr, pc)
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for (r1, c1), (r2, c2) in edges:
        pr1, pc1 = pix[(r1, c1)]
        pr2, pc2 = pix[(r2, c2)]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pix


def _extend_dest_to_edge(
    grid: np.ndarray,
    cell: tuple[int, int],
    pix: dict[tuple[int, int], tuple[int, int]],
    n: int,
) -> tuple[int, int]:
    H, W = grid.shape
    pr, pc = pix[cell]
    r, c = cell
    if r == 0:
        grid[0 : pr + n, pc : pc + n] = _RIVER
        return 0, pc
    if r == 3:
        grid[pr:H, pc : pc + n] = _RIVER
        return H - n, pc
    if c == 0:
        grid[pr : pr + n, 0 : pc + n] = _RIVER
        return pr, 0
    assert c == 3
    grid[pr : pr + n, pc:W] = _RIVER
    return pr, W - n


def _carve_choke(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    n: int,
    pix: dict[tuple[int, int], tuple[int, int]],
) -> None:
    pr1, pc1 = pix[a]
    pr2, pc2 = pix[b]
    narrow = random.randint(1, n - 1)
    remove = n - narrow
    side = random.randint(0, 1)

    if pr1 == pr2:
        gate_c = max(pc1, pc2)
        if side == 0:
            rows = slice(pr1, pr1 + remove)
        else:
            rows = slice(pr1 + n - remove, pr1 + n)
        grid[rows, gate_c] = _BG
    else:
        gate_r = max(pr1, pr2)
        if side == 0:
            cols = slice(pc1, pc1 + remove)
        else:
            cols = slice(pc1 + n - remove, pc1 + n)
        grid[gate_r, cols] = _BG


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _expected_output(grid: np.ndarray, mover_r: int, mover_c: int, dest_r: int, dest_c: int, n: int) -> tuple[np.ndarray, tuple[int, int]]:
    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible, defect = _compute_feasible_and_defect(trav, n)
    assert bool(feasible[mover_r, mover_c])
    tr, tc = _choose_target(feasible, defect, mover_r, mover_c, dest_r, dest_c)
    out = grid.copy()
    out[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out, (tr, tc)


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.45
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    min_size = max(10, 4 * n + 2)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    offsets_r = _sample_offsets(G, n, 4)
    offsets_c = _sample_offsets(G, n, 4)
    if offsets_r is None or offsets_c is None:
        return None

    rot = random.randrange(4)
    flip = bool(random.getrandbits(1))
    main_cells = _transform_path(list(_CANON_MAIN), rot, flip)
    branch_cells = _transform_path(list(_CANON_BRANCH), rot, flip)
    choke_a = main_cells[_CANON_CHOKE_EDGE_INDEX]
    choke_b = main_cells[_CANON_CHOKE_EDGE_INDEX + 1]

    grid = np.full((H, W), _BG, dtype=int)
    cells = set(main_cells)
    edges = _edge_list(main_cells)
    if not reached:
        cells.update(branch_cells)
        edges.extend(_edge_list(branch_cells))

    pix = _render_tree(grid, cells, edges, n, offsets_r, offsets_c)
    mover_r, mover_c = pix[main_cells[0]]
    dest_r, dest_c = _extend_dest_to_edge(grid, main_cells[-1], pix, n)

    if not reached:
        _carve_choke(grid, choke_a, choke_b, n, pix)
        if not _river_connected(grid):
            return None

    if not (0 < mover_r < H - n and 0 < mover_c < W - n):
        return None
    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible, defect = _compute_feasible_and_defect(trav, n)
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = bool(
        0 <= dest_r < feasible.shape[0]
        and 0 <= dest_c < feasible.shape[1]
        and feasible[dest_r, dest_c]
        and dist[dest_r, dest_c] >= 0
    )
    if reached != dest_ok:
        return None

    output, target = _expected_output(grid, mover_r, mover_c, dest_r, dest_c, n)
    if reached:
        if target != (dest_r, dest_c):
            return None
    else:
        old_target = _old_farthest_reachable(feasible, mover_r, mover_c)
        if target == old_target:
            return None
        path = _least_damage_path(defect, (mover_r, mover_c), (dest_r, dest_c))
        first_block_index = next((i for i, node in enumerate(path) if i > 0 and int(defect[node]) > 0), None)
        if first_block_index is None or path[first_block_index - 1] != target:
            return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")

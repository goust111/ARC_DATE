# Variant of 332f06d7
# Dimension: D1 — 通道导航方式 (path)
# Change: 沿蓝色水道走廊（corridor_following），方块在宽度>=n的通道内BFS搜索可行放置位置... → 直线射线导航

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components

# Design analysis for the D1 variant
# ----------------------------------
# spec_variant:
#   preserved:
#     grid: square 11..20
#     background: corner-majority green
#     river: single connected blue channel
#     objects:
#       - one black solid n×n square
#       - one red solid n×n square
#       - both squares have the same side length n in 1..4
#   changed_dimension:
#     name: path
#     from: corridor_following
#     to: straight_ray
#   concrete_behavior:
#     placement_space:
#       - keep the original "top-left placement of an n×n square" abstraction
#       - a placement is feasible iff the whole n×n window lies inside {river,mover,dest}
#     motion:
#       - from one feasible placement, the mover chooses one cardinal direction
#       - it slides in that direction as long as the next placement remains feasible
#       - it cannot stop early on interior cells of a straight run
#       - it can change direction only after the slide stops at a wall/boundary/dead-end
#     output:
#       - if the red square sits on a reachable stop-cell, black replaces red
#       - otherwise black moves to the reachable stop-cell with maximum slide count
#       - the old black position is repainted blue
#   behavior_trace:
#     original_on_t_junction:
#       - step-BFS can walk cell-by-cell along a horizontal corridor to the T-junction
#       - from the junction it can immediately turn into the vertical branch and reach red
#     variant_on_same_layout:
#       - ray motion from the left endpoint slides past the junction to the far right endpoint
#       - coming back slides to the far left endpoint
#       - the mover never stops at the branch column, so the vertical branch is unusable
#       - red stays uncovered and black ends at the farthest reachable endpoint
#   complexity_check:
#     - the visual river is still a winding corridor rather than a trivial open room
#     - reachable and unreachable cases both exist
#     - unreachable cases are specifically chosen so old BFS logic would often succeed,
#       making the D1 change visible instead of cosmetic
#   implicit_assumptions_audited:
#     - old solver assumed every feasible placement is a legal stopping point; no longer true
#     - old solver chose mover/destination by color ordering; that is removed here
#     - new solver assumes exactly one of the two equal-sized squares touches the outer border;
#       generator enforces that the red target is boundary-touching and the black mover is not
#     - new solver assumes the river color is the most common non-background, non-square color;
#       generator keeps a single blue river component
#   legality_and_clarity:
#     - generator constructs a sparse orthogonal placement mask, then validates the actual
#       feasible placements recovered from rendered pixels
#     - segments are spaced to avoid accidental merges; samples failing the ray predicates
#       are rejected
#     - no rule depends on color id ordering; target identity is visible from geometry

_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _touches_boundary(r: int, c: int, n: int, H: int, W: int) -> bool:
    return int(r) == 0 or int(c) == 0 or int(r + n) == int(H) or int(c + n) == int(W)


def _compute_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - n + 1
    w2 = W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _slide_endpoint(feasible: np.ndarray, r: int, c: int, dr: int, dc: int) -> tuple[int, int] | None:
    h2, w2 = feasible.shape
    rr, cc = int(r), int(c)
    moved = False
    while True:
        nr = rr + int(dr)
        nc = cc + int(dc)
        if not (0 <= nr < h2 and 0 <= nc < w2 and bool(feasible[nr, nc])):
            break
        rr, cc = int(nr), int(nc)
        moved = True
    if not moved:
        return None
    return int(rr), int(cc)


def _ray_bfs(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            nxt = _slide_endpoint(feasible, int(r), int(c), int(dr), int(dc))
            if nxt is None:
                continue
            rr, cc = nxt
            if int(dist[rr, cc]) != -1:
                continue
            dist[rr, cc] = int(dist[r, c]) + 1
            q.append((int(rr), int(cc)))
    return dist


def _step_bfs(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c]) + 1
                q.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a blue channel using straight-line ray moves.

    Input structure:
    - The corner-majority color is background, with one connected blue channel on top of it.
    - Two same-sized solid squares sit on that channel: a black mover inside the field and a red target touching an outer border.

    Rule:
    - Treat every top-left position whose full NxN footprint stays on the traversable colors as a feasible placement.
    - From any stop, the mover chooses one cardinal direction and keeps sliding until the next step would leave the feasible set.
    - If the red square is a reachable stopping point, the black square lands there; otherwise it lands on the reachable stop with the greatest number of ray slides.
    - The mover's starting footprint turns back into blue river.

    Key insight: Unlike the original free BFS through the corridor, interior cells of a straight run are pass-through only; turns are possible only at places where a slide naturally stops.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in squares:
        by_n[int(s["n"])].append(s)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    boundary_flags = [
        _touches_boundary(int(s["r0"]), int(s["c0"]), int(s["n"]), int(H), int(W))
        for s in cand
    ]
    assert sum(boundary_flags) == 1

    dest = cand[boundary_flags.index(True)]
    mover = cand[boundary_flags.index(False)]

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
    feasible = _compute_feasible(trav, int(n))
    h2, w2 = feasible.shape
    assert 0 <= sr < h2 and 0 <= sc < w2 and bool(feasible[sr, sc])

    dist = _ray_bfs(feasible, int(sr), int(sc))

    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        tr, tc = int(dr), int(dc)
    else:
        best = -1
        tr, tc = int(sr), int(sc)
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = int(r), int(c)

    out = g.copy()
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(river)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover["col"])
    return out


def _size_choices() -> tuple[int, int]:
    n = random.choice([1, 2, 3, 4])
    size_ranges = {
        1: (11, 16),
        2: (13, 17),
        3: (15, 19),
        4: (18, 20),
    }
    lo, hi = size_ranges[int(n)]
    G = random.randint(int(lo), int(hi))
    return int(n), int(G)


def _cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(int(c0), int(c1) + step, step):
            cells.append((int(r0), int(c)))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(int(r0), int(r1) + step, step):
            cells.append((int(r), int(c0)))
    return cells


def _build_mask(P: int, reached: bool) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]] | None:
    if P < 11:
        return None

    r_top = 2
    r_low = P - 3
    c0 = 1
    c1 = max(3, P // 3)
    c2 = max(c1 + 3, (2 * P) // 3)
    c3 = P - 2

    if not (0 < r_top < r_low < P - 1):
        return None
    if not (0 < c0 < c1 < c2 < c3 < P):
        return None

    start = (int(r_low), int(c0))
    turn1 = (int(r_top), int(c0))
    turn2 = (int(r_top), int(c1))
    turn3 = (int(r_low), int(c1))
    trunk_end = (int(r_low), int(c3))

    mask = np.zeros((P, P), dtype=bool)
    segments = [(start, turn1), (turn1, turn2), (turn2, turn3), (turn3, trunk_end)]

    if reached:
        dest = (0, int(c3))
        segments.append((trunk_end, dest))
    else:
        branch = (int(r_low), int(c2))
        dest = (0, int(c2))
        segments.append((branch, dest))

    for a, b in segments:
        for r, c in _cells_between(a, b):
            mask[r, c] = True

    return mask, start, dest


def _paint_mask(mask: np.ndarray, n: int) -> np.ndarray:
    P = mask.shape[0]
    G = P + n - 1
    grid = np.full((G, G), _BG, dtype=int)
    cells = np.argwhere(mask)
    for r, c in cells:
        grid[int(r) : int(r + n), int(c) : int(c + n)] = _RIVER
    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    if k:
        out = np.rot90(out, k)
    return out.copy()


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n, G = _size_choices()
    P = G - n + 1

    built = _build_mask(int(P), bool(reached))
    if built is None:
        return None
    mask, start, dest = built
    grid = _paint_mask(mask, int(n))

    sr, sc = start
    dr, dc = dest

    grid[sr : sr + n, sc : sc + n] = _MOVER
    grid[dr : dr + n, dc : dc + n] = _DEST
    grid = _random_transform(grid)

    H, W = grid.shape
    if not all(int(grid[r, c]) == int(_BG) for r, c in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]):
        return None

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _compute_feasible(trav, int(n))

    comps = []
    for comp in find_components(grid, bg=_BG, connectivity=4, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = grid[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        col = int(comp["color"])
        if bool((patch == int(col)).all()):
            comps.append((col, int(b[0]), int(b[1]), int(bbox_h(b))))
    squares = [s for s in comps if s[0] in (_MOVER, _DEST)]
    if len(squares) != 2:
        return None

    boundary = [s for s in squares if _touches_boundary(int(s[1]), int(s[2]), int(s[3]), int(H), int(W))]
    interior = [s for s in squares if not _touches_boundary(int(s[1]), int(s[2]), int(s[3]), int(H), int(W))]
    if len(boundary) != 1 or len(interior) != 1:
        return None

    mover = interior[0]
    dest_sq = boundary[0]
    if mover[3] != dest_sq[3]:
        return None

    sr, sc, n = int(mover[1]), int(mover[2]), int(mover[3])
    dr, dc = int(dest_sq[1]), int(dest_sq[2])
    if not bool(feasible[sr, sc]) or not bool(feasible[dr, dc]):
        return None

    ray_dist = _ray_bfs(feasible, int(sr), int(sc))
    step_dist = _step_bfs(feasible, int(sr), int(sc))
    ray_reaches = int(ray_dist[dr, dc]) >= 0
    step_reaches = int(step_dist[dr, dc]) >= 0

    if reached:
        if not ray_reaches:
            return None
        if int(ray_dist[dr, dc]) < 2:
            return None
    else:
        if ray_reaches:
            return None
        if not step_reaches:
            return None
        if int(step_dist[dr, dc]) < 3:
            return None

    out = main(grid)
    if out.shape != grid.shape:
        return None

    expected = grid.copy()
    river = _RIVER
    expected[sr : sr + n, sc : sc + n] = river
    if ray_reaches:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(feasible.shape[0]):
            for c in range(feasible.shape[1]):
                if int(ray_dist[r, c]) > best:
                    best = int(ray_dist[r, c])
                    tr, tc = int(r), int(c)
    expected[tr : tr + n, tc : tc + n] = _MOVER

    if not np.array_equal(out, expected):
        return None

    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

# Variant of 332f06d7
# Dimension: D15 — 可达性判定逻辑 (meta)
# Change: 通过BFS在可行放置位置空间上搜索，判断目标位置是否在起点的连通分量内... → 限步可达性

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D15 variant
# - Concrete behavior: the input keeps the original blue river, black mover square,
#   and red destination square. A new yellow straight bar is added in the background.
#   Its pixel length is the move budget in placement-space BFS steps. Example:
#   if the mover-to-destination shortest placement distance is 11 and the yellow bar
#   length is 7, the original task would still move to the red square because the path
#   is connected; this variant instead stops after 7 legal placement moves, at the
#   farthest feasible top-left still within the budget.
# - Complexity calibration: only one extra indicator object is introduced, and it uses
#   one simple degree of freedom (its length). The river remains a single winding
#   corridor, so the visual difference comes from comparing "distance along the channel"
#   versus "visible bar length", not from adding many unrelated objects.
# - Hidden assumptions from the original solver:
#   1. Sorting same-sized squares by color id is no longer acceptable. The new solver
#      assigns roles visually: the destination is the unique same-sized square touching
#      the grid border, while the mover is the corresponding interior square nearest an
#      edge. The generator guarantees this.
#   2. "Reachable iff connected in feasible-placement space" is replaced by
#      "reachable iff shortest-path distance <= budget". Therefore BFS distances must
#      be preserved, not just the visited mask.
#   3. The new solver assumes there is one non-river straight-line marker component.
#      The generator guarantees a single isolated 1-cell-thick bar whose area equals
#      the intended budget.
# - Placement legality: the generator first paints the river and squares, computes the
#   true feasible-placement distances, chooses a budget relative to the destination
#   distance, and only then places the yellow bar into a background-only slot with a
#   one-cell halo. This avoids overlaps, border-corner corruption, and accidental
#   component merges.
# - Visual observability: samples can now show both outcomes on fully passable rivers.
#   When the bar is long enough the black square reaches the red one; when the bar is
#   shorter, the square halts earlier even though the channel is visibly open all the
#   way through. That difference is readable from 2-4 examples.
# - Visual cleanliness: the bar stays detached from the river and from both squares,
#   and corners remain background so background detection stays stable.
# - Color inference: no color remapping is performed. The extra yellow bar carries a
#   structural meaning through its length and isolation, not through any hidden color
#   ordering.


_BG, _RIVER, _DEST, _MOVER, _BUDGET = 3, 1, 2, 0, 4


def _edge_distance(r0: int, c0: int, n: int, h: int, w: int) -> int:
    return min(r0, c0, h - (r0 + n), w - (c0 + n))


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _choose_target(dist: np.ndarray, sr: int, sc: int, dr: int, dc: int, limit: int) -> tuple[int, int]:
    h2, w2 = dist.shape
    if 0 <= dr < h2 and 0 <= dc < w2 and 0 <= dist[dr, dc] <= limit:
        return dr, dc

    tr, tc = sr, sc
    best = (-1, -10**9)
    for r in range(h2):
        for c in range(w2):
            d = int(dist[r, c])
            if d < 0 or d > limit:
                continue
            score = (d, -abs(r - dr) - abs(c - dc))
            if score > best:
                best = score
                tr, tc = r, c
    return tr, tc


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= total * 0.7:
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                break
            neighbors = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            valid = []
            for nb in neighbors:
                if nb == end:
                    valid.append(nb)
                    continue
                test_visited = visited | {nb}
                seen = {end}
                q = deque([end])
                while q:
                    cr, cc = q.popleft()
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr2, nc2 = cr + dr2, cc + dc2
                        if 0 <= nr2 < rows and 0 <= nc2 < cols and (nr2, nc2) not in test_visited and (nr2, nc2) not in seen:
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in test_visited and (ar, ac) in seen:
                        valid.append(nb)
                        break
            if not valid:
                break
            random.shuffle(valid)
            nxt = valid[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        if best_path and best_len >= total * 0.85:
            break
    return best_path


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _compute_offsets(gsz: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = gsz - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append((0, pc, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = h - (pr + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, h) for c in range(pc, pc + n)):
                candidates.append((h - n, pc, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append((pr, 0, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
                candidates.append((pr, 0, dist))
        else:
            dist = w - (pc + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, w)):
                candidates.append((pr, w - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return candidates[0][:2]


def _place_budget_bar(grid: np.ndarray, budget: int) -> bool:
    h, w = grid.shape
    candidates = []
    for r in (1, h - 2):
        if 0 <= r < h:
            for c0 in range(1, w - budget):
                candidates.append((r, c0, 1, budget))
    for c in (1, w - 2):
        if 0 <= c < w:
            for r0 in range(1, h - budget):
                candidates.append((r0, c, budget, 1))
    random.shuffle(candidates)
    for r0, c0, hh, ww in candidates:
        r1 = r0 + hh - 1
        c1 = c0 + ww - 1
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        if not bool((grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == _BG).all()):
            continue
        grid[r0 : r1 + 1, c0 : c1 + 1] = _BUDGET
        return True
    return False


def _feasible_mask(trav: np.ndarray, n: int) -> np.ndarray:
    h, w = trav.shape
    h2, w2 = h - n + 1, w - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _axis_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("Segment endpoints must share a row or a column")


def _join_segments(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _axis_segment(points[i], points[i + 1])
        if path:
            seg = seg[1:]
        path.extend(seg)
    return path


def _build_snake_path(h2: int, w2: int, side: str, compact: bool = False) -> list[tuple[int, int]] | None:
    if h2 < 6 or w2 < 6:
        return None
    left = 1
    right = w2 - 2
    top = 1
    bottom = h2 - 2
    if right <= left or bottom <= top:
        return None

    if side == "top":
        if compact:
            right = min(w2 - 2, left + random.randint(2, min(4, w2 - 3)))
        r0 = bottom
        r1 = max(1, (2 * h2) // 3)
        r2 = max(1, h2 // 3)
        if not (0 < r2 < r1 < r0):
            return None
        waypoints = [(r0, left), (r0, right), (r1, right), (r1, left), (r2, left), (r2, right), (0, right)]
    elif side == "bottom":
        if compact:
            right = min(w2 - 2, left + random.randint(2, min(4, w2 - 3)))
        r0 = top
        r1 = min(h2 - 2, h2 // 3)
        r2 = min(h2 - 2, (2 * h2) // 3)
        if not (r0 < r1 < r2 < h2 - 1):
            return None
        waypoints = [(r0, left), (r0, right), (r1, right), (r1, left), (r2, left), (r2, right), (h2 - 1, right)]
    elif side == "left":
        if compact:
            bottom = min(h2 - 2, top + random.randint(2, min(4, h2 - 3)))
        c0 = right
        c1 = max(1, (2 * w2) // 3)
        c2 = max(1, w2 // 3)
        if not (0 < c2 < c1 < c0):
            return None
        waypoints = [(top, c0), (bottom, c0), (bottom, c1), (top, c1), (top, c2), (bottom, c2), (bottom, 0)]
    else:
        if compact:
            bottom = min(h2 - 2, top + random.randint(2, min(4, h2 - 3)))
        c0 = left
        c1 = min(w2 - 2, w2 // 3)
        c2 = min(w2 - 2, (2 * w2) // 3)
        if not (c0 < c1 < c2 < w2 - 1):
            return None
        waypoints = [(top, c0), (bottom, c0), (bottom, c1), (top, c1), (top, c2), (bottom, c2), (bottom, w2 - 1)]

    path = _join_segments(waypoints)
    if len(set(path)) != len(path):
        return None
    return path


def _render_placement_path(grid: np.ndarray, path: list[tuple[int, int]], n: int) -> None:
    for r, c in path:
        grid[r : r + n, c : c + n] = _RIVER


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel, but only for as many steps as the budget bar allows.

    Input structure:
    - The corner-majority background surrounds one blue river corridor and one detached yellow straight bar.
    - Two same-sized solid squares sit on the river: one interior mover and one border-touching destination.

    Rule:
    - The mover can shift only through top-left placements whose full NxN footprint stays on river/destination/mover cells.
    - The yellow bar length is the maximum number of BFS placement steps the mover may spend.
    - If the destination is within that budget, the mover lands on it; otherwise it stops at the farthest feasible placement still inside the limit.
    - The mover's starting square turns back into river.

    Key insight:
    Connectivity alone is no longer enough; the solver must compare the shortest placement-path length against the visible step budget.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

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
        touch = int(b[0]) == 0 or int(b[1]) == 0 or int(b[2]) == h - 1 or int(b[3]) == w - 1
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": n, "touch": touch})

    assert squares

    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in squares:
        by_n[int(s["n"])].append(s)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    dest_cands = [s for s in cand if bool(s["touch"])]
    mover_cands = [s for s in cand if not bool(s["touch"])]
    assert len(dest_cands) == 1 and mover_cands
    dest = dest_cands[0]
    mover = min(
        mover_cands,
        key=lambda s: _edge_distance(int(s["r0"]), int(s["c0"]), int(s["n"]), h, w),
    )

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == bg:
            continue
        if int(col) in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    budget = None
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in (int(mover["col"]), int(dest["col"]), int(river)):
            continue
        b = comp["bbox"]
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        hh = int(bbox_h(b))
        ww = int(bbox_w(b))
        if hh != 1 and ww != 1:
            continue
        area = int(comp["area"])
        if budget is None or area > budget:
            budget = area
    assert budget is not None and budget >= 0

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2 = int(h - n + 1)
    w2 = int(w - n + 1)
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = _feasible_mask(trav, n)
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    tr, tc = _choose_target(dist, sr, sc, dr, dc, int(budget))

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _gen_attempt() -> np.ndarray | None:
    within_budget = random.random() < 0.5
    n = random.choice([2, 3, 4])

    min_size = max(12, 4 * n + 4)
    if min_size > 20:
        return None
    gsz = random.randint(min_size, 20)
    h = w = gsz
    grid = np.full((h, w), _BG, dtype=int)
    h2, w2 = h - n + 1, w - n + 1
    side = random.choice(["top", "bottom", "left", "right"])
    path = _build_snake_path(h2, w2, side, compact=within_budget)
    if path is None or len(path) < 10 or _count_turns(path) < 4:
        return None
    _render_placement_path(grid, path, n)

    river_frac = float(np.mean(grid == _RIVER))
    if not (0.16 <= river_frac <= 0.58):
        return None

    mover_r, mover_c = path[0]
    dest_r, dest_c = path[-1]
    if mover_r == 0 or mover_r == h2 - 1 or mover_c == 0 or mover_c == w2 - 1:
        return None
    if _edge_distance(mover_r, mover_c, n, h, w) > 2:
        return None
    if not (dest_r == 0 or dest_r == h2 - 1 or dest_c == 0 or dest_c == w2 - 1):
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _feasible_mask(trav, n)
    if not bool(feasible[mover_r, mover_c]) or not bool(feasible[dest_r, dest_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_dist = int(dist[dest_r, dest_c])
    if dest_dist < 5:
        return None

    budget_cap = min(18, max(h, w) - 2)
    if within_budget:
        if dest_dist > budget_cap:
            return None
        budget = random.randint(dest_dist, min(dest_dist + 3, budget_cap))
    else:
        if dest_dist <= 6:
            return None
        lo = max(2, dest_dist // 3)
        hi = min(dest_dist - 2, budget_cap)
        if hi < lo:
            return None
        budget = random.randint(lo, hi)
    if budget < 2:
        return None

    if not _place_budget_bar(grid, budget):
        return None

    tr, tc = _choose_target(dist, mover_r, mover_c, dest_r, dest_c, budget)
    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[tr : tr + n, tc : tc + n] = _MOVER

    if within_budget != (dest_dist <= budget):
        return None
    if within_budget and (tr, tc) != (dest_r, dest_c):
        return None
    if (not within_budget) and (tr, tc) == (dest_r, dest_c):
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")

# Variant of 332f06d7
# Dimension: D10 — 阻塞场景处理 (meta)
# Change: 不可达时（stuck）方块停在BFS可达的最远位置（距起点最远的可行放置点），目标方块保持不变... → 阻塞时反弹

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis
# a) Concrete behavior:
#    The visible objects remain the same as the base task: green background, blue river,
#    one black n×n mover, one red n×n destination. Reachable cases are unchanged.
#    The stuck cases change only at the final black position. Instead of resting at the
#    farthest reachable placement before a choke, the mover now advances toward the red
#    square, fails at a narrow gate, and rebounds to the latest junction on that route.
#    To make the rebound leg visible, the generator adds a short side spur early in the
#    river, so the returned junction is usually several steps away from the start.
# b) Complexity calibration:
#    The rule change is only in the stuck branch. Objects, colors, corridor geometry,
#    and movement model stay intact. The added spur is small and only appears in stuck
#    samples, which keeps the task close to the seed while making the new behavior
#    visually distinguishable from the original "stop at farthest point" outcome.
# c) Old assumptions audited:
#    - The seed solver picked mover/destination by color id ordering. That is not an
#      observable rule, so this variant identifies them by position instead: the red
#      target touches the border, while the black mover does not.
#    - The seed stuck branch assumed the correct target was the max-distance BFS node.
#      That assumption is exactly what D10 changes; we now keep BFS but replace the
#      target-selection logic with "closest reachable point to the red square, then
#      backtrack to the nearest earlier branchpoint or the start".
#    - BFS parent information becomes necessary because rebound depends on the attempted
#      forward route, not just the reachable set.
# d) Placement legality:
#    The generator keeps the squares same-sized, non-overlapping, and within bounds.
#    The side spur uses an unused coarse neighbor of the main walk, so it connects to
#    the river without creating extra solid objects. Chokes are only inserted after the
#    spur, ensuring the rebound junction lies on the attempted route.
# e) Visual observability:
#    Stuck samples are rejected unless the new rebound target differs from both the
#    original farthest-stop target and the start. This prevents silent degeneration into
#    the old task or into identity outputs that would hide the rule change.
# f) Visual cleanliness:
#    Only one extra branch is added, and it is short. The river remains a single
#    connected region, so the image still reads as one channel with one blocked route.
# g) Color inference:
#    No color remapping is introduced. Colors retain their original roles, and the
#    transformation depends only on visible geometry.
# h) Small behavior trace:
#    Original stuck logic: BFS cannot reach red -> choose max-distance reachable node ->
#    black stops just before the gate. Variant logic: BFS cannot reach red -> choose the
#    reachable node closest to red as the failed advance point -> follow parents backward
#    until the first placement with degree >= 3 (or the start) -> place black there.
# i) New solver assumptions checked against generator:
#    The solver assumes one same-sized border-touching square (destination), one
#    same-sized non-border square (mover), a river color distinct from both, and stuck
#    samples whose reachable graph contains an earlier junction on the route to the gate.
#    The generator explicitly constructs and validates those conditions.


DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _touches_edge(square: dict, H: int, W: int) -> bool:
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    n = int(square["n"])
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _bfs_with_parent(feasible: np.ndarray, sr: int, sc: int):
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent = np.full((h2, w2, 2), -1, dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                parent[rr, cc] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _placement_degree(feasible: np.ndarray, r: int, c: int) -> int:
    h2, w2 = feasible.shape
    deg = 0
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc]:
            deg += 1
    return deg


def _pick_farthest_target(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = sr, sc
    h2, w2 = dist.shape
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def _pick_bounce_target(
    feasible: np.ndarray,
    dist: np.ndarray,
    parent: np.ndarray,
    sr: int,
    sc: int,
    dr: int,
    dc: int,
) -> tuple[int, int]:
    h2, w2 = dist.shape
    advance = None
    advance_key = None
    for r in range(h2):
        for c in range(w2):
            d = int(dist[r, c])
            if d < 0:
                continue
            key = (abs(r - dr) + abs(c - dc), -d, abs(r - sr) + abs(c - sc), r, c)
            if advance_key is None or key < advance_key:
                advance_key = key
                advance = (r, c)
    assert not (advance is None)

    cur_r, cur_c = int(advance[0]), int(advance[1])
    while (cur_r, cur_c) != (sr, sc):
        pr, pc = parent[cur_r, cur_c]
        assert not (pr < 0 or pc < 0)
        pr = int(pr)
        pc = int(pc)
        if (pr, pc) == (sr, sc) or _placement_degree(feasible, pr, pc) >= 3:
            return pr, pc
        cur_r, cur_c = pr, pc
    return sr, sc


def _solve_target(
    feasible: np.ndarray,
    sr: int,
    sc: int,
    dr: int,
    dc: int,
) -> tuple[int, int, np.ndarray, np.ndarray, bool]:
    dist, parent = _bfs_with_parent(feasible, sr, sc)
    h2, w2 = feasible.shape
    dest_ok = 0 <= dr < h2 and 0 <= dc < w2 and feasible[dr, dc] and dist[dr, dc] >= 0
    if dest_ok:
        return dr, dc, dist, parent, True
    tr, tc = _pick_bounce_target(feasible, dist, parent, sr, sc, dr, dc)
    return tr, tc, dist, parent, False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding river toward a same-sized border target.

    Input structure:
    - A dominant background color fills the corners, and one river color forms a single corridor.
    - Two same-sized solid squares lie on that corridor: the interior one is the mover, and the
      border-touching one is the destination.

    Rule:
    - The mover can shift by one cell at a time through placements where its full NxN footprint
      remains inside the river system.
    - If the destination placement is reachable, the mover ends on the destination square.
    - If the route is blocked by a narrow gate, the mover does not stay at the last forward spot;
      it rebounds along the attempted route and stops at the nearest earlier junction (or at its
      start if there is no junction).
    - The mover's original location becomes river.

    Key insight:
    Work on the placement grid of valid NxN windows. Reachable cases are ordinary BFS, while
    blocked cases require following the best forward attempt toward the target and then backtracking
    to the latest branching placement.
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
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert not (len(squares) < 2)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in squares:
        by_n[int(s["n"])].append(s)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert not (len(cand) < 2)

    border = [s for s in cand if _touches_edge(s, H, W)]
    interior = [s for s in cand if not _touches_edge(s, H, W)]
    assert not (len(border) != 1 or len(interior) != 1)
    dest = border[0]
    mover = interior[0]

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
    assert not (river is None)

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert not (h2 <= 0 or w2 <= 0)
    assert not (not (0 <= sr < h2 and 0 <= sc < w2))

    feasible = _build_feasible(trav, n)
    assert not (not bool(feasible[sr, sc]))

    tr, tc, _dist, _parent, _reached = _solve_target(feasible, sr, sc, dr, dc)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _hamiltonian_walk(rows, cols, start, end):
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
            for dr, dc in DIR4:
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
                    for dr2, dc2 in DIR4:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in DIR4:
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if (
                        0 <= ar < rows
                        and 0 <= ac < cols
                        and (ar, ac) not in test_visited
                        and (ar, ac) in seen
                    ):
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


def _paint_segment(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int):
    pr1, pc1 = p1
    pr2, pc2 = p2
    if pr1 == pr2:
        c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
    else:
        r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER


def _render_channel(grid, path, n, offsets_r, offsets_c):
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(path) - 1):
        _paint_segment(grid, pixel_pos[i], pixel_pos[i + 1], n)
    return pixel_pos


def _add_chokepoint(grid, pixel_pos_i, pixel_pos_j, n, direction):
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, n - 1)
    remove = n - narrow_w
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        hold_len = random.randint(1, min(6, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + max(1, hold_len)
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if c < 0 or c >= W:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < H:
                    grid[rr, c] = _BG
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(6, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + max(1, hold_len)
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if r < 0 or r >= H:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < W:
                    grid[r, cc] = _BG


def _river_connected(grid):
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return num == 1


def _compute_offsets(G, n, num_coarse):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _add_bounce_spur(grid, walk, pixel_pos, offsets_r, offsets_c, n):
    walk_set = set(walk)
    coarse_rows = len(offsets_r)
    coarse_cols = len(offsets_c)
    end_r, end_c = walk[-1]

    limit = max(2, len(walk) // 3)
    indices = list(range(1, min(len(walk) - 1, limit)))
    random.shuffle(indices)

    best = None
    best_key = None
    for idx in indices:
        cr, cc = walk[idx]
        dirs = list(DIR4)
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < coarse_rows and 0 <= nc < coarse_cols):
                continue
            if (nr, nc) in walk_set:
                continue
            adj_to_walk = 0
            for dr2, dc2 in DIR4:
                if (nr + dr2, nc + dc2) in walk_set:
                    adj_to_walk += 1
            if adj_to_walk != 1:
                continue
            # Prefer branches that do not point toward the destination.
            key = (
                -(abs(nr - end_r) + abs(nc - end_c)),
                idx,
                nr,
                nc,
            )
            if best_key is None or key < best_key:
                best_key = key
                best = (idx, nr, nc)
    if best is None:
        return None

    idx, nr, nc = best
    pr, pc = offsets_r[nr], offsets_c[nc]
    grid[pr : pr + n, pc : pc + n] = _RIVER
    _paint_segment(grid, pixel_pos[idx], (pr, pc), n)
    return idx, (pr, pc)


def _extend_dest_to_edge(grid: np.ndarray, pr: int, pc: int, n: int):
    H, W = grid.shape
    candidates = []
    for direction in ["top", "bottom", "left", "right"]:
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append((0, pc, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = H - (pr + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(
                grid[r, c] in (_BG, _RIVER)
                for r in range(pr + n, H)
                for c in range(pc, pc + n)
                if c < W
            ):
                candidates.append((H - n, pc, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append((pr, 0, 0))
                continue
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                candidates.append((pr, 0, dist))
        else:
            dist = W - (pc + n)
            if dist == 0:
                candidates.append((pr, pc, 0))
                continue
            if all(
                grid[r, c] in (_BG, _RIVER)
                for r in range(pr, pr + n)
                for c in range(pc + n, W)
                if r < H
            ):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return candidates[0][:2]


def _gen_attempt(reached: bool | None = None):
    if reached is None:
        reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(3, (G - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, n, coarse_rows)
    offsets_c = _compute_offsets(G, n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < n:
            return None
    for i in range(len(offsets_c) - 1):
        if offsets_c[i + 1] - offsets_c[i] < n:
            return None

    edge_cells = set()
    for c in range(coarse_cols):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    result = _extend_dest_to_edge(grid, dest_r, dest_c, n)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    touching = mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = _RIVER
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = _RIVER
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = _RIVER
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    spur_idx = None
    if not reached:
        spur_info = _add_bounce_spur(grid, walk, pixel_pos, offsets_r, offsets_c, n)
        if spur_info is None:
            return None
        spur_idx, _spur_pixel = spur_info

    if not reached and n >= 2:
        min_seg = max(max(1, len(walk) // 4), int(spur_idx) + 2)
        available = list(range(min_seg, len(walk) - 1))
        if not available:
            return None
        random.shuffle(available)
        num_choke = random.randint(1, min(2, len(available)))
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _build_feasible(trav, n)
    if not feasible[mover_r, mover_c]:
        return None

    tr, tc, dist, parent, dest_ok = _solve_target(feasible, mover_r, mover_c, dest_r, dest_c)
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None
    if not reached:
        old_tr, old_tc = _pick_farthest_target(dist, mover_r, mover_c)
        if (tr, tc) == (old_tr, old_tc):
            return None
        if (tr, tc) == (mover_r, mover_c):
            return None
        if int(dist[tr, tc]) < 1:
            return None
        # Ensure the branchpoint actually lies on the backtracked route, not on an arbitrary open area.
        cur = _pick_bounce_target(feasible, dist, parent, mover_r, mover_c, dest_r, dest_c)
        if cur != (tr, tc):
            return None

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[tr : tr + n, tc : tc + n] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    target_reached = random.random() < 0.5
    for _ in range(300):
        result = _gen_attempt(reached=target_reached)
        if result is not None:
            return result
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

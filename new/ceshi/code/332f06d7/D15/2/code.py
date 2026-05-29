# Variant of 332f06d7
# Dimension: D15 — 可达性判定逻辑 (meta)
# Change: 通过BFS在可行放置位置空间上搜索，判断目标位置是否在起点的连通分量内... → 时间衰减可达性

from __future__ import annotations

from collections import Counter, defaultdict, deque
from functools import lru_cache
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D15 variant.
#
# a) Concrete behavior and trace:
#    - The visible objects stay the same as the source task: green background, blue river,
#      one black NxN mover square, one red NxN destination square.
#    - The key change is temporal. The black square still moves in unit top-left steps over
#      feasible NxN placements, but whenever it leaves some blue channel pixels behind, those
#      vacated pixels collapse to green and remain unusable.
#    - This affects both path search and the final picture. In the source task, a reachable
#      target kept the river intact except at the start cell; here the traversed prefix of the
#      river disappears, so the output records the irreversible motion.
#    - Small trace for n=2 on a horizontal segment:
#        start at columns 5-6, move to 4-5, then 3-4.
#        Source logic: previously occupied blue pixels stay blue, so the whole corridor remains.
#        Variant logic: after the first move, column 6 collapses to green; after the second
#        move, column 5 collapses too. The square can no longer return to the right.
#
# b) Complexity calibration:
#    - n is restricted to 2-4 in the generator so the decay is visible. For n=1 the dynamic
#      effect is much less informative and often looks too similar to the source task.
#    - The river remains a long winding corridor with repeated straight segments and several
#      turns, so the solver still has to infer a geometric movement rule instead of matching a
#      one-off pattern.
#
# c) Old implicit assumptions that no longer hold:
#    - Static BFS reachability is no longer sufficient: reachability now depends on the set of
#      already collapsed pixels, so the state must include both current placement and collapse.
#    - The original code used color-id ordering to decide mover vs destination. That is banned
#      here. The new solver instead uses a visible geometric cue supplied by the generator:
#      the destination touches the border, the mover stays near a border without touching it.
#    - Filling only the original mover cell with river is no longer enough; the whole traversed
#      prefix must be collapsed step by step.
#
# d) Placement legality guaranteed by the generator:
#    - The generator still paints the river first, then overlays the two NxN squares.
#    - Only one same-sized square touches the border (destination), and the mover is near an
#      edge but not touching it. They never overlap and always lie on the river.
#    - Stuck cases still use a true geometric chokepoint narrower than n, so the final stop is
#      physically well-defined even under the decay rule.
#
# e) Visual observability:
#    - The new rule is visible in output because much of the blue path vanishes behind the mover.
#      This makes the time-decay constraint inferable from 2-4 examples without relying on
#      hidden metadata such as color ids or generation order.
#
# f) Visual cleanliness:
#    - Only four colors are used, as in the source task. The river is a single connected blue
#      structure; the two solid squares remain easy to identify.
#    - The generator keeps the destination on a border and the mover near a border, which makes
#      their roles visually stable across samples.
#
# g) New solver assumptions checked against generation:
#    - There are exactly two relevant solid squares of the chosen size.
#    - Exactly one of them touches the border.
#    - The river color is the most common remaining non-background color.
#    - The feasible placement graph is small enough that a memoized search over
#      (top-left placement, collapsed-pixel bitmask) remains practical.


def _touches_edge(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _edge_distance(r0: int, c0: int, n: int, H: int, W: int) -> int:
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _window_bitmask(r0: int, c0: int, n: int, W: int) -> int:
    bits = 0
    for rr in range(r0, r0 + n):
        row_base = rr * W
        for cc in range(c0, c0 + n):
            bits |= 1 << (row_base + cc)
    return bits


def _set_bits_to_color(arr: np.ndarray, bits: int, color: int) -> None:
    H, W = arr.shape
    idx = 0
    while bits:
        if bits & 1:
            r, c = divmod(idx, W)
            arr[r, c] = color
        bits >>= 1
        idx += 1


def _detect_roles(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        n = int(bbox_h(b))
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": n})

    assert not (len(squares) < 2)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in squares:
        by_n[int(s["n"])].append(s)

    ranked = sorted(by_n.items(), key=lambda kv: (-len(kv[1]), -kv[0]))
    for nside, cand in ranked:
        edge = [s for s in cand if _touches_edge(int(s["r0"]), int(s["c0"]), int(nside), H, W)]
        inner = [s for s in cand if not _touches_edge(int(s["r0"]), int(s["c0"]), int(nside), H, W)]
        if len(edge) == 1 and len(inner) >= 1:
            mover = min(
                inner,
                key=lambda s: (
                    _edge_distance(int(s["r0"]), int(s["c0"]), int(nside), H, W),
                    int(s["r0"]),
                    int(s["c0"]),
                ),
            )
            return mover, edge[0]

    assert False


def _search_decay_path(
    feasible: np.ndarray,
    foot_masks: dict[tuple[int, int], int],
    start: tuple[int, int],
    dest: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    h2, w2 = feasible.shape
    dest_r, dest_c = dest

    def ordered_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]):
                out.append((rr, cc))
        out.sort(key=lambda p: (abs(p[0] - dest_r) + abs(p[1] - dest_c), p[0], p[1]))
        return out

    def stop_key(pos: tuple[int, int], steps: int) -> tuple[int, int, int, int]:
        return (
            int(steps),
            -abs(int(pos[0]) - dest_r) - abs(int(pos[1]) - dest_c),
            -int(pos[0]),
            -int(pos[1]),
        )

    @lru_cache(maxsize=None)
    def dfs(r: int, c: int, collapsed: int) -> tuple[bool, int, tuple[int, int] | None, int, tuple[int, int], tuple[int, int] | None]:
        pos = (int(r), int(c))
        current_mask = foot_masks[pos]
        reach_dest = pos == dest
        best_dest_len = 0 if reach_dest else 10**9
        best_dest_next: tuple[int, int] | None = None

        best_stop_steps = 0
        best_stop_pos = pos
        best_stop_next: tuple[int, int] | None = None
        best_stop_rank = stop_key(best_stop_pos, best_stop_steps)

        for nxt in ordered_neighbors(int(r), int(c)):
            next_mask = foot_masks[nxt]
            if next_mask & int(collapsed):
                continue
            new_collapsed = int(collapsed) | (int(current_mask) & ~int(next_mask))
            child = dfs(int(nxt[0]), int(nxt[1]), int(new_collapsed))

            if bool(child[0]):
                cand_len = int(child[1]) + 1
                cand_rank = (cand_len, int(nxt[0]), int(nxt[1]))
                cur_rank = (best_dest_len, 10**9 if best_dest_next is None else int(best_dest_next[0]), 10**9 if best_dest_next is None else int(best_dest_next[1]))
                if (not reach_dest) or cand_rank < cur_rank:
                    reach_dest = True
                    best_dest_len = cand_len
                    best_dest_next = nxt

            cand_stop_steps = int(child[3]) + 1
            cand_stop_pos = child[4]
            cand_stop_rank = stop_key(cand_stop_pos, cand_stop_steps)
            if cand_stop_rank > best_stop_rank:
                best_stop_steps = cand_stop_steps
                best_stop_pos = cand_stop_pos
                best_stop_next = nxt
                best_stop_rank = cand_stop_rank

        return (
            bool(reach_dest),
            int(best_dest_len),
            best_dest_next,
            int(best_stop_steps),
            best_stop_pos,
            best_stop_next,
        )

    path = [start]
    cur = start
    collapsed = 0
    root = dfs(int(cur[0]), int(cur[1]), int(collapsed))
    reached = bool(root[0])
    while True:
        state = dfs(int(cur[0]), int(cur[1]), int(collapsed))
        nxt = state[2] if reached else state[5]
        if nxt is None:
            break
        collapsed |= foot_masks[cur] & ~foot_masks[nxt]
        cur = nxt
        path.append(cur)
        if reached and cur == dest:
            break
    return path, reached


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a solid square through a blue channel while the used-up river collapses behind it.

    Input structure:
    - The background is the corner-majority color, and a blue corridor forms the only route.
    - Two same-sized solid squares sit on that corridor: a black mover near an edge and a red
      destination touching the border.

    Rule:
    - The mover advances in unit top-left steps through placements where its full NxN body fits.
    - After each step, any channel pixels the mover has just left become background permanently.
    - If the destination can still be reached under that irreversible collapse, the mover ends on
      the red square; otherwise it stops at the deepest terminal placement it can force.

    Key insight:
    - Reachability is no longer a static connected-component question. The solver must track both
      the current placement and which traversed river pixels have already disappeared.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _detect_roles(g, bg)

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
    assert not (river is None)

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert not (h2 <= 0 or w2 <= 0)
    assert not (not (0 <= sr < h2 and 0 <= sc < w2 and 0 <= dr < h2 and 0 <= dc < w2))

    feasible = np.zeros((h2, w2), dtype=bool)
    foot_masks: dict[tuple[int, int], int] = {}
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
                foot_masks[(int(r), int(c))] = _window_bitmask(int(r), int(c), int(n), int(W))

    assert not (not bool(feasible[sr, sc]))
    assert not (not bool(feasible[dr, dc]))

    path, _reached = _search_decay_path(feasible, foot_masks, (int(sr), int(sc)), (int(dr), int(dc)))
    tr, tc = path[-1]

    out = g.copy()
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(river)
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        vacated = int(foot_masks[(int(r1), int(c1))]) & ~int(foot_masks[(int(r2), int(c2))])
        _set_bits_to_color(out, int(vacated), int(bg))
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover["col"])
    return out


# --- Generator ---

_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _static_bfs(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((int(rr), int(cc)))
    return dist


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= int(total * 0.7):
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
        if best_path and best_len >= int(total * 0.85):
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
        pr, pc = offsets_r[int(cr)], offsets_c[int(cc)]
        pixel_pos.append((int(pr), int(pc)))
        grid[int(pr) : int(pr + n), int(pc) : int(pc + n)] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[int(pr1) : int(pr1 + n), int(c_lo) : int(c_hi)] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[int(r_lo) : int(r_hi), int(pc1) : int(pc1 + n)] = _RIVER
    return pixel_pos


def _add_chokepoint(
    grid: np.ndarray,
    pixel_pos_i: tuple[int, int],
    pixel_pos_j: tuple[int, int],
    n: int,
    direction: str,
) -> None:
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
        total_len = 2 * remove + hold_len
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if not (0 <= c < W):
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
                    grid[int(rr), int(c)] = _BG
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(6, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if not (0 <= r < H):
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
                    grid[int(r), int(cc)] = _BG


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _num_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns += 1
    return turns


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.55
    n = random.choice([2, 3, 4])

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
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
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
    if walk is None or len(walk) < 5 or _num_turns(walk) < 3:
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

    def _extend_to_edge(pr: int, pc: int) -> tuple[int, int] | None:
        candidates = []
        for direction in ("top", "bottom", "left", "right"):
            if direction == "top":
                dist = pr
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = H - (pr + n)
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n)):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
                    candidates.append((pr, 0, dist))
            else:
                dist = W - (pc + n)
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W)):
                    candidates.append((pr, W - n, dist))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[2])
        return int(candidates[0][0]), int(candidates[0][1])

    result = _extend_to_edge(dest_r, dest_c)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[int(r_lo) : int(r_hi), int(dest_c) : int(dest_c + n)] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[int(dest_r) : int(dest_r + n), int(c_lo) : int(c_hi)] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if not _touches_edge(dest_r, dest_c, n, H, W):
        return None

    if _touches_edge(mover_r, mover_c, n, H, W):
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
    if _touches_edge(mover_r, mover_c, n, H, W):
        return None
    if _edge_distance(mover_r, mover_c, n, H, W) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not reached:
        num_choke = random.randint(1, min(3, len(walk) - 2))
        min_seg = max(1, len(walk) // 4)
        available = list(range(min_seg, len(walk) - 1))
        random.shuffle(available)
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    counts = Counter(map(int, grid.flatten()))
    bg = int(Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0])
    river = None
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (_MOVER, _DEST):
            continue
        river = int(col)
        break
    if river is None:
        return None

    trav = (grid == river) | (grid == _MOVER) | (grid == _DEST)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not feasible[mover_r, mover_c] or not feasible[dest_r, dest_c]:
        return None

    static_dist = _static_bfs(feasible, mover_r, mover_c)
    static_reach = int(static_dist[dest_r, dest_c]) >= 0
    if reached and not static_reach:
        return None
    if not reached and static_reach:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out == _MOVER):
        return None
    if not np.array_equal(main(grid), out):
        return None

    mover_cells = np.argwhere(out == _MOVER)
    if mover_cells.size == 0:
        return None
    tr, tc = mover_cells.min(axis=0)
    reached_now = int(tr) == int(dest_r) and int(tc) == int(dest_c)
    if reached != reached_now:
        return None

    collapsed_blue = int(((grid == _RIVER) & (out == _BG)).sum())
    if collapsed_blue < n:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

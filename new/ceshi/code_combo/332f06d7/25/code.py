# Combo variant of 332f06d7 (index 25)
# combo_spec: D2/1+D13/1
# Name: 渐变宽度+停在中间
#
# Design analysis
# ---------------
# a) Concrete form + behavior trace:
#    - Visuals stay close to the base task: green background, one connected blue river,
#      one black n×n mover, one red n×n destination of the same size.
#    - The fused change is concentrated on the final straight tail leading to the red square.
#      One river wall stays fixed while the opposite wall steps inward monotonically,
#      so the tail width visibly shrinks as it approaches the target.
#    - Reachable samples taper from width n+extra down to width n, so the mover still
#      reaches the target.
#    - Blocked samples taper from width n+extra down to a spine of width 1..n-1.
#      The first place where the width drops below n defines the effective chokepoint.
#      Instead of stopping at the last reachable placement right before that chokepoint,
#      the mover stops at the midpoint of the BFS path from its start to that frontier.
#    - Small trace with n=3 and a horizontal tail:
#      base task widths: 3,3,3,1,1,... -> stop at the last width-3 placement.
#      D2/1 only:       4,4,3,2,2,... -> frontier is the last placement still fitting 3×3.
#      fused combo:     same frontier as D2/1, but output stop is halfway from the start
#      to that frontier, so if the frontier is at BFS distance 8 the output stop is at 4.
# b) Complexity calibration:
#    - Only one taper zone is introduced, always on the final straight tail.
#    - The taper is monotone and one-sided; widths use only a few integer levels, so the
#      shrinking trend is visible without clutter.
#    - Blocked cases require a frontier distance of at least 4 and a midpoint distinct from
#      the frontier, making the "stop in the middle" behavior visible in 2-4 examples.
# c) Implicit-assumption review of main():
#    - Square detection still assumes the mover and destination are solid same-sized squares.
#      The generator preserves this.
#    - River-color detection still assumes the remaining dominant non-background color is
#      the river. No extra marker colors are introduced.
#    - Feasible-window BFS remains the right geometric primitive: it already captures the
#      n×n clearance test needed for gradual narrowing.
#    - The base task assumed "unreachable -> farthest BFS distance". That is no longer valid.
#      Here the meaningful blocked frontier is the reachable placement closest to the target,
#      because the taper can create upstream slack before the true narrowing point.
#    - The new midpoint logic assumes that chosen frontier is unique enough to be stable.
#      The generator enforces a single taper, no loops, and a unique best frontier under the
#      same target-closeness key used by the solver.
# d) Placement legality:
#    - The winding prefix is painted first; the taper is only added into clear space beyond
#      the final coarse cell, so it does not overlap the prefix.
#    - The destination still touches exactly one outer edge and avoids corners.
#    - The mover is kept near an edge but not touching it after a one-cell inward adjustment.
#    - The river remains pixel-connected in blocked cases because the narrow spine still links
#      the prefix to the destination pocket even when n×n movement is impossible.
# e) Visual observability:
#    - Humans can see the one-sided staircase narrowing near the target.
#    - Comparing reachable and blocked samples shows that the mover does not simply stop at
#      the narrowing; it stops noticeably earlier, roughly halfway along the usable route.
#    - No hidden ordering by color id is required to infer the rule; the behavior follows
#      directly from the visible geometry of the river.
# f) Visual cleanliness:
#    - There is still only one river and one mover/target pair.
#    - The taper is axis-aligned and isolated on the tail, avoiding noisy overlaps with the
#      winding prefix.
#    - Width expansion is capped at a small extra band, so the tail looks like a clean funnel
#      instead of a large blob.
# g) Change inferability:
#    - No new color mapping is added.
#    - The critical cue is structural: a monotone funnel plus a midpoint stop in blocked cases.
#      That cue repeats consistently across generated examples.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _detect_square_pair(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append({"col": int(col), "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})
    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2
    cand.sort(key=lambda s: int(s["col"]))
    return cand[0], cand[1]


def _choose_river_color(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover_col), int(dest_col)):
            continue
        return int(col)
    raise AssertionError("river color not found")


def _compute_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _bfs_with_parents(feasible: np.ndarray, sr: int, sc: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent_r = -np.ones((h2, w2), dtype=int)
    parent_c = -np.ones((h2, w2), dtype=int)
    q = deque([(int(sr), int(sc))])
    dist[int(sr), int(sc)] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                parent_r[rr, cc] = int(r)
                parent_c[rr, cc] = int(c)
                q.append((rr, cc))
    return dist, parent_r, parent_c


def _closest_reachable_frontier(dist: np.ndarray, dr: int, dc: int) -> tuple[int, int, tuple[int, int, int, int]]:
    best_key = None
    best_pos = (0, 0)
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            d = int(dist[r, c])
            if d < 0:
                continue
            key = (abs(r - dr) + abs(c - dc), -d, r, c)
            if best_key is None or key < best_key:
                best_key = key
                best_pos = (int(r), int(c))
    assert best_key is not None
    return best_pos[0], best_pos[1], best_key


def _midpoint_from_frontier(
    dist: np.ndarray,
    parent_r: np.ndarray,
    parent_c: np.ndarray,
    frontier_r: int,
    frontier_c: int,
) -> tuple[int, int]:
    target_dist = int(dist[int(frontier_r), int(frontier_c)]) // 2
    tr, tc = int(frontier_r), int(frontier_c)
    while int(dist[tr, tc]) > target_dist:
        nr, nc = int(parent_r[tr, tc]), int(parent_c[tr, tc])
        assert nr >= 0 and nc >= 0
        tr, tc = nr, nc
    assert int(dist[tr, tc]) == target_dist
    return int(tr), int(tc)


def _select_target(feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int) -> dict:
    assert bool(feasible[int(sr), int(sc)])
    dist, parent_r, parent_c = _bfs_with_parents(feasible, sr, sc)
    h2, w2 = feasible.shape
    dest_ok = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0
    if dest_ok:
        return {
            "dist": dist,
            "dest_ok": True,
            "tr": int(dr),
            "tc": int(dc),
            "frontier_r": int(dr),
            "frontier_c": int(dc),
            "frontier_d": int(dist[int(dr), int(dc)]),
            "frontier_key": (0, -int(dist[int(dr), int(dc)]), int(dr), int(dc)),
        }

    fr, fc, frontier_key = _closest_reachable_frontier(dist, dr, dc)
    tr, tc = _midpoint_from_frontier(dist, parent_r, parent_c, fr, fc)
    return {
        "dist": dist,
        "dest_ok": False,
        "tr": int(tr),
        "tc": int(tc),
        "frontier_r": int(fr),
        "frontier_c": int(fc),
        "frontier_d": int(dist[int(fr), int(fc)]),
        "frontier_key": frontier_key,
    }


def _analyze_grid(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int, int, int, int]:
    g = np.array(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _detect_square_pair(g, bg)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    river = _choose_river_color(g, bg, int(mover["col"]), int(dest["col"]))
    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _compute_feasible(trav, n)
    assert bool(feasible[int(sr), int(sc)])
    return g, river, int(mover["col"]), n, sr, sc, dr, dc, feasible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel toward a same-sized target square.

    Combo rule:
    - The tail can gradually taper as it approaches the destination.
    - Reachable case: unchanged, the mover covers the destination.
    - Blocked case: find the reachable placement closest to the destination (the upstream
      frontier induced by the taper), then stop halfway along the BFS path from start to that
      frontier.
    """
    g, river, mover_col, n, sr, sc, dr, dc, feasible = _analyze_grid(grid)
    choice = _select_target(feasible, sr, sc, dr, dc)
    tr, tc = int(choice["tr"]), int(choice["tc"])

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover_col)
    return out


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= total * 0.65:
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                break
            neighbors = []
            for dr, dc in _DIR4:
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
                    for dr2, dc2 in _DIR4:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
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
        if best_path and best_len >= total * 0.8:
            break
    return best_path


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns += 1
    return turns


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
    pixel_pos: list[tuple[int, int]] = []
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


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return num == 1


def _profile_widths(body_len: int, start_w: int, end_w: int, taper_len: int) -> list[int]:
    assert body_len >= taper_len >= 1
    if taper_len == 1:
        lead = [int(end_w)]
    else:
        lead = []
        diff = start_w - end_w
        for i in range(taper_len):
            w = start_w - (diff * i) // (taper_len - 1)
            lead.append(int(w))
        lead[-1] = int(end_w)
    return lead + [int(end_w)] * (body_len - taper_len)


def _tail_clear(
    grid: np.ndarray,
    pr: int,
    pc: int,
    n: int,
    direction: str,
    side: int,
    extra: int,
) -> bool:
    H, W = grid.shape
    if direction == "right":
        rows = slice(pr - extra, pr + n) if side < 0 else slice(pr, pr + n + extra)
        cols = slice(pc + n, W)
    elif direction == "left":
        rows = slice(pr - extra, pr + n) if side < 0 else slice(pr, pr + n + extra)
        cols = slice(0, pc)
    elif direction == "down":
        rows = slice(pr + n, H)
        cols = slice(pc - extra, pc + n) if side < 0 else slice(pc, pc + n + extra)
    else:
        rows = slice(0, pr)
        cols = slice(pc - extra, pc + n) if side < 0 else slice(pc, pc + n + extra)
    if rows.start < 0 or cols.start < 0 or rows.stop > H or cols.stop > W:
        return False
    patch = grid[rows, cols]
    return bool((patch == _BG).all())


def _pick_tail_plan(grid: np.ndarray, pr: int, pc: int, n: int, reached: bool) -> dict | None:
    H, W = grid.shape
    candidates: list[dict] = []
    spine_choices = [n] if reached else list(range(1, n))

    for direction in random.sample(["right", "left", "down", "up"], 4):
        if direction in ("right", "left") and (pr <= 0 or pr + n >= H):
            continue
        if direction in ("down", "up") and (pc <= 0 or pc + n >= W):
            continue

        if direction == "right":
            dest_r, dest_c = pr, W - n
            body_len = dest_c - pc
            room_neg = pr
            room_pos = H - (pr + n)
        elif direction == "left":
            dest_r, dest_c = pr, 0
            body_len = pc - dest_c
            room_neg = pr
            room_pos = H - (pr + n)
        elif direction == "down":
            dest_r, dest_c = H - n, pc
            body_len = dest_r - pr
            room_neg = pc
            room_pos = W - (pc + n)
        else:
            dest_r, dest_c = 0, pc
            body_len = pr - dest_r
            room_neg = pc
            room_pos = W - (pc + n)

        if body_len < 5:
            continue

        for side, room in [(-1, room_neg), (1, room_pos)]:
            if room < 1:
                continue
            max_extra = min(room, 2)
            min_extra = 1 if n == 1 else 1
            for extra in range(max_extra, min_extra - 1, -1):
                if not _tail_clear(grid, pr, pc, n, direction, side, extra):
                    continue
                start_w = n + extra
                for spine_w in spine_choices:
                    if not reached and spine_w >= n:
                        continue
                    diff = start_w - spine_w
                    min_taper = max(3, diff + 1)
                    if body_len < min_taper + 1:
                        continue
                    taper_len = random.randint(min_taper, body_len)
                    candidates.append(
                        {
                            "direction": direction,
                            "side": side,
                            "extra": int(extra),
                            "start_w": int(start_w),
                            "spine_w": int(spine_w),
                            "taper_len": int(taper_len),
                            "dest_r": int(dest_r),
                            "dest_c": int(dest_c),
                            "body_len": int(body_len),
                        }
                    )
    if not candidates:
        return None
    candidates.sort(key=lambda plan: (plan["body_len"], plan["start_w"] - plan["spine_w"], plan["extra"]), reverse=True)
    return random.choice(candidates[: min(8, len(candidates))])


def _paint_tapered_tail(grid: np.ndarray, pr: int, pc: int, n: int, plan: dict) -> tuple[int, int]:
    H, W = grid.shape
    direction = str(plan["direction"])
    side = int(plan["side"])
    start_w = int(plan["start_w"])
    spine_w = int(plan["spine_w"])
    taper_len = int(plan["taper_len"])
    dest_r = int(plan["dest_r"])
    dest_c = int(plan["dest_c"])

    if direction in ("right", "left"):
        body_len = abs(dest_c - pc)
    else:
        body_len = abs(dest_r - pr)
    widths = _profile_widths(body_len, start_w, spine_w, taper_len)

    if direction == "right":
        for offset, c in enumerate(range(pc, dest_c)):
            w = widths[offset]
            r0 = pr + n - w if side < 0 else pr
            grid[r0 : r0 + w, c] = _RIVER
        grid[pr : pr + n, dest_c:W] = _RIVER
    elif direction == "left":
        cols = list(range(dest_c + n, pc + n))
        for offset, c in enumerate(reversed(cols)):
            w = widths[offset]
            r0 = pr + n - w if side < 0 else pr
            grid[r0 : r0 + w, c] = _RIVER
        grid[pr : pr + n, : dest_c + n] = _RIVER
    elif direction == "down":
        for offset, r in enumerate(range(pr, dest_r)):
            w = widths[offset]
            c0 = pc + n - w if side < 0 else pc
            grid[r, c0 : c0 + w] = _RIVER
        grid[dest_r:H, pc : pc + n] = _RIVER
    else:
        rows = list(range(dest_r + n, pr + n))
        for offset, r in enumerate(reversed(rows)):
            w = widths[offset]
            c0 = pc + n - w if side < 0 else pc
            grid[r, c0 : c0 + w] = _RIVER
        grid[: dest_r + n, pc : pc + n] = _RIVER
    return int(dest_r), int(dest_c)


def _build_walk(G: int, n: int) -> tuple[list[tuple[int, int]], list[int], list[int]] | None:
    min_spacing = n + 1
    max_coarse = max(4, (G - 2) // min_spacing)
    coarse_dim = random.randint(4, min(6, max_coarse))
    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
        return None

    edge_cells = set()
    for c in range(coarse_dim):
        edge_cells.add((0, c))
        edge_cells.add((coarse_dim - 1, c))
    for r in range(1, coarse_dim - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_dim - 1))
    edge_cells = list(edge_cells)
    interior_cells = [(r, c) for r in range(1, coarse_dim - 1) for c in range(1, coarse_dim - 1)]
    if not interior_cells:
        return None

    random.shuffle(edge_cells)
    random.shuffle(interior_cells)
    for start_coarse in edge_cells[: min(8, len(edge_cells))]:
        ranked_ends = sorted(
            interior_cells,
            key=lambda cell: -(abs(cell[0] - start_coarse[0]) + abs(cell[1] - start_coarse[1])),
        )
        for end_coarse in ranked_ends[: min(10, len(ranked_ends))]:
            walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
            if walk is None or len(walk) < 5 or _turn_count(walk) < 3:
                continue
            return walk, offsets_r, offsets_c
    return None


def _adjust_mover_inward(grid: np.ndarray, mover_r: int, mover_c: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
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
    return int(mover_r), int(mover_c)


def _count_same_frontier_key(dist: np.ndarray, dr: int, dc: int, frontier_key: tuple[int, int, int, int]) -> int:
    count = 0
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            d = int(dist[r, c])
            if d < 0:
                continue
            key = (abs(r - dr) + abs(c - dc), -d, r, c)
            if key == frontier_key:
                count += 1
    return count


def _gen_attempt(return_meta: bool = False):
    reached = random.random() < 0.45
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return (None, None) if return_meta else None
    G = random.randint(min_size, 20)
    H = W = G

    built = _build_walk(G, n)
    if built is None:
        return (None, None) if return_meta else None
    walk, offsets_r, offsets_c = built

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)
    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.55 or np.any(mask, axis=0).sum() / W < 0.55:
        return (None, None) if return_meta else None

    mover_r, mover_c = pixel_pos[0]
    tail_r, tail_c = pixel_pos[-1]
    plan = _pick_tail_plan(grid, tail_r, tail_c, n, reached)
    if plan is None:
        return (None, None) if return_meta else None
    dest_r, dest_c = _paint_tapered_tail(grid, tail_r, tail_c, n, plan)

    adjusted = _adjust_mover_inward(grid, mover_r, mover_c, n)
    if adjusted is None:
        return (None, None) if return_meta else None
    mover_r, mover_c = adjusted

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST
    if not _river_connected(grid):
        return (None, None) if return_meta else None

    g, river, mover_col, n2, sr, sc, dr, dc, feasible = _analyze_grid(grid)
    if n2 != n or river != _RIVER or mover_col != _MOVER:
        return (None, None) if return_meta else None
    choice = _select_target(feasible, sr, sc, dr, dc)
    dest_ok = bool(choice["dest_ok"])
    if dest_ok != reached:
        return (None, None) if return_meta else None

    if not dest_ok:
        frontier_d = int(choice["frontier_d"])
        if frontier_d < 4:
            return (None, None) if return_meta else None
        if (int(choice["tr"]), int(choice["tc"])) == (int(choice["frontier_r"]), int(choice["frontier_c"])):
            return (None, None) if return_meta else None
        if _count_same_frontier_key(choice["dist"], dr, dc, tuple(choice["frontier_key"])) != 1:
            return (None, None) if return_meta else None

        old_best = -1
        old_tr, old_tc = sr, sc
        for r in range(choice["dist"].shape[0]):
            for c in range(choice["dist"].shape[1]):
                if int(choice["dist"][r, c]) > old_best:
                    old_best = int(choice["dist"][r, c])
                    old_tr, old_tc = int(r), int(c)
        if (old_tr, old_tc) == (int(choice["tr"]), int(choice["tc"])):
            return (None, None) if return_meta else None

    output = g.copy()
    output[sr : sr + n, sc : sc + n] = _RIVER
    tr, tc = int(choice["tr"]), int(choice["tc"])
    output[tr : tr + n, tc : tc + n] = _MOVER
    if not np.array_equal(main(grid), output):
        return (None, None) if return_meta else None

    meta = {
        "reached": bool(reached),
        "n": int(n),
        "direction": str(plan["direction"]),
        "side": int(plan["side"]),
        "start_width": int(plan["start_w"]),
        "spine_width": int(plan["spine_w"]),
        "taper_len": int(plan["taper_len"]),
        "frontier_dist": int(choice["frontier_d"]),
        "midpoint_dist": int(choice["dist"][tr, tc]),
        "frontier": (int(choice["frontier_r"]), int(choice["frontier_c"])),
        "stop": (int(tr), int(tc)),
    }
    if return_meta:
        return grid, meta
    return grid


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")

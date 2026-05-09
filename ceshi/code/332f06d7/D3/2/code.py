# Variant of 332f06d7
# Dimension: D3 — 轨迹涂色 (trajectory)
# Change: 无轨迹涂色——方块移走后原位置填充河流色（蓝色），不留下行进痕迹... → 间歇性轨迹

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D3 variant
# ----------------------------------
# spec_variant:
#   preserved_structure:
#     grid: 10-20 square grid
#     colors:
#       bg: 3
#       river: 1
#       mover: 0
#       dest: 2
#     geometry:
#       river remains a single winding corridor
#       mover/dest remain same-sized solid n x n squares
#       chokepoints still decide reachable vs stuck
#   changed_dimension:
#     trajectory:
#       old: mover disappears from all vacated cells; source becomes river
#       new: simulate the slide step by step and keep only every other vacated
#         1-pixel-thick edge strip in mover color, producing a dashed trail
#   pixel_level_rule:
#     path_model:
#       - BFS still runs on feasible n x n top-left placements
#       - reconstruct one shortest placement path from mover to final target
#       - each unit move exposes exactly one 1-cell-thick strip
#     marking_rule:
#       - number moves from 0
#       - for moves 0,2,4,... keep the newly vacated strip black
#       - for moves 1,3,5,... restore the vacated strip to river
#       - final placement is always the full mover square
#     why_strip_not_full_square:
#       - coloring every other full n x n placement would overlap heavily when
#         n>1 because consecutive placements differ by one cell only
#       - that would visually collapse into a nearly solid black bar and make
#         the "intermittent" change hard to observe
#       - using exposed edge strips keeps the dash pattern visible for all n
#   small_trace_example:
#     - n=2, mover top-left goes (5,3)->(5,4)->(5,5)->(6,5)
#     - original output: source becomes blue, only final 2x2 square is black
#     - variant output:
#         move 0 right: exposed col 3 stays black
#         move 1 right: exposed col 4 becomes blue
#         move 2 down : exposed row 5 at cols 5..6 stays black
#         final square at (6,5) is black
#     - difference: the output now contains a dashed black trace on the river
#   complexity_calibration:
#     - only one extra rule is introduced: parity on movement steps
#     - the path itself is still determined exactly as before by the river and
#       chokepoints, so D3 is the only logical novelty
#     - generator rejects cases with too-short paths so the dashes are visible
#   implicit_assumptions_review:
#     - old code sorted square colors numerically to choose mover/dest; that is
#       an invisible color-id dependency and is removed here
#     - new solver identifies roles structurally:
#         destination = only same-sized square touching the grid edge
#         mover       = same-sized square not touching the grid edge
#     - generator must therefore guarantee exactly one edge-touching square
#     - BFS feasibility logic is unchanged because the input still has no trail
#   placement_legality:
#     - generator keeps mover off the boundary and extends destination to one
#       boundary edge
#     - chokepoints are still carved only in the river and connectivity is
#       checked before placing mover/dest
#     - solver reconstructs a path only after verifying the chosen target is
#       reachable in placement space
#   visual_observability:
#     - the output shows repeated black dash segments on blue river cells
#     - both reachable and stuck cases still appear, so the original motion
#       rule remains learnable while the new D3 behavior is clearly visible
#   cleanliness:
#     - trail uses only mover color and only on the river corridor
#     - no extra marker color is introduced, avoiding visual clutter


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _touches_edge(square: dict, H: int, W: int) -> bool:
    return (
        int(square["r0"]) == 0
        or int(square["c0"]) == 0
        or int(square["r0"]) + int(square["n"]) == H
        or int(square["c0"]) + int(square["n"]) == W
    )


def _detect_roles(g: np.ndarray, bg: int) -> tuple[dict, dict]:
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
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert not (len(squares) < 2)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for square in squares:
        by_n[int(square["n"])].append(square)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert not (len(cand) < 2)

    edge = [square for square in cand if _touches_edge(square, H, W)]
    inner = [square for square in cand if not _touches_edge(square, H, W)]
    assert not (len(edge) != 1 or len(inner) != 1)
    return inner[0], edge[0]


def _find_river(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover_col), int(dest_col)):
            continue
        return int(col)
    raise AssertionError("river color not found")


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert not (h2 <= 0 or w2 <= 0)
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _bfs_with_parent(feasible: np.ndarray, sr: int, sc: int):
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent_r = -np.ones((h2, w2), dtype=int)
    parent_c = -np.ones((h2, w2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    for_start = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        r, c = q.popleft()
        for dr, dc in for_start:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                parent_r[rr, cc] = int(r)
                parent_c[rr, cc] = int(c)
                q.append((rr, cc))
    return dist, parent_r, parent_c


def _pick_target(dist: np.ndarray, feasible: np.ndarray, dr: int, dc: int, sr: int, sc: int) -> tuple[int, int]:
    h2, w2 = feasible.shape
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        return int(dr), int(dc)

    best = -1
    tr, tc = int(sr), int(sc)
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def _reconstruct_path(parent_r: np.ndarray, parent_c: np.ndarray, start: tuple[int, int], target: tuple[int, int]):
    sr, sc = start
    tr, tc = target
    path = [(int(tr), int(tc))]
    while (tr, tc) != (sr, sc):
        pr = int(parent_r[tr, tc])
        pc = int(parent_c[tr, tc])
        assert not (pr < 0 or pc < 0)
        tr, tc = pr, pc
        path.append((int(tr), int(tc)))
    path.reverse()
    return path


def _solve_state(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    mover, dest = _detect_roles(g, bg)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    river = _find_river(g, bg, int(mover["col"]), int(dest["col"]))

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _build_feasible(trav, n)
    h2, w2 = feasible.shape
    assert not (not (0 <= sr < h2 and 0 <= sc < w2))
    assert not (not bool(feasible[sr, sc]))

    dist, parent_r, parent_c = _bfs_with_parent(feasible, sr, sc)
    tr, tc = _pick_target(dist, feasible, dr, dc, sr, sc)
    path = _reconstruct_path(parent_r, parent_c, (sr, sc), (tr, tc))

    return {
        "grid": g,
        "H": H,
        "W": W,
        "bg": bg,
        "river": int(river),
        "mover": mover,
        "dest": dest,
        "n": n,
        "start": (sr, sc),
        "target": (tr, tc),
        "path": path,
        "dist": dist,
    }


def _paint_vacated_strip(out: np.ndarray, pos_a: tuple[int, int], pos_b: tuple[int, int], n: int, color: int) -> None:
    r0, c0 = pos_a
    r1, c1 = pos_b
    if r1 == r0 + 1 and c1 == c0:
        out[r0, c0 : c0 + n] = int(color)
    elif r1 == r0 - 1 and c1 == c0:
        out[r0 + n - 1, c0 : c0 + n] = int(color)
    elif c1 == c0 + 1 and r1 == r0:
        out[r0 : r0 + n, c0] = int(color)
    elif c1 == c0 - 1 and r1 == r0:
        out[r0 : r0 + n, c0 + n - 1] = int(color)
    else:
        raise AssertionError("non-unit move in path")


def _render_output(state: dict, intermittent: bool) -> np.ndarray:
    g = state["grid"]
    river = int(state["river"])
    mover_col = int(state["mover"]["col"])
    n = int(state["n"])
    sr, sc = state["start"]
    tr, tc = state["target"]
    path = state["path"]

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)

    if intermittent:
        for step, (cur, nxt) in enumerate(zip(path, path[1:])):
            if step % 2 == 0:
                _paint_vacated_strip(out, cur, nxt, n, mover_col)
            else:
                _paint_vacated_strip(out, cur, nxt, n, river)

    out[tr : tr + n, tc : tc + n] = int(mover_col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel and leave a dashed wake.

    Input structure:
    - The corner-majority background surrounds one blue river corridor.
    - Two same-sized solid squares sit on that corridor: the mover is inside the grid, while the destination touches an edge.

    Rule:
    - Treat each top-left position where the full NxN mover fits inside river/mover/destination cells as a placement state.
    - Move the square along a shortest 4-neighbor placement path to the destination; if the destination is blocked by a narrow choke, stop at the farthest reachable placement instead.
    - After each unit move, only every other newly exposed 1-cell-thick strip stays black, so the wake becomes a dashed trajectory; the final placement is fully black.

    Key insight: The path is still determined by feasible NxN placements, but the visible change is no longer just the endpoint: the vacated edge strips reveal an alternating trail.
    """
    state = _solve_state(np.array(grid, dtype=int))
    return _render_output(state, intermittent=True)


def _bfs_placement(feasible, sr, sc):
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


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
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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
                    for dr2, dc2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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


def _render_channel(grid, path, n, offsets_r, offsets_c):
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
                if 0 <= rr < H and 0 <= c < W:
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
                if 0 <= r < H and 0 <= cc < W:
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


def _trail_metric(inp: np.ndarray, out: np.ndarray) -> int:
    state = _solve_state(inp)
    tr, tc = state["target"]
    n = int(state["n"])
    mover_col = int(state["mover"]["col"])
    trail = out == mover_col
    trail[tr : tr + n, tc : tc + n] = False
    return int(trail.sum())


def _gen_attempt():
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

    def _extend_to_edge(pr, pc):
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
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                    candidates.append((pr, 0, dist))
            elif direction == "right":
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
                    candidates.append((pr, W - n, dist))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[2])
        return candidates[0][:2]

    result = _extend_to_edge(dest_r, dest_c)
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
        elif mover_c + n == W:
            grid[mover_r : mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not reached and n >= 2:
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

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if trav[r : r + n, c : c + n].all():
                feasible[r, c] = True
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = 0 <= dest_r < h2 and 0 <= dest_c < w2 and feasible[dest_r, dest_c] and dist[dest_r, dest_c] >= 0
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    state = _solve_state(grid)
    path_len = len(state["path"]) - 1
    if path_len < 3:
        return None

    output = _render_output(state, intermittent=True)
    legacy = _render_output(state, intermittent=False)
    if np.array_equal(output, legacy):
        return None

    trail_pixels = _trail_metric(grid, output)
    if trail_pixels < 2 * n:
        return None

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

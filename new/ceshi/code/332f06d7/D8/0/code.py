# Variant of 332f06d7
# Dimension: D8 — 移动体与目标的配对规则 (meta)
# Change: 按颜色值升序排序，颜色最小的为移动体（黑色0），次小的为目标（红色2），通过相同尺寸n配对... → 按位置配对

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components

# Design analysis for the D8 variant
# ----------------------------------
# concrete_behavior:
#   - The scene still contains one winding river (color 1) on a dominant background
#     (color 3), plus exactly two solid same-sized squares placed on the river.
#   - The visual change is not the path geometry but the role assignment:
#     the square whose center is nearer to the grid center is the mover, and the
#     farther one is the destination.
#   - To make this visible, the generator alternates color assignment between
#     samples. Some grids have black near the center and red farther away, while
#     others deliberately invert that color-role relation so red becomes the
#     mover and black becomes the target.
#   - Example trace:
#       original logic:
#         two 2x2 squares at (4,4) red and (0,9) black in a 12x12 grid
#         -> color sort picks black as mover because 0 < 2
#       variant logic:
#         center-distance scores are red: 2, black: 50
#         -> red is mover because it is closer to the center
#       difference:
#         the moving color now depends on geometry, so color ids alone are
#         insufficient.
# complexity_calibration:
#   - Only two role-bearing squares are used, preserving the original puzzle's
#     low object count and keeping the center-based comparison unambiguous.
#   - Distances are computed from square centers, and the generator rejects ties
#     or near-ties so the role contrast is easy to see by eye.
#   - Color variety is intentionally limited to the original pair {0, 2}; this
#     keeps the puzzle readable while still breaking the old color-order rule.
# implicit_assumptions_review:
#   - The original `cand.sort(key=lambda s: s["col"])` depended on color ids
#     encoding roles. That assumption is invalid here and is replaced by center
#     distance ordering.
#   - River detection still excludes the two chosen square colors; this remains
#     valid because the generator still uses exactly one river color plus two
#     square colors.
#   - BFS still assumes the chosen mover location is a feasible n x n placement.
#     The generator preserves this by placing both squares on the painted river.
# placement_legality:
#   - Squares remain non-overlapping, within bounds, and aligned to the river
#     because they reuse the original placement path construction.
#   - The generator extends the farther square to the boundary as before, keeps
#     the nearer square slightly inset, and then verifies the new solver output.
# visual_observability:
#   - The key clue is that the more central square moves, even when it is red in
#     one sample and black in another.
#   - This avoids any hidden dependency on numeric color ids and keeps the rule
#     recoverable from a few input/output pairs.
# visual_cleanliness:
#   - The grid still has only background, one river, and two solid squares, so
#     no additional overlays are introduced.
# color_inference:
#   - Colors themselves do not transform. Their only purpose is to preserve the
#     moved square's identity in the output, which is inferable from position.
# main_new_assumptions_checked:
#   - `main()` now assumes the two same-sized squares have distinct center
#     distances. `generate_input()` enforces a minimum distance margin.
#   - `main()` chooses the nearest and farthest squares within the dominant size
#     group. The generator keeps exactly one such pair, so no ambiguity remains.


def _center_distance_sq(r0: int, c0: int, n: int, H: int, W: int) -> int:
    """Squared distance from the square center to the grid center in doubled coordinates."""
    dr = 2 * int(r0) + int(n) - int(H)
    dc = 2 * int(c0) + int(n) - int(W)
    return int(dr * dr + dc * dc)


def _role_key(square: dict, H: int, W: int) -> tuple[int, int, int]:
    return (
        _center_distance_sq(int(square["r0"]), int(square["c0"]), int(square["n"]), H, W),
        int(square["r0"]),
        int(square["c0"]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the more central solid square through a winding channel toward the farther square.

    Input structure:
    - A dominant background surrounds one river-like corridor.
    - Two same-sized solid squares sit on that corridor, but their colors no longer encode
      which one moves and which one is the goal.

    Rule:
    - Compare the two equal-sized squares by distance to the grid center.
    - The square closer to the center is the mover; the farther one is the destination.
    - Slide the mover through river cells using only placements where its full NxN footprint fits.
    - If the destination is reachable, the mover lands there; otherwise it stops at the farthest
      reachable placement, and its start position turns back into river.

    Key insight:
    - Role assignment must be inferred from geometry, not color ids, while narrow channel sections
      still block an NxN square exactly as in the original task.
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

    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for square in squares:
        by_n[int(square["n"])].append(square)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    cand.sort(key=lambda s: _role_key(s, H, W))
    mover = cand[0]
    dest = cand[-1]

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if col == bg:
            continue
        if col in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))

    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = -np.ones((h2, w2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    for_r = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        r, c = q.popleft()
        for dr2, dc2 in for_r:
            rr = int(r + dr2)
            cc = int(c + dc2)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))

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
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


# --- Generator ---

_BG = 3
_RIVER = 1
_ROLE_COLORS = (0, 2)
_GENERATE_CALLS = 0


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
        total_len = max(1, 2 * remove + hold_len)
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
        total_len = max(1, 2 * remove + hold_len)
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
    mask = grid != _BG
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


def _distance_margin_ok(mover_r, mover_c, dest_r, dest_c, n, H, W):
    mover_d = _center_distance_sq(mover_r, mover_c, n, H, W)
    dest_d = _center_distance_sq(dest_r, dest_c, n, H, W)
    return mover_d + 4 <= dest_d


def _paint_roles(grid, mover_r, mover_c, dest_r, dest_c, n, mover_col, dest_col):
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_col
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = dest_col


def _gen_attempt(reverse_colors: bool):
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
    if not _distance_margin_ok(mover_r, mover_c, dest_r, dest_c, n, H, W):
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

    mover_col, dest_col = (_ROLE_COLORS[1], _ROLE_COLORS[0]) if reverse_colors else _ROLE_COLORS
    _paint_roles(grid, mover_r, mover_c, dest_r, dest_c, n, mover_col, dest_col)

    trav = (grid == _RIVER) | (grid == mover_col) | (grid == dest_col)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = 0 <= dest_r < h2 and 0 <= dest_c < w2 and bool(feasible[dest_r, dest_c]) and dist[dest_r, dest_c] >= 0
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    if dest_ok:
        tr, tc = dest_r, dest_c
    else:
        best_d = -1
        tr, tc = mover_r, mover_c
        for r in range(h2):
            for c in range(w2):
                if dist[r, c] > best_d:
                    best_d = dist[r, c]
                    tr, tc = r, c

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[tr : tr + n, tc : tc + n] = mover_col

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    global _GENERATE_CALLS
    reverse_colors = (_GENERATE_CALLS % 2) == 0
    for _ in range(300):
        result = _gen_attempt(reverse_colors=reverse_colors)
        if result is not None:
            _GENERATE_CALLS += 1
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

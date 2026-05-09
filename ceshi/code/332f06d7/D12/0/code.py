# Variant of 332f06d7
# Dimension: D12 — 背景与通道的颜色关系 (meta)
# Change: 固定配色：背景=绿色(3)，通道=蓝色(1)，移动体=黑色(0)，目标=红色(2)... → 可变配色方案
#
# Design analysis
# spec_variant:
#   unchanged_core:
#     - Grid stays square, 10..20 on each side.
#     - A single winding corridor still connects the two square regions.
#     - The mover is still an n x n solid square and the destination is another
#       same-sized solid square.
#     - Movement is still BFS over feasible n x n top-left placements.
#     - Reachable cases still land on the destination; stuck cases still stop at
#       the farthest reachable placement before a narrow gate.
#   D12_change:
#     - The four semantic roles use four distinct colors sampled from the ARC palette.
#     - No role is tied to a fixed color id.
#     - The solver must infer roles from structure, not numeric color ordering.
#   concrete_pixel_story:
#     - One color dominates the corners and fills the outside background.
#     - A second color forms one connected winding corridor.
#     - Two other colors form solid equal-sized squares sitting on the corridor.
#     - The destination square is the only same-sized square touching the grid
#       boundary; the mover square is near an edge but does not touch it.
#   behavior_trace:
#     original:
#       - Detect the two equal-sized squares.
#       - Sort them by color id and declare lower color = mover, higher color = dest.
#     variant:
#       - Detect the two equal-sized squares.
#       - Use geometry instead: edge-touching square = destination, interior square = mover.
#       - Everything after role assignment stays the same.
#     difference:
#       - If mover_color > dest_color, the original solver swaps roles and moves
#         the wrong object. The variant solver still succeeds because position,
#         not color id, encodes the roles.
#   complexity_calibration:
#     - Keep exactly four colors so the scene remains readable.
#     - Randomize only the color-role mapping; do not add extra markers or extra
#       objects that would change the puzzle family.
#     - Preserve the strong repeated cues: corner-majority background, one winding
#       corridor color, one interior square, one border-touching square.
#   implicit_assumption_audit:
#     - Corner-majority background detection remains valid because the generator
#       still keeps all four corners as background.
#     - Solid-square detection remains valid because only the two role objects are
#       filled monochromatic squares of the repeated size n.
#     - The old color sort is invalid under D12 and is removed.
#     - The new solver assumes exactly one candidate square of size n touches an
#       edge and at least one other candidate of the same size does not; the
#       generator enforces that.
#     - River inference as the most frequent remaining non-background color stays
#       valid because the corridor occupies more cells than either square and the
#       generator uses no extra colors.
#   placement_legality:
#     - Squares are painted only onto corridor cells and never overlap each other.
#     - The destination is extended to a boundary but still lies on the corridor.
#     - The mover is shifted off the boundary if needed and then required to be
#       within distance <= 2 from some edge, preserving the structural cue.
#     - Chokepoints only erase corridor cells to background and are verified not
#       to disconnect the river-color region.
#   visual_observability:
#     - Humans can infer the randomized palette because the background owns the
#       corners and largest empty area, the corridor is the long connected path,
#       the mover is the interior square that disappears from its start, and the
#       destination is the same-sized square attached to the boundary.
#     - No hidden ordering by color id is used anywhere in the transformation.
#   cleanliness:
#     - Still only four flat colors and three object types on screen.
#     - No overlapping labels, no internal textures, no extra noise.

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


def _touches_edge(H: int, W: int, r0: int, c0: int, n: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _edge_distance(H: int, W: int, r0: int, c0: int, n: int) -> int:
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


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
        squares.append(
            {
                "col": col,
                "r0": r0,
                "c0": c0,
                "n": n,
                "touches": _touches_edge(H, W, r0, c0, n),
                "edge_dist": _edge_distance(H, W, r0, c0, n),
            }
        )

    assert not (len(squares) < 2)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert not (len(cand) < 2)

    touching = [sq for sq in cand if bool(sq["touches"])]
    interior = [sq for sq in cand if not bool(sq["touches"])]
    assert not (not touching or not interior)

    dest = min(touching, key=lambda sq: (sq["edge_dist"], sq["r0"], sq["c0"]))
    mover = min(interior, key=lambda sq: (sq["edge_dist"], sq["r0"], sq["c0"]))
    return mover, dest


def _infer_river_color(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    candidates = [
        int(col)
        for col in counts
        if int(col) not in (int(bg), int(mover_col), int(dest_col))
    ]
    assert not (not candidates)
    river = candidates[0]
    best_count = int(counts[int(river)])
    for col in candidates[1:]:
        ct = int(counts[int(col)])
        if ct > best_count:
            river = int(col)
            best_count = ct
    return int(river)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide one inferred square through a color-shuffled corridor until it reaches the matching-size border square.

    Input structure:
    - One color dominates the corners and acts as the background, but its id is not fixed.
    - A second color forms a single winding corridor through the background.
    - Two other colors form equal-sized solid squares on that corridor: one interior square and one edge-touching square.

    Rule:
    - Infer the mover from geometry: it is the interior square, while the boundary-touching square is the destination.
    - Move the mover through all n x n placements whose full area lies inside corridor-or-square cells.
    - If the destination placement is reachable, the mover lands there; otherwise it stops at the farthest reachable placement.
    - Repaint the mover's starting area with the corridor color.

    Key insight:
    - The palette is shuffled from puzzle to puzzle, so color ids are unreliable; the stable cues are corner-majority background, the connected corridor, and the fact that only the destination square touches the boundary.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _find_role_squares(g, bg)
    river = _infer_river_color(g, bg, int(mover["col"]), int(dest["col"]))

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))

    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert not (h2 <= 0 or w2 <= 0)
    assert not (not (0 <= sr < h2 and 0 <= sc < w2))

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all()):
                feasible[int(r), int(c)] = True
    assert not (not bool(feasible[int(sr), int(sc)]))

    dist = -np.ones((h2, w2), dtype=int)
    q = deque([(int(sr), int(sc))])
    dist[int(sr), int(sc)] = 0
    while q:
        r, c = q.popleft()
        for dr2, dc2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr = int(r + dr2)
            cc = int(c + dc2)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((int(rr), int(cc)))

    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0:
        tr, tc = int(dr), int(dc)
    else:
        best = -1
        tr, tc = int(sr), int(sc)
        for r in range(h2):
            for c in range(w2):
                if int(dist[int(r), int(c)]) > best:
                    best = int(dist[int(r), int(c)])
                    tr, tc = int(r), int(c)

    out = g.copy()
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(river)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover["col"])
    return out


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
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


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
    river: int,
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = river
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = river
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = river
    return pixel_pos


def _add_chokepoint(
    grid: np.ndarray,
    pixel_pos_i: tuple[int, int],
    pixel_pos_j: tuple[int, int],
    n: int,
    direction: str,
    bg: int,
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
                    grid[rr, c] = bg
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
                    grid[r, cc] = bg


def _river_connected(grid: np.ndarray, river: int, mover: int, dest: int) -> bool:
    mask = (grid == river) | (grid == mover) | (grid == dest)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return num == 1


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _gen_attempt() -> np.ndarray | None:
    bg, river, mover_col, dest_col = random.sample(list(range(10)), 4)

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

    grid = np.full((H, W), bg, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c, river)

    mask = grid != bg
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    def _extend_to_edge(pr: int, pc: int) -> tuple[int, int] | None:
        candidates = []
        for direction in ["top", "bottom", "left", "right"]:
            if direction == "top":
                dist = pr
                if dist == 0:
                    candidates.append((0, pc, 0))
                    continue
                if all(grid[r, c] in (bg, river) for r in range(pr) for c in range(pc, pc + n) if c < W):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = H - (pr + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                    continue
                if all(grid[r, c] in (bg, river) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                    continue
                if all(grid[r, c] in (bg, river) for r in range(pr, pr + n) for c in range(pc) if r < H):
                    candidates.append((pr, 0, dist))
            elif direction == "right":
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                    continue
                if all(grid[r, c] in (bg, river) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
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
        grid[r_lo:r_hi, dest_c : dest_c + n] = river
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = river
    dest_r, dest_c = new_dest_r, new_dest_c
    if not _touches_edge(H, W, dest_r, dest_c, n):
        return None

    touching = _touches_edge(H, W, mover_r, mover_c, n)
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = river
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = river
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = river
            mover_c = 1
        elif mover_c + n == W:
            grid[mover_r : mover_r + n, W - 1] = river
            mover_c = W - n - 1
    if _touches_edge(H, W, mover_r, mover_c, n):
        return None
    if _edge_distance(H, W, mover_r, mover_c, n) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = river
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = river

    if not reached and n >= 2:
        num_choke = random.randint(1, min(3, len(walk) - 2))
        min_seg = max(1, len(walk) // 4)
        available = list(range(min_seg, len(walk) - 1))
        random.shuffle(available)
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction, bg)
        if not _river_connected(grid, river, mover_col, dest_col):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_col
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = dest_col

    trav = (grid == river) | (grid == mover_col) | (grid == dest_col)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = bool(
        0 <= dest_r < h2
        and 0 <= dest_c < w2
        and feasible[dest_r, dest_c]
        and dist[dest_r, dest_c] >= 0
    )
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
                if int(dist[r, c]) > best_d:
                    best_d = int(dist[r, c])
                    tr, tc = r, c

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = river
    output[tr : tr + n, tc : tc + n] = mover_col

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

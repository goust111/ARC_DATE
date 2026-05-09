# Variant of 332f06d7
# Dimension: D6 — 碰撞/覆盖变换 (transform)
# Change: 可达时移动体覆盖目标方块位置（目标消失），不可达时停在最远处（目标保留），原位置填充河流色... → 重新着色变换

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D6 variant
# ----------------------------------
# a) Concretization and behavior trace
#    - Input keeps the original scene: green background, blue river, one black n×n mover,
#      one red n×n destination, same size, destination touching the outer border.
#    - Output changes only when the mover can reach the destination placement.
#      Original behavior at the critical write-back step:
#        1. paint mover start back to blue river
#        2. paint final placement with black
#        3. if final placement == destination, red disappears under black
#      Variant behavior:
#        1. paint mover start back to blue river
#        2. if destination is reachable, paint the destination placement with orange
#           so the collision site is explicitly marked
#        3. otherwise keep the old stuck behavior: mover stops at the farthest reachable
#           placement in black, and the red destination remains untouched
#      Example: black 2×2 at (1,4), red 2×2 at (8,0), blue channel connects them.
#      If reachable, original output would place black 2×2 at (8,0); variant output places
#      orange 2×2 at (8,0). If a 1-pixel chokepoint blocks a 2×2 mover, both versions stop
#      at the last feasible placement and leave the red target in place.
# b) Complexity calibration
#    - Only one extra color is introduced: orange (7). This is enough to make the new rule
#      visible without creating a large color vocabulary.
#    - Reached vs stuck remains the only branch. That keeps the task close to the source
#      while making the transformed branch unmistakable.
# c) Hidden-assumption audit
#    - The original solver sorted same-sized squares by color id to decide mover vs destination.
#      That is not acceptable here because role assignment must be visually inferable.
#      We replace it with geometry: destination touches the border, mover is near a border
#      but does not touch it.
#    - The original write-back line unconditionally painted the final square with mover color.
#      That relied on "covering the target" being correct. In this variant that line would
#      erase the new collision marker, so reachable and stuck writes must be split.
#    - River detection still assumes exactly one non-background, non-square filler color.
#      The generator preserves that.
# d) Placement legality
#    - Generator still builds one connected river corridor, places exactly two same-sized
#      solid squares, keeps them in-bounds and non-overlapping, and verifies the solver.
#    - Destination is extended to the border; mover is nudged to be within distance 1 or 2
#      from some border but not touching it. These constraints make role detection robust.
# e) Visual observability
#    - The new orange square appears only in reachable cases, exactly where red used to be.
#      In stuck cases the output retains the old red target plus black stopped mover.
#      With 2-4 examples, a human can see that "successful contact produces orange".
# f) Visual cleanliness
#    - No extra overlay glyphs are added. The orange marker fully occupies the target square,
#      so the output remains a clean river-and-squares scene.
# g) Color-change inferability
#    - The color mapping is single-condition and single-target:
#      successful collision -> destination area becomes orange.
#      The condition is structurally signaled by whether the black square can traverse the
#      full-width river. No hidden color-id lookup is needed.
# New solver assumptions introduced by this variant
#    - Among same-sized solid squares, exactly one touches the border (destination) and
#      exactly one is near the border but not touching (mover).
#    - The collision marker color does not appear in the input.
#    - Generator enforces both conditions.


_BG, _RIVER, _MOVER, _DEST, _HIT = 3, 1, 0, 2, 7


def _edge_gap(r0: int, c0: int, n: int, H: int, W: int) -> int:
    return int(min(r0, c0, H - (r0 + n), W - (c0 + n)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel and recolor the collision site on contact.

    Input structure:
    - The corner-dominant background surrounds one blue river corridor.
    - Two same-sized solid squares sit on that corridor: an interior mover and a border-touching destination.

    Rule:
    - The mover slides through placements where its full NxN footprint stays inside the traversable river system.
    - If the destination placement is reachable, the mover arrives there and both objects are represented by a new orange NxN collision square.
    - If the destination is blocked by a narrow section, the mover stops at the farthest reachable placement, the red destination stays, and the start area becomes river.

    Key insight: The path search is unchanged; only the final write-back differs, and the reachable branch must emit a third color instead of overwriting the target with the mover color.
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
        n = int(bbox_h(b))
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": n})

    assert not (len(squares) < 2)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in squares:
        by_n[int(s["n"])].append(s)

    mover = None
    dest = None
    best_key = None
    for nside, group in by_n.items():
        edge_squares = []
        near_edge_squares = []
        for s in group:
            gap = _edge_gap(int(s["r0"]), int(s["c0"]), int(s["n"]), H, W)
            if gap == 0:
                edge_squares.append(s)
            elif gap <= 2:
                near_edge_squares.append(s)
        if not edge_squares or not near_edge_squares:
            continue
        key = (len(group), int(nside))
        if best_key is None or key > best_key:
            best_key = key
            dest = min(edge_squares, key=lambda s: (int(s["r0"]), int(s["c0"])))
            mover = min(near_edge_squares, key=lambda s: (_edge_gap(int(s["r0"]), int(s["c0"]), int(s["n"]), H, W), int(s["r0"]), int(s["c0"])))

    assert mover is not None and dest is not None

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
        for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr2)
            cc = int(c + dc2)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((int(rr), int(cc)))

    reached = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0
    if reached:
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
    if reached:
        out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(_HIT)
    else:
        out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover["col"])
    return out


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


def _render_channel(grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]) -> list[tuple[int, int]]:
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


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str) -> None:
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


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return bool(num == 1)


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

    def _extend_to_edge(pr: int, pc: int) -> tuple[int, int] | None:
        candidates = []
        for direction in ("top", "bottom", "left", "right"):
            if direction == "top":
                dist = pr
                if dist == 0:
                    candidates.append((0, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = H - (pr + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                    candidates.append((pr, 0, dist))
            else:
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
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
    if _edge_gap(mover_r, mover_c, n, H, W) > 2:
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
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = 0 <= dest_r < h2 and 0 <= dest_c < w2 and bool(feasible[dest_r, dest_c]) and int(dist[dest_r, dest_c]) >= 0
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

    expected = grid.copy()
    expected[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    if dest_ok:
        expected[tr : tr + n, tc : tc + n] = _HIT
    else:
        expected[tr : tr + n, tc : tc + n] = _MOVER

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

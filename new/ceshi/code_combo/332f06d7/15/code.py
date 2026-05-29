# Combo variant of 332f06d7 (index 15)
# combo_spec: D6/3+D13/2
# Name: 碰撞消除+回退起点

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for combo D6/3 + D13/2
======================================

a) Concrete behavior and trace
   - Base task: a black n×n mover slides through blue river pixels toward a red n×n
     destination. Motion is BFS over top-left placements whose whole n×n window lies in
     {river, mover, destination}.
   - Original reachable behavior: mover reaches the red square and overwrites it with
     black; start square turns blue.
   - Original stuck behavior: mover advances to the farthest reachable placement before
     the sub-n chokepoint; start turns blue.
   - Combo reachable behavior (D6/3): when BFS can reach the destination placement,
     both the black mover and red destination are erased to river color. The whole
     corridor becomes pure blue.
   - Combo stuck behavior (D13/2): when BFS cannot reach the destination, the mover does
     not advance at all. Output is identical to input.
   - Small trace with n=2:
       input: black 2x2 near one edge, red 2x2 at the far edge, blue path between them
       reachable branch:
         original -> start becomes blue, dest becomes black
         combo    -> start becomes blue, dest becomes blue
       stuck branch with a 1-pixel gate:
         original -> black moves forward to the last 2x2 placement before the gate
         combo    -> black stays at start, red stays at end, output equals input
   - The fusion is not two unrelated edits: the two branches are now maximally
     separated visually. Reachable examples erase both salient blocks; stuck examples
     preserve both blocks exactly.

b) Complexity calibration
   - Reachable and stuck outputs are easy to distinguish by eye:
       reachable -> no black/red blocks remain
       stuck     -> the scene is unchanged
   - This is strong enough to be learnable from 2-4 examples without adding extra
     markers or hidden bookkeeping.
   - To avoid degeneracy, stuck samples are rejected unless the original task would have
     moved the black square forward by at least one placement. That guarantees the combo
     differs materially from the base task on blocked cases.

c) Implicit assumption audit for main()
   - Component scan still assumes the two salient non-background objects are solid
     monochromatic squares. Generator preserves that.
   - Picking mover/destination by color order remains safe only because generation uses
     the fixed palette black=0, red=2 and introduces no extra same-sized solid squares.
   - River inference as the most common non-background color still holds because blue is
     the only large non-square structure.
   - Feasibility logic still relies on the full n×n window being traversable in
     {river,mover,destination}. The fusion does not change motion geometry.
   - The original unreachable branch depended on scanning for the farthest reachable
     distance. D13/2 invalidates that assumption, so main() must not choose a fallback
     target there.
   - The original reachable renderer always painted the destination with mover color.
     D6/3 invalidates that assumption, so main() must erase the destination instead.
   - New solver assumption: leaving the output unchanged on the stuck branch is legal.
     Generator guarantees the starting placement itself is feasible, so no cleanup is
     needed to keep the mover inside the river envelope.

d) Placement legality
   - Generator keeps square grids of size 10..20 and uses the original winding-channel
     construction.
   - Destination is extended to a boundary edge, mover is near but not touching an edge,
     and both occupy disjoint n×n river placements before coloring.
   - Chokepoints are added only after carving the river and before placing colored
     squares, so no object is partially overwritten.
   - Oracle validation uses the combo main(), and blocked samples are additionally
     checked against the original fallback distance to ensure the combo behavior is
     really exercised.

e) Visual observability
   - Human-visible evidence is strong:
       reachable -> both blocks disappear at contact
       stuck     -> nothing changes despite a visible downstream red goal
   - No branch depends on invisible numeric properties like color ids; the generator only
     uses fixed ARC colors to keep roles consistent across samples.

f) Visual cleanliness
   - The output never overlays multiple symbols in the same place. A destination either
     stays red in blocked cases or becomes blue in reachable cases.
   - The river remains a single connected blue structure, which keeps the scene readable
     even when the reachable output becomes very minimal.

g) New assumptions main() introduces and generator guarantees
   - If destination is unreachable, returning the unchanged grid must still differ from
     the original task. Generator enforces old farthest reachable distance >= 1.
   - If destination is reachable, erasing both squares must leave a recognizable river.
     Generator preserves a single coherent channel so the all-blue reachable output is
     not visually ambiguous.
"""


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


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int, int]:
    best_d = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best_d:
                best_d = int(dist[r, c])
                tr, tc = r, c
    return tr, tc, best_d


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - reachable  -> mover and destination both erase to river color
    - unreachable -> output stays unchanged
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(
        Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0]
    )

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

    cand.sort(key=lambda s: int(s["col"]))
    mover = cand[0]
    dest = cand[1]

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

    dist = _bfs_placement(feasible, sr, sc)
    dest_reached = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0

    if not dest_reached:
        return g.copy()

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[dr : dr + n, dc : dc + n] = river
    return out


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]):
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
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
    return num == 1


def _compute_offsets(G: int, n: int, num_coarse: int):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int):
    H, W = grid.shape
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
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
                grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W
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
                grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H
            ):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return candidates[0][:2]


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

    result = _extend_to_edge(grid, dest_r, dest_c, n)
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
    dest_ok = 0 <= dest_r < h2 and 0 <= dest_c < w2 and bool(feasible[dest_r, dest_c]) and dist[dest_r, dest_c] >= 0
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    old_tr, old_tc, old_best_d = _farthest_reachable(dist, mover_r, mover_c)
    if not reached and (old_best_d < 1 or (old_tr, old_tc) == (mover_r, mover_c)):
        return None

    if reached:
        output = grid.copy()
        output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
        output[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER
    else:
        output = grid.copy()

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

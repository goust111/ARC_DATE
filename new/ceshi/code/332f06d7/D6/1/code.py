# Variant of 332f06d7
# Dimension: D6 — 碰撞/覆盖变换 (transform)
# Change: 可达时移动体覆盖目标方块位置（目标消失），不可达时停在最远处（目标保留），原位置填充河流色... → 移动体消失变换

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D6 variant
# a) Concrete behavior and trace:
#    - The input still shows one blue winding river on a green background, one black n×n mover,
#      and one red n×n target of the same size.
#    - The only rule change is the reachable collision outcome. Original: black arrives at the red
#      square and overwrites it. Variant: black reaches the red square, then disappears, so the
#      destination remains visibly red and only the black source area turns blue.
#    - Small trace: if a 2×2 black square can slide through a 2-wide river to a 2×2 red dock,
#      the original output would end with a black 2×2 at the dock. The variant instead shows no
#      black square at all: source becomes blue, dock stays red. In a blocked case nothing else
#      changes: the black square still stops at the farthest reachable placement and the red square
#      remains at the end.
# b) Complexity calibration:
#    - The pathfinding, choke logic, and square-size reasoning stay unchanged. This keeps the task
#      at the same structural difficulty and isolates the innovation to the transform step.
#    - To make the new branch visible in a few examples, the generator alternates preferred
#      reachable/stuck samples across calls instead of leaving the split fully random.
# c) Implicit-assumption audit of the original main():
#    - Original `cand.sort(key=lambda s: s["col"])` relied on invisible color ids. That is not
#      allowed for the variant. We replace it with a visible cue already present in the task:
#      the destination touches the outer border while the mover starts one cell away from it.
#    - Original final writeback always painted the mover at `(tr, tc)`. Under the variant this is
#      wrong in the reachable branch because the mover must vanish after delivery. The writeback now
#      depends on reachability.
#    - The traversable mask can still include both mover and destination colors; the path search is
#      unchanged because the variant only changes the post-collision rendering.
# d) Placement legality guaranteed by the generator:
#    - Exactly two same-sized solid squares are created.
#    - The red destination is extended to touch one grid edge; the black mover is nudged off the
#      edge so it remains near the boundary but not touching it.
#    - Both squares stay within the river corridor footprint used by the placement BFS.
# e) Visual observability:
#    - Reachable examples now visibly differ from the original task because the black square
#      disappears instead of replacing the red one.
#    - Stuck examples preserve the black square away from the dock, so a small training set can
#      contrast "delivered => vanish" against "not delivered => still visible".
# f) Visual cleanliness:
#    - No extra marks or colors are added. The grid still contains only background, river, mover,
#      and destination, so the variant remains easy to parse.
# g) Color-change inferability:
#    - There is no arbitrary color remapping. The only color change is local and structural:
#      source black cells become blue, and reachable delivery suppresses the black redraw.
# h) New solver assumptions to enforce:
#    - Among the candidate same-sized squares, the destination is the one closest to the border,
#      ideally touching it; the mover is the closest non-destination square.
#    - The generator explicitly constructs samples satisfying that geometry, so role assignment is
#      stable without referring to color ids.


def _edge_distance(square: dict, H: int, W: int) -> int:
    n = int(square["n"])
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _pick_roles(candidates: list[dict], H: int, W: int) -> tuple[dict, dict]:
    annotated = sorted(
        candidates,
        key=lambda s: (_edge_distance(s, H, W), int(s["r0"]), int(s["c0"])),
    )
    dest = annotated[0]
    movers = [s for s in annotated[1:] if _edge_distance(s, H, W) > _edge_distance(dest, H, W)]
    if not movers:
        movers = annotated[1:]
    assert movers
    mover = min(
        movers,
        key=lambda s: (_edge_distance(s, H, W), int(s["r0"]), int(s["c0"])),
    )
    return mover, dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a river corridor toward a same-sized dock square.

    Input structure:
    - A corner-dominant background surrounds one blue winding channel.
    - Two solid n×n squares lie on that channel: a black movable block and a red destination block.
    - The red square sits on the outer border, while the black square starts just inside the border.

    Rule:
    - Move the black square through the blue corridor using only placements where its full n×n body fits.
    - If the destination placement is reachable, the delivery completes and the black square disappears,
      leaving the red destination unchanged.
    - If a chokepoint narrower than n blocks the route, the black square stops at the farthest reachable
      placement instead, and the red destination stays where it is.
    - The mover's starting area is always refilled with the river color.

    Key insight:
    - Search on top-left placements of the whole n×n square. The pathfinding is unchanged from the
      original task; only the final reachable writeback flips from "cover the dock" to "vanish on delivery".
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
    for s in squares:
        by_n[int(s["n"])].append(s)

    eligible: list[tuple[int, int, int]] = []
    for nside, group in by_n.items():
        if len(group) < 2:
            continue
        dists = [_edge_distance(s, H, W) for s in group]
        eligible.append((int(any(d == 0 for d in dists)), len(group), int(nside)))
    assert eligible
    _touch_flag, _count, nside = max(eligible)
    cand = by_n[int(nside)]
    mover, dest = _pick_roles(cand, H, W)

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
    assert not (h2 <= 0 or w2 <= 0)
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    assert bool(feasible[sr, sc])

    dist = -np.ones((h2, w2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr2)
            cc = int(c + dc2)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))

    reached_dest = (
        0 <= dr < h2
        and 0 <= dc < w2
        and bool(feasible[dr, dc])
        and int(dist[dr, dc]) >= 0
    )

    if reached_dest:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    if not reached_dest:
        out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


# --- Generator ---

_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_NEXT_REACHED = True


def _bfs_placement(feasible, sr, sc):
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
                    grid[rr, c] = _BG
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

    def _extend_to_edge(pr, pc):
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
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W):
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
            if trav[r : r + n, c : c + n].all():
                feasible[r, c] = True
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = (
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
                if dist[r, c] > best_d:
                    best_d = int(dist[r, c])
                    tr, tc = r, c

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    if not dest_ok:
        output[tr : tr + n, tc : tc + n] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid, reached


def generate_input():
    global _NEXT_REACHED

    preferred = _NEXT_REACHED
    schedules = ((preferred, 180), (not preferred, 80), (None, 40))
    for forced_reached, budget in schedules:
        for _ in range(budget):
            result = _gen_attempt(forced_reached)
            if result is not None:
                grid, actual_reached = result
                _NEXT_REACHED = not actual_reached
                return grid
    raise RuntimeError("Failed to generate valid input after 300 attempts")

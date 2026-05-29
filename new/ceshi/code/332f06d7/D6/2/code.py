# Variant of 332f06d7
# Dimension: D6 — 碰撞/覆盖变换 (transform)
# Change: 可达时移动体覆盖目标方块位置（目标消失），不可达时停在最远处（目标保留），原位置填充河流色... → 合并变换

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D6 variant
# a) Concrete behavior and trace
#    - Input still contains one winding blue river on green background plus two same-sized solid squares.
#    - Exactly one square touches exactly one border cell-strip: that border-touching square is the destination.
#      The other same-sized square sits near an edge but does not touch it: that is the mover.
#    - Reachable case no longer ends with overwrite at the destination top-left. Instead, the mover reaches the
#      n×n river slot immediately inside the border-touching target, and the two n×n squares fuse into one 2n-long
#      rectangle colored like the mover. If the target touches the top border, the result is a 2n×n vertical block;
#      if it touches left/right, the result is an n×2n horizontal block.
#    - Small trace for n=2, top-edge target:
#      original rule: mover path reaches the red 2×2 location -> output paints only that 2×2 black.
#      new rule: mover path reaches the 2×2 slot directly below the red target -> output paints both the red 2×2
#      and that lower 2×2 black, producing one 4×2 black rectangle. Stuck case stays unchanged except the red
#      target remains because no merge happened.
# b) Complexity calibration
#    - The merged result has only two shape families: vertical 2n×n or horizontal n×2n. That is enough freedom to
#      be visibly different from the seed task without creating many branches.
#    - Border side varies across samples, so the solver must infer direction from geometry, not from a hidden label.
# c) Old assumptions that must change
#    - Old code sorted candidate squares by color id to decide mover/destination. That is not allowed here because
#      color ordering is not a visible cue. We instead use border contact: border-touching square = destination.
#    - Old reachable target was the destination top-left itself. That would collapse the new merge into the old
#      overwrite behavior. The reachable anchor is now the interior n×n slot adjacent to the destination.
#    - Old output wrote one n×n mover patch at the final location. The new reachable output must write two adjacent
#      n×n patches: the original destination square plus the interior merge slot.
# d) Placement legality guaranteed by generator
#    - Destination is extended to exactly one border and rejected if it lands on a corner.
#    - Grid size stays large enough that the inward adjacent merge slot always exists.
#    - The inward merge slot is kept blue in the input, so the solver can reach it if and only if the corridor is
#      wide enough; the red target remains a separate solid square until output time.
# e) Visual observability
#    - From 2-4 examples, a human sees that reachable outputs create a doubled rectangle attached to the border
#      target, while stuck outputs keep the red target unchanged. The only varying factor is which border is used.
# f) Visual cleanliness
#    - Only three active structures remain: river, mover square, destination square. The merge output is a single
#      clean rectangle, not an overlap with mixed colors.
# g) Color inference
#    - No new color mapping is introduced. The merged rectangle always takes the mover's color, which is directly
#      visible because the surviving large shape matches the moving square's color.


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _edge_contacts(r0: int, c0: int, n: int, H: int, W: int) -> list[str]:
    edges: list[str] = []
    if r0 == 0:
        edges.append("top")
    if r0 + n == H:
        edges.append("bottom")
    if c0 == 0:
        edges.append("left")
    if c0 + n == W:
        edges.append("right")
    return edges


def _merge_slot(dest_r: int, dest_c: int, n: int, H: int, W: int) -> tuple[int, int]:
    edges = _edge_contacts(dest_r, dest_c, n, H, W)
    assert len(edges) == 1
    edge = edges[0]
    if edge == "top":
        return dest_r + n, dest_c
    if edge == "bottom":
        return dest_r - n, dest_c
    if edge == "left":
        return dest_r, dest_c + n
    return dest_r, dest_c - n


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


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
        n = int(bbox_h(b))
        r0, c0 = int(b[0]), int(b[1])
        squares.append({"col": col, "r0": r0, "c0": c0, "n": n, "edges": _edge_contacts(r0, c0, n, H, W)})

    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) == 2

    border_touch = [sq for sq in cand if len(sq["edges"]) == 1]
    interior = [sq for sq in cand if len(sq["edges"]) == 0]
    assert len(border_touch) == 1
    assert len(interior) == 1
    mover = interior[0]
    dest = border_touch[0]
    assert int(mover["n"]) == int(dest["n"])
    return mover, dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel and fuse it with the border target when contact is possible.

    Input structure:
    - The corner-majority background surrounds a blue river corridor.
    - Two same-sized solid squares sit on that corridor: one square touches exactly one border, and the other
      remains inside the frame near an edge.

    Rule:
    - The interior square is the mover and slides through positions whose full n×n footprint stays on river/blocks.
    - If it can reach the n×n slot immediately inside the border-touching square, the two squares merge into one
      larger 2n-long rectangle colored like the mover.
    - If that merge slot is unreachable because the river narrows too much, the mover stops at the farthest
      reachable placement, the target square stays in place, and the mover's start area turns back into river.

    Key insight:
    - The destination is identified geometrically by its single-border contact, and that same border determines the
      only valid merge direction.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _detect_roles(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    mr, mc = _merge_slot(dr, dc, n, H, W)

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if col == bg or col == int(mover["col"]) or col == int(dest["col"]):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2
    assert 0 <= mr < h2 and 0 <= mc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    merge_ok = bool(feasible[mr, mc]) and int(dist[mr, mc]) >= 0

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    if merge_ok:
        out[dr : dr + n, dc : dc + n] = int(mover["col"])
        out[mr : mr + n, mc : mc + n] = int(mover["col"])
    else:
        tr, tc = _farthest_reachable(dist, sr, sc)
        out[tr : tr + n, tc : tc + n] = int(mover["col"])
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
            if cur == end and len(path) >= total * 0.7:
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
                    grid[r, cc] = _BG


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
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


def _extend_to_single_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for direction in ["top", "bottom", "left", "right"]:
        if direction == "top":
            if pc == 0 or pc + n == W:
                continue
            dist = pr
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            if pc == 0 or pc + n == W:
                continue
            dist = H - (pr + n)
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n)):
                candidates.append((H - n, pc, dist))
        elif direction == "left":
            if pr == 0 or pr + n == H:
                continue
            dist = pc
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
                candidates.append((pr, 0, dist))
        else:
            if pr == 0 or pr + n == H:
                continue
            dist = W - (pc + n)
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W)):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    best_dist = min(c[2] for c in candidates)
    near = [c for c in candidates if c[2] <= best_dist + n]
    return random.choice(near)[:2]


def _placement_grid(grid: np.ndarray, n: int) -> np.ndarray:
    H, W = grid.shape
    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.7
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

    extended = _extend_to_single_edge(grid, dest_r, dest_c, n)
    if extended is None:
        return None
    new_dest_r, new_dest_c = extended
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if len(_edge_contacts(dest_r, dest_c, n, H, W)) != 1:
        return None

    merge_r, merge_c = _merge_slot(dest_r, dest_c, n, H, W)
    if not (0 <= merge_r <= H - n and 0 <= merge_c <= W - n):
        return None

    touching = bool(_edge_contacts(mover_r, mover_c, n, H, W))
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
    if bool(_edge_contacts(mover_r, mover_c, n, H, W)):
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

    feasible = _placement_grid(grid, n)
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    merge_ok = bool(feasible[merge_r, merge_c]) and int(dist[merge_r, merge_c]) >= 0
    if reached and not merge_ok:
        return None
    if not reached and merge_ok:
        return None

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    if merge_ok:
        output[dest_r : dest_r + n, dest_c : dest_c + n] = _MOVER
        output[merge_r : merge_r + n, merge_c : merge_c + n] = _MOVER
    else:
        tr, tc = _farthest_reachable(dist, mover_r, mover_c)
        output[tr : tr + n, tc : tc + n] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

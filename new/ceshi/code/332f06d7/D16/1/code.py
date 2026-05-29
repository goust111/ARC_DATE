# Variant of 332f06d7
# Dimension: D16 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——角色由固定颜色值隐式编码（黑=移动体，红=目标，蓝=通道，绿=背景）... → 控制像素编码方向

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for this D16 variant
# -----------------------------------
# a) Concrete behavior and flow impact
#    - The river / chokepoint mechanics stay the same: a solid n x n square slides through
#      a blue corridor and either reaches the other square or stops at the farthest feasible
#      placement before a narrow gate.
#    - The changed part is role decoding. The two same-sized solid squares no longer have fixed
#      semantic colors. Instead, a tiny 2-cell control bar is placed on exactly one grid edge.
#      Its edge indicates the preferred inward direction:
#        top -> compare squares from top to bottom
#        bottom -> compare from bottom to top
#        left -> compare from left to right
#        right -> compare from right to left
#      The square encountered first along that directed axis is the mover; the other one is
#      the destination.
#    - Small example: if the control bar sits on the left border, the leftmost square is the
#      mover even if its color is numerically larger than the right square's color. The original
#      solver would have sorted colors; this variant must sort by geometry relative to the
#      control edge instead.
#
# b) Complexity calibration
#    - Only one new symbol system is introduced: a 2-cell border marker. That is enough to be
#      visually obvious without turning the task into a multi-symbol codebook.
#    - Square colors are randomized from a small palette so color can no longer reliably encode
#      role, but the total color count stays low and the scene remains easy to parse.
#    - The marker has exactly four interpretations (one per edge), which is a small repeated
#      vocabulary suitable for ARC-style induction.
#
# c) Hidden assumptions in the original main() that no longer hold
#    - `cand.sort(key=lambda s: int(s["col"]))` assumed role = color order. That is invalid now.
#      We replace it with directional ordering derived from the control marker.
#    - River-color detection originally excluded only bg and the two square colors. The control
#      marker adds another non-river color, so it must also be excluded.
#    - n=1 would have made a 1-pixel control marker ambiguous with 1x1 squares, so the variant
#      uses a 2-cell border bar. This keeps the marker visually distinct while preserving n=1.
#
# d) Placement legality guarantees enforced by the generator
#    - The control bar is always fully on one border, never on a corner, and only on background
#      cells so it cannot overlap the river or either square.
#    - The two role-candidate squares remain solid, same-sized, non-overlapping, and strictly
#      ordered along the chosen control direction, so the solver never faces a tie.
#    - The existing corridor construction and chokepoint checks are retained, so the n x n
#      placement BFS remains valid.
#
# e) Visual observability
#    - Across examples, the moving square's color changes, so humans cannot rely on "black moves".
#      The only stable instruction is the edge marker, which is large enough to notice and has a
#      clear spatial meaning.
#    - The output visibly confirms the rule because the square nearest the marked edge is the one
#      whose color travels through the river.
#
# f) Visual cleanliness
#    - The control bar uses its own color and sits on empty border cells, avoiding overlap with
#      the river mouth or the destination touching the edge.
#    - Using a 2-cell bar instead of scattered pixels keeps the added symbol crisp and easy to
#      distinguish from the squares and corridor.
#
# g) Color inference constraints
#    - No hidden color-id ordering is used anywhere in the solver. Colors only distinguish regions.
#      The semantic instruction is carried by the marker's border position, which is fully visible.
#
# New main() assumptions that the generator explicitly guarantees
#    - Exactly one 2-cell monochromatic border bar exists.
#    - Exactly two solid same-sized squares exist as the role candidates.
#    - The control direction gives a strict upstream/downstream order between those squares.


_BG = 3
_RIVER = 1
_BASE_MOVER = 0
_BASE_DEST = 2
_CTRL = 8
_SQUARE_PALETTE = [0, 2, 4, 5, 6, 7, 9]


def _control_side_from_bbox(bbox, H: int, W: int) -> str:
    r0, c0, r1, c1 = map(int, bbox)
    if r0 == 0:
        return "top"
    if r1 == H - 1:
        return "bottom"
    if c0 == 0:
        return "left"
    if c1 == W - 1:
        return "right"
    raise AssertionError("control marker must touch exactly one border")


def _direction_key(side: str, square: dict) -> tuple[int, int]:
    r0 = int(square["r0"])
    c0 = int(square["c0"])
    if side == "top":
        return (r0, c0)
    if side == "bottom":
        return (-r0, c0)
    if side == "left":
        return (c0, r0)
    if side == "right":
        return (-c0, r0)
    raise AssertionError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide one solid square through a winding channel, with the moving role chosen by an edge marker.

    Input structure:
    - bg dominates the corners; a river color forms the corridor.
    - Two same-sized solid squares sit on the river, but their colors no longer imply mover/goal.
    - A tiny 2-cell control bar on one border encodes the inward priority direction.

    Rule:
    - Read the marked border to decide which square is upstream: the nearer square along that
      inward direction is the mover and the other square is the destination.
    - The mover travels through positions where the full n x n footprint stays on the traversable
      river-plus-square mask.
    - If the destination placement is reachable, the mover replaces it; otherwise it stops at the
      farthest reachable placement, and its starting cells turn back into river.

    Key insight: The border marker decides role assignment, while chokepoints narrower than n still
    determine whether the moving square can actually finish the trip.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    control = None
    squares: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]

        # The control marker is the unique 2-cell border bar.
        if area == 2 and ((h == 1 and w == 2) or (h == 2 and w == 1)):
            if int(b[0]) == 0 or int(b[1]) == 0 or int(b[2]) == H - 1 or int(b[3]) == W - 1:
                if bool((patch == int(comp["color"])).all()):
                    assert control is None
                    control = {"col": int(comp["color"]), "bbox": b}
                    continue

        if h != w:
            continue
        if not bool((patch == int(comp["color"])).all()):
            continue
        squares.append({"col": int(comp["color"]), "r0": int(b[0]), "c0": int(b[1]), "n": h})

    assert control is not None

    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in squares:
        by_n[int(s["n"])].append(s)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    side = _control_side_from_bbox(control["bbox"], H, W)
    cand = sorted(cand, key=lambda s: _direction_key(side, s))
    mover, dest = cand[0], cand[1]

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    excluded = {int(bg), int(control["col"]), int(mover["col"]), int(dest["col"])}
    for col, _ct in counts.most_common():
        if int(col) in excluded:
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))

    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert not (h2 <= 0 or w2 <= 0)
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all()):
                feasible[int(r), int(c)] = True
    assert bool(feasible[int(sr), int(sc)])

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
    mask = (grid == _RIVER) | (grid == _BASE_MOVER) | (grid == _BASE_DEST)
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


def _choose_control_side(mover_r, mover_c, dest_r, dest_c, n, H, W):
    distances = {
        "top": (mover_r, dest_r),
        "bottom": (H - (mover_r + n), H - (dest_r + n)),
        "left": (mover_c, dest_c),
        "right": (W - (mover_c + n), W - (dest_c + n)),
    }
    viable = []
    for side, (mover_d, dest_d) in distances.items():
        if mover_d < dest_d:
            viable.append((dest_d - mover_d, -mover_d, side))
    if not viable:
        return None
    viable.sort(reverse=True)
    return viable[0][2]


def _place_control_bar(grid, side, anchor_r, anchor_c):
    H, W = grid.shape
    if side in ("top", "bottom"):
        r = 0 if side == "top" else H - 1
        candidates = list(range(1, W - 2))
        center = anchor_c + 0.5
        candidates.sort(key=lambda c: abs((c + 0.5) - center))
        for c in candidates:
            if bool((grid[r, c : c + 2] == _BG).all()):
                grid[r, c : c + 2] = _CTRL
                return True
    else:
        c = 0 if side == "left" else W - 1
        candidates = list(range(1, H - 2))
        center = anchor_r + 0.5
        candidates.sort(key=lambda r: abs((r + 0.5) - center))
        for r in candidates:
            if bool((grid[r : r + 2, c] == _BG).all()):
                grid[r : r + 2, c] = _CTRL
                return True
    return False


def _sample_role_colors():
    a, b = random.sample(_SQUARE_PALETTE, 2)
    if random.random() < 0.5:
        return a, b
    return b, a


def _apply_role_colors(grid, mover_color, dest_color):
    out = grid.copy()
    mover_mask = out == _BASE_MOVER
    dest_mask = out == _BASE_DEST
    out[mover_mask] = int(mover_color)
    out[dest_mask] = int(dest_color)
    return out


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
    if walk is None or len(walk) < 4:
        return None
    if len({cr for cr, _ in walk}) < coarse_rows or len({cc for _, cc in walk}) < coarse_cols:
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
            elif direction == "right":
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H):
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

    base = grid.copy()
    base[mover_r : mover_r + n, mover_c : mover_c + n] = _BASE_MOVER
    base[dest_r : dest_r + n, dest_c : dest_c + n] = _BASE_DEST

    trav = (base == _RIVER) | (base == _BASE_MOVER) | (base == _BASE_DEST)
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

    control_side = _choose_control_side(mover_r, mover_c, dest_r, dest_c, n, H, W)
    if control_side is None:
        return None

    mover_color, dest_color = _sample_role_colors()
    final_grid = _apply_role_colors(base, mover_color, dest_color)
    if not _place_control_bar(final_grid, control_side, mover_r, mover_c):
        return None

    expected = final_grid.copy()
    expected[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    expected[tr : tr + n, tc : tc + n] = mover_color

    if not np.array_equal(main(final_grid), expected):
        return None
    return final_grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

# Variant of 332f06d7
# Dimension: D16 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——角色由固定颜色值隐式编码（黑=移动体，红=目标，蓝=通道，绿=背景）... → 图例区域编码

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D16 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     - Add one compact legend box on the top edge.
#     - The legend is a 3x9 black panel whose middle row contains 4 colored cells.
#     - Left-to-right order is fixed and visible: background, river, mover, destination.
#   why_this_is_observable:
#     - The moving square, target square, river, and dominant background are all easy to identify
#       from the input/output pair, so a solver can align those observed roles with the 4 legend slots.
#     - Colors vary per instance, so the role mapping can no longer be recovered from color ids.
#   behavior_trace:
#     - Original: detect the two same-sized solid squares, sort by color id, and infer river as the
#       most common remaining color.
#     - Variant: first detect the black legend panel, read its 4 key cells, and thereby learn the
#       actual colors for bg/river/mover/destination; only then run the original sliding logic.
#     - Difference: color-id ordering is invalid now; role lookup is explicit and must happen before
#       component selection and traversability construction.
#   complexity_calibration:
#     - The legend contributes exactly four key cells and one stable frame color (black), so the
#       symbolic overhead is visible but still small.
#     - Black is reserved for the legend only; task roles are sampled from 1..9, which prevents
#       accidental ambiguity between the instruction area and the scene objects.
#   hidden_assumptions_checked_in_main:
#     - Corner-majority background is no longer sufficient as the primary role source; the legend
#       must be parsed first, then the detected legend area is neutralized back to bg before
#       component analysis.
#     - Sorting squares by color is invalid and replaced by legend-guided color matching.
#     - Inferring river from frequency is invalid and replaced by the legend's river key.
#     - Traversability must ignore the legend even if it contains role colors, so the legend bbox is
#       overwritten with bg in an internal working copy before BFS.
#   placement_legality:
#     - The legend touches the top edge but never the corners, so bg still dominates the corners and
#       the legend stays visually separate from the river scene.
#     - The scene is generated entirely below the legend strip, so the moving square and the river
#       never overlap the instruction area.
#   visual_cleanliness:
#     - The legend is one compact horizontal strip with fixed spacing; it does not stack extra marks
#       onto the main puzzle area.
#   generator_guarantees_for_new_assumptions:
#     - Exactly one legend panel exists and black appears nowhere else.
#     - The four legend cells are distinct colors matching the actual bg/river/mover/destination.
#     - The mover and destination are the only solid same-sized squares of their colors.


LEGEND_COLOR = 0
LEGEND_H = 3
LEGEND_W = 9
LEGEND_KEY_ROW = 1
LEGEND_KEY_COLS = (1, 3, 5, 7)
ROLE_ORDER = ("bg", "river", "mover", "dest")
SCENE_TOP = LEGEND_H
PALETTE = tuple(range(1, 10))


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == int(LEGEND_COLOR)
    labeled, num = scipy_label(mask.astype(int), structure=np.ones((3, 3), dtype=int))
    best = None
    for idx in range(1, int(num) + 1):
        coords = np.argwhere(labeled == int(idx))
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        sub = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        if not bool((sub != int(LEGEND_COLOR)).any()):
            continue
        area = int(coords.shape[0])
        cand = (area, int(r0), int(c0), int(r1), int(c1))
        if best is None or cand > best:
            best = cand
    assert not (best is None)
    return int(best[1]), int(best[2]), int(best[3]), int(best[4])


def _decode_legend(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = map(int, bbox)
    panel = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    keys = []
    for rr in range(panel.shape[0]):
        for cc in range(panel.shape[1]):
            col = int(panel[int(rr), int(cc)])
            if col != int(LEGEND_COLOR):
                keys.append((int(rr), int(cc), int(col)))
    assert not (len(keys) != 4)
    rows = {int(rr) for rr, _cc, _col in keys}
    assert not (len(rows) != 1)
    keys.sort(key=lambda t: int(t[1]))
    return {role: int(col) for role, (_rr, _cc, col) in zip(ROLE_ORDER, keys)}


def _scene_without_legend(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = map(int, bbox)
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel after decoding a legend strip.

    Input structure:
    - A small black legend panel sits on the top edge, with four colored key cells.
    - Outside that panel, one color is the background, one is the river, and two equal-sized solid squares
      mark the mover and the destination.

    Rule:
    - Read the legend left-to-right to learn which colors mean background, river, mover, and destination.
    - The mover square slides through the river only where the full NxN footprint remains on river-or-square cells.
    - If the destination placement is reachable, the mover covers it; otherwise the mover stops at the farthest
      reachable placement, and its original location becomes river.

    Key insight: The geometry is unchanged from the original task; the new challenge is that role colors are no
    longer fixed, so the legend must be interpreted before any square matching or path search is valid.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    legend_bbox = _find_legend_bbox(g)
    roles = _decode_legend(g, legend_bbox)
    bg = int(roles["bg"])
    river = int(roles["river"])
    mover_col = int(roles["mover"])
    dest_col = int(roles["dest"])

    corner_bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    assert not (corner_bg != int(bg))

    scene = _scene_without_legend(g, bg, legend_bbox)

    squares: list[dict] = []
    for comp in find_components(scene, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = scene[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append({"col": int(col), "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert not (len(squares) < 2)

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    valid_sizes = [
        int(nside)
        for nside, comps in by_n.items()
        if any(int(c["col"]) == int(mover_col) for c in comps) and any(int(c["col"]) == int(dest_col) for c in comps)
    ]
    assert not (not valid_sizes)

    nside = max(valid_sizes)
    cand = by_n[int(nside)]
    mover_candidates = [sq for sq in cand if int(sq["col"]) == int(mover_col)]
    dest_candidates = [sq for sq in cand if int(sq["col"]) == int(dest_col)]
    assert not (len(mover_candidates) != 1 or len(dest_candidates) != 1)

    mover = mover_candidates[0]
    dest = dest_candidates[0]

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (scene == int(river)) | (scene == int(mover_col)) | (scene == int(dest_col))

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
    for_river = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while q:
        r, c = q.popleft()
        for dr2, dc2 in for_river:
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
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover_col)
    return out


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((int(rr), int(cc)))
    return dist


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = int(rows * cols)
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
                nr, nc = int(cur[0] + dr), int(cur[1] + dc)
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((int(nr), int(nc)))
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
                        nr2, nc2 = int(cr + dr2), int(cc + dc2)
                        if (0 <= nr2 < rows and 0 <= nc2 < cols and
                                (nr2, nc2) not in test_visited and (nr2, nc2) not in seen):
                            seen.add((int(nr2), int(nc2)))
                            q.append((int(nr2), int(nc2)))
                for dr2, dc2 in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ar, ac = int(nb[0] + dr2), int(nb[1] + dc2)
                    if (0 <= ar < rows and 0 <= ac < cols and
                            (ar, ac) not in test_visited and (ar, ac) in seen):
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
    row_offset: int,
    col_offset: int,
    river: int,
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr = int(row_offset + offsets_r[int(cr)])
        pc = int(col_offset + offsets_c[int(cc)])
        pixel_pos.append((int(pr), int(pc)))
        grid[int(pr) : int(pr + n), int(pc) : int(pc + n)] = int(river)
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[int(pr1) : int(pr1 + n), int(c_lo) : int(c_hi)] = int(river)
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[int(r_lo) : int(r_hi), int(pc1) : int(pc1 + n)] = int(river)
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
    pr1, pc1 = map(int, pixel_pos_i)
    pr2, pc2 = map(int, pixel_pos_j)
    narrow_w = random.randint(1, n - 1)
    remove = int(n - narrow_w)
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        hold_len = random.randint(1, min(6, max(1, c_hi - c_lo - 2 * remove)))
        total_len = int(2 * remove + hold_len)
        c_mid = int((c_lo + c_hi) // 2)
        c_start = int(max(c_lo, c_mid - total_len // 2))
        for step in range(total_len):
            c = int(c_start + step)
            if not (0 <= c < W):
                continue
            if step < remove:
                depth = int(step + 1)
            elif step >= total_len - remove:
                depth = int(total_len - step)
            else:
                depth = int(remove)
            for k in range(depth):
                rr = int(pr1 + k) if side == 0 else int(pr1 + n - 1 - k)
                if 0 <= rr < H:
                    grid[int(rr), int(c)] = int(bg)
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(6, max(1, r_hi - r_lo - 2 * remove)))
        total_len = int(2 * remove + hold_len)
        r_mid = int((r_lo + r_hi) // 2)
        r_start = int(max(r_lo, r_mid - total_len // 2))
        for step in range(total_len):
            r = int(r_start + step)
            if not (0 <= r < H):
                continue
            if step < remove:
                depth = int(step + 1)
            elif step >= total_len - remove:
                depth = int(total_len - step)
            else:
                depth = int(remove)
            for k in range(depth):
                cc = int(pc1 + k) if side == 0 else int(pc1 + n - 1 - k)
                if 0 <= cc < W:
                    grid[int(r), int(cc)] = int(bg)


def _river_connected(grid: np.ndarray, river: int, mover: int, dest: int) -> bool:
    mask = (grid == int(river)) | (grid == int(mover)) | (grid == int(dest))
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(length: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = int(length - n - 1)
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _draw_legend(grid: np.ndarray, c0: int, role_colors: tuple[int, int, int, int]) -> None:
    grid[0:LEGEND_H, int(c0) : int(c0 + LEGEND_W)] = int(LEGEND_COLOR)
    for idx, col in enumerate(role_colors):
        cc = int(c0 + LEGEND_KEY_COLS[idx])
        grid[int(LEGEND_KEY_ROW), int(cc)] = int(col)


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    bg, river, mover, dest = random.sample(PALETTE, 4)

    min_size = max(12, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    scene_rows = int(H - SCENE_TOP)
    scene_cols = int(W)
    if scene_rows <= n + 2:
        return None

    min_spacing = int(n + 1)
    max_coarse_rows = max(3, (scene_rows - 2) // min_spacing)
    max_coarse_cols = max(3, (scene_cols - 2) // min_spacing)
    coarse_rows = random.randint(3, min(6, max_coarse_rows))
    coarse_cols = random.randint(3, min(6, max_coarse_cols))

    offsets_r = _compute_offsets(scene_rows, n, coarse_rows)
    offsets_c = _compute_offsets(scene_cols, n, coarse_cols)
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
        edge_cells.add((coarse_rows - 1, c))
    for r in range(coarse_rows):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_cells.discard((0, 0))
    edge_cells.discard((0, coarse_cols - 1))
    edge_cells = list(edge_cells)
    if len(edge_cells) < 2:
        return None
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len({cr for cr, _cc in walk}) < coarse_rows or len({cc for _cr, cc in walk}) < coarse_cols:
        return None

    grid = np.full((H, W), int(bg), dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c, SCENE_TOP, 0, river)

    scene_mask = grid[SCENE_TOP:, :] != int(bg)
    if np.any(scene_mask, axis=1).sum() / scene_rows < 0.70 or np.any(scene_mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = map(int, pixel_pos[0])
    dest_r, dest_c = map(int, pixel_pos[-1])

    def _extend_to_edge(pr: int, pc: int) -> tuple[int, int] | None:
        candidates = []
        dist = H - (pr + n)
        if dist == 0:
            candidates.append((int(pr), int(pc), 0))
        elif all(grid[r, c] in (bg, river) for r in range(pr + n, H) for c in range(pc, pc + n)):
            candidates.append((int(H - n), int(pc), int(dist)))

        dist = pc
        if dist == 0:
            candidates.append((int(pr), 0, 0))
        elif all(grid[r, c] in (bg, river) for r in range(pr, pr + n) for c in range(pc)):
            candidates.append((int(pr), 0, int(dist)))

        dist = W - (pc + n)
        if dist == 0:
            candidates.append((int(pr), int(pc), 0))
        elif all(grid[r, c] in (bg, river) for r in range(pr, pr + n) for c in range(pc + n, W)):
            candidates.append((int(pr), int(W - n), int(dist)))

        if not candidates:
            return None
        candidates.sort(key=lambda x: int(x[2]))
        return int(candidates[0][0]), int(candidates[0][1])

    result = _extend_to_edge(dest_r, dest_c)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[int(r_lo) : int(r_hi), int(dest_c) : int(dest_c + n)] = int(river)
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[int(dest_r) : int(dest_r + n), int(c_lo) : int(c_hi)] = int(river)
    dest_r, dest_c = int(new_dest_r), int(new_dest_c)
    if not (dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    touching = (mover_r + n == H or mover_c == 0 or mover_c + n == W)
    if touching:
        if mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = int(river)
            mover_r = int(H - n - 1)
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = int(river)
            mover_c = 1
        elif mover_c + n == W:
            grid[mover_r : mover_r + n, W - 1] = int(river)
            mover_c = int(W - n - 1)
    if mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_c, W - (mover_c + n), H - (mover_r + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = int(river)
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = int(river)

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
        if not _river_connected(grid, river, mover, dest):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = int(mover)
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = int(dest)

    trav = (grid == int(river)) | (grid == int(mover)) | (grid == int(dest))
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all()):
                feasible[int(r), int(c)] = True
    if not bool(feasible[int(mover_r), int(mover_c)]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    dest_ok = (
        0 <= dest_r < h2 and
        0 <= dest_c < w2 and
        bool(feasible[int(dest_r), int(dest_c)]) and
        int(dist[int(dest_r), int(dest_c)]) >= 0
    )
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    if dest_ok:
        tr, tc = int(dest_r), int(dest_c)
    else:
        best_d = -1
        tr, tc = int(mover_r), int(mover_c)
        for r in range(h2):
            for c in range(w2):
                if int(dist[int(r), int(c)]) > best_d:
                    best_d = int(dist[int(r), int(c)])
                    tr, tc = int(r), int(c)

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = int(river)
    output[tr : tr + n, tc : tc + n] = int(mover)

    legend_c0 = random.randint(1, W - LEGEND_W - 1)
    role_colors = (int(bg), int(river), int(mover), int(dest))
    _draw_legend(grid, legend_c0, role_colors)
    _draw_legend(output, legend_c0, role_colors)

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")

# Combo variant of 332f06d7 (index 12)
# combo_spec: D5/2+D20/0
# Name: 边缘停止+起点在边缘
#
# Design analysis
# - Concrete behavior:
#   Original 332f06d7 shows a black n×n square starting inside a blue winding river,
#   moving toward a red n×n destination. Here both changed dimensions interact:
#   the red target disappears (D5/2), and the black mover is placed directly on one
#   grid edge (D20/0). The output erases the black start square back to blue and
#   repaints it either on the opposite edge when the corridor stays at least n wide,
#   or at the farthest reachable interior placement before a narrow gate.
# - Behavior trace on a small example:
#   Suppose n=2, the mover starts on the top edge, and the blue river winds downward
#   to the bottom edge. The original solver would search for one red 2×2 target. The
#   D5/2-only variant would accept any reachable boundary placement, but that becomes
#   wrong once the mover itself already starts on a boundary: the start cell would be
#   a trivial "success". The fused rule therefore identifies the mover's entry side
#   (top here) and only treats placements touching the opposite side (bottom) as a
#   successful crossing. If a width-1 chokepoint appears midway, the mover cannot
#   fit through; it stops just before the gate instead of snapping back to its start.
# - Complexity calibration:
#   Keeping only three colors {background, river, mover} is acceptable because the
#   geometry now supplies the missing structure: one boundary entry, one opposite
#   boundary exit, a winding corridor, and optional chokepoints. The generator keeps
#   path turns and broad spatial coverage so the task still reads as a traversal
#   problem rather than a trivial straight slide.
# - Implicit assumption audit:
#   1. Original main() assumed at least two same-sized solid squares and sorted them
#      by color to pick mover vs destination. That is invalid; this variant instead
#      finds the unique solid square that touches exactly one edge.
#   2. Original traversability included destination color. That object no longer
#      exists, so traversability is just {river ∪ mover}.
#   3. Original stopping logic targeted one fixed destination placement. D5/2 changed
#      that to "some boundary placement", but D20/0 breaks that assumption because
#      the start is already on a boundary. The fused solver therefore searches only
#      the opposite boundary side.
#   4. The farthest-reachable fallback still works, but now it must compete only with
#      opposite-edge placements, not arbitrary edge placements.
# - Placement legality:
#   The generator first paints an internal winding corridor, then extends its first
#   and last coarse cells to opposite grid edges using width-n boundary strips. The
#   mover is placed on the start strip after the river exists, so its initial n×n
#   placement is always feasible. Chokepoints are inserted only on interior segments
#   and never disconnect river pixels entirely; they only block n×n placements.
# - Visual observability / cleanliness:
#   The input visibly differs from both the original and D5/2-only versions: the
#   black square starts flush against an edge, there is no red square, and the river
#   appears to enter from one side and potentially leave from the opposite side. In
#   reachable examples, the output square lands on the opposite frame; in blocked
#   examples, it stops inside just before a narrow section. No overlapping markers or
#   hidden color ordering are required.
# - New solver assumptions guaranteed by the generator:
#   There is exactly one solid non-background square component, it touches exactly one
#   edge and not a corner, the river is the only other non-background color, the river
#   has one meaningful entry side and one meaningful opposite exit side, and the mover
#   start placement is a legal n×n placement on the traversable mask.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


def _detect_mover(g: np.ndarray, bg: int) -> dict:
    candidates: list[dict] = []
    H, W = g.shape
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        r0, c0, r1, c1 = map(int, b)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == col).all()):
            continue
        touches = []
        if r0 == 0:
            touches.append("top")
        if c0 == 0:
            touches.append("left")
        if r1 == H - 1:
            touches.append("bottom")
        if c1 == W - 1:
            touches.append("right")
        if len(touches) != 1:
            continue
        candidates.append({"col": col, "r0": r0, "c0": c0, "n": int(bbox_h(b)), "side": touches[0]})
    assert len(candidates) == 1
    return candidates[0]


def _opposite_side(side: str) -> str:
    return {
        "top": "bottom",
        "bottom": "top",
        "left": "right",
        "right": "left",
    }[side]


def _positions_on_side(dist: np.ndarray, side: str) -> list[tuple[int, int]]:
    h2, w2 = dist.shape
    positions: list[tuple[int, int]] = []
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) < 0:
                continue
            if side == "top" and r == 0:
                positions.append((r, c))
            elif side == "bottom" and r == h2 - 1:
                positions.append((r, c))
            elif side == "left" and c == 0:
                positions.append((r, c))
            elif side == "right" and c == w2 - 1:
                positions.append((r, c))
    return positions


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = int(sr), int(sc)
    h2, w2 = dist.shape
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square from one grid edge through a winding channel.

    The mover begins on exactly one edge. It can occupy only n×n placements fully
    covered by river or mover pixels. If some reachable placement touches the edge
    opposite the mover's start side, the mover stops at the farthest such opposite
    placement. Otherwise it stops at the farthest reachable placement before the
    blockage. The original mover position is filled with river color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover = _detect_mover(g, bg)
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    start_side = str(mover["side"])
    exit_side = _opposite_side(start_side)

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == int(bg) or int(col) == int(mover["col"]):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == int(river)) | (g == int(mover["col"]))
    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    exit_positions = _positions_on_side(dist, exit_side)
    if exit_positions:
        tr, tc = max(exit_positions, key=lambda pos: (int(dist[pos[0], pos[1]]), -abs(pos[0] - sr) - abs(pos[1] - sc)))
    else:
        tr, tc = _farthest_reachable(dist, sr, sc)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


_BG, _RIVER, _MOVER = 3, 1, 0


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        d1 = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        d2 = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if d1 != d2:
            turns += 1
    return turns


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


def _render_channel(grid: np.ndarray, path, n: int, offsets_r: list[int], offsets_c: list[int]):
    pixel_pos = []
    for cr, cc in path:
        pr, pc = int(offsets_r[cr]), int(offsets_c[cc])
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo : c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo : r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str):
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
    mask = (grid == _RIVER) | (grid == _MOVER)
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, n: int, num_coarse: int):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _boundary_extension(grid: np.ndarray, pr: int, pc: int, n: int, side: str) -> tuple[int, int]:
    H, W = grid.shape
    if side == "top":
        grid[0 : pr + n, pc : pc + n] = _RIVER
        return 0, pc
    if side == "bottom":
        grid[pr:H, pc : pc + n] = _RIVER
        return H - n, pc
    if side == "left":
        grid[pr : pr + n, 0 : pc + n] = _RIVER
        return pr, 0
    grid[pr : pr + n, pc:W] = _RIVER
    return pr, W - n


def _pick_start_end(coarse_rows: int, coarse_cols: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    orientation = random.choice(["tb", "lr"])
    if orientation == "tb":
        start_side = random.choice(["top", "bottom"])
        start_c = random.randrange(coarse_cols)
        end_choices = list(range(coarse_cols))
        end_choices.sort(key=lambda c: -abs(c - start_c))
        end_c = random.choice(end_choices[: max(1, min(3, len(end_choices)))])
        if start_side == "top":
            return start_side, (0, start_c), (coarse_rows - 1, end_c)
        return start_side, (coarse_rows - 1, start_c), (0, end_c)
    start_side = random.choice(["left", "right"])
    start_r = random.randrange(coarse_rows)
    end_choices = list(range(coarse_rows))
    end_choices.sort(key=lambda r: -abs(r - start_r))
    end_r = random.choice(end_choices[: max(1, min(3, len(end_choices)))])
    if start_side == "left":
        return start_side, (start_r, 0), (end_r, coarse_cols - 1)
    return start_side, (start_r, coarse_cols - 1), (end_r, 0)


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

    start_side, start_coarse, end_coarse = _pick_start_end(coarse_rows, coarse_cols)
    exit_side = _opposite_side(start_side)
    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4 or _turn_count(walk) < 3:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)
    mover_r, mover_c = _boundary_extension(grid, pixel_pos[0][0], pixel_pos[0][1], n, start_side)
    _edge_r, _edge_c = _boundary_extension(grid, pixel_pos[-1][0], pixel_pos[-1][1], n, exit_side)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    if not reached and n >= 2:
        min_seg = max(1, len(walk) // 4)
        max_seg = max(min_seg, len(walk) - 2)
        available = list(range(min_seg, max_seg))
        random.shuffle(available)
        chosen = available[: random.randint(1, min(3, len(available)))]
        for seg_idx in chosen:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[seg_idx + 1]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER

    trav = (grid == _RIVER) | (grid == _MOVER)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    exit_positions = _positions_on_side(dist, exit_side)
    exit_ok = bool(exit_positions)
    if reached and not exit_ok:
        return None
    if not reached and exit_ok:
        return None

    if exit_ok:
        tr, tc = max(exit_positions, key=lambda pos: (int(dist[pos[0], pos[1]]), -abs(pos[0] - mover_r) - abs(pos[1] - mover_c)))
    else:
        tr, tc = _farthest_reachable(dist, mover_r, mover_c)

    out = grid.copy()
    out[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    out[tr : tr + n, tc : tc + n] = _MOVER
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")

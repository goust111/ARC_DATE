# Variant of 332f06d7
# Dimension: D5 — 终点条件 (dst)
# Change: 同尺寸红色实心方块位于通道末端（接触网格边缘），可达时方块覆盖目标位置... → 堆叠终点

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components

# Design analysis for the D5 variant
# a) Concrete behavior and trace:
#    The red square is no longer a cell the mover covers. Instead, it is a border-touching marker
#    for an adjacent same-sized "parking slot" on its inward side:
#    - top-edge red square -> mover parks immediately below it
#    - bottom-edge red square -> mover parks immediately above it
#    - left-edge red square -> mover parks immediately to its right
#    - right-edge red square -> mover parks immediately to its left
#    Small trace with n=2: old logic ended with black at red top-left (0, 6). New logic keeps the
#    red 2x2 at (0, 6) and places the black 2x2 at (2, 6), touching the red along one full side.
#    The stuck branch is unchanged: if the parking slot is unreachable because some river section is
#    narrower than n, the mover still stops at the farthest reachable placement.
# b) Complexity calibration:
#    The new degree of freedom is only the stack direction, and it is limited to four visible cases
#    determined by the edge touched by the red square. This is enough to make the variant clearly
#    different from the original without introducing hidden category mappings.
# c) Old implicit assumptions that no longer hold:
#    - Sorting candidate squares by color id is no longer acceptable; the visible role cue is now
#      border contact. The solver therefore identifies the destination as the unique same-sized
#      solid square touching exactly one grid edge, and the mover as the same-sized interior square.
#    - Using the destination top-left as the final placement is wrong in this variant; the final
#      placement is the inward-adjacent slot derived from the touched edge.
#    - Treating the red square as traversable is unnecessary and can blur the new endpoint logic.
#      The traversable set is the river plus the mover's current square.
#    - The output may not overwrite the red square; both objects must remain visible when reached.
# d) Placement legality:
#    The generator guarantees that the destination touches exactly one non-corner edge, the inward
#    stack slot stays within bounds, and the river fills that slot as a legal n×n placement. The
#    mover and destination never overlap, and the slot is part of the river corridor.
# e) Visual observability:
#    From 2-4 examples, the human can see that the black square consistently ends flush against the
#    border-touching red square on the inward side, instead of replacing it. No role depends on
#    invisible facts like color-id ordering.
# f) Visual cleanliness:
#    The reachable endpoint is a full n×n parking bay contiguous with the corridor, so the output
#    shows two separate solid squares sharing one side, not tangled overlap.
# g) Color-change inference:
#    No new color mapping is introduced. Colors keep the same semantic roles as in the source task;
#    only the destination condition changes.


def _touch_sides(r0: int, c0: int, n: int, H: int, W: int) -> list[str]:
    sides: list[str] = []
    if r0 == 0:
        sides.append("top")
    if r0 + n == H:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c0 + n == W:
        sides.append("right")
    return sides


def _detect_roles(g: np.ndarray, bg: int, river: int) -> tuple[dict, dict]:
    H, W = g.shape
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col == int(river):
            continue
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
                "touch": _touch_sides(r0, c0, n, H, W),
            }
        )

    assert squares

    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in squares:
        by_n[int(s["n"])].append(s)

    candidates: list[tuple[int, int, int, list[dict]]] = []
    for n, comps in by_n.items():
        edge = [s for s in comps if len(s["touch"]) == 1]
        inner = [s for s in comps if len(s["touch"]) == 0]
        if edge and inner:
            candidates.append((len(edge) + len(inner), len(edge), n, comps))
    assert candidates

    _total, _edge_count, _n, comps = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    edge = [s for s in comps if len(s["touch"]) == 1]
    inner = [s for s in comps if len(s["touch"]) == 0]
    assert len(edge) == 1 and len(inner) == 1
    return inner[0], edge[0]


def _stack_slot(dest: dict, H: int, W: int) -> tuple[int, int]:
    n = int(dest["n"])
    r0 = int(dest["r0"])
    c0 = int(dest["c0"])
    sides = dest["touch"]
    assert len(sides) == 1
    side = sides[0]
    if side == "top":
        return n, c0
    if side == "bottom":
        return H - 2 * n, c0
    if side == "left":
        return r0, n
    return r0, W - 2 * n


def _river_color(g: np.ndarray, bg: int, mover_col: int | None = None, dest_col: int | None = None) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if mover_col is not None and int(col) == int(mover_col):
            continue
        if dest_col is not None and int(col) == int(dest_col):
            continue
        return int(col)
    raise AssertionError("river color not found")


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = sr, sc
    h2, w2 = dist.shape
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid square through a winding channel and park it beside a border target.

    Input structure:
    - bg fills most of the canvas. A river-colored corridor forms a winding route.
    - One solid square is the mover. Another same-sized solid square is the destination marker.
    - The destination square touches exactly one grid edge; the inward-adjacent slot is the stack bay.

    Rule:
    - The mover travels through river positions where the whole n×n footprint fits.
    - If the inward stack bay next to the destination is reachable, move the square there and keep
    the red destination visible.
    - If a narrow chokepoint blocks the route, stop at the farthest reachable placement instead.
    - Refill the mover's starting square with the river color.

    Key insight: The red square marks the end of the route, but the true reachable endpoint is the
    same-sized bay immediately inside the grid on the side opposite the touched border.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    river = _river_color(g, bg)
    mover, dest = _detect_roles(g, bg, river)

    n = int(mover["n"])
    assert n == int(dest["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    gr, gc = _stack_slot(dest, H, W)
    trav = (g == river) | (g == int(mover["col"]))

    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert 0 <= sr < h2 and 0 <= sc < w2
    assert 0 <= gr < h2 and 0 <= gc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    if bool(feasible[gr, gc]) and int(dist[gr, gc]) >= 0:
        tr, tc = gr, gc
    else:
        tr, tc = _farthest_reachable(dist, sr, sc)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


# --- Generator ---

_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


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
                        if (0 <= nr2 < rows and 0 <= nc2 < cols and
                                (nr2, nc2) not in test_visited and (nr2, nc2) not in seen):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ar, ac = nb[0] + dr2, nb[1] + dc2
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


def _render_channel(grid: np.ndarray, path: list[tuple[int, int]], n: int,
                    offsets_r: list[int], offsets_c: list[int]) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr:pr + n, pc:pc + n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1:pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1:pc1 + n] = _RIVER
    return pixel_pos


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int],
                    n: int, direction: str) -> None:
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
        total_len = 2 * remove + hold_len
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
    mask = (grid == _RIVER)
    if not bool(mask.any()):
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


def _edge_side(cell: tuple[int, int], rows: int, cols: int) -> str | None:
    r, c = cell
    if r == 0:
        return "top"
    if r == rows - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == cols - 1:
        return "right"
    return None


def _goal_and_dest(side: str, H: int, W: int, n: int, pr: int, pc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (n, pc), (0, pc)
    if side == "bottom":
        return (H - 2 * n, pc), (H - n, pc)
    if side == "left":
        return (pr, n), (pr, 0)
    return (pr, W - 2 * n), (pr, W - n)


def _paint_tail_to_goal(grid: np.ndarray, side: str, n: int,
                        end_pos: tuple[int, int], goal_pos: tuple[int, int]) -> None:
    pr, pc = end_pos
    gr, gc = goal_pos
    if side in ("top", "bottom"):
        r_lo, r_hi = min(pr, gr), max(pr, gr) + n
        grid[r_lo:r_hi, pc:pc + n] = _RIVER
    else:
        c_lo, c_hi = min(pc, gc), max(pc, gc) + n
        grid[pr:pr + n, c_lo:c_hi] = _RIVER


def _expected_output(grid: np.ndarray, mover_r: int, mover_c: int, n: int, tr: int, tc: int) -> np.ndarray:
    out = grid.copy()
    out[mover_r:mover_r + n, mover_c:mover_c + n] = _RIVER
    out[tr:tr + n, tc:tc + n] = _MOVER
    return out


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

    edge_cells = []
    for c in range(1, coarse_cols - 1):
        edge_cells.append((0, c))
        edge_cells.append((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, coarse_cols - 1))
    if len(edge_cells) < 2:
        return None
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]
    end_side = _edge_side(end_coarse, coarse_rows, coarse_cols)
    if end_side is None:
        return None

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = (grid != _BG)
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    end_r, end_c = pixel_pos[-1]
    (goal_r, goal_c), (dest_r, dest_c) = _goal_and_dest(end_side, H, W, n, end_r, end_c)
    if goal_r < 0 or goal_c < 0 or goal_r + n > H or goal_c + n > W:
        return None
    if dest_r < 0 or dest_c < 0 or dest_r + n > H or dest_c + n > W:
        return None

    _paint_tail_to_goal(grid, end_side, n, (end_r, end_c), (goal_r, goal_c))

    touching = (mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W)
    if touching:
        if mover_r == 0:
            grid[0, mover_c:mover_c + n] = _RIVER
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c:mover_c + n] = _RIVER
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r:mover_r + n, 0] = _RIVER
            mover_c = 1
        else:
            grid[mover_r:mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    goal_slice = grid[goal_r:goal_r + n, goal_c:goal_c + n]
    dest_slice = grid[dest_r:dest_r + n, dest_c:dest_c + n]
    if not bool(np.isin(goal_slice, [_BG, _RIVER]).all()):
        return None
    if not bool(np.isin(dest_slice, [_BG, _RIVER]).all()):
        return None

    grid[mover_r:mover_r + n, mover_c:mover_c + n] = _RIVER
    grid[goal_r:goal_r + n, goal_c:goal_c + n] = _RIVER

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

    grid[mover_r:mover_r + n, mover_c:mover_c + n] = _MOVER
    grid[dest_r:dest_r + n, dest_c:dest_c + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER)
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r:r + n, c:c + n].all()):
                feasible[r, c] = True
    if not bool(feasible[mover_r, mover_c]):
        return None
    if len(_touch_sides(dest_r, dest_c, n, H, W)) != 1:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    goal_ok = bool(feasible[goal_r, goal_c]) and int(dist[goal_r, goal_c]) >= 0
    if reached and not goal_ok:
        return None
    if not reached and goal_ok:
        return None

    if goal_ok:
        tr, tc = goal_r, goal_c
    else:
        tr, tc = _farthest_reachable(dist, mover_r, mover_c)

    output = _expected_output(grid, mover_r, mover_c, n, tr, tc)
    if not np.array_equal(main(grid), output):
        return None
    if goal_ok and (tr != goal_r or tc != goal_c):
        return None
    if goal_ok and (abs(tr - dest_r) + abs(tc - dest_c) != n):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")

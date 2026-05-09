# Combo variant of 332f06d7 (index 19)
# combo_spec: D8/3+D18/0
# Name: 形状线索方向+边缘控制像素

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete visual change and behavior trace:
#    - The destination remains a solid n×n square.
#    - The mover is no longer identified by color order. It is the unique n×n square
#      body with one extra same-colored tip pixel on one side, forming a directional cue.
#    - A single red control pixel is placed on one non-corner boundary cell. Its edge
#      position encodes the forced first step: top->down, bottom->up, left->right,
#      right->left.
#    - The start region is generated with exactly two exits: the original corridor exit
#      and one short dead-end stub. The tip points to one of these exits, while the red
#      control pixel selects the actual first step. When they disagree, the control pixel
#      overrides the tip. The tip therefore still matters visually, because some samples
#      show agreement and others show conflict.
#    - Small trace:
#      original: sort colors -> pick lower-color square -> BFS from start over all exits.
#      combo: detect arrow-shaped mover -> read its tip direction -> read border control
#      direction -> force the first step to the control direction and forbid returning to
#      the start -> slide only the square body; erase the input-only tip back to river.
# b) Complexity calibration:
#    - Only two extra cues are added: one protrusion pixel and one edge control pixel.
#    - n is restricted to 2..4. This keeps the arrow decomposition legible and avoids
#      confusion between 1×1 objects and 1×1 control pixels.
#    - The generator enforces exactly two exits from the start after adding the stub, so
#      "agreement vs conflict" is easy to see and does not depend on hidden tie-breaking.
# c) Hidden assumptions broken from the original main():
#    - "Both special objects are solid squares" fails because the mover has one tip.
#    - "Lower color moves" fails because mover/destination colors are randomized.
#    - "All reachable branches from the start are fair game" fails because the border
#      control pixel forces a specific first step and the start cannot be revisited.
#    - "Only the destination uses the red color" fails because the red control pixel is
#      separate from the destination. The solver must detect the destination by shape, not
#      by red color.
# d) Placement legality:
#    - The tip is painted onto an already-river cell adjacent to the mover body, so it
#      never changes the true traversable geometry.
#    - The control pixel is always on a non-corner boundary background cell and is kept
#      4-disconnected from the destination square.
#    - The stub is carved only through background cells and must leave the start with
#      exactly two feasible exits.
# e) Visual observability:
#    - Across examples, humans can see one arrowed object, one plain target square, one
#      red boundary dot, and outputs that sometimes follow the arrow and sometimes defy it.
#      This makes the priority rule inferable from a few examples without color arithmetic.
# f) Cleanliness:
#    - The extra cues are sparse and non-overlapping: one tip attached to the mover and
#      one isolated boundary pixel.
#    - The generator rejects layouts where extra branches or overlaps would muddy the two
#      relevant exit choices.
# g) New solver assumptions introduced here and guaranteed by the generator:
#    - Exactly one non-control component is a valid "square + one side tip" mover.
#    - Exactly one other non-control component is a solid square of the same side length.
#    - After removing the red control pixel, the mover start has exactly two feasible
#      neighboring placements: the tip-indicated exit and the alternate exit.
#    - The control direction is always one of those exits, and the tip direction is the
#      other or the same exit when the cues agree.


_BG, _RIVER, _CTRL = 3, 1, 2
_OBJECT_COLORS = [0, 4, 5, 6, 7, 8, 9]

DIR4 = {
    "D": (1, 0),
    "U": (-1, 0),
    "R": (0, 1),
    "L": (0, -1),
}
STEP_TO_NAME = {v: k for k, v in DIR4.items()}


def _solid_square_info(g: np.ndarray, comp: dict) -> dict | None:
    col = int(comp["color"])
    b = comp["bbox"]
    if bbox_h(b) != bbox_w(b):
        return None
    patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
    if not bool((patch == int(col)).all()):
        return None
    return {
        "col": col,
        "r0": int(b[0]),
        "c0": int(b[1]),
        "n": int(bbox_h(b)),
        "coords": {(int(r), int(c)) for r, c in comp["coords"]},
    }


def _arrow_square_info(comp: dict) -> dict | None:
    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    if len(coords) < 5:
        return None

    matches: list[dict] = []
    for tip_r, tip_c in coords:
        rem = coords - {(tip_r, tip_c)}
        rows = [r for r, _ in rem]
        cols = [c for _, c in rem]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        n = int(r1 - r0 + 1)
        if n != int(c1 - c0 + 1):
            continue
        if len(rem) != n * n:
            continue

        full_square = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if (r, c) not in rem:
                    full_square = False
                    break
            if not full_square:
                break
        if not full_square:
            continue

        direction = None
        if tip_r == r0 - 1 and c0 <= tip_c <= c1:
            direction = "U"
        elif tip_r == r1 + 1 and c0 <= tip_c <= c1:
            direction = "D"
        elif tip_c == c0 - 1 and r0 <= tip_r <= r1:
            direction = "L"
        elif tip_c == c1 + 1 and r0 <= tip_r <= r1:
            direction = "R"
        if direction is None:
            continue

        matches.append(
            {
                "col": int(comp["color"]),
                "r0": int(r0),
                "c0": int(c0),
                "n": int(n),
                "dir": direction,
                "tip": (int(tip_r), int(tip_c)),
                "coords": set(coords),
            }
        )

    if len(matches) != 1:
        return None
    return matches[0]


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _branch_dist(feasible: np.ndarray, sr: int, sc: int, step: tuple[int, int]) -> tuple[np.ndarray | None, tuple[int, int]]:
    dr, dc = step
    h2, w2 = feasible.shape
    fr, fc = sr + dr, sc + dc
    if not (0 <= fr < h2 and 0 <= fc < w2):
        return None, (fr, fc)
    if not feasible[fr, fc]:
        return None, (fr, fc)
    allowed = feasible.copy()
    allowed[sr, sc] = False
    if not allowed[fr, fc]:
        return None, (fr, fc)
    return _bfs_placement(allowed, fr, fc), (fr, fc)


def _pick_farthest(dist: np.ndarray, default: tuple[int, int]) -> tuple[int, int]:
    best = -1
    tr, tc = default
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def _control_to_step(H: int, W: int, ctrl_r: int, ctrl_c: int) -> tuple[int, int]:
    hits: list[tuple[int, int]] = []
    if ctrl_r == 0:
        hits.append((1, 0))
    if ctrl_r == H - 1:
        hits.append((-1, 0))
    if ctrl_c == 0:
        hits.append((0, 1))
    if ctrl_c == W - 1:
        hits.append((0, -1))
    assert len(hits) == 1
    return hits[0]


def _simulate_direction(
    feasible: np.ndarray, sr: int, sc: int, dest_r: int, dest_c: int, step: tuple[int, int]
) -> tuple[int, int, bool, np.ndarray | None]:
    dist, start_pos = _branch_dist(feasible, sr, sc, step)
    if dist is None:
        return int(sr), int(sc), False, None
    if 0 <= dest_r < dist.shape[0] and 0 <= dest_c < dist.shape[1] and int(dist[dest_r, dest_c]) >= 0:
        return int(dest_r), int(dest_c), True, dist
    tr, tc = _pick_farthest(dist, start_pos)
    return int(tr), int(tc), False, dist


def _neighbor_steps(feasible: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for step in DIR4.values():
        rr, cc = r + step[0], c + step[1]
        if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and feasible[rr, cc]:
            out.append(step)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    ctrl_pts: list[tuple[int, int]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        if int(comp["color"]) != _CTRL or int(comp["area"]) != 1:
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        if r0 in (0, H - 1) or c0 in (0, W - 1):
            ctrl_pts.append((int(r0), int(c0)))
    assert len(ctrl_pts) == 1
    control_step = _control_to_step(H, W, ctrl_pts[0][0], ctrl_pts[0][1])

    g_work = g.copy()
    g_work[ctrl_pts[0][0], ctrl_pts[0][1]] = bg

    mover_candidates: list[dict] = []
    dest_candidates: list[dict] = []
    for comp in find_components(g_work, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        if int(comp["color"]) == _CTRL:
            continue
        mover_info = _arrow_square_info(comp)
        if mover_info is not None:
            mover_candidates.append(mover_info)
            continue
        square_info = _solid_square_info(g_work, comp)
        if square_info is not None:
            dest_candidates.append(square_info)

    pairs = [
        (m, d)
        for m in mover_candidates
        for d in dest_candidates
        if int(m["n"]) == int(d["n"]) and int(m["col"]) != int(d["col"])
    ]
    assert len(pairs) == 1
    mover, dest = pairs[0]

    counts = Counter(map(int, g_work.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) in (bg, _CTRL, int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g_work == int(river)) | (g_work == int(mover["col"])) | (g_work == int(dest["col"]))

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

    exits = _neighbor_steps(feasible, sr, sc)
    tip_step = DIR4[str(mover["dir"])]
    assert len(exits) == 2
    assert control_step in exits
    assert tip_step in exits

    tr, tc, _dest_ok, _dist = _simulate_direction(feasible, sr, sc, dr, dc, control_step)

    out = g.copy()
    for r, c in mover["coords"]:
        out[int(r), int(c)] = int(river)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover["col"])
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
            for dr, dc in DIR4.values():
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
                    for dr2, dc2 in DIR4.values():
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in DIR4.values():
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


def _render_channel(
    grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]
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
    grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str
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
    mask = grid == _RIVER
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


def _feasible_from_river(grid: np.ndarray, n: int) -> np.ndarray:
    H, W = grid.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool((grid[r : r + n, c : c + n] == _RIVER).all()):
                feasible[r, c] = True
    return feasible


def _choose_tip_cell(grid: np.ndarray, r0: int, c0: int, n: int, direction: str) -> tuple[int, int] | None:
    offsets = sorted(range(n), key=lambda k: (abs(k - (n - 1) / 2.0), k))
    for off in offsets:
        if direction == "U":
            rr, cc = r0 - 1, c0 + off
        elif direction == "D":
            rr, cc = r0 + n, c0 + off
        elif direction == "L":
            rr, cc = r0 + off, c0 - 1
        else:
            rr, cc = r0 + off, c0 + n
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) == _RIVER:
            return int(rr), int(cc)
    return None


def _paint_stub(grid: np.ndarray, sr: int, sc: int, n: int, step: tuple[int, int], length: int) -> None:
    dr, dc = step
    if dr == 1:
        grid[sr : sr + n + length, sc : sc + n] = _RIVER
    elif dr == -1:
        grid[sr - length : sr + n, sc : sc + n] = _RIVER
    elif dc == 1:
        grid[sr : sr + n, sc : sc + n + length] = _RIVER
    else:
        grid[sr : sr + n, sc - length : sc + n] = _RIVER


def _max_bg_stub_length(grid: np.ndarray, sr: int, sc: int, n: int, step: tuple[int, int]) -> int:
    H, W = grid.shape
    dr, dc = step
    if dr == 1:
        max_len = H - (sr + n)
        for length in range(max_len, 0, -1):
            if bool((grid[sr + n : sr + n + length, sc : sc + n] == _BG).all()):
                return int(length)
    elif dr == -1:
        max_len = sr
        for length in range(max_len, 0, -1):
            if bool((grid[sr - length : sr, sc : sc + n] == _BG).all()):
                return int(length)
    elif dc == 1:
        max_len = W - (sc + n)
        for length in range(max_len, 0, -1):
            if bool((grid[sr : sr + n, sc + n : sc + n + length] == _BG).all()):
                return int(length)
    else:
        max_len = sc
        for length in range(max_len, 0, -1):
            if bool((grid[sr : sr + n, sc - length : sc] == _BG).all()):
                return int(length)
    return 0


def _pick_control_cell(
    grid: np.ndarray, step: tuple[int, int], mover_r: int, mover_c: int, n: int
) -> tuple[int, int] | None:
    H, W = grid.shape
    pref_r = mover_r + n // 2
    pref_c = mover_c + n // 2

    if step == (1, 0):
        row = 0
        cols = sorted(range(1, W - 1), key=lambda c: abs(c - pref_c))
        for c in cols:
            if grid[row, c] != _BG:
                continue
            if row + 1 < H and grid[row + 1, c] != _BG:
                continue
            return int(row), int(c)
    elif step == (-1, 0):
        row = H - 1
        cols = sorted(range(1, W - 1), key=lambda c: abs(c - pref_c))
        for c in cols:
            if grid[row, c] != _BG:
                continue
            if row - 1 >= 0 and grid[row - 1, c] != _BG:
                continue
            return int(row), int(c)
    elif step == (0, 1):
        col = 0
        rows = sorted(range(1, H - 1), key=lambda r: abs(r - pref_r))
        for r in rows:
            if grid[r, col] != _BG:
                continue
            if col + 1 < W and grid[r, col + 1] != _BG:
                continue
            return int(r), int(col)
    else:
        col = W - 1
        rows = sorted(range(1, H - 1), key=lambda r: abs(r - pref_r))
        for r in rows:
            if grid[r, col] != _BG:
                continue
            if col - 1 >= 0 and grid[r, col - 1] != _BG:
                continue
            return int(r), int(col)
    return None


def _sample_object_colors() -> tuple[int, int]:
    mover_col, dest_col = random.sample(_OBJECT_COLORS, 2)
    return int(mover_col), int(dest_col)


def _gen_attempt() -> np.ndarray | None:
    main_reaches = random.random() < 0.5
    n = random.choice([2, 3, 4])

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
        if offsets_r[i + 1] - offsets_r[i] < n or offsets_c[i + 1] - offsets_c[i] < n:
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
        for direction in ["top", "bottom", "left", "right"]:
            if direction == "top":
                dist = pr
                if dist == 0:
                    candidates.append((0, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = H - (pr + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n)):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
                    candidates.append((pr, 0, dist))
            else:
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W)):
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
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    if not main_reaches:
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

    feasible_base = _feasible_from_river(grid, n)
    if not feasible_base[mover_r, mover_c]:
        return None

    main_dirs = _neighbor_steps(feasible_base, mover_r, mover_c)
    if len(main_dirs) != 1:
        return None
    main_step = main_dirs[0]

    main_tr, main_tc, main_dest_ok, _main_dist = _simulate_direction(
        feasible_base, mover_r, mover_c, dest_r, dest_c, main_step
    )
    if main_reaches and (main_tr, main_tc) != (dest_r, dest_c):
        return None
    if (not main_reaches) and main_dest_ok:
        return None

    decoy_candidates: list[tuple[int, tuple[int, int]]] = []
    for step in DIR4.values():
        if step == main_step:
            continue
        max_len = _max_bg_stub_length(grid, mover_r, mover_c, n, step)
        if max_len <= 0:
            continue
        lengths = {1, max_len, min(max_len, 2), min(max_len, 3)}
        for length in sorted(lengths):
            decoy_candidates.append((int(length), step))
    random.shuffle(decoy_candidates)

    feasible = None
    decoy_step = None
    decoy_target = None
    for length, step in decoy_candidates:
        trial = grid.copy()
        _paint_stub(trial, mover_r, mover_c, n, step, length)
        feasible_trial = _feasible_from_river(trial, n)
        if not feasible_trial[mover_r, mover_c]:
            continue
        exits = _neighbor_steps(feasible_trial, mover_r, mover_c)
        if len(exits) != 2 or main_step not in exits or step not in exits:
            continue

        trial_main = _simulate_direction(feasible_trial, mover_r, mover_c, dest_r, dest_c, main_step)
        if main_reaches and (trial_main[0], trial_main[1]) != (dest_r, dest_c):
            continue
        if (not main_reaches) and trial_main[2]:
            continue

        dtr, dtc, decoy_hits_dest, _ddist = _simulate_direction(feasible_trial, mover_r, mover_c, dest_r, dest_c, step)
        if decoy_hits_dest:
            continue
        if (dtr, dtc) == (mover_r, mover_c):
            continue
        if (dtr, dtc) == (trial_main[0], trial_main[1]):
            continue

        grid = trial
        feasible = feasible_trial
        decoy_step = step
        decoy_target = (int(dtr), int(dtc))
        main_tr, main_tc = int(trial_main[0]), int(trial_main[1])
        break

    if feasible is None or decoy_step is None or decoy_target is None:
        return None

    use_main = random.random() < 0.5
    control_step = main_step if use_main else decoy_step
    agree = random.random() < 0.5
    arrow_step = control_step if agree else (decoy_step if control_step == main_step else main_step)

    tip = _choose_tip_cell(grid, mover_r, mover_c, n, STEP_TO_NAME[arrow_step])
    if tip is None:
        return None

    tr, tc, _dest_ok, _dist = _simulate_direction(feasible, mover_r, mover_c, dest_r, dest_c, control_step)
    if use_main:
        if (tr, tc) != (main_tr, main_tc):
            return None
    else:
        if (tr, tc) != decoy_target:
            return None

    mover_col, dest_col = _sample_object_colors()
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_col
    grid[tip[0], tip[1]] = mover_col
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = dest_col

    ctrl_cell = _pick_control_cell(grid, control_step, mover_r, mover_c, n)
    if ctrl_cell is None:
        return None
    grid[ctrl_cell[0], ctrl_cell[1]] = _CTRL

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[tip[0], tip[1]] = _RIVER
    output[tr : tr + n, tc : tc + n] = mover_col

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")

# Combo variant of 332f06d7 (index 18)
# combo_spec: D8/1+D9/1
# Name: 按面积区分+大尺寸方块

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label as scipy_label

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components

# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Scene: one winding blue river on a green background, plus two solid squares
#      of different sizes. The smaller square is the mover, the larger square is
#      the destination region. Colors are not role-defining.
#    - Concrete trace:
#      original task: two same-sized 3x3 squares -> solver groups same size,
#      sorts by color id, moves the color-smaller square toward the other 3x3.
#      combo task: a 6x6 square and a 7x7 square on a 28x28 grid -> solver must
#      ignore color order, pick the 6x6 as mover, run BFS with a 6x6 footprint,
#      and accept any reachable 6x6 top-left fully contained inside the 7x7
#      target. If a choke narrows below 6, the mover stops at the farthest
#      reachable 6x6 placement instead of entering the target.
#    - Difference summary: D8 breaks the old color-order identity rule, while D9
#      makes the mover footprint large enough that chokepoints become visually
#      strong and materially affect reachability.
# b) Complexity calibration
#    - To keep the area cue obvious but controlled, sizes come from only three
#      pairs: (5,6), (6,7), (7,8). The mover size N is always 5..7, matching the
#      large-size requirement; the destination is exactly one size larger.
#    - Grid size stays in 25..30 and correlates with mover size, so large squares
#      occupy a meaningful fraction of the board without overfilling it.
#    - Object colors are sampled independently from a fixed palette, so examples
#      repeatedly show that size, not color id, determines roles.
# c) Implicit assumption audit of main()
#    - Solid-square detection still assumes monochrome filled squares. Generator
#      guarantees that both object components are perfect squares.
#    - Original code assumed at least two squares of the same size. That breaks
#      here, so role selection is rewritten to require one unique smallest square
#      and one unique largest square.
#    - Original target logic assumed a single landing top-left equal to the
#      destination square's top-left. That breaks because the target is larger;
#      solver now enumerates all mover-sized windows fully inside the target bbox.
#    - BFS feasibility still assumes movement is determined solely by the mover
#      footprint. This remains valid because the generator carves the river using
#      the mover size N, not the destination size.
#    - River-color inference still assumes there is exactly one non-background,
#      non-object color left. Generator keeps only bg, river, mover, destination.
# d) Placement legality
#    - Generator renders the river from mover-sized placement cells, so every
#      traversable position corresponds to an actual N x N legal window.
#    - Destination placement is constrained to touch a border and to overlap only
#      a simple terminal river strip; this avoids accidental overlap with other
#      river branches and keeps all objects inside bounds.
#    - Chokepoints are added before object colors are painted, and connectivity as
#      well as reachability are revalidated afterward.
# e) Visual observability
#    - Across samples, the stable cue is that the smaller square is the one that
#      moves, even when it has a numerically larger color id than the target.
#    - The larger target remains partially visible after a successful move, so the
#      "small square entering a larger square" relation is directly observable.
#    - Big mover sizes 5..7 make narrow gates visibly narrower than the mover.
# f) Visual cleanliness
#    - The target overlaps only the straight terminal river strip, not arbitrary
#      turns, so the final boards stay legible instead of having heavy region
#      overpainting.
#    - There are still only four colors per grid and exactly two non-river square
#      objects, keeping the scene sparse enough to parse by eye.
# g) Change inferability
#    - No invisible ordering rule is used: not color id, not component order.
#      Human-solvable evidence comes from repeated size differences and from the
#      large mover footprint interacting with visibly narrow chokepoints.
# 9) New solver assumptions and generator guarantees
#    - Solver assumes exactly one smallest and one largest solid square component.
#      Generator emits exactly two square objects with different sizes.
#    - Solver assumes the destination square is strictly larger than the mover and
#      solid, so every contained N x N window is a valid candidate landing area.
#      Generator enforces size pairs with m = n + 1 and solid fill.
#    - Solver assumes the mover's start top-left is itself feasible under the N x
#      N traversal rule. Generator verifies this before accepting a sample.


_BG = 3
_RIVER = 1
_OBJECT_COLORS = [0, 2, 4, 5, 6, 7, 8, 9]
_GRID_BY_N = {
    5: (25, 27),
    6: (27, 29),
    7: (29, 30),
}
_SIZE_PAIRS = [(5, 6), (6, 7), (7, 8)]


def _square_components(g: np.ndarray, bg: int) -> list[dict]:
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
    return squares


def _select_mover_dest(squares: list[dict]) -> tuple[dict, dict]:
    assert len(squares) >= 2
    sizes = sorted({int(s["n"]) for s in squares})
    assert len(sizes) >= 2
    mover_cands = [s for s in squares if int(s["n"]) == int(sizes[0])]
    dest_cands = [s for s in squares if int(s["n"]) == int(sizes[-1])]
    assert len(mover_cands) == 1 and len(dest_cands) == 1
    mover = mover_cands[0]
    dest = dest_cands[0]
    assert int(mover["n"]) < int(dest["n"])
    return mover, dest


def _river_color(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover_col), int(dest_col)):
            continue
        return int(col)
    raise AssertionError("river color not found")


def _feasible_placements(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return dist


def _target_positions(dest: dict, mover_n: int) -> list[tuple[int, int]]:
    m = int(dest["n"])
    r0 = int(dest["r0"])
    c0 = int(dest["c0"])
    gap = int(m - mover_n)
    assert gap >= 1
    pos = []
    for r in range(r0, r0 + gap + 1):
        for c in range(c0, c0 + gap + 1):
            pos.append((int(r), int(c)))
    return pos


def _landing_position(
    feasible: np.ndarray, dist: np.ndarray, sr: int, sc: int, dest: dict, mover_n: int
) -> tuple[int, int, bool]:
    reachable_targets = []
    for r, c in _target_positions(dest, mover_n):
        if 0 <= r < feasible.shape[0] and 0 <= c < feasible.shape[1] and bool(feasible[r, c]) and int(dist[r, c]) >= 0:
            reachable_targets.append((int(dist[r, c]), int(r), int(c)))
    if reachable_targets:
        reachable_targets.sort()
        _d, tr, tc = reachable_targets[0]
        return int(tr), int(tc), True

    best = -1
    tr, tc = int(sr), int(sc)
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return int(tr), int(tc), False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares = _square_components(g, bg)
    mover, dest = _select_mover_dest(squares)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    river = _river_color(g, bg, int(mover["col"]), int(dest["col"]))

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _feasible_placements(trav, n)
    h2, w2 = feasible.shape
    assert 0 <= sr < h2 and 0 <= sc < w2
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    tr, tc, _reached = _landing_position(feasible, dist, sr, sc, dest, n)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(120):
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


def _render_channel(
    grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]
) -> list[tuple[int, int]]:
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
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str) -> None:
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, max(1, n - 2))
    remove = int(n - narrow_w)
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        span = max(1, c_hi - c_lo)
        hold_cap = min(max(3, n + 1), max(1, span - 2 * remove))
        hold_len = random.randint(1, hold_cap)
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
        span = max(1, r_hi - r_lo)
        hold_cap = min(max(3, n + 1), max(1, span - 2 * remove))
        hold_len = random.randint(1, hold_cap)
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


def _river_connected(grid: np.ndarray) -> bool:
    mask = grid == _RIVER
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


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
        if direction == "top":
            dist = pr
            if dist == 0:
                candidates.append((0, pc, dist))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = H - (pr + n)
            if dist == 0:
                candidates.append((pr, pc, dist))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n)):
                candidates.append((H - n, pc, dist))
        elif direction == "left":
            dist = pc
            if dist == 0:
                candidates.append((pr, 0, dist))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
                candidates.append((pr, 0, dist))
        else:
            dist = W - (pc + n)
            if dist == 0:
                candidates.append((pr, pc, dist))
            elif all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W)):
                candidates.append((pr, W - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    new_r, new_c, _d = candidates[0]
    if new_r != pr:
        r_lo, r_hi = min(new_r, pr), max(new_r, pr) + n
        grid[r_lo:r_hi, pc : pc + n] = _RIVER
    if new_c != pc:
        c_lo, c_hi = min(new_c, pc), max(new_c, pc) + n
        grid[pr : pr + n, c_lo:c_hi] = _RIVER
    return int(new_r), int(new_c)


def _rectilinear_overlap_ok(grid: np.ndarray, dest_r: int, dest_c: int, m: int, n: int) -> bool:
    patch = grid[dest_r : dest_r + m, dest_c : dest_c + m]
    mask = patch != _BG
    if not bool(mask.any()):
        return False
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if int(mask.sum()) != h * w:
        return False
    return (h == n and w >= n) or (w == n and h >= n)


def _destination_candidates(anchor_r: int, anchor_c: int, H: int, W: int, n: int, m: int) -> list[tuple[int, int]]:
    gap = int(m - n)
    roles = [(0, 0), (0, gap), (gap, 0), (gap, gap)]
    out = []
    for off_r, off_c in roles:
        dest_r = int(anchor_r - off_r)
        dest_c = int(anchor_c - off_c)
        if not (0 <= dest_r <= H - m and 0 <= dest_c <= W - m):
            continue
        if not (dest_r == 0 or dest_c == 0 or dest_r + m == H or dest_c + m == W):
            continue
        out.append((dest_r, dest_c))
    return out


def _choose_dest_square(grid: np.ndarray, anchor_r: int, anchor_c: int, n: int, m: int) -> tuple[int, int] | None:
    H, W = grid.shape
    candidates = _destination_candidates(anchor_r, anchor_c, H, W, n, m)
    random.shuffle(candidates)
    good = []
    for dest_r, dest_c in candidates:
        if _rectilinear_overlap_ok(grid, dest_r, dest_c, m, n):
            good.append((dest_r, dest_c))
    if not good:
        return None
    good.sort(key=lambda rc: (abs(rc[0] - anchor_r) + abs(rc[1] - anchor_c), rc[0], rc[1]))
    return good[0]


def _pick_colors() -> tuple[int, int]:
    a, b = random.sample(_OBJECT_COLORS, 2)
    if random.random() < 0.5:
        return int(a), int(b)
    return int(b), int(a)


def _choose_grid_size(n: int) -> int:
    lo, hi = _GRID_BY_N[int(n)]
    return random.randint(lo, hi)


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n, m = random.choice(_SIZE_PAIRS)
    mover_col, dest_col = _pick_colors()

    G = _choose_grid_size(n)
    H = W = G

    max_coarse = max(3, (G - 2) // n)
    coarse_dim = random.randint(3, min(4, max_coarse))
    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < n or offsets_c[i + 1] - offsets_c[i] < n:
            return None

    edge_cells = set()
    for c in range(coarse_dim):
        edge_cells.add((0, c))
        edge_cells.add((coarse_dim - 1, c))
    for r in range(1, coarse_dim - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_dim - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_dim or len(set(cc for _, cc in walk)) < coarse_dim:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.68 or np.any(mask, axis=0).sum() / W < 0.68:
        return None

    mover_r, mover_c = pixel_pos[0]
    anchor_r, anchor_c = pixel_pos[-1]

    result = _extend_to_edge(grid, anchor_r, anchor_c, n)
    if result is None:
        return None
    anchor_r, anchor_c = result

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

    if not reached:
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

    dest_pos = _choose_dest_square(grid, anchor_r, anchor_c, n, m)
    if dest_pos is None:
        return None
    dest_r, dest_c = dest_pos

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_col
    grid[dest_r : dest_r + m, dest_c : dest_c + m] = dest_col

    squares = _square_components(grid, _BG)
    mover, dest = _select_mover_dest(squares)
    if int(mover["col"]) != int(mover_col) or int(dest["col"]) != int(dest_col):
        return None

    trav = (grid == _RIVER) | (grid == mover_col) | (grid == dest_col)
    feasible = _feasible_placements(trav, n)
    if not bool(feasible[mover_r, mover_c]):
        return None
    dist = _bfs_placement(feasible, mover_r, mover_c)
    tr, tc, dest_ok = _landing_position(feasible, dist, mover_r, mover_c, dest, n)
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    output = grid.copy()
    output[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    output[tr : tr + n, tc : tc + n] = mover_col
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")

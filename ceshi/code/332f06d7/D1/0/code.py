# Variant of 332f06d7
# Dimension: D1 — 通道导航方式 (path)
# Change: 沿蓝色水道走廊（corridor_following），方块在宽度>=n的通道内BFS搜索可行放置位置... → 重力流动导航

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D1 variant
# ----------------------------------
# yaml:
#   reference_task:
#     source_task: 28a6681f
#     status: "evaluation json exists, but no local solver source file was found in the allowed workspace"
#     consequence: "the gravity baseline below is designed from the variant brief plus visible ARC-style behavior, without reading forbidden code"
#   concretization:
#     objects:
#       - "background keeps dominating the corners"
#       - "blue river is still a single connected waterway rendered as n-pixel-wide shafts and shelves"
#       - "one solid n×n mover square starts away from the border"
#       - "one solid n×n destination square touches exactly one outer border, so target-vs-mover is visually recoverable"
#     gravity_rule:
#       - "state space still uses top-left placements of the n×n block"
#       - "the full n×n window must lie inside {river,mover,dest}"
#       - "from a placement, if the cell one row lower is feasible, gravity forces a downward move"
#       - "only when directly below is blocked may the block move left or right on the same row"
#       - "upward motion is never allowed, so connected river geometry alone no longer implies reachability"
#     traced_difference_example:
#       original: "if a connected side branch goes downward and then upward to the red square, 4-neighbor BFS can climb back up and reach red"
#       variant: "the block may descend into the branch base, but cannot take the upward segment, so it continues downward and settles at the deepest reachable shelf"
#       visual_effect: "the new samples show red chambers above a lower basin; humans can see the river is connected but the black square still cannot climb"
#   complexity_calibration:
#     - "the generator alternates shelves and drops so examples visibly require both falling and lateral flow"
#     - "about half the cases include an upward-only side branch to create a clear contrast with the original free-BFS behavior"
#     - "branching is limited to one upward chamber, keeping the rule inferable from 2-4 examples"
#   original_main_assumptions_audited:
#     - "sorting same-sized squares by color id was only safe because mover=0 and dest=2 were fixed; this variant removes that invisible dependency and instead uses border-touching geometry"
#     - "4-direction BFS on placements encoded free navigation; this is replaced by a directed gravity graph"
#     - "choosing the farthest BFS node no longer matches the physics; the new fallback is the deepest reachable placement under gravity, with distance only as a tie-break"
#   placement_legality:
#     - "generator constructs placement cells first, then paints each placement as an n×n river window"
#     - "all path cells are contiguous by unit steps, so river connectivity is guaranteed by construction"
#     - "mover is always interior; destination is always border-touching but never in a corner"
#     - "the oracle recomputes feasibility on the rendered pixels and rejects any sample whose actual dynamics differ from the intended one"
#   observability_and_cleanliness:
#     - "unreachable cases use a single clean upward branch from a shelf to the border red square"
#     - "reachable cases end on the bottom shelf, so the output reads as a gravity-driven descent rather than arbitrary corridor traversal"
#     - "no color remapping or color-id-based visible behavior is used"


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _touches_edge(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _detect_squares(g: np.ndarray, bg: int) -> list[dict]:
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


def _choose_mover_and_dest(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
    squares = _detect_squares(g, bg)
    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    options = []
    for nside, group in by_n.items():
        edge = [sq for sq in group if _touches_edge(int(sq["r0"]), int(sq["c0"]), int(sq["n"]), H, W)]
        inner = [sq for sq in group if sq not in edge]
        if edge and inner:
            options.append((min(len(edge), len(inner)), len(group), int(nside), inner, edge))
    assert options

    _pairs, _count, _nside, inner, edge = max(options, key=lambda x: (x[0], x[1], x[2]))
    mover = min(inner, key=lambda sq: (sq["r0"], sq["c0"]))
    dest = min(edge, key=lambda sq: (sq["r0"], sq["c0"]))
    return mover, dest


def _river_color(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if col == bg or col == mover_col or col == dest_col:
            continue
        return int(col)
    raise AssertionError("river color not found")


def _feasible_windows(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _gravity_reach(feasible: np.ndarray, sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    seen = np.zeros((h2, w2), dtype=bool)
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    dist[sr, sc] = 0

    while q:
        r, c = q.popleft()
        nexts: list[tuple[int, int]] = []
        if r + 1 < h2 and bool(feasible[r + 1, c]):
            nexts.append((r + 1, c))
        else:
            if c - 1 >= 0 and bool(feasible[r, c - 1]):
                nexts.append((r, c - 1))
            if c + 1 < w2 and bool(feasible[r, c + 1]):
                nexts.append((r, c + 1))
        for rr, cc in nexts:
            if not bool(seen[rr, cc]):
                seen[rr, cc] = True
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))
    return seen, dist


def _gravity_target(feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    seen, dist = _gravity_reach(feasible, sr, sc)
    if 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and bool(seen[dr, dc]):
        return int(dr), int(dc), seen, dist

    best = (int(sr), int(sc))
    best_key = (int(sr), int(dist[sr, sc]), -abs(int(sc) - int(dc)))
    for r in range(feasible.shape[0]):
        for c in range(feasible.shape[1]):
            if not bool(seen[r, c]):
                continue
            key = (int(r), int(dist[r, c]), -abs(int(c) - int(dc)))
            if key > best_key:
                best_key = key
                best = (int(r), int(c))
    return best[0], best[1], seen, dist


def _undirected_reach(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    seen = np.zeros((h2, w2), dtype=bool)
    q = deque([(sr, sc)])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and not bool(seen[rr, cc]):
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let a solid square drift through a blue waterway under gravity to reach a same-sized target square.

    Input structure:
    - The corner-majority color is background, while one blue connected waterway forms shafts and shelves.
    - Two same-sized solid squares sit on the waterway: the interior one is the mover, and the border-touching one is the destination.

    Rule:
    - The mover may only occupy placements where its full NxN footprint lies inside water, itself, or the destination.
    - Gravity is one-way: if the next row down is feasible, the square must fall; only when blocked below may it slide left or right.
    - If the destination placement is reachable under those gravity-constrained moves, the mover ends there; otherwise it settles at the deepest reachable placement.
    - The mover's starting footprint turns back into river color.

    Key insight:
    - A connected channel is not sufficient anymore: any route that would require climbing back upward is impossible, even though ordinary BFS would still see connectivity.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _choose_mover_and_dest(g, bg)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    river = _river_color(g, bg, int(mover["col"]), int(dest["col"]))

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _feasible_windows(trav, n)
    assert bool(feasible[sr, sc])

    tr, tc, _seen, _dist = _gravity_target(feasible, sr, sc, dr, dc)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    return out


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], n: int) -> None:
    for r, c in cells:
        grid[r : r + n, c : c + n] = _RIVER


def _compute_output(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    bg = int(Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0])
    mover, dest = _choose_mover_and_dest(grid, bg)
    n = int(mover["n"])
    river = _river_color(grid, bg, int(mover["col"]), int(dest["col"]))
    trav = (grid == river) | (grid == int(mover["col"])) | (grid == int(dest["col"]))
    feasible = _feasible_windows(trav, n)
    tr, tc, seen, dist = _gravity_target(feasible, int(mover["r0"]), int(mover["c0"]), int(dest["r0"]), int(dest["c0"]))
    out = grid.copy()
    out[int(mover["r0"]) : int(mover["r0"]) + n, int(mover["c0"]) : int(mover["c0"]) + n] = river
    out[tr : tr + n, tc : tc + n] = int(mover["col"])
    old_seen = _undirected_reach(feasible, int(mover["r0"]), int(mover["c0"]))
    info = {
        "n": n,
        "mover": (int(mover["r0"]), int(mover["c0"])),
        "dest": (int(dest["r0"]), int(dest["c0"])),
        "target": (int(tr), int(tc)),
        "gravity_reaches_dest": bool(seen[int(dest["r0"]), int(dest["c0"])]),
        "undirected_reaches_dest": bool(old_seen[int(dest["r0"]), int(dest["c0"])]),
        "deepest_row": int(max((r for r, c in zip(*np.where(seen)) if seen[r, c]), default=int(mover["r0"]))),
        "max_dist": int(dist[tr, tc]),
    }
    return out, info


def _build_shelf_path(h2: int, w2: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    assert h2 >= 7 and w2 >= 6
    sr = random.randint(1, min(2, h2 - 3))
    start_left = random.random() < 0.5
    if start_left:
        sc = random.randint(1, min(2, w2 - 3))
        side = 1
    else:
        sc = random.randint(max(1, w2 - 3), w2 - 2)
        side = -1

    num_shelves = random.randint(2, min(4, h2 - sr))
    rows = [sr]
    cur_r = sr
    for remain in range(num_shelves - 1, 0, -1):
        min_next = cur_r + 2
        max_next = h2 - 1 - 2 * (remain - 1)
        if min_next > max_next:
            raise AssertionError("not enough vertical room for shelves")
        cur_r = random.randint(min_next, max_next)
        rows.append(cur_r)
    rows[-1] = h2 - 1

    left_limit = max(1, w2 // 3)
    right_start = min(w2 - 2, max(left_limit + 2, w2 - 1 - w2 // 3))
    if right_start > w2 - 2:
        right_start = w2 - 2

    cells: list[tuple[int, int]] = [(sr, sc)]
    shelf_cells: list[tuple[int, int]] = []
    cur_c = sc
    for i, row in enumerate(rows):
        if i == 0:
            row = sr
        if i > 0:
            for rr in range(rows[i - 1] + 1, row + 1):
                cells.append((rr, cur_c))
        if side > 0:
            lo = max(cur_c + 2, right_start)
            hi = w2 - 2
        else:
            lo = 1
            hi = min(cur_c - 2, left_limit)
        if i == len(rows) - 1:
            if side > 0:
                lo = max(cur_c, right_start - 1)
                hi = w2 - 2
            else:
                lo = 1
                hi = min(cur_c, left_limit + 1)
        if lo > hi:
            target_c = w2 - 2 if side > 0 else 1
        else:
            target_c = random.randint(lo, hi)
        step = 1 if target_c >= cur_c else -1
        for cc in range(cur_c + step, target_c + step, step):
            cells.append((row, cc))
            shelf_cells.append((row, cc))
        if target_c == cur_c:
            shelf_cells.append((row, cur_c))
        cur_c = target_c
        side *= -1
    return cells, shelf_cells, (sr, sc)


def _grid_from_cells(H: int, W: int, n: int, cells: set[tuple[int, int]], mover: tuple[int, int], dest: tuple[int, int]) -> np.ndarray:
    grid = np.full((H, W), _BG, dtype=int)
    _paint_cells(grid, cells, n)
    mr, mc = mover
    dr, dc = dest
    grid[mr : mr + n, mc : mc + n] = _MOVER
    grid[dr : dr + n, dc : dc + n] = _DEST
    return grid


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4])
    min_size = max(10, 4 * n + 4)
    if min_size > 20:
        return None
    H = W = random.randint(min_size, 20)
    h2, w2 = H - n + 1, W - n + 1
    if h2 < 7 or w2 < 6:
        return None

    try:
        main_cells, shelf_cells, mover = _build_shelf_path(h2, w2)
    except AssertionError:
        return None
    cells = set(main_cells)
    mr, mc = mover

    if reached:
        dest = main_cells[-1]
    else:
        branch_candidates = [cell for cell in shelf_cells if 2 <= cell[0] <= h2 - 3 and 1 <= cell[1] <= w2 - 2]
        if not branch_candidates:
            return None
        junction = random.choice(branch_candidates)
        br, bc = junction
        for rr in range(0, br):
            cells.add((rr, bc))
        dest = (0, bc)

    if dest == mover or _touches_edge(mr, mc, n, H, W) or not _touches_edge(dest[0], dest[1], n, H, W):
        return None
    if dest[1] in (0, w2 - 1) and dest[0] in (0, h2 - 1):
        return None

    grid = _grid_from_cells(H, W, n, cells, mover, dest)
    river_area = int((grid == _RIVER).sum() + (grid == _MOVER).sum() + (grid == _DEST).sum())
    frac = river_area / (H * W)
    if not (0.18 <= frac <= 0.62):
        return None

    try:
        out, info = _compute_output(grid)
    except Exception:
        return None

    if reached != bool(info["gravity_reaches_dest"]):
        return None
    if not reached and not bool(info["undirected_reaches_dest"]):
        return None
    if tuple(info["mover"]) != mover or tuple(info["dest"]) != dest:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")

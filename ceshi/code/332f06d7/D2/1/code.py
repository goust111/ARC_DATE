# Variant of 332f06d7
# Dimension: D2 — 通道宽度约束 (path)
# Change: 通道宽度>=n时可通过，<n时形成阻塞点（chokepoint），方块被卡住... → 渐变宽度约束

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the D2 variant
# ==================================
# source_task: None
#
# concrete_rule:
# - The black square starts larger than the red square.
# - It slides one cell at a time through the blue corridor.
# - After every move, its side length shrinks by 1 pixel until it matches the
#   red square's side length; after that it stays at that minimum size.
# - A placement is legal only if the square of its current size fits entirely
#   inside {river ∪ mover ∪ destination}.
# - Success means reaching the red square's top-left while having exactly the
#   red square's side length. Otherwise the square stops at the farthest
#   reachable state. The output always paints the final black square at its
#   current size, so the output mover is usually smaller than the input mover.
#
# behavior_trace:
# - Example with start size 4 and target size 2:
#   step 0 at (r,c): size 4
#   step 1 at (r,c+1): size 3
#   step 2 at (r,c+2): size 2
#   later steps: still size 2
# - In the original task, a 4x4 mover could never enter a 3-wide or 2-wide
#   section. In this variant, the same path becomes traversable after enough
#   motion because the mover has already shrunk before it reaches the narrow
#   section.
#
# complexity_calibration:
# - Only one shrink schedule is used: shrink by 1 per move until the red size.
# - Start size n0 is restricted to {3,4}; target size m is {1,2} with m<n0.
#   This keeps the effect visible without introducing many cases.
# - The destination is identified visually by touching the border; the mover
#   is visibly inset from the border. This replaces the original color-id sort,
#   which is not allowed as an invisible cue.
#
# old_assumptions_that_break:
# - "mover and destination have the same size" is no longer true.
# - "choose the pair by most common size" is no longer valid.
# - "sort by color to decide mover vs destination" is no longer valid.
# - "all BFS states use a fixed NxN footprint" is no longer valid.
# - "output mover size equals input mover size" is no longer valid.
#
# placement_legality:
# - The generator builds a long orthogonal polyline in top-left coordinate
#   space, then renders the river as the union of the mover's size-dependent
#   squares along that path.
# - Parallel bands are separated by more than the maximum start size so
#   unrelated parts of the river do not merge into accidental shortcuts.
# - The destination square always touches exactly one border in the canonical
#   construction; the mover is near the opposite side but does not touch any
#   border.
# - Stuck cases carve one late gate to width m-1 on an already size-m section,
#   so shrinking helps earlier but still cannot pass the final gate.
#
# visual_observability:
# - The input already shows a larger black square, a smaller red square, and a
#   corridor that becomes narrower away from the start. The output then shows
#   the black square arriving smaller than it began, or stopping before a gate
#   that is thinner than the red size.
# - No extra marker colors are introduced. The only semantic cues are visible:
#   size difference, border contact, and corridor width.
#
# cleanliness_and_color_reasoning:
# - Colors keep their original roles: background, river, mover, destination.
# - There is no hidden color mapping. The rule is entirely geometric.


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2


def _touches_edge(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _detect_squares(g: np.ndarray, bg: int) -> tuple[dict, dict]:
    H, W = g.shape
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        r0, c0, r1, c1 = map(int, b)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == col).all()):
            continue
        n = int(bbox_h(b))
        squares.append({"col": col, "r0": r0, "c0": c0, "n": n})

    assert len(squares) >= 2

    touching = [s for s in squares if _touches_edge(int(s["r0"]), int(s["c0"]), int(s["n"]), H, W)]
    inset = [s for s in squares if not _touches_edge(int(s["r0"]), int(s["c0"]), int(s["n"]), H, W)]
    assert touching and inset

    dest = max(touching, key=lambda s: int(s["n"]))
    mover = max(inset, key=lambda s: int(s["n"]))
    return mover, dest


def _infer_river(g: np.ndarray, bg: int, mover_col: int, dest_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        col = int(col)
        if col == bg or col == mover_col or col == dest_col:
            continue
        return col
    raise AssertionError("river color not found")


def _can_place(trav: np.ndarray, r: int, c: int, n: int) -> bool:
    H, W = trav.shape
    if n <= 0 or r < 0 or c < 0 or r + n > H or c + n > W:
        return False
    return bool(trav[r : r + n, c : c + n].all())


def _solve_variant(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _detect_squares(g, bg)
    river = _infer_river(g, bg, int(mover["col"]), int(dest["col"]))

    sr, sc, n0 = int(mover["r0"]), int(mover["c0"]), int(mover["n"])
    dr, dc, m = int(dest["r0"]), int(dest["c0"]), int(dest["n"])
    assert n0 > m >= 1

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    assert _can_place(trav, sr, sc, n0)
    assert _can_place(trav, dr, dc, m)

    dist = -np.ones((n0 + 1, H, W), dtype=int)
    q = deque([(sr, sc, n0)])
    dist[n0, sr, sc] = 0
    best_state = (sr, sc, n0)
    best_dist = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        r, c, s = q.popleft()
        cur_d = int(dist[s, r, c])
        if cur_d > best_dist:
            best_dist = cur_d
            best_state = (r, c, s)
        ns = max(m, s - 1)
        for dr2, dc2 in dirs:
            rr = int(r + dr2)
            cc = int(c + dc2)
            if not _can_place(trav, rr, cc, ns):
                continue
            if int(dist[ns, rr, cc]) != -1:
                continue
            dist[ns, rr, cc] = cur_d + 1
            q.append((rr, cc, ns))

    reached = int(dist[m, dr, dc]) >= 0
    if reached:
        tr, tc, ts = dr, dc, m
    else:
        tr, tc, ts = best_state

    out = g.copy()
    out[sr : sr + n0, sc : sc + n0] = river
    out[tr : tr + ts, tc : tc + ts] = int(mover["col"])

    info = {
        "bg": bg,
        "river": river,
        "mover_color": int(mover["col"]),
        "dest_color": int(dest["col"]),
        "start": (sr, sc, n0),
        "dest": (dr, dc, m),
        "final": (tr, tc, ts),
        "reached": reached,
        "max_steps": best_dist if not reached else int(dist[m, dr, dc]),
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a shrinking square through a winding channel toward a smaller border target.

    Input structure:
    - The corner-majority background surrounds one blue orthogonal river.
    - A larger solid black square starts on the river away from the border.
    - A smaller solid red square lies on the same river and touches the border.

    Rule:
    - The black square moves one cell at a time through traversable blue/red/black cells.
    - After each move its side length drops by 1, until it matches the red square's size.
    - Each new placement must fit completely inside the traversable region at that current size.
    - If it can arrive at the red square with that smaller size, it replaces the red square; otherwise it stops at the farthest reachable state, and its starting area turns blue.

    Key insight:
    - Late corridor sections can be narrower than the starting square because the mover may have already shrunk enough by the time it reaches them.
    """
    out, _info = _solve_variant(np.array(grid, dtype=int))
    return out


def _enumerate_polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0 + step, c1 + step, step):
                path.append((r0, c))
        else:
            step = 1 if r1 > r0 else -1
            for r in range(r0 + step, r1 + step, step):
                path.append((r, c0))
    return path


def _render_gradient_river(grid: np.ndarray, path: list[tuple[int, int]], n0: int, m: int) -> list[int]:
    sizes: list[int] = []
    for idx, (r, c) in enumerate(path):
        s = max(m, n0 - idx)
        sizes.append(s)
        grid[r : r + s, c : c + s] = _RIVER
    return sizes


def _build_canonical_path(H: int, W: int, n0: int, m: int) -> list[tuple[int, int]] | None:
    row_low = H - n0 - 2
    row_mid_min = n0 + 1
    row_mid_max = row_low - (n0 + 1)
    if row_mid_max < row_mid_min:
        return None
    row_mid = random.randint(row_mid_min, row_mid_max)

    col_left = 1
    col_mid_max = max(2, W // 3)
    col_mid = random.randint(2, col_mid_max)
    turn_right = W - n0 - 4
    dest_c = W - m - 2
    if turn_right - col_mid < n0 + 2:
        return None
    if dest_c - col_mid < max(5, n0 + 1):
        return None

    points = [
        (row_low, col_left),
        (row_low, turn_right),
        (row_mid, turn_right),
        (row_mid, col_mid),
        (0, col_mid),
        (0, dest_c),
    ]
    return _enumerate_polyline(points)


def _carve_gate(grid: np.ndarray, path: list[tuple[int, int]], sizes: list[int], gate_width: int) -> bool:
    m = sizes[-1]
    if gate_width >= m:
        return False

    for i in range(len(path) - 8, 2, -1):
        if sizes[i] != m or sizes[i + 1] != m:
            continue
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        if r0 == r1:
            run_end = i + 1
            while run_end + 1 < len(path) and path[run_end + 1][0] == r0 and sizes[run_end + 1] == m:
                run_end += 1
            if run_end - i < 4:
                continue
            c_lo = min(path[i][1], path[run_end][1]) + 1
            c_hi = max(path[i][1], path[run_end][1]) + m - 1
            if c_hi - c_lo < 2:
                continue
            side = random.randint(0, 1)
            if side == 0:
                grid[r0 : r0 + (m - gate_width), c_lo : c_hi + 1] = _BG
            else:
                grid[r0 + gate_width : r0 + m, c_lo : c_hi + 1] = _BG
            return True
        if c0 == c1:
            run_end = i + 1
            while run_end + 1 < len(path) and path[run_end + 1][1] == c0 and sizes[run_end + 1] == m:
                run_end += 1
            if run_end - i < 4:
                continue
            r_lo = min(path[i][0], path[run_end][0]) + 1
            r_hi = max(path[i][0], path[run_end][0]) + m - 1
            if r_hi - r_lo < 2:
                continue
            side = random.randint(0, 1)
            if side == 0:
                grid[r_lo : r_hi + 1, c0 : c0 + (m - gate_width)] = _BG
            else:
                grid[r_lo : r_hi + 1, c0 + gate_width : c0 + m] = _BG
            return True
    return False


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    k = random.randint(0, 3)
    out = np.rot90(grid, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _gen_attempt() -> np.ndarray | None:
    reachable = random.random() < 0.55
    n0 = random.choice([3, 4])
    if reachable:
        m = random.choice([1, 2])
    else:
        m = 2
    if m >= n0:
        return None

    Gmin = max(14, 3 * n0 + 4)
    if Gmin > 20:
        return None
    G = random.randint(Gmin, 20)
    H = W = G

    path = _build_canonical_path(H, W, n0, m)
    if path is None or len(path) < (n0 - m + 6):
        return None

    grid = np.full((H, W), _BG, dtype=int)
    sizes = _render_gradient_river(grid, path, n0, m)

    if not reachable:
        if not _carve_gate(grid, path, sizes, m - 1):
            return None

    sr, sc = path[0]
    dr, dc = path[-1]
    grid[sr : sr + n0, sc : sc + n0] = _MOVER
    grid[dr : dr + m, dc : dc + m] = _DEST

    grid = _apply_random_symmetry(grid)

    try:
        expected, info = _solve_variant(grid)
    except Exception:
        return None

    if info["start"][2] <= info["dest"][2]:
        return None
    if info["final"][2] > info["start"][2]:
        return None
    if info["reached"] != reachable:
        return None
    if info["final"][2] != info["dest"][2] and info["reached"]:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    # Ensure the new mechanic is actually used: the start is larger than the
    # destination, so the river must contain a later section narrower than the
    # initial mover but at least as wide as the destination.
    start_n = int(info["start"][2])
    dest_n = int(info["dest"][2])
    has_narrow_section = False
    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    for r in range(H):
        run = 0
        for c in range(W):
            run = run + 1 if bool(trav[r, c]) else 0
            if dest_n <= run < start_n:
                has_narrow_section = True
                break
        if has_narrow_section:
            break
    if not has_narrow_section:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 400 attempts")

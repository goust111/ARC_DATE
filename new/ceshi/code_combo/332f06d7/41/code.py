# Combo variant of 332f06d7 (index 41)
# combo_spec: D1/0+D2/3+D16/0
# Name: 直线射线+动态障碍+显式障碍

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    _cand = _parent / "solvers" / "arc_utils.py"
    if _cand.exists():
        sys.path.append(str(_cand.parent))
        break

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for combo D1/0 + D2/3 + D16/0
==============================================

a) Concrete behavior and small trace
------------------------------------
- Keep the original visible object vocabulary:
  - background = green 3
  - river / track = blue 1
  - mover = black solid n x n square
  - destination = red solid n x n square of the same size
  - explicit obstacles = teal 8
- The fused task removes the winding-BFS motion model. The mover and the destination are aligned
  on one axis, so the intended movement is a single horizontal or vertical ray.
- The crucial obstacle change is twofold:
  1. the blocker is no longer a green geometric pinch; it is an explicit teal bar on the blue
     track,
  2. there are additional scattered teal bars on side branches / downstream tail, so the input
     is not just "one wall in a line".
- Small blocked trace:
  - Example: n=2, mover top-left (3,1), destination top-left (3,12), main river horizontal.
  - A teal bar sits at pixel column 8 across the 2 river rows. That makes top-left (3,7)
    the last valid straight placement.
  - A side branch starts at top-left column 5 and goes downward for 4 placements.
  - Old 332f06d7 logic: BFS over feasible placements would happily turn into that branch and pick
    its tip as the farthest reachable point.
  - Fused logic here: infer direction from black to red, test only (3,2), (3,3), ..., and stop
    at (3,7) because the next straight step would overlap the teal obstacle. The branch is ignored.
- Difference from the original task:
  - original blocker source: river narrows below width n,
  - fused blocker source: river stays straight and readable, but explicit obstacle color occupies
    the forbidden cells.

b) Complexity calibration
-------------------------
- A plain line with one obstacle would be too weak: the original BFS solver would often produce
  the same output, so the D1 change would not really matter.
- To keep the fusion visible, each generated sample contains:
  - one straight main strip aimed from black toward red,
  - one mandatory side branch before the first blocking obstacle, long enough that BFS would
    prefer its tip over the straight stop,
  - one or more extra teal obstacles elsewhere on the river geometry.
- Freedom is intentionally small:
  - orientation: horizontal or vertical,
  - direction: positive or negative,
  - square size: n in {2,3,4},
  - 1 to 2 side branches,
  - 2 to 4 teal bars total.
- That is enough variation for ARC while keeping the rule legible from a few examples.

c) Hidden-assumption audit of the original main()
-------------------------------------------------
- Original "sort same-sized squares by color" is replaced. The fused solver uses a visible rule:
  the destination touches a border, the mover does not.
- Original "most common remaining color is river" still holds, but now only because the generator
  guarantees blue area > teal obstacle area. The solver therefore treats every remaining color
  after the largest one as obstacle.
- Original feasible-window BFS is no longer the motion rule:
  - it assumes turning is allowed,
  - that is false after D1/0.
- Original fallback "pick farthest BFS position" is invalid:
  - side branches would attract the mover,
  - the fused rule must ignore them and stay on the black->red ray.
- New solver assumptions introduced here:
  - exactly one repeated-size pair of salient solid squares exists,
  - among that pair, exactly one square touches an edge,
  - mover and destination align horizontally or vertically in top-left placement coordinates,
  - blue is the largest non-(bg/squares) color and any extra remaining color is an obstacle,
  - mover's start position is a valid n x n placement on the passable mask.
- The generator explicitly enforces all of these.

d) Placement legality
---------------------
- Generation is analytical, not search-heavy:
  - pick the straight main strip first,
  - choose the stop position on that strip,
  - place the first teal bar exactly one step beyond the desired stop,
  - add a longer side branch before that blocker,
  - optionally add a second branch and extra teal bars farther away.
- Every river segment is axis-aligned and painted before squares are placed.
- Teal bars are always 1-pixel thick lines crossing an n-wide strip, so they block n x n windows
  but never masquerade as same-sized solid squares.
- Branches and extra barriers are only added when there is enough space, so they stay in-bounds
  and do not overlap the mover or destination.

e) Visual observability
-----------------------
- From a few examples, a human can see:
  - black and red are same-sized squares aligned on one straight corridor,
  - teal blocks are explicit obstacles inside the blue track,
  - extra side branches exist but the output still stays on the straight line,
  - the black square stops just before the first teal blocker on that ray.
- No invisible ordering is needed to infer the rule. Edge-touching red gives the motion target,
  and the first teal bar on that axis explains the stop.

f) Visual cleanliness
---------------------
- Only five colors are used.
- Blue geometry is rectilinear and sparse: one main strip, one required branch, optional extra
  branch, and a small number of teal bars.
- Obstacles never stack on top of each other, and they stay embedded inside blue strips rather
  than scattered arbitrarily across the background.

g) Change inferability
----------------------
- The only output recoloring is still simple and structural:
  - the black start square becomes blue,
  - black reappears at the terminal straight-ray placement,
  - red stays visible because the generated samples are deliberately blocked before the goal.
- The important inference is geometric, not numeric: "follow the straight shot toward red until the
  first explicit obstacle on that shot."

h) Anti-degeneration metric used after implementation
-----------------------------------------------------
- `branch_divergence`: compare the fused straight-ray stop with the old BFS-style stop computed on
  the same obstacle-aware traversable mask. The generator requires them to differ.
- `obstacle_effect`: compare the fused stop with the straight-ray stop obtained when teal is treated
  as passable. The generator requires them to differ too.
- If either metric is false, the sample has collapsed into an uninteresting subcase and is rejected.

i) Post-edit audit of new main() assumptions
--------------------------------------------
- `_pick_roles(...)` assumes one repeated-size pair has one edge-touching and one interior member.
  The generator creates exactly that pattern.
- `_ray_direction(...)` assumes axis alignment. The generator always places red on the same row or
  column as black in placement coordinates.
- `_scene_info(...)` assumes at least two non-excluded colors remain and blue dominates them. The
  generator always paints blue first and keeps teal area smaller.
- `_ray_terminal(...)` assumes the start window is feasible. The generator validates this before
  returning a sample.
"""


_BG, _RIVER, _MOVER, _DEST, _OBST = 3, 1, 0, 2, 8
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _solid_squares(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})
    return squares


def _touches_edge(square: dict, H: int, W: int) -> bool:
    r0, c0, n = int(square["r0"]), int(square["c0"]), int(square["n"])
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _edge_gap(square: dict, H: int, W: int) -> int:
    r0, c0, n = int(square["r0"]), int(square["c0"]), int(square["n"])
    return min(r0, c0, H - (r0 + n), W - (c0 + n))


def _pick_roles(squares: list[dict], H: int, W: int) -> tuple[dict, dict]:
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    sizes = [n for n, group in by_n.items() if len(group) >= 2]
    assert sizes
    nside = max(sizes, key=lambda n: (len(by_n[n]), n))
    cand = list(by_n[int(nside)])

    edge_touching = [sq for sq in cand if _touches_edge(sq, H, W)]
    interior = [sq for sq in cand if not _touches_edge(sq, H, W)]
    assert edge_touching and interior

    dest = min(edge_touching, key=lambda sq: (_edge_gap(sq, H, W), int(sq["r0"]), int(sq["c0"])))
    mover = min(interior, key=lambda sq: (_edge_gap(sq, H, W), int(sq["r0"]), int(sq["c0"])))
    return mover, dest


def _remaining_colors(g: np.ndarray, excluded: set[int]) -> list[tuple[int, int]]:
    counts = Counter(map(int, g.flatten()))
    items = [(int(col), int(ct)) for col, ct in counts.items() if int(col) not in excluded]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items


def _scene_info(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dest = _pick_roles(_solid_squares(g, bg), H, W)
    rem = _remaining_colors(g, {bg, int(mover["col"]), int(dest["col"])})
    assert len(rem) >= 2

    river = int(rem[0][0])
    obstacle_colors = {int(col) for col, _ in rem[1:]}
    assert obstacle_colors

    return {
        "grid": g,
        "H": int(H),
        "W": int(W),
        "bg": int(bg),
        "mover": mover,
        "dest": dest,
        "river": int(river),
        "obstacles": obstacle_colors,
        "n": int(mover["n"]),
        "sr": int(mover["r0"]),
        "sc": int(mover["c0"]),
        "dr": int(dest["r0"]),
        "dc": int(dest["c0"]),
    }


def _trav_mask(info: dict, *, obstacle_passable: bool = False) -> np.ndarray:
    g = info["grid"]
    passable = {int(info["river"]), int(info["mover"]["col"]), int(info["dest"]["col"])}
    if obstacle_passable:
        passable |= {int(col) for col in info["obstacles"]}
    return np.isin(g, list(passable))


def _window_ok(trav: np.ndarray, r: int, c: int, n: int) -> bool:
    H, W = trav.shape
    if not (0 <= int(r) <= int(H - n) and 0 <= int(c) <= int(W - n)):
        return False
    return bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all())


def _ray_direction(sr: int, sc: int, dr: int, dc: int) -> tuple[int, int]:
    if sr == dr and sc != dc:
        return 0, 1 if dc > sc else -1
    if sc == dc and sr != dr:
        return 1 if dr > sr else -1, 0
    raise AssertionError("mover and destination must align on exactly one axis")


def _ray_terminal(trav: np.ndarray, sr: int, sc: int, dr: int, dc: int, n: int) -> tuple[int, int]:
    assert _window_ok(trav, sr, sc, n)
    step_r, step_c = _ray_direction(sr, sc, dr, dc)

    tr, tc = int(sr), int(sc)
    while True:
        nr = int(tr + step_r)
        nc = int(tc + step_c)
        if not _window_ok(trav, nr, nc, n):
            break
        tr, tc = int(nr), int(nc)
        if tr == int(dr) and tc == int(dc):
            break
    return int(tr), int(tc)


def _feasible_mask(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_terminal(trav: np.ndarray, sr: int, sc: int, dr: int, dc: int, n: int) -> tuple[int, int]:
    feasible = _feasible_mask(trav, n)
    assert feasible[int(sr), int(sc)]

    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr2, dc2 in _DIR4:
            rr, cc = r + dr2, c + dc2
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((rr, cc))

    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0:
        return int(dr), int(dc)

    best = -1
    tr, tc = int(sr), int(sc)
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return int(tr), int(tc)


def _behavior_metrics(grid: np.ndarray) -> dict:
    info = _scene_info(grid)
    sr, sc = int(info["sr"]), int(info["sc"])
    dr, dc = int(info["dr"]), int(info["dc"])
    n = int(info["n"])

    trav = _trav_mask(info, obstacle_passable=False)
    trav_open = _trav_mask(info, obstacle_passable=True)

    ray_stop = _ray_terminal(trav, sr, sc, dr, dc, n)
    bfs_stop = _bfs_terminal(trav, sr, sc, dr, dc, n)
    open_ray_stop = _ray_terminal(trav_open, sr, sc, dr, dc, n)
    return {
        "ray_stop": ray_stop,
        "bfs_stop": bfs_stop,
        "open_ray_stop": open_ray_stop,
        "branch_divergence": ray_stop != bfs_stop,
        "obstacle_effect": ray_stop != open_ray_stop,
        "obstacle_pixels": int((grid == _OBST).sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - detect the black mover and the red destination by visible square geometry,
    - infer the unique straight ray from black toward red,
    - treat explicit non-blue obstacle colors as solid blockers,
    - slide only along that ray until the next n x n placement would hit a blocker,
    - repaint the start square back to blue.
    """
    info = _scene_info(np.array(grid, dtype=int))
    trav = _trav_mask(info, obstacle_passable=False)

    sr, sc = int(info["sr"]), int(info["sc"])
    dr, dc = int(info["dr"]), int(info["dc"])
    n = int(info["n"])
    river = int(info["river"])
    mover_col = int(info["mover"]["col"])

    tr, tc = _ray_terminal(trav, sr, sc, dr, dc, n)

    out = np.array(info["grid"], copy=True)
    out[sr : sr + n, sc : sc + n] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover_col)
    return out


def _size_to_grid(n: int) -> int:
    if n == 2:
        return random.randint(12, 18)
    if n == 3:
        return random.randint(15, 20)
    return random.randint(18, 20)


def _between_positions(start: int, end: int, step: int) -> list[int]:
    dist = abs(end - start)
    return [int(start + step * k) for k in range(1, dist)]


def _paint_main_strip(grid: np.ndarray, sr: int, sc: int, dr: int, dc: int, n: int) -> None:
    if sr == dr:
        c0 = min(sc, dc)
        c1 = max(sc, dc) + n
        grid[sr : sr + n, c0:c1] = _RIVER
    else:
        r0 = min(sr, dr)
        r1 = max(sr, dr) + n
        grid[r0:r1, sc : sc + n] = _RIVER


def _paint_branch(
    grid: np.ndarray,
    orient: str,
    fixed_r: int,
    fixed_c: int,
    branch_pos: int,
    branch_len: int,
    branch_side: int,
    n: int,
) -> None:
    if orient == "h":
        if branch_side > 0:
            grid[fixed_r : fixed_r + branch_len + n, branch_pos : branch_pos + n] = _RIVER
        else:
            grid[fixed_r - branch_len : fixed_r + n, branch_pos : branch_pos + n] = _RIVER
    else:
        if branch_side > 0:
            grid[branch_pos : branch_pos + n, fixed_c : fixed_c + branch_len + n] = _RIVER
        else:
            grid[branch_pos : branch_pos + n, fixed_c - branch_len : fixed_c + n] = _RIVER


def _paint_barrier(grid: np.ndarray, strip_orient: str, fixed: int, line_pos: int, n: int) -> None:
    if strip_orient == "h":
        grid[fixed : fixed + n, line_pos] = _OBST
    else:
        grid[line_pos, fixed : fixed + n] = _OBST


def _line_for_next_step(stop_pos: int, step: int, n: int) -> int:
    return int(stop_pos + n) if step > 0 else int(stop_pos - 1)


def _main_case() -> np.ndarray | None:
    n = random.choice([2, 3, 4])
    H = W = _size_to_grid(n)
    grid = np.full((H, W), _BG, dtype=int)

    orient = random.choice(["h", "v"])

    if orient == "h":
        positive = random.choice([True, False])
        step = 1 if positive else -1

        mover_c = random.randint(1, 2) if positive else W - n - random.randint(1, 2)
        dest_c = W - n if positive else 0
        total_steps = abs(dest_c - mover_c)
        if total_steps < 7:
            return None

        tail = random.randint(4, min(6, total_steps - 3))
        stop_c = int(dest_c - step * tail)
        if abs(stop_c - mover_c) < 3:
            return None

        delta = random.randint(2, min(3, abs(stop_c - mover_c) - 1))
        branch_c = int(stop_c - step * delta)

        row_side_pairs = []
        for r in range(1, H - n):
            if r >= delta + 1:
                row_side_pairs.append((r, -1))
            if H - (r + n) >= delta + 1:
                row_side_pairs.append((r, 1))
        if not row_side_pairs:
            return None
        mover_r, branch_side = random.choice(row_side_pairs)
        dest_r = mover_r

        _paint_main_strip(grid, mover_r, mover_c, dest_r, dest_c, n)

        avail = mover_r if branch_side < 0 else H - (mover_r + n)
        branch_len = random.randint(delta + 1, min(avail, delta + 4))
        _paint_branch(grid, "h", mover_r, 0, branch_c, branch_len, branch_side, n)

        extra_positions = [p for p in _between_positions(mover_c, stop_c, step) if abs(p - branch_c) >= 2]
        extra_branch = None
        if extra_positions and random.random() < 0.7:
            other_c = random.choice(extra_positions)
            side_options = []
            if mover_r >= 2:
                side_options.append(-1)
            if H - (mover_r + n) >= 2:
                side_options.append(1)
            if side_options:
                other_side = random.choice(side_options)
                other_avail = mover_r if other_side < 0 else H - (mover_r + n)
                other_len = random.randint(2, min(other_avail, 4))
                _paint_branch(grid, "h", mover_r, 0, other_c, other_len, other_side, n)
                extra_branch = ("v", other_c, mover_r, other_side, other_len)

        _paint_barrier(grid, "h", mover_r, _line_for_next_step(stop_c, step, n), n)

        downstream = [int(stop_c + step * k) for k in range(2, tail)]
        random.shuffle(downstream)
        for pos in downstream[: random.randint(1, min(2, len(downstream))) if downstream else 0]:
            _paint_barrier(grid, "h", mover_r, _line_for_next_step(pos, step, n), n)

        if extra_branch is not None and random.random() < 0.8:
            _orient, fixed_c, root_r, side_sign, blen = extra_branch
            if blen >= 2:
                bstop = int(root_r + side_sign * random.randint(1, blen - 1))
                _paint_barrier(grid, "v", fixed_c, _line_for_next_step(bstop, side_sign, n), n)

        mover_r0, mover_c0 = mover_r, mover_c
        dest_r0, dest_c0 = dest_r, dest_c
        expected_stop = (mover_r, stop_c)

    else:
        positive = random.choice([True, False])
        step = 1 if positive else -1

        mover_r = random.randint(1, 2) if positive else H - n - random.randint(1, 2)
        dest_r = H - n if positive else 0
        total_steps = abs(dest_r - mover_r)
        if total_steps < 7:
            return None

        tail = random.randint(4, min(6, total_steps - 3))
        stop_r = int(dest_r - step * tail)
        if abs(stop_r - mover_r) < 3:
            return None

        delta = random.randint(2, min(3, abs(stop_r - mover_r) - 1))
        branch_r = int(stop_r - step * delta)

        col_side_pairs = []
        for c in range(1, W - n):
            if c >= delta + 1:
                col_side_pairs.append((c, -1))
            if W - (c + n) >= delta + 1:
                col_side_pairs.append((c, 1))
        if not col_side_pairs:
            return None
        mover_c, branch_side = random.choice(col_side_pairs)
        dest_c = mover_c

        _paint_main_strip(grid, mover_r, mover_c, dest_r, dest_c, n)

        avail = mover_c if branch_side < 0 else W - (mover_c + n)
        branch_len = random.randint(delta + 1, min(avail, delta + 4))
        _paint_branch(grid, "v", 0, mover_c, branch_r, branch_len, branch_side, n)

        extra_positions = [p for p in _between_positions(mover_r, stop_r, step) if abs(p - branch_r) >= 2]
        extra_branch = None
        if extra_positions and random.random() < 0.7:
            other_r = random.choice(extra_positions)
            side_options = []
            if mover_c >= 2:
                side_options.append(-1)
            if W - (mover_c + n) >= 2:
                side_options.append(1)
            if side_options:
                other_side = random.choice(side_options)
                other_avail = mover_c if other_side < 0 else W - (mover_c + n)
                other_len = random.randint(2, min(other_avail, 4))
                _paint_branch(grid, "v", 0, mover_c, other_r, other_len, other_side, n)
                extra_branch = ("h", other_r, mover_c, other_side, other_len)

        _paint_barrier(grid, "v", mover_c, _line_for_next_step(stop_r, step, n), n)

        downstream = [int(stop_r + step * k) for k in range(2, tail)]
        random.shuffle(downstream)
        for pos in downstream[: random.randint(1, min(2, len(downstream))) if downstream else 0]:
            _paint_barrier(grid, "v", mover_c, _line_for_next_step(pos, step, n), n)

        if extra_branch is not None and random.random() < 0.8:
            _orient, fixed_r, root_c, side_sign, blen = extra_branch
            if blen >= 2:
                bstop = int(root_c + side_sign * random.randint(1, blen - 1))
                _paint_barrier(grid, "h", fixed_r, _line_for_next_step(bstop, side_sign, n), n)

        mover_r0, mover_c0 = mover_r, mover_c
        dest_r0, dest_c0 = dest_r, dest_c
        expected_stop = (stop_r, mover_c)

    grid[mover_r0 : mover_r0 + n, mover_c0 : mover_c0 + n] = _MOVER
    grid[dest_r0 : dest_r0 + n, dest_c0 : dest_c0 + n] = _DEST

    if Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0] != _BG:
        return None

    try:
        out = main(grid)
        info = _scene_info(grid)
    except AssertionError:
        return None

    metrics = _behavior_metrics(grid)
    if not metrics["branch_divergence"]:
        return None
    if not metrics["obstacle_effect"]:
        return None
    if metrics["obstacle_pixels"] < 2 * n:
        return None

    sr, sc = int(info["sr"]), int(info["sc"])
    river = int(info["river"])
    tr, tc = metrics["ray_stop"]
    if (tr, tc) != expected_stop:
        return None

    expected = np.array(grid, copy=True)
    expected[sr : sr + n, sc : sc + n] = river
    expected[tr : tr + n, tc : tc + n] = _MOVER
    if not np.array_equal(out, expected):
        return None

    if (tr, tc) == (int(info["dr"]), int(info["dc"])):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _main_case()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 600 attempts")

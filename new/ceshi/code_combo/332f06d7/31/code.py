# Combo variant of 332f06d7 (index 31)
# combo_spec: D1/0+D5/2+D14/0
# Name: 直线射线+边缘停止+稀疏窄通道

from __future__ import annotations

from collections import Counter
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
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
------------------------------
- Original 332f06d7:
  - green background, blue winding corridor, black n x n mover, red n x n destination,
  - solve by BFS over all feasible n x n placements inside the blue corridor,
  - if red is reachable, black covers red; otherwise it stops at the farthest reachable placement.
- Fused variant:
  - remove the red destination completely,
  - keep one black n x n mover inside a sparse blue channel system,
  - the channel geometry is now a straight main ray plus one or two perpendicular side branches,
  - the black square does not BFS through turns; it continues only along the unique longest straight
    ray that starts at its current placement,
  - it stops when that ray reaches the grid edge.
- Small trace:
  - Example: n=2 on a 14 x 14 grid. The mover sits at top-left (5,2). The main blue strip runs
    horizontally to the right edge, with only a one-step blue stub behind the mover. Farther right,
    at column 8, a vertical branch goes upward to the top edge.
  - Old corridor-following logic with edge-stop would BFS through the turn and could prefer the
    upward edge because it is farther in path length.
  - The fused rule instead compares the straight rays leaving the mover: left has length 1, right
    has length 10, vertical directions have length 0 at the mover. Therefore the mover slides only
    right and finishes flush against the right border.

b) Complexity calibration
-------------------------
- If the grid contained only one isolated straight line, the task would become too close to a toy
  case and the straight-ray constraint would not be visibly necessary.
- The side branches matter because they create legal traversable turns that a BFS-based edge-stop
  solver would use, but the fused rule forbids turning.
- To keep the rule learnable, the generator restricts variation to a small set:
  - orientation: horizontal or vertical,
  - motion direction: one of the four cardinal directions,
  - short opposite-side stub length: 0 to 2,
  - one required edge-reaching side branch and an optional short second branch.
- This is enough variation for ARC-style pattern induction without creating too many cases.

c) Hidden-assumption audit of the original main()
-------------------------------------------------
- Original assumption: there are at least two same-sized solid squares.
  - Broken by D5/2. The fused solver must detect a single solid mover square.
- Original assumption: the river plus mover plus destination define traversability.
  - Broken by D5/2. Traversability is now just river union mover.
- Original assumption: BFS over 4-neighbor placement moves is the correct motion model.
  - Broken by D1/0. In the fused task, turning into a side branch is illegal even when feasible.
- Original assumption: the chosen goal is either the red square or the farthest BFS-reachable
  placement.
  - Broken by D1/0 and D5/2 together. The goal is the end of the longest straight ray from the
    mover, and the generator guarantees that this ray ends on the boundary.
- New solver assumptions introduced here:
  - exactly one solid non-background square component exists,
  - feasible motion from the mover is axis-aligned,
  - among the four cardinal directions there is a unique longest straight feasible ray,
  - that longest ray reaches the boundary,
  - side branches never touch the mover directly, so they do not create local ambiguity.
  - The generator explicitly enforces these constraints.

d) Placement legality
---------------------
- All geometry is built in placement space first and then painted as n-pixel-thick strips.
- The main strip always keeps the mover fully in bounds and always reaches exactly one intended
  stopping edge without using a corner.
- The backward stub is short and stays interior, so the opposite direction is visible but inferior.
- Branches attach only at interior points of the main strip, never at the mover, never at corners,
  and never overlap each other except at their single junction cells with the main strip.
- The occupied non-background area is constrained to 10-20% of the grid, matching the sparse-narrow
  requirement.

e) Visual observability
-----------------------
- Humans can see one black square, no red target, a dominant green background, and thin blue
  rectilinear channels.
- The output repeatedly shows the black square continuing straight to a border instead of turning
  into visibly available side branches.
- The direction cue is geometric rather than based on color ids: the mover sits on a straight line
  with a very short stub on one side and a much longer continuation on the other.

f) Visual cleanliness
---------------------
- Only three colors remain: green background, blue channel, black mover.
- All blue shapes are thin axis-aligned strips with low density; the grid stays mostly background.
- Branch count is limited to at most two, which keeps the scene sparse and legible.

g) Color-change inferability
----------------------------
- There is no hidden color remapping. The only output change is that the black square vacates its
  start location back to blue and reappears at the ray endpoint on the border.

h) Anti-degeneration target
---------------------------
- To prove the fusion is not just "edge-stop on any corridor", every generated sample includes a
  required branch that makes unrestricted BFS edge-stop disagree with the fused straight-ray rule.
- Concretely, the required branch reaches a border through a longer path than the straight main ray,
  so a BFS edge-stop baseline would turn into that branch, while the fused solver must stay straight.

i) Generator obligations for the new solver
-------------------------------------------
- The generator must guarantee:
  - exactly one salient black solid square,
  - sparse blue coverage ratio in [10%, 20%],
  - a unique longest straight ray from the mover,
  - that longest ray ends on a border but not a corner,
  - at least one side branch exists and creates a different BFS-edge-stop answer,
  - oracle check: main(input) matches the analytically constructed expected output.
"""


_BG, _RIVER, _MOVER = 3, 1, 0
_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))


def _detect_mover(g: np.ndarray, bg: int) -> dict:
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
    assert squares
    return max(squares, key=lambda s: (int(s["n"]), -int(s["col"])))


def _find_river(g: np.ndarray, bg: int, mover_col: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) in (int(bg), int(mover_col)):
            continue
        return int(col)
    raise AssertionError("river color not found")


def _feasible_windows(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert not (h2 <= 0 or w2 <= 0)
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all()):
                feasible[int(r), int(c)] = True
    return feasible


def _touches_boundary(r: int, c: int, n: int, H: int, W: int) -> bool:
    return int(r) == 0 or int(c) == 0 or int(r + n) == H or int(c + n) == W


def _trace_ray(feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int) -> tuple[int, int, int]:
    h2, w2 = feasible.shape
    steps = 0
    r, c = int(sr), int(sc)
    while True:
        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < h2 and 0 <= nc < w2 and bool(feasible[nr, nc])):
            return steps, int(r), int(c)
        steps += 1
        r, c = int(nr), int(nc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - detect the single solid mover square,
    - compute all feasible n x n placements inside the sparse channel,
    - compare the four straight rays leaving the mover,
    - choose the unique longest straight ray and slide to its endpoint,
    - repaint the start square back to river.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    mover = _detect_mover(g, bg)
    river = _find_river(g, bg, int(mover["col"]))

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    trav = (g == int(river)) | (g == int(mover["col"]))
    feasible = _feasible_windows(trav, n)
    assert bool(feasible[int(sr), int(sc)])

    rays = []
    for dr, dc in _DIRS:
        steps, er, ec = _trace_ray(feasible, sr, sc, dr, dc)
        rays.append(
            {
                "steps": int(steps),
                "end_r": int(er),
                "end_c": int(ec),
                "edge": int(_touches_boundary(er, ec, n, H, W)),
            }
        )

    best = max(rays, key=lambda item: (int(item["steps"]), int(item["edge"])))
    tr, tc = int(best["end_r"]), int(best["end_c"])

    out = g.copy()
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(river)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover["col"])
    return out


def _size_to_grid(n: int) -> int:
    if n == 1:
        return random.randint(10, 13)
    if n == 2:
        return random.randint(12, 16)
    if n == 3:
        return random.randint(15, 18)
    return random.randint(19, 20)


def _paint_main_strip(grid: np.ndarray, orient: str, fixed: int, a0: int, a1: int, n: int) -> None:
    lo, hi = sorted((int(a0), int(a1)))
    if orient == "h":
        grid[int(fixed) : int(fixed + n), int(lo) : int(hi + n)] = _RIVER
    else:
        grid[int(lo) : int(hi + n), int(fixed) : int(fixed + n)] = _RIVER


def _paint_branch_to_edge(grid: np.ndarray, orient: str, fixed: int, anchor: int, side: int, n: int) -> int:
    G = int(grid.shape[0])
    if orient == "h":
        if side < 0:
            grid[0 : int(fixed + n), int(anchor) : int(anchor + n)] = _RIVER
            return int(fixed)
        grid[int(fixed) : G, int(anchor) : int(anchor + n)] = _RIVER
        return int(G - n - fixed)
    if side < 0:
        grid[int(anchor) : int(anchor + n), 0 : int(fixed + n)] = _RIVER
        return int(fixed)
    grid[int(anchor) : int(anchor + n), int(fixed) : G] = _RIVER
    return int(G - n - fixed)


def _paint_partial_branch(grid: np.ndarray, orient: str, fixed: int, anchor: int, side: int, length: int, n: int) -> None:
    length = int(length)
    if orient == "h":
        if side < 0:
            grid[int(fixed - length) : int(fixed + n), int(anchor) : int(anchor + n)] = _RIVER
        else:
            grid[int(fixed) : int(fixed + n + length), int(anchor) : int(anchor + n)] = _RIVER
    else:
        if side < 0:
            grid[int(anchor) : int(anchor + n), int(fixed - length) : int(fixed + n)] = _RIVER
        else:
            grid[int(anchor) : int(anchor + n), int(fixed) : int(fixed + n + length)] = _RIVER


def _farthest_edge_stop(grid: np.ndarray, mover_r: int, mover_c: int, n: int) -> tuple[int, int]:
    trav = (grid == _RIVER) | (grid == _MOVER)
    feasible = _feasible_windows(trav, n)
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(mover_r), int(mover_c)] = 0
    q = [(int(mover_r), int(mover_c))]
    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        for dr, dc in _DIRS:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                q.append((int(rr), int(cc)))

    edge_positions: list[tuple[int, int]] = []
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) < 0:
                continue
            if _touches_boundary(r, c, n, grid.shape[0], grid.shape[1]):
                edge_positions.append((int(r), int(c)))
    assert edge_positions
    return max(edge_positions, key=lambda pos: (int(dist[pos[0], pos[1]]), -abs(pos[0] - mover_r) - abs(pos[1] - mover_c)))


def _occupancy_ratio(grid: np.ndarray) -> float:
    return float((grid != _BG).sum()) / float(grid.shape[0] * grid.shape[1])


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([1, 2, 3, 4])
    G = _size_to_grid(n)
    H = W = G
    orient = random.choice(["h", "v"])
    sign = random.choice([-1, 1])
    grid = np.full((H, W), _BG, dtype=int)

    if orient == "h":
        row_candidates = [r for r in range(1, H - n) if 1 <= r <= H - n - 1]
        row_candidates = [r for r in row_candidates if max(r, H - n - r) >= 2]
        if not row_candidates:
            return None
        fixed = random.choice(row_candidates)

        if sign > 0:
            start_candidates = [c for c in range(1, W - n) if W - n - c >= 5]
            if not start_candidates:
                return None
            start_axis = random.choice(start_candidates)
            back_max = min(2, start_axis - 1)
            edge_axis = W - n
        else:
            start_candidates = [c for c in range(0, W - n) if c >= 5]
            start_candidates = [c for c in start_candidates if c <= W - n - 1]
            if not start_candidates:
                return None
            start_axis = random.choice(start_candidates)
            back_max = min(2, W - n - 1 - start_axis)
            edge_axis = 0
    else:
        col_candidates = [c for c in range(1, W - n) if 1 <= c <= W - n - 1]
        col_candidates = [c for c in col_candidates if max(c, W - n - c) >= 2]
        if not col_candidates:
            return None
        fixed = random.choice(col_candidates)

        if sign > 0:
            start_candidates = [r for r in range(1, H - n) if H - n - r >= 5]
            if not start_candidates:
                return None
            start_axis = random.choice(start_candidates)
            back_max = min(2, start_axis - 1)
            edge_axis = H - n
        else:
            start_candidates = [r for r in range(0, H - n) if r >= 5]
            start_candidates = [r for r in start_candidates if r <= H - n - 1]
            if not start_candidates:
                return None
            start_axis = random.choice(start_candidates)
            back_max = min(2, H - n - 1 - start_axis)
            edge_axis = 0

    forward_steps = abs(int(edge_axis - start_axis))
    if forward_steps < 5:
        return None

    back_steps = random.randint(0, back_max)
    back_axis = int(start_axis - sign * back_steps)
    _paint_main_strip(grid, orient, fixed, back_axis, edge_axis, n)

    if orient == "h":
        side_lengths = [(-1, int(fixed)), (1, int(H - n - fixed))]
    else:
        side_lengths = [(-1, int(fixed)), (1, int(W - n - fixed))]
    side_lengths = [(side, steps) for side, steps in side_lengths if int(steps) >= 2]
    if not side_lengths:
        return None

    primary_options: list[tuple[int, int, int]] = []
    for side, branch_steps in side_lengths:
        for offset in range(2, forward_steps - 1):
            if int(offset + branch_steps) > int(forward_steps):
                primary_options.append((int(offset), int(side), int(branch_steps)))
    if not primary_options:
        return None
    primary_offset, primary_side, _primary_steps = random.choice(primary_options)
    primary_anchor = int(start_axis + sign * primary_offset)
    _paint_branch_to_edge(grid, orient, fixed, primary_anchor, primary_side, n)

    if random.random() < 0.55:
        secondary_offsets = [off for off in range(2, forward_steps - 1) if abs(off - primary_offset) >= 2]
        secondary_options = []
        for off in secondary_offsets:
            for side, max_steps in side_lengths:
                if int(max_steps) >= 2:
                    secondary_options.append((int(off), int(side), int(max_steps)))
        if secondary_options:
            off2, side2, max_steps2 = random.choice(secondary_options)
            len2 = random.randint(2, max_steps2)
            _paint_partial_branch(grid, orient, fixed, int(start_axis + sign * off2), side2, len2, n)

    ratio = _occupancy_ratio(grid)
    if not (0.10 <= ratio <= 0.20):
        return None

    if orient == "h":
        mover_r, mover_c = int(fixed), int(start_axis)
        final_r, final_c = int(fixed), int(edge_axis)
    else:
        mover_r, mover_c = int(start_axis), int(fixed)
        final_r, final_c = int(edge_axis), int(fixed)

    grid[int(mover_r) : int(mover_r + n), int(mover_c) : int(mover_c + n)] = _MOVER
    expected = grid.copy()
    expected[int(mover_r) : int(mover_r + n), int(mover_c) : int(mover_c + n)] = _RIVER
    expected[int(final_r) : int(final_r + n), int(final_c) : int(final_c + n)] = _MOVER

    if Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0] != _BG:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    bfs_r, bfs_c = _farthest_edge_stop(grid, mover_r, mover_c, n)
    if (int(bfs_r), int(bfs_c)) == (int(final_r), int(final_c)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo variant after 800 attempts")

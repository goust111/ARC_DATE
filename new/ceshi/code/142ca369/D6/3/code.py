# Variant of 142ca369
# Dimension: D6 -- 射线终止条件 (dst)
# Change: 射线离开网格边界时终止（out of grid）... -> 反射次数耗尽时终止

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


"""
Design analysis for the D6 variant
==================================

a) Concrete behavior and step trace
-----------------------------------
Original task:
- Input contains monochrome 3-cell L arrows plus colored wall pixels/segments.
- Every arrow keeps reflecting until it exits the grid.

Variant:
- Keep the same wall physics and color inheritance, but arrows now encode a
  reflection budget inside the arrow itself.
- The arrow still occupies a 2x2 box with one missing corner. The right-angle
  elbow is the only "payload" color (1..7): that is the ray's starting color.
- The two arm-end pixels are resource markers:
  - color 9 = one usable reflection token
  - color 8 = empty slot
  - number of 9s in the two arm cells = max reflections for that arrow
- Walls never use colors 8 or 9, so the arrow remains visually separable from
  walls without adding a new object type.

Small trace:
- Suppose an arrow points down-right, elbow color is 3, arm markers are [9, 8].
  Then its budget is 1.
- In the original task, if the ray meets two side walls, it bends twice.
- In the variant, it bends at the first wall, adopts that wall color, spends its
  only token, then continues straight. Later walls are ignored: the ray keeps
  the current color/direction and simply passes through their cells if needed.

b) Complexity calibration
-------------------------
- The new freedom is deliberately small: each arrow has exactly two visible
  marker slots, so the budget range is 0..2.
- That is enough to create clearly non-original behavior when we place more wall
  opportunities than the budget allows.
- Using fixed marker colors 8/9 avoids an arbitrary color->number mapping that
  would be hard to infer from 2-4 ARC examples.

c) Implicit-assumption audit against the original solver
--------------------------------------------------------
- Original arrow detection grouped connected components per color. That relied on
  arrows being monochrome. This is no longer true, so the variant scans 2x2
  windows for a 3-cell L whose elbow is a normal color and whose two arm cells
  are in {8, 9}.
- Original `start_col` was the arrow component color. Now only the elbow color
  should seed the ray; arm marker colors are metadata and must never leak into
  the ray.
- Original wall handling always reflected when a wall was hit. That assumption
  breaks under D6. Every reflection check must now be guarded by
  `used_reflections < max_reflections`.
- Original side-touch logic could flip both axes in one step if both side walls
  existed. The generator never creates such cases. The solver keeps a
  deterministic priority (`dr`-wall before `dc`-wall) once budgets are active.

d) Placement legality
---------------------
- Arrows still fit in a 2x2 box and never overlap walls or other arrows.
- Walls remain size 1 or 3, axis-aligned, and sit one cell off the diagonal ray.
- The generator is still output-first: it plans the intended ray path, places the
  supporting walls, commits the input, then rejects the sample unless
  `main(input) == output`.
- Walls avoid colors 8/9 so arrow parsing is stable.

e) Visual observability
-----------------------
- Human-visible cue: the two arm cells look like "resource slots".
- Human-visible effect: some rays bend fewer times than the number of nearby
  walls would suggest, then continue straight with the post-bounce color.
- This makes the D6 change visible even if the rest of the task is unchanged.

f) Visual cleanliness
---------------------
- No extra markers are added outside the arrow body.
- Marker colors 8/9 are reserved for arrows only, so walls/rays are not visually
  conflated with the reflection-budget encoding.

g) Color-change inferability
----------------------------
- Color inheritance is unchanged: on a consumed reflection, the ray adopts the
  wall color and the bounce cell shows that wall color.
- Ignored walls do not alter ray color. This contrast is itself a clue that the
  budget, not the wall color, controls whether a bend happens.

h) Generator anti-degeneration rule
-----------------------------------
- At least one arrow per generated grid is forced to encounter more wall
  opportunities than its budget allows. This prevents the generator from
  collapsing back to the original "always reflect" behavior.

i) New solver assumptions introduced by the variant
---------------------------------------------------
- Arrow arms must use only colors 8/9 and the elbow must use a non-marker color.
  The generator guarantees this.
- Walls must not use 8/9. The generator guarantees this too.
- The solver interprets budget as "number of reflections that may actually alter
  direction"; after the budget is exhausted, later walls are ignored.
"""


_EMPTY_TOKEN = 8
_FULL_TOKEN = 9
_MARKER_COLORS = {_EMPTY_TOKEN, _FULL_TOKEN}
_RAY_COLORS = list(range(1, 8))


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _arrow_from_window(g: np.ndarray, r0: int, c0: int, bg: int):
    cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
    pts = [p for p in cells if int(g[p]) != int(bg)]
    if len(pts) != 3:
        return None

    pts_set = set(pts)
    deg = {
        p: sum(
            (nr, nc) in pts_set
            for nr, nc in ((p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1))
        )
        for p in pts
    }
    elbows = [p for p, d in deg.items() if d == 2]
    if len(elbows) != 1:
        return None

    elbow = elbows[0]
    arm_pts = [p for p in pts if p != elbow]
    elbow_col = int(g[elbow])
    if elbow_col in _MARKER_COLORS or elbow_col == int(bg):
        return None
    if any(int(g[p]) not in _MARKER_COLORS for p in arm_pts):
        return None

    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    if max(rs) - min(rs) != 1 or max(cs) - min(cs) != 1:
        return None
    corners = [(min(rs), min(cs)), (min(rs), max(cs)), (max(rs), min(cs)), (max(rs), max(cs))]
    missing = [p for p in corners if p not in pts_set]
    if len(missing) != 1:
        return None
    mr, mc = missing[0]
    dr = 1 if mr == min(rs) else -1
    dc = 1 if mc == min(cs) else -1
    quota = sum(int(g[p]) == _FULL_TOKEN for p in arm_pts)
    return {
        "pts": pts,
        "start": elbow,
        "dir": (int(dr), int(dc)),
        "start_col": elbow_col,
        "quota": int(quota),
    }


def _find_arrows(g: np.ndarray, bg: int):
    h, w = map(int, g.shape)
    arrows = []
    arrow_cells: set[tuple[int, int]] = set()
    seen: set[frozenset[tuple[int, int]]] = set()

    for r in range(h - 1):
        for c in range(w - 1):
            info = _arrow_from_window(g, r, c, bg)
            if info is None:
                continue
            key = frozenset(info["pts"])
            if key in seen:
                continue
            seen.add(key)
            arrows.append(info)
            arrow_cells |= set(info["pts"])

    return arrows, arrow_cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    arrows, arrow_cells = _find_arrows(g, bg)
    wall_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for arrow in arrows:
        dr, dc = map(int, arrow["dir"])
        cur = (int(arrow["start"][0]), int(arrow["start"][1]))
        cur_col = int(arrow["start_col"])
        max_reflections = int(arrow["quota"])
        used_reflections = 0
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col), int(used_reflections))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not _in_bounds(h, w, nr, nc):
                break
            if (nr, nc) in arrow_cells:
                break

            # With budget left, a head-on wall still reflects as in the original task.
            if used_reflections < max_reflections and (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[cur]) == int(bg):
                    out[cur] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                used_reflections += 1
                continue

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if used_reflections >= max_reflections:
                continue

            wall_pos = None
            flip_axis = None
            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wall_pos = (nr + 1, nc)
                flip_axis = "dr"
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wall_pos = (nr - 1, nc)
                flip_axis = "dr"
            elif int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wall_pos = (nr, nc + 1)
                flip_axis = "dc"
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wall_pos = (nr, nc - 1)
                flip_axis = "dc"

            if wall_pos is None:
                continue

            wcol = int(g[wall_pos])
            if int(g[nr, nc]) == int(bg):
                out[nr, nc] = int(wcol)
            cur_col = int(wcol)
            if flip_axis == "dr":
                dr = -int(dr)
            else:
                dc = -int(dc)
            used_reflections += 1

    return out


def _max_diag_steps(h: int, w: int, r: int, c: int, dr: int, dc: int) -> int:
    return min((h - 1 - r) if dr == 1 else r, (w - 1 - c) if dc == 1 else c)


def _arrow_cells(sr: int, sc: int, dr: int, dc: int, base_col: int, quota: int):
    arm_pts = [(int(sr - dr), int(sc)), (int(sr), int(sc - dc))]
    arms = arm_pts[:]
    _rng.shuffle(arms)
    token_map = {}
    for i, p in enumerate(arms):
        token_map[p] = _FULL_TOKEN if i < quota else _EMPTY_TOKEN
    cells = [
        (int(sr), int(sc), int(base_col)),
        (int(arm_pts[0][0]), int(arm_pts[0][1]), int(token_map[arm_pts[0]])),
        (int(arm_pts[1][0]), int(arm_pts[1][1]), int(token_map[arm_pts[1]])),
    ]
    return cells


def _wall_options(
    h: int,
    w: int,
    cur_r: int,
    cur_c: int,
    cur_dr: int,
    cur_dc: int,
    blocked: set[tuple[int, int]],
):
    options: list[tuple[list[tuple[int, int]], str]] = []

    wr, wc = cur_r + cur_dr, cur_c
    if 0 <= wr < h:
        seg3 = [(wr, wc + d) for d in (-1, 0, 1)]
        if all(0 <= cc < w for _, cc in seg3):
            options.append((seg3, "dr"))
        options.append(([(wr, wc)], "dr"))

    wr2, wc2 = cur_r, cur_c + cur_dc
    if 0 <= wc2 < w:
        seg3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
        if all(0 <= rr < h for rr, _ in seg3v):
            options.append((seg3v, "dc"))
        options.append(([(wr2, wc2)], "dc"))

    _rng.shuffle(options)
    prefer_long = _rng.random() < 0.6
    options.sort(key=lambda item: len(item[0]), reverse=prefer_long)
    return [
        (cells, axis)
        for cells, axis in options
        if not any((int(r), int(c)) in blocked for r, c in cells)
    ]


def _plan_one_arrow(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    base_col: int,
    must_have_ignored_wall: bool,
):
    quota = _rng.choices([0, 1, 2], weights=[1, 3, 3])[0]
    extra_ignored = _rng.choices([0, 1, 2], weights=[1, 3, 2])[0]
    if must_have_ignored_wall:
        extra_ignored = max(1, extra_ignored)
    total_events = quota + extra_ignored

    dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
    r_lo, r_hi = (1 if dr == 1 else 0), (h - 1 if dr == 1 else h - 2)
    c_lo, c_hi = (1 if dc == 1 else 0), (w - 1 if dc == 1 else w - 2)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(160):
        sr = _rng.randint(r_lo, r_hi)
        sc = _rng.randint(c_lo, c_hi)
        arrow = _arrow_cells(sr, sc, dr, dc, base_col, quota)
        arrow_set = {(int(r), int(c)) for r, c, _ in arrow}
        if any((r, c) in occupied for r, c in arrow_set):
            continue

        cur_r, cur_c = int(sr), int(sc)
        cur_dr, cur_dc = int(dr), int(dc)
        cur_col = int(base_col)
        ray_colors: dict[tuple[int, int], int] = {}
        ray_set: set[tuple[int, int]] = set()
        walls: list[tuple[list[tuple[int, int]], int]] = []
        wall_cells: set[tuple[int, int]] = set()
        ok = True

        for ev_idx in range(total_events):
            max_steps = _max_diag_steps(h, w, cur_r, cur_c, cur_dr, cur_dc)
            if max_steps < 2:
                ok = False
                break

            upper = max_steps - 1
            if upper < 2:
                ok = False
                break
            seg_len = _rng.randint(2, upper)

            for _step in range(seg_len):
                nr = int(cur_r + cur_dr)
                nc = int(cur_c + cur_dc)
                if not _in_bounds(h, w, nr, nc):
                    ok = False
                    break
                if (
                    (nr, nc) in occupied
                    or (nr, nc) in arrow_set
                    or (nr, nc) in ray_set
                    or (nr, nc) in wall_cells
                ):
                    ok = False
                    break
                cur_r, cur_c = nr, nc
                ray_set.add((cur_r, cur_c))
                ray_colors[(cur_r, cur_c)] = int(cur_col)
            if not ok:
                break

            blocked = set(occupied) | arrow_set | ray_set | wall_cells
            options = _wall_options(h, w, cur_r, cur_c, cur_dr, cur_dc, blocked)
            if not options:
                ok = False
                break

            chosen_wall = None
            for wpxs, axis in options:
                next_dr, next_dc = int(cur_dr), int(cur_dc)
                if ev_idx < quota:
                    if axis == "dr":
                        next_dr = -int(next_dr)
                    else:
                        next_dc = -int(next_dc)
                if _max_diag_steps(h, w, cur_r, cur_c, next_dr, next_dc) < (
                    2 if ev_idx + 1 < total_events else 1
                ):
                    continue
                chosen_wall = (wpxs, axis, next_dr, next_dc)
                break
            if chosen_wall is None:
                ok = False
                break

            wpxs, axis, next_dr, next_dc = chosen_wall
            wcol = int(_rng.choice(_RAY_COLORS))
            walls.append(([(int(r), int(c)) for r, c in wpxs], int(wcol)))
            wall_cells.update((int(r), int(c)) for r, c in wpxs)

            if ev_idx < quota:
                ray_colors[(cur_r, cur_c)] = int(wcol)
                cur_col = int(wcol)
                cur_dr, cur_dc = int(next_dr), int(next_dc)

        if not ok:
            continue

        while True:
            nr = int(cur_r + cur_dr)
            nc = int(cur_c + cur_dc)
            if not _in_bounds(h, w, nr, nc):
                break
            if (
                (nr, nc) in occupied
                or (nr, nc) in arrow_set
                or (nr, nc) in ray_set
                or (nr, nc) in wall_cells
            ):
                ok = False
                break
            cur_r, cur_c = nr, nc
            ray_set.add((cur_r, cur_c))
            ray_colors[(cur_r, cur_c)] = int(cur_col)
        if not ok or len(ray_set) < 4:
            continue

        new_cells = set(arrow_set) | set(ray_set) | set(wall_cells)
        if new_cells & occupied:
            continue

        return {
            "arrow": arrow,
            "walls": walls,
            "ray_colors": ray_colors,
            "occupied": new_cells,
            "used_ignored": extra_ignored > 0,
            "quota": quota,
            "total_events": total_events,
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    h, w = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 4)
    colors = _RAY_COLORS[:]
    _rng.shuffle(colors)
    arrow_colors = colors[:n_arrows]

    grid = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    forced_ignored_idx = _rng.randrange(n_arrows)
    any_ignored = False

    for ai, base_col in enumerate(arrow_colors):
        placed = None
        for _ in range(60):
            plan = _plan_one_arrow(
                h=h,
                w=w,
                occupied=occupied,
                base_col=int(base_col),
                must_have_ignored_wall=(ai == forced_ignored_idx and not any_ignored),
            )
            if plan is None:
                continue
            placed = plan
            break

        if placed is None:
            return None

        for r, c, col in placed["arrow"]:
            grid[r, c] = int(col)
            out[r, c] = int(col)
        for wpxs, wcol in placed["walls"]:
            for r, c in wpxs:
                grid[r, c] = int(wcol)
                out[r, c] = int(wcol)
        for (r, c), col in placed["ray_colors"].items():
            out[r, c] = int(col)

        occupied |= placed["occupied"]
        any_ignored = any_ignored or bool(placed["used_ignored"])

    if not any_ignored:
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

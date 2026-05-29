# Variant of da515329
# Dimension: D1 — 螺旋方向 (path)
# Change: 四个端点统一按顺时针方向螺旋展开... → 方向由端点位置决定（上下顺时针、左右逆时针）

import math
import os
import random
import sys

import numpy as np

_SOLVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "solvers")
)
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis
# ---------------
# variant_spec:
#   input:
#     - Same as the seed task: one hollow teal plus-sign cross on black.
#     - No extra markers are added, so the only observable change must come from the
#       geometry of the emitted paths.
#   concretization:
#     - Keep the original spacing model intact: one turtle per arm tip, right-hand
#       wall following, turn-right > straight > turn-left, with 2-cell straights and
#       1-cell moves immediately after a turn.
#     - Change only the launch side. Vertical tips keep the original clockwise launch
#       (initial heading = right of outward); horizontal tips launch from the mirrored
#       side (initial heading = left of outward).
#     - This is the smallest D1-only change that still produces opposite local winding
#       families. Experiments showed that fully reversing the horizontal turtles'
#       turn-priority creates many dead-end layouts and very sparse outputs.
#   behavioral_trace:
#     - Original left tip:
#       outward = west, initial heading = north, then the right-hand rule folds the
#       path upward before it joins the rectangular corridors.
#     - Variant left tip:
#       outward = west, initial heading = south, while the top/bottom tips still
#       launch on their clockwise flank. The later wall-following rule is unchanged,
#       but this mirrored entry point propagates through the same spacing constraints
#       and changes a large fraction of the final wall pixels.
#   complexity_calibration:
#     - The rule still has only two categories, repeated four times per example:
#       vertical tips behave one way, horizontal tips another way.
#     - This is visible in 2-4 examples because every output contains both categories
#       simultaneously; no hidden color ordering or single-instance exception is used.
#   implicit_assumption_audit:
#     - Center detection by row/column argmax remains valid because the input cross is
#       unchanged.
#     - Arm collection along the four cardinal rays remains valid for the same reason.
#     - The original parity/border post-processing is not retained. Those cleanups
#       were fitted to the uniform-clockwise outer ring and hard-coded asymmetric caps
#       such as the top-left completion and bottom-right singleton; after the launch
#       side is mirrored for horizontal tips, those assumptions no longer hold.
#     - The move legality checks (empty target + 1-cell orthogonal gap except along
#       the predecessor edge) remain valid and are the main mechanism keeping the
#       mixed-handed output visually clean.
#   placement_legality:
#     - The generator keeps the original size formula and minimum tip-to-border gap,
#       so all four turtles have enough runway before touching the boundary.
#     - Candidate centers are validated by actually running the new solver, checking
#       fill fraction, determinism, and divergence from the all-clockwise baseline.
#   observability_and_cleanliness:
#     - Generator rejection uses a visible-difference metric: compare the mixed-launch
#       output against the original all-clockwise launch on the same input and require
#       a sizeable Hamming distance. This prevents accidental regression to outputs
#       that would look like the seed task.
#     - Fill is constrained to a mid-density band so the result stays legible rather
#       than degenerating into tiny stubs or nearly solid blocks.


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _solve_with_launch_policy(grid: np.ndarray, launch_dir_for_outward) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = 8
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == 8:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": launch_dir_for_outward(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1)
            agent["phase"] = None
            agent["dir"] = launch_dir_for_outward(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                do_move(agent, nd, 1)
                agent["p"] = 0
                return True

        agent["done"] = True
        return False

    changed = True
    guard = 0
    while changed and guard < 200000:
        guard += 1
        changed = False
        for ag in agents:
            if step_agent(ag):
                changed = True
        if all(ag["done"] for ag in agents):
            break

    return out


def _variant_launch_dir(outward):
    return _right(outward) if outward[0] != 0 else _left(outward)


def _uniform_clockwise_launch_dir(outward):
    return _right(outward)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four spiral wall paths from the tips of a teal cross, with launch direction tied to tip orientation.

    Input structure:
    - bg = 0 (black). A single teal (8) cross has a one-cell empty hole at its center.

    Rule:
    - Top and bottom tips launch from the clockwise side of their arm; left and right tips launch from the mirrored side.
    - After launch, every path uses the same right-hand preference: turn right > go straight > turn left (left only away from the border).
    - Straight runs normally advance 2 cells; the first move after a turn is only 1 cell.
    - Every newly painted cell must keep a 1-cell orthogonal gap from existing walls except along the path predecessor edge.

    Key insight: The wall-following mechanics stay the same as the seed task, but flipping the launch side for the horizontal tips creates two endpoint-dependent winding families that visibly diverge from the uniform-clockwise baseline.
    """

    return _solve_with_launch_policy(grid, _variant_launch_dir)


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.42, 0.54
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
# --- end CFG -----------------------------------------------------------


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _variant_metrics(inp: np.ndarray) -> tuple[np.ndarray, float, int]:
    out = main(inp)
    base = _solve_with_launch_policy(inp, _uniform_clockwise_launch_dir)
    fill = float(np.mean(out == CROSS_COLOR))
    diff = int(np.sum(out != base))
    return out, fill, diff


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    min_visible_diff = max(12, size // 2)
    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        try:
            out, fill, diff = _variant_metrics(inp)
        except Exception:
            continue
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if diff < min_visible_diff:
            continue
        if np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

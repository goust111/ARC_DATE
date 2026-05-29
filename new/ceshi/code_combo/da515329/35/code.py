# Combo variant of da515329 (index 35)
# combo_spec: D1/1+D16/1
# Name: 向侧面展开 + 方向由位置决定

import math
import os
import random
import sys

import numpy as np

_SOLVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "solvers")
)
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis
# ---------------
# variant_spec:
#   input:
#     - Same object vocabulary as the seed task: one hollow teal plus-sign on black.
#     - No extra markers are introduced, so every visible difference must come from the
#       path geometry itself.
#   concretization:
#     - D1/1 is implemented with the stable launch-side interpretation from the single
#       variant: vertical tips use the clockwise flank of their arm (`right(outward)`),
#       horizontal tips use the mirrored flank (`left(outward)`).
#     - D16/1 changes the first painted move, not the later wall-following policy. In
#       the seed logic, a tip with heading `launch_dir` usually paints its first cell by
#       turning right, which is a one-step retreat toward the cross. Here we force the
#       first move to go along `launch_dir` itself, i.e. sideways relative to the arm.
#     - For arm_len == 1, a pure side step is too easy to absorb back into the same
#       corridor family. The combo therefore keeps a short-arm special case: after the
#       forced side step, that tip also takes one outward step before the usual spiral.
#   behavioral_trace:
#     - Example: top tip at `(cr-2, cc)`.
#       Seed / D1-only local start:
#         launch_dir = east, then the priority list tries south first, so the first new
#         cell is `(cr-1, cc)`, a backward retreat.
#       Combo local start:
#         the same top tip first paints `(cr-2, cc+1)`, a lateral offset, then resumes
#         the original right-hand spacing rule from that shifted anchor.
#     - Example: left tip at `(cr, cc-2)`.
#       D1/1 already mirrors its launch side to south; the combo now makes that south
#       cell the first painted cell, so horizontal and vertical tips differ both in
#       launch side and in the visible offset of the first corridor.
#   complexity_calibration:
#     - There are only two repeated endpoint categories: vertical tips vs horizontal
#       tips. Every sample contains both, so a human can compare them directly.
#     - The side-step rule is repeated four times in each output. This makes D16 visible
#       without introducing extra hidden state or many-case branching.
#   implicit_assumption_audit:
#     - Cross-center detection by row/column argmax still holds because the input object
#       is unchanged.
#     - Arm collection on the four cardinal rays still holds for the same reason.
#     - The seed task's parity-specific border cleanup is intentionally removed. Those
#       patches were fitted to the uniform-clockwise outer ring and become unreliable
#       once horizontal tips mirror their launch side and every tip shifts sideways.
#     - `can_move` and the 1-cell orthogonal spacing rule remain valid; they are the
#       mechanism that keeps the fused pattern legible instead of collapsing into solid
#       blocks.
#   placement_legality:
#     - The generator keeps the original size formula and minimum tip-to-border gap, so
#       all side-start and short-arm outward steps stay in-bounds for accepted samples.
#     - Candidate centers are validated by running the combo solver itself, checking
#       fill fraction, determinism, and visible divergence from two baselines.
#   observability_and_cleanliness:
#     - Baseline A = D1-only: position-dependent launch side, but no forced side step.
#       Rejecting low `diff_vs_d1` prevents D16 from collapsing away.
#     - Baseline B = uniform launch: all tips use the seed task's clockwise flank and no
#       side step. Rejecting low `diff_vs_uniform` prevents D1 from collapsing away.
#     - Fill is kept in a mid-density band so the result remains a clean corridor system
#       rather than sparse stubs or a nearly solid rectangle.


ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.42, 0.56
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
MIN_SIDE_DIFF = 20
MIN_DIRECTION_DIFF = 20


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    return int(np.argmax(row_counts)), int(np.argmax(col_counts))


def _collect_arms(grid: np.ndarray, cr: int, cc: int) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))
    return arms


def _solve_with_launch_policy(
    grid: np.ndarray,
    launch_dir_for_outward,
    *,
    force_side_start: bool,
) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc = _find_cross_center(grid)
    arms = _collect_arms(grid, cr, cc)

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
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        initial_dir = launch_dir_for_outward(outward)
        phase = None
        if force_side_start:
            phase = "side_then_out" if len(cells) == 1 else "side_first"
        agents.append(
            {
                "cur": cells[-1],
                "dir": initial_dir,
                "initial_dir": initial_dir,
                "out_dir": outward,
                "phase": phase,
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "side_first":
            if can_move(agent["cur"], agent["initial_dir"], 1):
                do_move(agent, agent["initial_dir"], 1)
            agent["phase"] = None
            agent["dir"] = agent["initial_dir"]
            agent["p"] = 0
            return True

        if agent["phase"] == "side_then_out":
            if can_move(agent["cur"], agent["initial_dir"], 1):
                do_move(agent, agent["initial_dir"], 1)
            agent["phase"] = "out_after_side"
            agent["dir"] = agent["initial_dir"]
            agent["p"] = 0
            return True

        if agent["phase"] == "out_after_side":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1)
            agent["phase"] = None
            agent["dir"] = agent["initial_dir"]
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


def _launch_dir_uniform(outward):
    return _right(outward)


def _launch_dir_position_based(outward):
    return _right(outward) if outward[0] != 0 else _left(outward)


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _variant_metrics(inp: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    out = main(inp)
    d1_only = _solve_with_launch_policy(
        inp, _launch_dir_position_based, force_side_start=False
    )
    uniform = _solve_with_launch_policy(inp, _launch_dir_uniform, force_side_start=False)
    fill = float(np.mean(out == CROSS_COLOR))
    diff_vs_d1 = int(np.sum(out != d1_only))
    diff_vs_uniform = int(np.sum(out != uniform))
    return out, fill, diff_vs_d1, diff_vs_uniform


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four spaced spiral tracks from a hollow teal cross, using side-launch starts whose side depends on tip position.

    Input structure:
    - bg = 0 (black). The only foreground object is a teal (8) plus-sign with a one-cell hole at its center.

    Rule:
    - Top and bottom tips choose the clockwise side of their arm; left and right tips choose the mirrored side.
    - Each tip first paints one forced step along that side direction, i.e. sideways relative to the arm. If the arm length is 1, it then also takes one outward step before continuing.
    - After the launch offset, all paths use the same wall-following rule: prefer turn-right, then straight, then turn-left when not already on the border.
    - Straight moves are length 2 except immediately after a turn, when the move length is 1, and every new cell must keep a 1-cell orthogonal gap from existing teal cells except along the predecessor edge.

    Key insight: The fusion changes the launch geometry, not the later spacing engine. Position decides which flank each tip uses, and the forced lateral first step replaces the seed task's initial backward retreat, so the very first corridor near each tip becomes visibly rotated and asymmetric.
    """

    return _solve_with_launch_policy(
        grid, _launch_dir_position_based, force_side_start=True
    )


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        try:
            out, fill, diff_vs_d1, diff_vs_uniform = _variant_metrics(inp)
        except Exception:
            continue
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if diff_vs_d1 < MIN_SIDE_DIFF:
            continue
        if diff_vs_uniform < MIN_DIRECTION_DIFF:
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

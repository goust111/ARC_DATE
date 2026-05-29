# Combo variant of da515329 (index 1)
# combo_spec: D1/1+D12/0
# Name: 方向由端点位置决定 + 两组不等长臂

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
# combo_spec:
#   input:
#     - Keep the seed task's object vocabulary: one hollow teal cross on black.
#     - Apply D12/0 literally to the cross geometry: up/down arms share a vertical
#       length, left/right arms share a horizontal length, and those two lengths differ.
#       No extra color markers are introduced, so the arm-length asymmetry itself must
#       be the visible clue for the spatial asymmetry in the output.
#   concretization:
#     - Preserve the seed task's spacing engine: each tip grows one self-avoiding path
#       using 2-cell straight steps, 1-cell moves right after turns, and a mandatory
#       1-cell orthogonal gap from existing walls.
#     - Apply D1/1 as an endpoint-dependent launch-side rule:
#       vertical tips start on the clockwise flank of their arm (`right(outward)`),
#       while horizontal tips start on the mirrored flank (`left(outward)`).
#       The later move preference stays `turn right > straight > turn left`; changing
#       the initial flank is enough to create opposite visible winding families without
#       destabilizing the layout.
#   behavioral_trace:
#     - Small example: vertical arm = 3, horizontal arm = 1.
#       Seed task:
#         all four tips would begin from the clockwise side, and because every arm has
#         the same radius the first rectangular corridors are four-ways balanced.
#       Combo task:
#         the top tip starts three cells above the center and launches east; the left
#         tip starts only one cell left of the center and launches south. Those two
#         changes interact: the vertical tip's spiral enters a larger empty rectangle,
#         while the horizontal tip starts closer to the center and enters from the
#         mirrored side, so the occupied corridors in the left quadrants are not just
#         rotated copies of the top/bottom quadrants.
#       Visible consequence:
#         a solver must jointly infer `arm pair length -> launch radius / available
#         envelope` and `tip orientation -> launch side`.
#   complexity_calibration:
#     - The free variables are intentionally small and repeated:
#       only two arm-length categories (vertical vs horizontal) and two direction
#       categories (vertical tips vs horizontal tips).
#     - Every accepted sample shows both categories simultaneously, so 2-4 examples are
#       enough for a human to compare repeated evidence instead of guessing from a
#       one-off exception.
#   implicit_assumption_audit:
#     - Center detection by row/column argmax still holds: the center row contains
#       `2 * horizontal_arm` teal cells and every non-center row contains at most one;
#       similarly for the center column and the vertical arm.
#     - Arm collection along the four cardinal rays still holds because the input is
#       still exactly one cardinal cross with a center hole.
#     - The seed task's parity/border cleanups are removed. They were fitted to a fully
#       symmetric all-clockwise outer ring and assume a single scalar `arm_len`; after
#       mixing launch sides and allowing `vertical_arm != horizontal_arm`, those border
#       caps become unjustified and can erase valid asymmetry.
#     - The short-arm special case must remain per arm, not global: if only the
#       horizontal pair has length 1, only those two agents need the initial outward
#       nudge before joining the rectangular corridors.
#   placement_legality:
#     - The generator samples only a small catalog of unequal `(vertical_arm,
#       horizontal_arm)` pairs. This keeps the rule human-legible and avoids extreme
#       shapes that produce overly sparse outputs.
#     - Legal center ranges are computed separately for rows and columns:
#       the longer vertical arm constrains top/bottom clearance, while the longer
#       horizontal arm constrains left/right clearance.
#     - Each candidate is validated by running the solver and checking determinism.
#   observability_and_cleanliness:
#     - Generator rejection uses two visible Hamming distances:
#       `dir_diff` compares the combo output against the same unequal cross solved with
#       all-clockwise launches, proving the endpoint-dependent direction rule matters.
#       `arm_diff` compares against a symmetrized equal-arm cross solved by the combo
#       rule, proving the unequal arm lengths change more than just the center glyph.
#     - A mid-density fill band is enforced so accepted outputs look like structured
#       corridor systems rather than tiny dead-ends or near-solid blocks.
#   inferability_of_change:
#     - No hidden ordering depends on color ids or non-visual metadata.
#     - The only clues are visible geometry: one pair of arms is longer, and the four
#       resulting spiral families mirror differently by endpoint orientation.


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _find_cross(grid: np.ndarray) -> tuple[int, int, dict[tuple[int, int], int]]:
    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arm_lengths: dict[tuple[int, int], int] = {}
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        length = 0
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == 8:
            r += dr
            c += dc
            length += 1
        arm_lengths[d] = length
    return cr, cc, arm_lengths


def _solve_with_launch_policy(grid: np.ndarray, launch_dir_for_outward) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc, _ = _find_cross(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = cur[0] + d[0] * t, cur[1] + d[1] * t
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            out[r0 + d[0] * t, c0 + d[1] * t] = 8
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    arms = []
    for outward in _DIRS_URDL:
        dr, dc = outward
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == 8:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((outward, cells))

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
        for agent in agents:
            if step_agent(agent):
                changed = True
        if all(agent["done"] for agent in agents):
            break

    return out


def _combo_launch_dir(outward):
    return _right(outward) if outward[0] != 0 else _left(outward)


def _uniform_clockwise_launch_dir(outward):
    return _right(outward)


def _arm_pair_from_grid(grid: np.ndarray) -> tuple[int, int]:
    _, _, arm_lengths = _find_cross(grid)
    vertical_arm = arm_lengths[(-1, 0)]
    horizontal_arm = arm_lengths[(0, 1)]
    return vertical_arm, horizontal_arm


def _build_input(vertical_arm: int, horizontal_arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for t in range(1, vertical_arm + 1):
        grid[cr - t, cc] = 8
        grid[cr + t, cc] = 8
    for t in range(1, horizontal_arm + 1):
        grid[cr, cc - t] = 8
        grid[cr, cc + t] = 8
    return grid


def _symmetrized_input(grid: np.ndarray) -> np.ndarray:
    vertical_arm, horizontal_arm = _arm_pair_from_grid(grid)
    cr, cc, _ = _find_cross(grid)
    sym_arm = max(vertical_arm, horizontal_arm)
    return _build_input(sym_arm, sym_arm, grid.shape[0], cr, cc)


def _variant_metrics(inp: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    out = main(inp)
    dir_base = _solve_with_launch_policy(inp, _uniform_clockwise_launch_dir)
    dir_diff = int(np.sum(out != dir_base))
    arm_base = main(_symmetrized_input(inp))
    arm_diff = int(np.sum(out != arm_base))
    fill = float(np.mean(out == 8))
    return out, fill, dir_diff, arm_diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four endpoint-conditioned spiral walls from an asymmetric teal cross.

    Input structure:
    - bg = 0 (black). A single teal (8) cross has a one-cell empty center.
    - The vertical arm pair and horizontal arm pair are each internally symmetric, but the two pair lengths differ.

    Rule:
    - Each arm tip emits one path.
    - Top and bottom tips launch from the clockwise flank of their arm; left and right tips launch from the mirrored flank.
    - After launch, every path follows the same wall-following rule: turn right > go straight > turn left, with left turns allowed only away from the border.
    - Straight runs normally advance 2 cells, the first move after a turn advances 1 cell, and every new cell must keep a 1-cell orthogonal gap from existing walls except along its predecessor edge.

    Key insight: Two mappings must be fused at once: arm-pair length sets the starting radius and available corridor space, while endpoint orientation decides which mirrored spiral family that tip belongs to.
    """

    return _solve_with_launch_policy(np.asarray(grid, dtype=int), _combo_launch_dir)


# --------------- generator (combo D1/1 + D12/0) ---------------

ARM_PAIRS = [(1, 3), (3, 1), (2, 3), (3, 2)]
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.40, 0.54
MIN_DIR_DIFF_BASE = 12
MIN_ARM_DIFF_BASE = 20
MAX_OUTER_RETRIES = 400


def _grid_size_for_pair(vertical_arm: int, horizontal_arm: int) -> int:
    max_arm = max(vertical_arm, horizontal_arm)
    return 4 * max_arm + 2 * math.ceil(max_arm / 2) + GRID_SIZE_BASE


def _candidate_centers(size: int, vertical_arm: int, horizontal_arm: int) -> list[tuple[int, int]]:
    max_arm = max(vertical_arm, horizontal_arm)
    row_lo = vertical_arm + max_arm + TIP_GAP_MIN
    row_hi = size - (vertical_arm + max_arm + TIP_GAP_MIN) - 1
    col_lo = horizontal_arm + max_arm + TIP_GAP_MIN
    col_hi = size - (horizontal_arm + max_arm + TIP_GAP_MIN) - 1
    if row_lo > row_hi or col_lo > col_hi:
        return []
    centers = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(centers)
    return centers


def _gen_attempt() -> np.ndarray | None:
    vertical_arm, horizontal_arm = random.choice(ARM_PAIRS)
    size = _grid_size_for_pair(vertical_arm, horizontal_arm)
    min_dir_diff = max(MIN_DIR_DIFF_BASE, size // 2)
    min_arm_diff = max(MIN_ARM_DIFF_BASE, size)

    for cr, cc in _candidate_centers(size, vertical_arm, horizontal_arm):
        inp = _build_input(vertical_arm, horizontal_arm, size, cr, cc)
        try:
            out, fill, dir_diff, arm_diff = _variant_metrics(inp)
        except Exception:
            continue
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if dir_diff < min_dir_diff:
            continue
        if arm_diff < min_arm_diff:
            continue
        if np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")

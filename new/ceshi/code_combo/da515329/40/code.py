# Combo variant of da515329 (index 40)
# combo_spec: D6/3+D11/3
# Name: 反射后继续多遍历 + 反弹路径

import math
import random

import numpy as np


_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8


# Design analysis for the fused variant
# ------------------------------------
# 1) Concrete behavior trace:
#    Example: on a 16x16 grid with arm_len=2 and top tip at (5, 8), the original
#    solver would start east, then keep choosing right/straight/left turns to form
#    a global spiral. In this combo, that same tip still launches east (clockwise
#    from "up"), but it no longer turns opportunistically. It sweeps straight to
#    the right border, reflects, climbs two cells into the next inner lane, sweeps
#    back toward the cross-side wall, reflects again, and repeats. The path is
#    therefore bounce-driven (D6/3), and every reflection also starts another
#    inward pass on a parallel lane (D11/3).
# 2) Complexity calibration:
#    A pure single-pass bounce would look like four simple zigzags and would be
#    too weak visually. The added "advance two cells into the next lane after each
#    boundary hit" creates nested repeated traversals, which is visible in 2-4
#    examples without introducing many free parameters.
# 3) Hidden assumptions broken from the original code:
#    - The original `can_move/spacing_ok` logic assumes local self-avoidance grows
#      the spiral correctly. That premise is false after switching to reflected
#      sweeps, so the new solver uses explicit lane geometry instead.
#    - The original parity cleanup and border nubs are spiral-specific artifacts.
#      They are removed because the bounce construction already determines exactly
#      which border contacts should exist.
#    - The new solver assumes each of the four cardinal arms is present and that
#      the tip-to-border gaps are large enough for at least several reflected
#      passes to be visible. The generator enforces this with `MIN_TIP_GAP`.
# 4) Placement legality and visual cleanliness:
#    Each tip owns its outer quadrant only: top->north-east, right->south-east,
#    bottom->south-west, left->north-west. That keeps trajectories disjoint except
#    for touching their launch tip, so the output does not become visually muddy.
# 5) Human observability:
#    A solver can infer the rule from examples because the cross still marks the
#    four launch sites, the initial heading stays "clockwise from outward", and
#    every output shows the same two visible cues: border rebounds and repeated
#    inward lane shifts.


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    return int(np.argmax(row_counts)), int(np.argmax(col_counts))


def _collect_arms(grid: np.ndarray, cr: int, cc: int) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for dr, dc in _DIRS_URDL:
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while inb(r + dr, c + dc) and int(grid[r + dr, c + dc]) == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((int(r), int(c)))
        if cells:
            arms.append(((int(dr), int(dc)), cells))
    return arms


def _trace_horizontal_bounce(
    out: np.ndarray,
    start_r: int,
    start_c: int,
    outer_c: int,
    inner_c: int,
    slow_step: int,
) -> None:
    h, _w = out.shape
    r, c = int(start_r), int(start_c)
    move_to_outer = outer_c > inner_c

    while True:
        target = outer_c if move_to_outer else inner_c
        step = 1 if target > c else -1
        while c != target:
            c += step
            out[r, c] = CROSS_COLOR

        next_r = r + 2 * slow_step
        if not (0 <= next_r < h):
            break

        out[r + slow_step, c] = CROSS_COLOR
        out[next_r, c] = CROSS_COLOR
        r = next_r
        move_to_outer = not move_to_outer


def _trace_vertical_bounce(
    out: np.ndarray,
    start_r: int,
    start_c: int,
    outer_r: int,
    inner_r: int,
    slow_step: int,
) -> None:
    _h, w = out.shape
    r, c = int(start_r), int(start_c)
    move_to_outer = outer_r > inner_r

    while True:
        target = outer_r if move_to_outer else inner_r
        step = 1 if target > r else -1
        while r != target:
            r += step
            out[r, c] = CROSS_COLOR

        next_c = c + 2 * slow_step
        if not (0 <= next_c < w):
            break

        out[r, c + slow_step] = CROSS_COLOR
        out[r, next_c] = CROSS_COLOR
        c = next_c
        move_to_outer = not move_to_outer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four reflected sweep paths from the tips of a teal cross, and let each
    rebound spawn another inward traversal.

    Input structure:
    - bg = 0 (black). A single teal (8) plus-sign cross with a black center hole.
    - All four arms have the same length and sit far enough from the borders to
      leave visible outer quadrants for the reflected passes.

    Rule:
    - Each arm tip starts in the same clockwise tangent direction as the original
      task: top->east, right->south, bottom->west, left->north.
    - The tip only fills its outer quadrant. It runs straight to the quadrant's
      border, reflects, then moves two cells deeper into the quadrant before the
      next sweep. Repeating this creates parallel rebound lanes with 1-cell gaps.
    - Every arm follows the same mechanism, rotated to match its orientation.

    Key insight: The combo is not "spiral plus bounce" as two separate edits; the
    bounce itself is what advances the pattern into successive inner lanes, so one
    reflection both reverses the current sweep and triggers the next traversal.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc = _find_cross_center(grid)
    arms = _collect_arms(grid, cr, cc)

    for outward, cells in arms:
        tip_r, tip_c = cells[-1]

        if outward == (-1, 0):
            _trace_horizontal_bounce(
                out,
                start_r=tip_r,
                start_c=tip_c,
                outer_c=w - 1,
                inner_c=cc + 1,
                slow_step=-1,
            )
        elif outward == (0, 1):
            _trace_vertical_bounce(
                out,
                start_r=tip_r,
                start_c=tip_c,
                outer_r=h - 1,
                inner_r=cr + 1,
                slow_step=1,
            )
        elif outward == (1, 0):
            _trace_horizontal_bounce(
                out,
                start_r=tip_r,
                start_c=tip_c,
                outer_c=0,
                inner_c=cc - 1,
                slow_step=1,
            )
        elif outward == (0, -1):
            _trace_vertical_bounce(
                out,
                start_r=tip_r,
                start_c=tip_c,
                outer_r=0,
                inner_r=cr - 1,
                slow_step=-1,
            )

    return out


# --------------- generator (cx) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 8
GRID_SIZE_BONUS = (0, 2, 4)
MIN_TIP_GAP = 4
FILL_LO, FILL_HI = 0.40, 0.56
MIN_TOTAL_LANE_SHIFTS = 8
MIN_BORDER_CELLS = 14
MAX_OUTER_RETRIES = 300


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _tip_gaps(arm: int, size: int, cr: int, cc: int) -> tuple[int, int, int, int]:
    top = cr - arm
    right = (size - 1) - (cc + arm)
    bottom = (size - 1) - (cr + arm)
    left = cc - arm
    return int(top), int(right), int(bottom), int(left)


def _behavior_metrics(arm: int, size: int, cr: int, cc: int, out: np.ndarray) -> tuple[int, int]:
    gaps = _tip_gaps(arm, size, cr, cc)
    lane_shifts = int(sum(g // 2 for g in gaps))

    border_cells = int(np.sum(out[0] == CROSS_COLOR))
    border_cells += int(np.sum(out[-1] == CROSS_COLOR))
    border_cells += int(np.sum(out[:, 0] == CROSS_COLOR))
    border_cells += int(np.sum(out[:, -1] == CROSS_COLOR))
    border_cells -= int(out[0, 0] == CROSS_COLOR)
    border_cells -= int(out[0, -1] == CROSS_COLOR)
    border_cells -= int(out[-1, 0] == CROSS_COLOR)
    border_cells -= int(out[-1, -1] == CROSS_COLOR)
    return lane_shifts, border_cells


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE + random.choice(GRID_SIZE_BONUS)

    lo = arm + MIN_TIP_GAP
    hi = size - arm - MIN_TIP_GAP - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        fill = float(np.mean(out == CROSS_COLOR))
        lane_shifts, border_cells = _behavior_metrics(arm, size, cr, cc, out)
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if lane_shifts < MIN_TOTAL_LANE_SHIFTS or border_cells < MIN_BORDER_CELLS:
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

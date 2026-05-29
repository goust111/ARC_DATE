# Combo variant of da515329 (index 10)
# combo_spec: D5/2+D7/3
# Name: 单端点源 + 螺旋长度由臂长编码

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8

# Design analysis for the fused variant
# a) Concrete behavior:
#    With arm=2, size=16, center=(6,7), the four tip-to-edge distances are
#    top=4, right=6, bottom=7, left=5. Only the top tip is active, so the old
#    four-agent kaleidoscope collapses into one visible spiral attached to that
#    tip. The spiral still follows the original wall-following motion, but it
#    now halts once its trajectory reaches arm*size = 32 pixels. We count the
#    source tip as pixel 1, so this example paints 31 new cells beyond the
#    input cross.
# b) Complexity calibration:
#    The visual change is large enough to observe from a few examples: three
#    tips stay idle, one tip grows a long spiral, and the spiral length scales
#    with both arm length and board size. The only branching factor is which of
#    the four tips is uniquely closest to a border, which remains easy to infer.
# c) Assumption audit:
#    The original solver assumed four simultaneous agents, parity cleanup, and
#    near-50% fill. None of those assumptions survives the fusion. The retained
#    assumptions are only: one off-center hollow cross exists, all arms are
#    equal, and movement keeps a 1-cell orthogonal gap. The final segment may be
#    truncated to length 1 solely to hit the exact target count.
# d) Placement legality:
#    The generator keeps the original safe center band so the cross never leaves
#    the board, then filters to centers whose nearest tip is unique. For the
#    original size formula, every such center leaves enough room for the single
#    spiral to reach the encoded target exactly.
# e) Visual observability:
#    Humans can identify the active source by geometry alone: the chosen tip is
#    visibly nearest to one border. The output contains exactly one path-like
#    extension, so the source rule is not hidden behind symmetric clutter.
# f) Visual cleanliness:
#    Reusing the original spacing rule avoids overlaps and preserves a single
#    tidy self-avoiding path. Removing the old post-processing also avoids
#    border artifacts that only made sense for four-source full tilings.
# g) Inferability:
#    No invisible color ordering is used. The only control signals are arm
#    length, grid side length, and geometric proximity to borders, all directly
#    readable from the input.


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _add(pos, d, k=1):
    return pos[0] + d[0] * k, pos[1] + d[1] * k


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _extract_cross(grid: np.ndarray):
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while _inb((h, w), r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append({"outward": d, "cells": cells, "tip": cells[-1]})

    arm_len = max((len(arm["cells"]) for arm in arms), default=0)
    return cr, cc, arm_len, arms


def _nearest_border_distance(shape: tuple[int, int], cell: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    return min(r, h - 1 - r, c, w - 1 - c)


def _pick_active_arm(shape: tuple[int, int], arms: list[dict]):
    scored = []
    for idx, arm in enumerate(arms):
        dist = _nearest_border_distance(shape, arm["tip"])
        scored.append((dist, idx, arm))
    scored.sort(key=lambda item: (item[0], item[1]))
    active_dist, active_idx, active_arm = scored[0]
    unique = len(scored) == 1 or active_dist < scored[1][0]
    margin = None if len(scored) == 1 else scored[1][0] - active_dist
    return active_arm, active_idx, unique, margin


def _run_single_spiral(grid: np.ndarray, arm_len: int, active_arm: dict) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    target = arm_len * h

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if _inb((h, w), nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = _add(cur, d, t)
            if not _inb((h, w), rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(cur, d, length: int):
        r0, c0 = cur
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = CROSS_COLOR
        return r0 + d[0] * length, c0 + d[1] * length

    agent = {
        "cur": active_arm["tip"],
        "dir": _right(active_arm["outward"]),
        "out_dir": active_arm["outward"],
        "phase": "out_first" if arm_len == 1 else None,
        "p": 0,
    }

    painted = 1
    guard = 0
    while painted < target and guard < 200000:
        guard += 1

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                agent["cur"] = do_move(agent["cur"], agent["out_dir"], 1)
                painted += 1
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            continue

        cur = agent["cur"]
        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
            dirs.append(_left(agent["dir"]))

        moved = False
        for nd in dirs:
            turned = nd != agent["dir"]
            desired = 1 if turned else (2 if agent["p"] == 0 else 1)
            remaining = target - painted
            first_try = min(desired, remaining)

            if first_try > 0 and can_move(agent["cur"], nd, first_try):
                old_p = agent["p"]
                agent["cur"] = do_move(agent["cur"], nd, first_try)
                painted += first_try
                agent["dir"] = nd
                agent["p"] = 1 if turned and old_p == 0 else 0
                moved = True
                break

            if not turned and desired == 2 and first_try == 2 and remaining >= 1 and can_move(agent["cur"], nd, 1):
                agent["cur"] = do_move(agent["cur"], nd, 1)
                painted += 1
                agent["p"] = 0
                moved = True
                break

        if not moved:
            break

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow one clockwise spiral from the cross tip that sits closest to a border, then stop at an encoded path length.

    Input structure:
    - Background is black (0). A single teal (8) hollow plus sign appears off-center.
    - The four arms have equal length, so the only asymmetry is how close each tip is to the surrounding borders.

    Rule:
    - Measure each arm tip's distance to its nearest grid edge and activate only the nearest tip.
    - The active tip begins with the same clockwise wall-following dynamics as the base task: start 90 degrees clockwise from the outward arm direction, prefer right > straight > left, use straight steps of 2 unless just turned, and keep a 1-cell orthogonal gap from other teal cells.
    - Count the source tip as the first spiral pixel, and stop as soon as the full trajectory length reaches arm_length * grid_side.

    Key insight:
    - The input encodes two things at once: off-center placement selects which tip emits, and arm length determines the exact stopping count instead of letting the spiral tile the entire board.
    """

    grid = np.asarray(grid, dtype=int)
    _cr, _cc, arm_len, arms = _extract_cross(grid)
    if arm_len == 0 or not arms:
        return grid.copy()

    active_arm, _active_idx, _unique, _margin = _pick_active_arm(grid.shape, arms)
    return _run_single_spiral(grid, arm_len, active_arm)


# --------------- generator (combo) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
MAX_OUTER_RETRIES = 300


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _source_signature(size: int, cr: int, cc: int, arm: int):
    tips = [
        (cr - arm, cc),
        (cr, cc + arm),
        (cr + arm, cc),
        (cr, cc - arm),
    ]
    dists = [min(r, size - 1 - r, c, size - 1 - c) for r, c in tips]
    ranking = sorted((dist, idx) for idx, dist in enumerate(dists))
    source_idx = ranking[0][1]
    unique = len(ranking) == 1 or ranking[0][0] < ranking[1][0]
    margin = None if len(ranking) == 1 else ranking[1][0] - ranking[0][0]
    return source_idx, dists[source_idx], unique, margin


def _expected_output_pixels(arm: int, size: int) -> int:
    return 4 * arm + arm * size - 1


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    desired_source = random.randrange(4)
    candidates = []
    for cr in range(lo, hi + 1):
        for cc in range(lo, hi + 1):
            source_idx, _dist, unique, _margin = _source_signature(size, cr, cc, arm)
            if unique and source_idx == desired_source:
                candidates.append((cr, cc))
    if not candidates:
        for cr in range(lo, hi + 1):
            for cc in range(lo, hi + 1):
                source_idx, _dist, unique, _margin = _source_signature(size, cr, cc, arm)
                if unique:
                    candidates.append((cr, cc))
    random.shuffle(candidates)

    expected_pixels = _expected_output_pixels(arm, size)
    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        if int(np.count_nonzero(out == CROSS_COLOR)) != expected_pixels:
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

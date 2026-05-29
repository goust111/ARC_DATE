# Combo variant of da515329 (index 26)
# combo_spec: D3/1+D18/1
# Name: 斐波那契段长 + 每圈递增颜色

import math
import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
_ARC_UTILS_DIR = os.path.join(_ROOT, "arc2_solver_select")
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_FIB = [1, 1]
for _ in range(18):
    _FIB.append(_FIB[-1] + _FIB[-2])


# Design analysis for combo D3/1 + D18/1
# ----------------------------------------
# a) Concretization and behavior trace
# - The input stays a single hollow teal cross on black. The output is still made by
#   four clockwise self-avoiding spiral walkers starting from the four arm tips.
# - Original behavior: after each corner, the walker used a local 1-cell connector,
#   then kept extending with the fixed 2/1 alternation. This made outer shells grow
#   almost uniformly.
# - Combo behavior: keep the 1-cell corner connectors and right-turn preference, but
#   replace the major straight-run schedule by Fibonacci lengths 1, 1, 2, 3, 5, 8...
#   The first long run from every arm is length 1, the second is also 1, the third is
#   2, then 3, then 5, then 8, until the border/gap rule truncates the last few runs.
# - Coloring is integrated during construction instead of after the fact. One visible
#   colored shell consists of two consecutive major runs (typically one horizontal and
#   one vertical leg), so shell 1 carries the 1+1 pair, shell 2 carries the 2+3 pair,
#   shell 3 carries the 5+8 pair, etc. This makes the non-uniform growth directly
#   visible as increasingly wider colored spiral bands.
# - Concrete small trace for arm_len=3: the cross itself is recolored to 1. Each arm
#   then starts shell 1 by turning into the spiral and realizing straight lengths 1,1.
#   The next shell uses color 2 while the straight lengths become 2 then 3. The third
#   shell uses color 3 while lengths become 5 then 8. Compared with the base task, the
#   occupied geometry is no longer close to uniform concentric spacing, and compared
#   with the pure D3 variant, ring color is no longer a simple radius bucket.
#
# b) Complexity calibration
# - If the grid is too small or the cross is too short, the Fibonacci rule collapses
#   into just a few tiny shells and becomes visually indistinguishable from a generic
#   colored spiral. To avoid that, the generator only uses arm_len=3, which yields a
#   22x22 board where every accepted sample exhibits at least the full 1,1,2,3,5,8
#   prefix on all four walkers and at least 5 visible color shells.
# - This keeps the variant legible without introducing too many degrees of freedom.
#
# c) Implicit assumption audit
# - Any occupancy test from the original solver that relied on value 8 must become a
#   boolean wall mask, because output colors now vary by shell. `occ` stores geometry;
#   `out` stores colors.
# - The base task's parity/border cleanup was learned for the specific 2/1-growth
#   regime. Under Fibonacci growth that cleanup is no longer semantically justified and
#   can destroy valid colored shells, so it is removed entirely.
# - The solver assumes the input is exactly one hollow-centered plus with four equal
#   teal arms. The generator constructs only such inputs.
# - The solver also assumes the desired shell count fits ARC colors. The accepted
#   samples top out at color 6, so there is no color overflow.
#
# d) Placement legality
# - The generator keeps the original size formula for arm_len=3, so the board is 22x22.
# - Candidate centers are restricted so every tip stays at least arm+2 from each border,
#   preventing immediate border collisions and leaving room for the 1,1,2,3,5,8 prefix.
#
# e) Visual observability
# - The human-visible cue is not hidden state: longer and longer same-colored spiral
#   bands appear outward, and those band jumps repeat symmetrically in all four arms.
# - Because every accepted sample realizes the same Fibonacci prefix on all four arms,
#   the recurrence has repeated support inside a single grid, not just across examples.
#
# f) Visual cleanliness
# - Only one family of objects exists: the spiral walls themselves. There are no extra
#   markers, legends, or overlays. Colors are reserved exclusively for shell identity.
# - The 1-cell orthogonal gap rule is preserved, so adjacent shells remain separable.
#
# g) Inference of the color change
# - Colors increase monotonically from center to outer shells. The mapping is structural:
#   shell index -> color index. The solver never uses hidden IDs or non-visible ordering
#   unrelated to geometry. The generator rejects shallow cases where this layering would
#   be too weak to infer.


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = 8
    grid[cr + offsets, cc] = 8
    grid[cr, cc - offsets] = 8
    grid[cr, cc + offsets] = 8
    return grid


def _extract_cross(grid: np.ndarray):
    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

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

    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def _solve_variant(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    occ = grid == 8
    out = np.zeros_like(grid)
    out[occ] = 1

    cr, cc, arms, arm_len = _extract_cross(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and occ[nr, nc] and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or occ[rr, cc2]:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def best_len(cur, d, target: int) -> int:
        for length in range(target, 0, -1):
            if can_move(cur, d, length):
                return length
        return 0

    def shell_color(seg_idx: int) -> int:
        return 1 + seg_idx // 2

    def do_move(agent, d, length: int, color: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            occ[rr, cc2] = True
            out[rr, cc2] = color
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "seg_idx": 0,
                "straight_lengths": [],
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            got = best_len(agent["cur"], agent["out_dir"], 1)
            if got:
                do_move(agent, agent["out_dir"], got, 1)
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            return True

        cur = agent["cur"]
        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
            dirs.append(_left(agent["dir"]))

        color = shell_color(agent["seg_idx"])
        for nd in dirs:
            turned = nd != agent["dir"]
            target = 1 if turned else _FIB[agent["seg_idx"]]
            got = best_len(agent["cur"], nd, target)
            if got == 0:
                continue
            do_move(agent, nd, got, color)
            agent["dir"] = nd
            if not turned:
                agent["straight_lengths"].append(got)
                agent["seg_idx"] += 1
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

    prefix_lengths = []
    for ag in agents:
        prefix = 0
        for actual, expected in zip(ag["straight_lengths"], _FIB):
            if actual != expected:
                break
            prefix += 1
        prefix_lengths.append(prefix)

    meta = {
        "arm_len": arm_len,
        "fill": float(np.mean(out != 0)),
        "max_color": int(out.max(initial=0)),
        "straight_lengths": [tuple(ag["straight_lengths"]) for ag in agents],
        "prefix_lengths": tuple(prefix_lengths),
        "min_prefix": min(prefix_lengths, default=0),
        "max_prefix": max(prefix_lengths, default=0),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral walls from a hollow teal cross, but let shell growth follow Fibonacci-sized straight runs and color shells outward from 1 upward.

    Input structure:
    - Background is black (0).
    - A single teal (8) plus-sign cross with a missing center cell is the only object.
    - All four arms have equal length and serve as the four spiral launch tips.

    Rule:
    - Each tip starts a clockwise wall-walker whose initial heading is the arm's clockwise quarter-turn.
    - Walkers still obey the self-avoiding gap rule and prefer directions in the order right turn > straight > left.
    - Corner connectors stay 1 cell long, but the major straight runs now follow Fibonacci lengths 1, 1, 2, 3, 5, 8, ...
    - Every visible shell shares one color: shell 1 is color 1, shell 2 is color 2, shell 3 is color 3, and so on.

    Key insight: The task is no longer "uniform spiral then recolor by radius". The color boundaries are tied to the same construction process that uses Fibonacci straight-run lengths, so the widening shell bands reveal the recurrence.
    """

    out, _meta = _solve_variant(grid)
    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN = 3
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.47, 0.53
MIN_VISIBLE_SHELLS = 5
MIN_FIB_PREFIX = 6
MAX_OUTER_RETRIES = 300
# --- end CFG -----------------------------------------------------------


def _candidate_centers(size: int, arm: int):
    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    return [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]


def _accept(inp: np.ndarray) -> bool:
    out, meta = _solve_variant(inp)
    if not (FILL_LO <= meta["fill"] <= FILL_HI):
        return False
    if meta["max_color"] < MIN_VISIBLE_SHELLS:
        return False
    if meta["min_prefix"] < MIN_FIB_PREFIX:
        return False
    return np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    arm = ARM_LEN
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE
    candidates = _candidate_centers(size, arm)
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        if _accept(inp):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("generate_input: failed after retries")

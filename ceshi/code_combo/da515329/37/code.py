# Combo variant of da515329 (index 37)
# combo_spec: D2/3+D12/0
# Name: GCD间距 + 两组不等长臂

import math
import random

import numpy as np


_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8


# Design analysis for the fused variant
# a) Concrete behavior:
#    Example: on a 16x16 grid with vertical arm 4, horizontal arm 1 and center
#    at (6, 8), the center-to-edge distances are [6, 9, 8, 7].  The new gap is
#    gcd(min, max) = gcd(6, 9) = 3, so the original 2/1 straight schedule becomes
#    4/3 while turn-steps remain 1.  The cross is visibly taller than wide in the
#    input, and the output spirals leave larger corridors than the base task.
# b) Complexity calibration:
#    Only three new degrees of freedom matter: vertical arm length, horizontal arm
#    length, and the derived gcd gap.  Opposite arms still match each other, so
#    the input remains easy to parse and the output still looks like one coherent
#    spiral system instead of many unrelated rules.
# c) Hidden assumptions reviewed:
#    - The original single arm_len parity cleanup no longer applies, because top/
#      bottom and left/right arms may differ; it is removed.
#    - Center detection by max row/column counts still works, because the center
#      row contains all horizontal-arm cells and the center column contains all
#      vertical-arm cells.
#    - Fixed 2/1 movement lengths are replaced by gap-aware lengths derived from
#      the center distances, so generator must guarantee gap >= 2 to avoid
#      degenerating into the original fixed-spacing regime.
# d) Placement legality:
#    The generator uses separate row/column margins based on vertical/horizontal
#    arm lengths so every arm fits, no tips cross the border, and the center can
#    still achieve the target gcd gap.
# e) Visual observability:
#    Unequal arm groups are immediately visible in the input.  The gcd-derived gap
#    is visible in the output as wider or narrower rectangular corridors.
# f) Visual cleanliness:
#    Only one color is used and all output pixels remain path-like; no extra
#    overlays or markers are introduced.
# g) Inferability:
#    The generator rejects samples whose output is too close to a fixed-gap=1
#    baseline.  This makes the gcd spacing effect show up consistently across
#    examples instead of being a latent parameter with no visible consequence.


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _detect_center_and_arms(grid: np.ndarray):
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
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
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if not cells:
            raise ValueError("input must contain a four-armed teal cross")
        arms.append((d, cells))
    return cr, cc, arms


def _solve_core(grid: np.ndarray, forced_gap: int | None = None) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arms = _detect_center_and_arms(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    dists = [cr, h - 1 - cr, cc, w - 1 - cc]
    gap = max(1, math.gcd(min(dists), max(dists))) if forced_gap is None else forced_gap

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in ((r - 1, c), (r, c + 1), (r + 1, c), (r, c - 1)):
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
            out[r0 + d[0] * t, c0 + d[1] * t] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    # New hidden assumption made explicit:
    # each tip still launches a clockwise agent, but straight lengths now depend
    # on gap.  A normal straight move has length gap+1, while the first straight
    # after a turn has length gap.  Turn moves remain length 1.
    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
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
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (gap + 1 if agent["p"] == 0 else gap)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length > 1 and can_move(agent["cur"], nd, 1):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise rectangular spirals from an asymmetric teal cross.

    Input structure:
    - bg = 0 (black).  A single teal (8) cross with a center hole.
    - Top/bottom arms match each other; left/right arms match each other.
    - The vertical and horizontal arm lengths differ, so the cross is taller or
      wider instead of fully four-way symmetric.

    Rule:
    - Find the cross center from the densest teal row and column.
    - Compute gap = gcd(min(center-to-edge distances), max(center-to-edge distances)).
    - Each arm tip launches a clockwise turtle whose turn moves stay length 1.
    - Straight motion uses the gap-aware schedule: normal straight = gap + 1,
      first straight after a turn = gap; priority stays right > straight > left.

    Key insight:
    The unequal arm groups change which center placements are legal, and that in
    turn changes the gcd of the center distances.  The solver therefore has to
    read both the cross asymmetry and the center placement to recover the spiral
    spacing.
    """
    return _solve_core(grid)


# --------------- generator (cx) ---------------

ARM_MIN, ARM_MAX = 1, 4
SIZE_MIN, SIZE_MAX = 15, 27
CENTER_MARGIN = 2
GAP_MIN, GAP_MAX = 2, 5
FILL_LO, FILL_HI = 0.28, 0.52
DIVERGENCE_LO = 0.08
MAX_OUTER_RETRIES = 400


def _build_input(v_arm: int, h_arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for k in range(1, v_arm + 1):
        grid[cr - k, cc] = CROSS_COLOR
        grid[cr + k, cc] = CROSS_COLOR
    for k in range(1, h_arm + 1):
        grid[cr, cc - k] = CROSS_COLOR
        grid[cr, cc + k] = CROSS_COLOR
    return grid


def _gap_from_center(size: int, cr: int, cc: int) -> int:
    dists = [cr, size - 1 - cr, cc, size - 1 - cc]
    return math.gcd(min(dists), max(dists))


def _candidate_centers(size: int, v_arm: int, h_arm: int):
    lo_r = v_arm + CENTER_MARGIN
    hi_r = size - v_arm - CENTER_MARGIN - 1
    lo_c = h_arm + CENTER_MARGIN
    hi_c = size - h_arm - CENTER_MARGIN - 1
    if lo_r > hi_r or lo_c > hi_c:
        return []

    centers = []
    for cr in range(lo_r, hi_r + 1):
        for cc in range(lo_c, hi_c + 1):
            gap = _gap_from_center(size, cr, cc)
            if GAP_MIN <= gap <= GAP_MAX:
                centers.append((cr, cc, gap))
    random.shuffle(centers)
    return centers


def _difference_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a != b))


def _gen_attempt() -> np.ndarray | None:
    v_arm = random.randint(ARM_MIN, ARM_MAX)
    h_arm = random.randint(ARM_MIN, ARM_MAX)
    if v_arm == h_arm:
        return None

    size = random.randint(SIZE_MIN, SIZE_MAX)
    candidates = _candidate_centers(size, v_arm, h_arm)
    if not candidates:
        return None

    for cr, cc, gap in candidates:
        inp = _build_input(v_arm, h_arm, size, cr, cc)
        try:
            out = main(inp)
            baseline = _solve_core(inp, forced_gap=1)
        except Exception:
            continue

        fill = float(np.mean(out == CROSS_COLOR))
        divergence = _difference_ratio(out, baseline)

        # This is the anti-degeneration check for the fusion:
        # - arm groups must differ in the input
        # - gcd gap must be active (gap >= 2)
        # - output must visibly differ from the fixed-gap baseline
        if FILL_LO <= fill <= FILL_HI and divergence >= DIVERGENCE_LO:
            if np.array_equal(main(inp), out):
                return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

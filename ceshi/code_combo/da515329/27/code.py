# Combo variant of da515329 (index 27)
# combo_spec: D2/2+D18/2
# Name: 段长递增步幅由臂长决定 + 间距由臂长编码

import random

import numpy as np


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Let arm_len = 3. The input is still a single teal cross with a center hole.
#    In the original task, a spiral lane sits 1 black cell away from the previous
#    parallel lane, and its rectangular envelope grows in +2 increments. Here the
#    same arm length controls both parameters:
#    - each tip first steps 1 cell farther outward, so the first parallel lane is
#      3 black cells away from the cross;
#    - after that, turn-segments are length 3 and straight segments are length 6,
#      so every new rectangular layer expands by 6 cells instead of 2.
#    Example from the top tip:
#    original arm=3 would begin with a compact gap-1 spiral near the cross;
#    fused arm=3 goes from (cr-3, cc) -> (cr-4, cc) -> 6 cells east, so the first
#    visible lane is much farther away and the next layer is correspondingly larger.
# b) Complexity calibration:
#    The generator uses arm lengths 2..4. Arm=1 would collapse back to something
#    very close to the seed task, which would hide the fusion. Using 2/3/4 gives
#    three visibly distinct gap/increment settings while keeping the cross as the
#    only input object.
# c) Hidden-assumption audit:
#    The original code assumes a single equal-arm cross and relies on local
#    avoidance to infer the spiral. This rewrite keeps the same object assumption,
#    but the local validity test now depends on the movement direction: every new
#    cell must keep an arm_len-wide perpendicular corridor to existing walls.
#    The "all arms take one outward step first" rule is also new; without it, the
#    first parallel lane would sit too close to the cross when arm_len > 1.
# d) Placement legality:
#    The generator constrains the cross center to [2*arm, size-1-2*arm] in both
#    axes. That guarantees each of the four first straight legs can realize its
#    full 2*arm length without leaving the grid.
# e/f/g) Observability and visual cleanliness:
#    The task remains monochrome, so the rule must be readable from geometry
#    alone. The accepted samples are filtered to ensure two measurable signatures:
#    the outward launch gap equals arm_len and the first straight leg equals
#    2*arm_len. This keeps the fused behavior visible in just a few examples and
#    avoids clutter from heavily truncated first segments.


_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _extract_cross(grid: np.ndarray):
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    for d in _DIRS_URDL:
        r, c = cr, cc
        cells = []
        while inb(r + d[0], c + d[1]) and grid[r + d[0], c + d[1]] == CROSS_COLOR:
            r += d[0]
            c += d[1]
            cells.append((r, c))
        if not cells:
            return cr, cc, []
        arms.append((d, cells))
    return cr, cc, arms


def _measure_launch_signature(inp: np.ndarray, out: np.ndarray):
    cr, cc, arms = _extract_cross(inp)
    if len(arms) != 4:
        return None
    arm = max(len(cells) for _, cells in arms)
    h, w = out.shape

    top_tip = cr - arm
    bottom_tip = cr + arm
    left_tip = cc - arm
    right_tip = cc + arm

    top_r = None
    for r in range(top_tip - 1, -1, -1):
        if out[r, cc] == CROSS_COLOR:
            top_r = r
            break
    bottom_r = None
    for r in range(bottom_tip + 1, h):
        if out[r, cc] == CROSS_COLOR:
            bottom_r = r
            break
    left_c = None
    for c in range(left_tip - 1, -1, -1):
        if out[cr, c] == CROSS_COLOR:
            left_c = c
            break
    right_c = None
    for c in range(right_tip + 1, w):
        if out[cr, c] == CROSS_COLOR:
            right_c = c
            break

    if None in (top_r, bottom_r, left_c, right_c):
        return None

    def leg_until_turn(r: int, c: int, d, turn_probe) -> int | None:
        rr, cc2 = r, c
        for step in range(1, max(h, w) + 1):
            rr += d[0]
            cc2 += d[1]
            if not (0 <= rr < h and 0 <= cc2 < w and out[rr, cc2] == CROSS_COLOR):
                return None
            tr = rr + turn_probe[0]
            tc = cc2 + turn_probe[1]
            if 0 <= tr < h and 0 <= tc < w and out[tr, tc] == CROSS_COLOR:
                return step
        return None

    gaps = (
        cr - top_r - 1,
        bottom_r - cr - 1,
        cc - left_c - 1,
        right_c - cc - 1,
    )
    first_legs = (
        leg_until_turn(top_r, cc, (0, 1), (1, 0)),
        leg_until_turn(cr, right_c, (1, 0), (0, -1)),
        leg_until_turn(bottom_r, cc, (0, -1), (-1, 0)),
        leg_until_turn(cr, left_c, (-1, 0), (0, 1)),
    )
    return arm, gaps, first_legs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four clockwise spiral walls from the teal cross tips, with the arm length
    jointly controlling corridor width and layer growth.

    Input structure:
    - bg = 0 (black). A single teal (8) plus-sign cross with a center hole.
    - All four arms have the same length; that length is the latent parameter.

    Rule:
    - Each tip first moves 1 cell farther outward, then starts a clockwise turtle
      whose initial heading is 90° to the right of the arm's outward direction.
    - Movement priority stays right-turn > straight > left-turn, with left-turns
      disallowed on the border.
    - If the turtle turns, it advances arm_len cells; if it keeps going straight,
      it advances 2*arm_len cells (falling back to arm_len when the border clips
      the long move).
    - Every new segment must preserve an arm_len-cell orthogonal background corridor
      from all previously drawn walls except the predecessor chain.

    Key insight:
    - The same count read from the input cross explains two independent geometric
      features in the output: wider arm lengths create wider spiral gaps and also
      make each rectangular layer grow by a larger 2*arm_len increment.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    cr, cc, arms = _extract_cross(grid)
    if len(arms) != 4:
        return out

    arm_len = max(len(cells) for _, cells in arms)
    gap = arm_len

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr = cur[0] + d[0] * t
            cc2 = cur[1] + d[1] * t
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            for nr, nc in _neighbors4(rr, cc2):
                if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != cur:
                    return False

            perps = [(-1, 0), (1, 0)] if d[0] == 0 else [(0, -1), (0, 1)]
            for pd in perps:
                for dist in range(2, gap + 1):
                    nr = rr + pd[0] * dist
                    nc = cc2 + pd[1] * dist
                    if inb(nr, nc) and out[nr, nc] != 0:
                        return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr = r0 + d[0] * t
            cc2 = c0 + d[1] * t
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first",
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
            length = gap if turned else (2 * gap if agent["p"] == 0 else gap)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 * gap and can_move(agent["cur"], nd, gap):
                do_move(agent, nd, gap)
                agent["p"] = 0
                return True

        agent["done"] = True
        return False

    guard = 0
    changed = True
    while changed and guard < h * w * 16:
        guard += 1
        changed = False
        for ag in agents:
            if step_agent(ag):
                changed = True
        if all(ag["done"] for ag in agents):
            break

    return out


# --------------- generator ---------------

ARM_CHOICES = (2, 3, 4)
SIZE_BY_ARM = {
    2: 14,
    3: 15,
    4: 19,
}
FILL_LO, FILL_HI = 0.18, 0.45
MAX_OUTER_RETRIES = 100


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _candidate_centers(arm: int, size: int):
    lo = 2 * arm
    hi = size - 1 - 2 * arm
    if lo > hi:
        return []
    centers = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(centers)
    return centers


def _gen_attempt() -> np.ndarray | None:
    arm = random.choice(ARM_CHOICES)
    size = SIZE_BY_ARM[arm]

    for cr, cc in _candidate_centers(arm, size):
        inp = _build_input(arm, size, cr, cc)
        out = main(inp)
        fill = float(np.mean(out == CROSS_COLOR))
        sig = _measure_launch_signature(inp, out)
        if sig is None:
            continue
        sig_arm, gaps, first_legs = sig
        if sig_arm != arm:
            continue
        if not all(g == arm for g in gaps):
            continue
        if not all(v == 2 * arm for v in first_legs):
            continue
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

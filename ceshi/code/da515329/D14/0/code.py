# Variant of da515329
# Dimension: D14 — 背景与前景反转 (meta)
# Change: 背景为黑色(0)，螺旋轨迹为青色(8)，螺旋在黑色背景上涂青色... → 反转涂色：青色背景上擦除螺旋路径

# Design analysis (YAML-style notes kept with the code as requested)
# design_analysis:
#   source_task: none
#   concretization:
#     input:
#       - square even grid, same size formula as the original task
#       - visible background is teal (8)
#       - one black (0) plus-sign cross with a teal center hole marks the four launch tips
#     output:
#       - same spiral geometry as the base task
#       - the visible action is carving: teal cells along the spiral become black
#       - therefore output is the 0/8 color complement of the original task's output on the complemented input
#     trace_example:
#       - original local step: a tip-adjacent black cell is empty, so the agent paints it teal
#       - variant local step: the corresponding tip-adjacent teal cell is untouched background, so the agent erases it to black
#       - difference: geometry and turn schedule are identical; only occupancy polarity and write color are reversed
#   complexity_calibration:
#     - arm length remains 1..4 and center placement still varies, so the task keeps the same parity and border branches
#     - the inversion is visually obvious because every example starts mostly teal and ends with a dense black labyrinth
#     - no extra object types or per-sample mode switches are introduced
#   old_assumptions_to_flip:
#     - cross detection via `(grid == 8)` no longer matches the visible input; solve in complemented space
#     - movement checks such as `out[cell] != 0` used to mean "occupied wall"; in visible space that would misread the teal background as blocked
#     - all post-processing assignments to 0/8 are polarity-sensitive; they must run before converting back
#   placement_legality:
#     - generator still uses the original size formula and tip-to-border gap rule, so spirals have enough room
#     - input construction is now full teal with four black arms; no overlap or out-of-bounds risk beyond the original one
#     - solver validation inside generation guarantees accepted inputs remain solvable and deterministic
#   visual_observability:
#     - from 2-3 examples a human can infer "black paths are being cut out of a teal field from the black cross tips"
#     - the rule is not encoded via invisible ids; it is visible as a foreground/background polarity swap in both input and output
#   visual_cleanliness:
#     - only two colors are used, and the cross remains distinct because its hole stays teal while the arms are black
#     - the carved spiral is still one-pixel-spaced, so the output remains readable rather than muddy
#   color_inference:
#     - there is a single global color mapping: teal behaves as erasable field, black behaves as carved structure
#     - no conditional recoloring by rank/order is used
#   new_main_assumptions:
#     - valid task instances use only colors 0 and 8
#     - the four black arms remain equal-length and axis-aligned with a teal center hole
#     - generate_input enforces those assumptions by direct construction and by running `main()` for acceptance

import math
import random
import sys

import numpy as np

_SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if _SOLVER_UTILS not in sys.path:
    sys.path.append(_SOLVER_UTILS)

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

VISIBLE_BG = 8
VISIBLE_PATH = 0
DRAW_BG = 0
DRAW_PATH = 8


def _to_draw_space(grid: np.ndarray) -> np.ndarray:
    """Map the visible inverted task into the original draw-on-black occupancy space."""
    return np.where(grid == VISIBLE_BG, DRAW_BG, DRAW_PATH).astype(int, copy=False)


def _from_draw_space(grid: np.ndarray) -> np.ndarray:
    """Map the internal occupancy result back to the visible carve-on-teal task."""
    return np.where(grid == DRAW_PATH, VISIBLE_PATH, VISIBLE_BG).astype(int, copy=False)


def _solve_draw_space(grid: np.ndarray) -> np.ndarray:
    """Original spiral construction, kept in black-background / teal-path occupancy space."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    row_counts = np.sum(grid == DRAW_PATH, axis=1)
    col_counts = np.sum(grid == DRAW_PATH, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def right(d):
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]

    def left(d):
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != DRAW_BG and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != DRAW_BG:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = DRAW_PATH
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == DRAW_PATH:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    even_center = (cr + cc) % 2 == 0
    arm_len = max((len(cells) for _, cells in arms), default=0)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": right(outward),
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
            agent["dir"] = right(agent["out_dir"])
            agent["p"] = 0
            return True

        cur = agent["cur"]
        dirs = [right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
            dirs.append(left(agent["dir"]))

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

    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = DRAW_BG
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != DRAW_PATH:
                    continue
                if not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == DRAW_PATH:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = DRAW_BG

        br = (h - 1, w - 1)
        if out[br] == DRAW_BG:
            if all(
                not (inb(nr, nc) and out[nr, nc] == DRAW_PATH)
                for nr, nc in neighbors4(br[0], br[1])
            ):
                out[br] = DRAW_PATH
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = DRAW_PATH
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = DRAW_PATH
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == DRAW_PATH and out[h - 2, c] == DRAW_BG:
                    out[h - 1, c] = DRAW_BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Carve four clockwise spiral trenches from the tips of a black cross.

    Input structure:
    - bg = 8 (teal). A single black (0) cross with a teal center hole sits inside the grid.

    Rule:
    - Treat the black arms as the initial carved path and launch one turtle from each tip.
    - The turtles still prefer right turns, then straight motion, with left turns allowed only away from the border.
    - Step lengths and spacing constraints stay the same as the base task; the only visible difference is that traversed teal cells are erased to black.

    Key insight: This variant changes polarity, not geometry. If the grid is complemented into the original occupancy space, the same spiral generator and cleanup rules can run unchanged, then the result is complemented back.
    """

    grid = np.asarray(grid, dtype=int)
    draw_space = _to_draw_space(grid)
    solved = _solve_draw_space(draw_space)
    return _from_draw_space(solved)


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MAX_OUTER_RETRIES = 300
CROSS_COLOR = VISIBLE_PATH
BACKGROUND_COLOR = VISIBLE_BG
# --- end CFG -----------------------------------------------------------


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    """Build a square teal grid with a black plus-sign cross (center hole left teal)."""
    grid = np.full((size, size), BACKGROUND_COLOR, dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Try one arm-length, enumerate valid centers, and return the first deterministic solvable input."""
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
            out = main(inp)
        except Exception:
            continue
        fill = float(np.mean(out == CROSS_COLOR))
        if FILL_LO <= fill <= FILL_HI:
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

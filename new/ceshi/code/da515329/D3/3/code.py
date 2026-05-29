# Variant of da515329
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: 螺旋轨迹全程使用单一青色(8)连续涂色... → 反射变色（每次转弯时改变颜色）

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Design analysis for the D3 variant
# a) Concrete behavior and trace
#    - Input stays the same: one black grid with a single teal (8) cross whose center cell is empty.
#    - Output keeps the same four clockwise self-avoiding spirals, but the painted path now alternates
#      between teal (8) and red (2). A color switch happens exactly when an agent successfully changes
#      heading by 90 degrees. Straight continuation never switches color.
#    - Example: for the top arm, the first eastbound run is teal; when the agent turns south, that short
#      turning step and the following southbound segment become red; the next turn to west switches back
#      to teal. The geometry matches the original task, only the segment color attached to each turn event
#      changes.
# b) Complexity calibration
#    - The palette is restricted to two visible colors so the rule is inferable from a few examples.
#    - The geometry remains dense and turn-rich, so multiple color alternations appear in every accepted
#      sample; this avoids near-monochrome outputs that would look too close to the source task.
# c) Old assumptions that break under recoloring
#    - Any test like "cell == 8 means occupied" is no longer safe once path cells can be 2 or 8.
#    - Any post-process assignment that hard-codes 8 must be adapted, otherwise it creates local color
#      discontinuities unrelated to turns.
#    - Center detection still relies on the input cross being entirely color 8, so the input is kept
#      monochrome on purpose.
# d) Placement legality
#    - The generator still uses the original size formula and center-range constraints, so the cross,
#      launched spirals, and parity-based cleanup remain non-overlapping and in bounds.
#    - Added post-process cells inherit a neighboring path color when possible, which keeps border caps
#      attached to the visible segment they extend.
# e) Visual observability
#    - The new rule is visible directly in the output because long straight runs share one color and every
#      bend flips to the other color. Across multiple examples, the switch consistently lines up with
#      corner locations rather than with absolute position or hidden metadata.
# f) Visual cleanliness
#    - No extra markers are added. The variant only recolors existing path segments, so the grid stays as
#      readable as the base task.
# g) Color-change inferability
#    - The mapping is fixed and binary: teal <-> red. This is a small enough hypothesis space for ARC,
#      and the generator filters for outputs with enough red coverage to make the alternation obvious.
# 9) New assumptions introduced here
#    - main() assumes the input still contains exactly one teal cross with equal arms; generate_input()
#      constructs only that family.
#    - Recolored occupancy is treated as "nonzero" everywhere that the base solver previously treated
#      only teal as occupied.
#    - Generator-side acceptance requires multiple successful turns and a nontrivial red fraction, which
#      guarantees that the new turn-triggered coloring is actually exercised.

PRIMARY_COLOR = 8
ALT_COLOR = 2
PATH_COLORS = (PRIMARY_COLOR, ALT_COLOR)

# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MAX_OUTER_RETRIES = 300
CROSS_COLOR = PRIMARY_COLOR
MIN_TURN_COUNT = 6
MIN_ALT_FRAC, MAX_ALT_FRAC = 0.35, 0.48
# --- end CFG -----------------------------------------------------------


def _other_color(color: int) -> int:
    return ALT_COLOR if color == PRIMARY_COLOR else PRIMARY_COLOR


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    stats = {"turns": 0}

    row_counts = np.sum(grid == PRIMARY_COLOR, axis=1)
    col_counts = np.sum(grid == PRIMARY_COLOR, axis=0)
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

    def do_move(agent, d, length: int, color: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = color
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)
        agent["color"] = color

    def neighbor_colors(r: int, c: int) -> list[int]:
        return [
            int(out[nr, nc])
            for nr, nc in neighbors4(r, c)
            if inb(nr, nc) and out[nr, nc] != 0
        ]

    def paint_added_cells(cells: list[tuple[int, int]], fallback: int = PRIMARY_COLOR) -> None:
        pending = [(int(r), int(c)) for r, c in cells if inb(int(r), int(c)) and out[int(r), int(c)] == 0]
        while pending:
            progressed = False
            still_pending = []
            for r, c in pending:
                nbrs = neighbor_colors(r, c)
                if nbrs:
                    out[r, c] = nbrs[0]
                    progressed = True
                else:
                    still_pending.append((r, c))
            if not progressed:
                for r, c in still_pending:
                    out[r, c] = fallback
                break
            pending = still_pending

    arms = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == PRIMARY_COLOR:
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
                "color": PRIMARY_COLOR,
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1, agent["color"])
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
                color = _other_color(agent["color"]) if turned else agent["color"]
                do_move(agent, nd, length, color)
                if turned:
                    stats["turns"] += 1
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                do_move(agent, nd, 1, agent["color"])
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
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = 0
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] == 0:
                    continue
                if not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] != 0:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0 and all(out[nr, nc] == 0 for nr, nc in neighbors4(br[0], br[1]) if inb(nr, nc)):
            out[br] = PRIMARY_COLOR
    else:
        if arm_len >= 2:
            paint_added_cells([(r, 0) for r in range(min(h, arm_len - 1))])
        if arm_len >= 1:
            paint_added_cells([(0, c) for c in range(max(0, w - arm_len), w)])
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] != 0 and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    stats["primary_cells"] = int(np.sum(out == PRIMARY_COLOR))
    stats["alt_cells"] = int(np.sum(out == ALT_COLOR))
    stats["painted_cells"] = int(np.sum(out != 0))
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral wall paths from the tips of a teal cross, flipping path color at every 90-degree turn.

    Input structure:
    - bg = 0 (black). A single teal (8) cross with an empty center is placed inside the grid.

    Rule:
    - Each arm tip launches a turtle whose initial heading is 90 degrees clockwise from the arm's outward direction.
    - Movement priority stays right-turn > straight > left-turn (left only away from the border), with straight steps of length 2 except immediately after a turn, when the next advance is length 1.
    - Whenever a chosen move is a turn, that move and all later straight continuation use the other path color (teal 8 <-> red 2) until the next turn.
    - Every new painted cell must keep a one-cell orthogonal gap from earlier path cells, except for the predecessor and the launch attachment.

    Key insight: The spiral geometry is still determined entirely by the original right-biased self-avoiding walker; the new behavior is visible because the same turning events that shape the spiral now also drive a two-color segment alternation.
    """

    out, _stats = _solve_with_stats(grid)
    return out


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


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
            out, stats = _solve_with_stats(inp)
        except Exception:
            continue
        fill = float(np.mean(out != 0))
        alt_frac = stats["alt_cells"] / max(1, stats["painted_cells"])
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if stats["turns"] < MIN_TURN_COUNT:
            continue
        if not (MIN_ALT_FRAC <= alt_frac <= MAX_ALT_FRAC):
            continue
        if stats["alt_cells"] == 0 or stats["primary_cells"] == 0:
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

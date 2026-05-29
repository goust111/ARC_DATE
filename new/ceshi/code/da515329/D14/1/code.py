# Variant of da515329
# Dimension: D14 — 背景与前景反转 (meta)
# Change: 背景为黑色(0)，螺旋轨迹为青色(8)，螺旋在黑色背景上涂青色... → 螺旋涂色使用多种颜色（非单一色）

# Design analysis
# source_task: none
# concretization:
#   input:
#     same as the original task: a single teal plus-sign cross with a center hole on black.
#   output:
#     same wall geometry as the original task, but each of the four spirals keeps a fixed
#     direction-based color: up-tip spiral=red(2), right-tip spiral=blue(1),
#     down-tip spiral=green(3), left-tip spiral=yellow(4). The original teal cross remains teal.
#   small_trace:
#     if the upper arm tip starts a spiral, the original code would mark every visited cell as 8.
#     In the variant, the occupancy test is unchanged, but the write step paints those visited
#     cells as 2. The same happens independently for the other three tips with their own colors.
# complexity_calibration:
#   exactly four spiral colors are used, tied to the four visible arm directions. This is enough
#   to make the variant obvious without introducing many free color cases or ambiguous mappings.
# implicit_assumptions_review:
#   - center detection still relies on the input containing only one teal cross; generator keeps that.
#   - collision and spacing logic must depend on occupancy (nonzero) rather than equality to 8.
#   - post-processing can no longer write literal 8 on added border cells; new cells need an owner
#     so they inherit a spiral color from neighboring path cells or, when isolated, from the nearest
#     spiral frontier.
# placement_legality:
#   generator keeps the original size formula and center-gap constraints, so spirals still fit,
#   remain non-overlapping, and satisfy the same spacing rule.
# visual_observability:
#   the mapping from endpoint direction to color is stable across all samples, while the cross
#   moves around the canvas. That lets a human infer the rule from a few examples.
# visual_cleanliness:
#   only five visible foreground colors appear in outputs: teal cross plus four spiral colors.
#   No extra markers or overlapping layers are introduced.
# generator_checks:
#   accept only inputs whose solved output uses all four spiral colors and keeps the same ~50%
#   occupied-area range as the original task.
# new_main_assumptions_checked_against_generator:
#   - there are four non-empty arms, one in each cardinal direction.
#   - every added non-cross pixel can be assigned to one spiral owner.
#   - the input contains no stray nonzero cells beyond the teal cross.

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_DIR_TO_COLOR = {
    (-1, 0): 2,  # up tip -> red
    (0, 1): 1,   # right tip -> blue
    (1, 0): 3,   # down tip -> green
    (0, -1): 4,  # left tip -> yellow
}

BG = 0
CROSS_COLOR = 8
SPIRAL_COLORS = tuple(_DIR_TO_COLOR[d] for d in _DIRS_URDL)
BG_OWNER = -2
CROSS_OWNER = -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral wall paths from the tips of a teal cross, using four spiral colors.

    Input structure:
    - bg = 0 (black). A single teal (8) cross with a 1-cell center hole is the only foreground object.

    Rule:
    - Each arm tip launches one spiral turtle; the launch direction is still 90° clockwise from the arm's outward direction.
    - Movement priority and spacing stay unchanged: turn right > go straight > turn left, with 2-cell straight steps,
      1-cell turn steps, and a mandatory 1-cell orthogonal gap from other occupied cells.
    - The new twist is only the paint: the four arm directions own four fixed colors, so each turtle paints its whole spiral
      in its direction's color while the original cross remains teal.

    Key insight: The geometry is identical to the original self-avoiding rectangular spirals; only the write operation changes,
    so solving requires separating "where a spiral can go" from "which endpoint owns that spiral's cells".
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    owner = np.full((h, w), BG_OWNER, dtype=int)
    owner[grid == CROSS_COLOR] = CROSS_OWNER

    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
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

    def erase_cell(r: int, c: int) -> None:
        out[r, c] = BG
        owner[r, c] = BG_OWNER

    def paint_cell(r: int, c: int, agent_id: int) -> None:
        out[r, c] = agents[agent_id]["color"]
        owner[r, c] = agent_id

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != BG and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != BG:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            paint_cell(rr, cc2, agent["id"])
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

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

    even_center = (cr + cc) % 2 == 0
    arm_len = max((len(cells) for _, cells in arms), default=0)

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "id": len(agents),
                "cur": cells[-1],
                "dir": right(outward),
                "out_dir": outward,
                "color": _DIR_TO_COLOR[outward],
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
            }
        )

    def choose_owner_for_new_cell(cell: tuple[int, int]) -> int:
        r, c = cell
        adjacent_ids = []
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and owner[nr, nc] >= 0:
                adjacent_ids.append(int(owner[nr, nc]))
        if adjacent_ids:
            counts = {}
            for agent_id in adjacent_ids:
                counts[agent_id] = counts.get(agent_id, 0) + 1
            return min(
                counts,
                key=lambda agent_id: (
                    -counts[agent_id],
                    abs(agents[agent_id]["cur"][0] - r) + abs(agents[agent_id]["cur"][1] - c),
                    agent_id,
                ),
            )
        return min(
            range(len(agents)),
            key=lambda agent_id: (
                abs(agents[agent_id]["cur"][0] - r) + abs(agents[agent_id]["cur"][1] - c),
                agent_id,
            ),
        )

    def add_post_cell(r: int, c: int) -> None:
        if out[r, c] != BG:
            return
        paint_cell(r, c, choose_owner_for_new_cell((r, c)))

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
        mask = (rr % 2 == rp) & (ccg % 2 == cp)
        out[mask] = BG
        owner[mask] = BG_OWNER
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] == BG or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] != BG:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            erase_cell(r, c)

        br = (h - 1, w - 1)
        if out[br] == BG and all(out[nr, nc] == BG for nr, nc in neighbors4(br[0], br[1]) if inb(nr, nc)):
            add_post_cell(*br)
    else:
        if arm_len >= 2:
            for r in range(min(h, arm_len - 1) - 1, -1, -1):
                add_post_cell(r, 0)
        if arm_len >= 1:
            for c in range(max(0, w - arm_len), w):
                add_post_cell(0, c)
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] != BG and out[h - 2, c] == BG:
                    erase_cell(h - 1, c)

    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MAX_OUTER_RETRIES = 300
MIN_SPIRAL_PIXELS = 4
# --- end CFG -----------------------------------------------------------


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    """Build a square grid with a plus-sign cross (center hole) at (cr, cc)."""
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Try one arm-length: compute grid size, enumerate center candidates, return first valid input."""
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
        fill = float(np.mean(out != BG))
        spiral_counts = {color: int(np.sum(out == color)) for color in SPIRAL_COLORS}
        if not all(count >= MIN_SPIRAL_PIXELS for count in spiral_counts.values()):
            continue
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

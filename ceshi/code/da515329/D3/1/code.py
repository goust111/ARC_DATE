# Variant of da515329
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: 螺旋轨迹全程使用单一青色(8)连续涂色... → 每圈递增颜色编号

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the D3 variant
# ----------------------------------
# spec:
#   source_task: none
#   invariant_geometry:
#     input remains a single hollow-centered teal cross on black
#     spiral growth, spacing rule, and parity/border cleanup stay unchanged
#   D3_change:
#     the final occupied spiral is no longer monochrome
#     occupied cells are recolored by concentric rectangular ring depth
#     ring 1 uses color 1, ring 2 uses color 2, ring 3 uses color 3, etc.
#   concretization:
#     use the cross hole as the shared center
#     for any occupied cell, let radius = max(abs(r-cr), abs(c-cc))
#     because walls are 1-pixel thick with 1-pixel gaps, each visible ring spans
#     two consecutive radii: 1-2 -> ring 1, 3-4 -> ring 2, 5-6 -> ring 3 ...
#     ring_index = 1 + (radius - 1) // 2
#   behavior_trace:
#     original: a short-arm sample ends with all occupied cells still equal to 8
#     variant: the same occupancy is produced first, then inner cross/near-center
#     turns become color 1, the next rectangular band becomes color 2, and outer
#     bands continue upward, so the output differs only in layer coloring
#   complexity_calibration:
#     the rule is visible because every sample repeats each color across long
#     horizontal/vertical runs; generator rejects degenerate samples with too few
#     visible layers
#   implicit_assumption_audit:
#     any solver line that previously tested or wrote value 8 inside the evolving
#     output now works on a boolean occupancy mask instead
#     this avoids breaking spacing checks and post-processing once colors vary
#   placement_legality:
#     generator keeps the original size formula and center-placement bounds, so
#     arms do not hit borders too early and the occupancy solver remains valid
#   visual_cleanliness:
#     only color changes; no extra markers or overlapping objects are introduced
#   color_inference:
#     the mapping is driven by visible ring depth only, never by hidden ids/order
#     max layer count under the original size bounds stays within ARC colors 1..8


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _solve_occupancy(grid: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    occ = grid == 8

    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

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

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            occ[rr, cc2] = True
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

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

    even_center = (cr + cc) % 2 == 0
    arm_len = max((len(cells) for _, cells in arms), default=0)

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

        cur = agent["cur"]
        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*cur):
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

    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        occ[(rr % 2 == rp) & (ccg % 2 == cp)] = False
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if not occ[r, c] or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and occ[nr, nc]:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            occ[r, c] = False

        br = (h - 1, w - 1)
        if not occ[br]:
            if all(not (inb(nr, nc) and occ[nr, nc]) for nr, nc in neighbors4(br[0], br[1])):
                occ[br] = True
    else:
        if arm_len >= 2:
            occ[0 : min(h, arm_len - 1), 0] = True
        if arm_len >= 1:
            occ[0, max(0, w - arm_len) : w] = True
        if h >= 2:
            for c in range(w):
                if occ[h - 1, c] and not occ[h - 2, c]:
                    occ[h - 1, c] = False

    return occ, cr, cc, arm_len


def _colorize_by_ring(occ: np.ndarray, cr: int, cc: int) -> np.ndarray:
    out = np.zeros_like(occ, dtype=int)
    for r, c in zip(*np.where(occ)):
        radius = max(abs(r - cr), abs(c - cc))
        layer = 1 + (radius - 1) // 2
        out[r, c] = layer
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a teal cross into four clockwise spiral tracks, then color each ring by depth.

    Input structure:
    - Background is black (0). One teal (8) hollow-centered plus appears in the grid.
    - The four equal arms mark the seed of the future spiral walls.

    Rule:
    - Each arm tip launches a turtle whose initial heading is a clockwise quarter-turn from the arm's outward direction.
    - Motion still prefers right turn > straight > left, with straight lengths 2/1 and a mandatory 1-cell orthogonal gap from older walls.
    - After the original spiral geometry and border cleanup are finished, recolor every occupied cell by rectangular ring depth: radii 1-2 use color 1, 3-4 use color 2, 5-6 use color 3, and so on.

    Key insight: The spatial construction is the same as the base task; the new information is a visible layer-to-color mapping on the finished concentric spiral bands.
    """

    occ, cr, cc, _arm_len = _solve_occupancy(grid)
    return _colorize_by_ring(occ, cr, cc)


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MIN_VISIBLE_LAYERS = 4
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
# --- end CFG -----------------------------------------------------------


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
            out = main(inp)
        except Exception:
            continue
        fill = float(np.mean(out != 0))
        visible_layers = int(out.max(initial=0))
        if FILL_LO <= fill <= FILL_HI and visible_layers >= MIN_VISIBLE_LAYERS:
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

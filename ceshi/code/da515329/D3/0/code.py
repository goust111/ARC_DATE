# Variant of da515329
# Dimension: D3 — 轨迹涂色模式 (trajectory)
# Change: 螺旋轨迹全程使用单一青色(8)连续涂色... → 周期性颜色循环

import math
import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_UTILS) not in sys.path:
    sys.path.append(str(_SOLVER_UTILS))

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the D3 variant (YAML-ish notes kept in-code as requested):
# design:
#   concrete_behavior:
#     input:
#       - Keep the original single centered plus-sign cross with a hollow center.
#       - Add a top-left 2x2 + 2x2 legend that explicitly lists the trajectory color cycle.
#       - The cycle is short and visible: [8, secondary]. The first color stays teal so the
#         seed object still matches the original task family.
#     output:
#       - Keep the original spiral geometry exactly: four clockwise self-avoiding tracks grow
#         from the four arm tips.
#       - Change only D3: new trajectory cells are no longer monochrome; each visible path cell
#         along a branch alternates according to the legend, starting from the tip outward.
#       - The cross itself stays teal; only grown trajectory cells participate in the cycle.
#   behavior_trace:
#     original:
#       - do_move() wrote literal 8 into every newly occupied cell.
#       - fill checks, border cleanup, and center detection all quietly assumed "wall == 8".
#     variant:
#       - Geometry is solved first on a boolean occupancy mask.
#       - After final occupancy is known, each branch is replayed from its tip and colored with
#         cycle[0], cycle[1], cycle[0], ...
#       - This avoids color-specific bugs in spacing checks and makes the visible alternation easy
#         to read from examples.
#   complexity_calibration:
#     - Use a 2-color cycle only. More colors would make few-shot ARC induction weaker.
#     - Vary only the secondary color, chosen from a small high-contrast set, so examples show
#       that the legend color matters while the rule stays structurally simple.
#   implicit_assumptions_review:
#     - row/col argmax on color 8 no longer works because the legend may also contain 8.
#       Replaced with explicit hollow-cross detection.
#     - out[nr, nc] != 0 is now the occupancy predicate everywhere; no logic branches on color 8.
#     - generator fill-fraction must count nonzero cells, not just teal cells, otherwise all
#       multicolor outputs would be incorrectly rejected.
#   placement_legality:
#     - The legend occupies rows 0-1 and cols 0-4 only.
#     - Candidate centers keep the cross well away from the border, so the legend never overlaps
#       the cross and never interferes with spiral growth.
#   observability:
#     - The input legend gives a direct, visible key for the output palette.
#     - Alternation happens along every branch, so the color-cycle rule has repeated support inside
#       each sample instead of depending on one isolated special case.
#   visual_cleanliness:
#     - The legend is removed from the output, preventing top-left clutter from competing with the
#       spiral itself.
#     - Generator prefers odd-parity centers; that branch has the cleanest occupancy cleanup and
#       avoids isolated decorative correction pixels.


# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
SECONDARY_COLORS = (1, 2, 3, 4, 6, 7)
LEGEND_TOP = 0
LEGEND_LEFT = 0
LEGEND_GAP = 1
MIN_SECONDARY_PIXELS = 10
# --- end CFG -----------------------------------------------------------


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _add(pos, d, k=1):
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _legend_block_cols(idx: int) -> tuple[int, int]:
    c0 = LEGEND_LEFT + idx * (2 + LEGEND_GAP)
    return c0, c0 + 1


def _place_cycle_legend(grid: np.ndarray, cycle: list[int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx, color in enumerate(cycle):
        c0, c1 = _legend_block_cols(idx)
        grid[LEGEND_TOP : LEGEND_TOP + 2, c0 : c1 + 1] = color
        for r in range(LEGEND_TOP, LEGEND_TOP + 2):
            for c in range(c0, c1 + 1):
                cells.append((r, c))
    return cells


def _extract_cycle_legend(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    colors: list[int] = []
    cells: set[tuple[int, int]] = set()
    seen_cols: set[int] = set()
    if h < 2:
        return colors, cells
    for c in range(0, max(0, w - 1)):
        if c in seen_cols:
            continue
        block = grid[LEGEND_TOP : LEGEND_TOP + 2, c : c + 2]
        if block.shape != (2, 2):
            continue
        v = int(block[0, 0])
        if v == 0:
            continue
        if np.all(block == v):
            colors.append(v)
            seen_cols.update({c, c + 1})
            for rr in range(LEGEND_TOP, LEGEND_TOP + 2):
                for cc in range(c, c + 2):
                    cells.add((rr, cc))
    return colors, cells


def _find_cross(grid: np.ndarray) -> dict:
    h, w = grid.shape
    best = None

    for cr in range(1, h - 1):
        for cc in range(1, w - 1):
            if int(grid[cr, cc]) != 0:
                continue

            color = None
            arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
            ok = True
            for d in _DIRS_URDL:
                dr, dc = d
                r, c = cr, cc
                cells = []
                while _inb(h, w, r + dr, c + dc) and int(grid[r + dr, c + dc]) != 0:
                    r += dr
                    c += dc
                    v = int(grid[r, c])
                    if color is None:
                        color = v
                    elif v != color:
                        ok = False
                        break
                    cells.append((r, c))
                if not ok or not cells:
                    ok = False
                    break
                arms.append((d, cells))
            if not ok or color is None:
                continue

            arm_len = len(arms[0][1])
            if any(len(cells) != arm_len for _, cells in arms):
                continue

            for _, cells in arms:
                for i, (r, c) in enumerate(cells):
                    allowed = set()
                    if i > 0:
                        allowed.add(cells[i - 1])
                    if i + 1 < len(cells):
                        allowed.add(cells[i + 1])
                    for nr, nc in neighbors4(r, c):
                        if _inb(h, w, nr, nc) and int(grid[nr, nc]) == color:
                            if (nr, nc) not in allowed:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break

            if not ok:
                continue

            cand = {
                "center": (cr, cc),
                "color": color,
                "arms": arms,
                "arm_len": arm_len,
            }
            if best is None or arm_len > best["arm_len"]:
                best = cand

    if best is None:
        raise ValueError("failed to detect hollow cross")
    return best


def _solve_geometry(grid: np.ndarray, cross: dict) -> np.ndarray:
    h, w = grid.shape
    cr, cc = cross["center"]
    arm_len = cross["arm_len"]
    arms = cross["arms"]
    even_center = (cr + cc) % 2 == 0

    occ = np.zeros((h, w), dtype=bool)
    for _, cells in arms:
        for r, c in cells:
            occ[r, c] = True

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if _inb(h, w, nr, nc) and occ[nr, nc] and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = _add(cur, d, t)
            if not _inb(h, w, rr, cc2) or occ[rr, cc2]:
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
                    if _inb(h, w, nr, nc) and occ[nr, nc]:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            occ[r, c] = False

        br = (h - 1, w - 1)
        if not occ[br]:
            if all(not (_inb(h, w, nr, nc) and occ[nr, nc]) for nr, nc in neighbors4(br[0], br[1])):
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

    return occ


def _follow_visible_path(
    occ: np.ndarray,
    start_tip: tuple[int, int],
    inward: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    h, w = occ.shape
    cur = start_tip
    prev = inward
    order: list[tuple[int, int]] = []
    seen = {(cur, prev)}

    while True:
        nxts = []
        for nr, nc in neighbors4(cur[0], cur[1]):
            if not _inb(h, w, nr, nc) or not occ[nr, nc]:
                continue
            if prev is not None and (nr, nc) == prev:
                continue
            nxts.append((nr, nc))
        if not nxts:
            break

        if len(nxts) == 1:
            nxt = nxts[0]
        else:
            # Prefer the continuation that is farther from the cross interior.
            nxt = max(nxts, key=lambda cell: abs(cell[0] - start_tip[0]) + abs(cell[1] - start_tip[1]))
        state = (nxt, cur)
        if state in seen:
            break
        seen.add(state)
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _color_geometry(occ: np.ndarray, cross: dict, cycle: list[int]) -> np.ndarray:
    h, w = occ.shape
    out = np.zeros((h, w), dtype=int)

    for _, cells in cross["arms"]:
        for r, c in cells:
            out[r, c] = cross["color"]

    for _, cells in cross["arms"]:
        tip = cells[-1]
        inward = cells[-2] if len(cells) >= 2 else None
        path = _follow_visible_path(occ, tip, inward)
        for idx, (r, c) in enumerate(path):
            out[r, c] = cycle[idx % len(cycle)]

    leftover = np.argwhere(occ & (out == 0))
    if leftover.size:
        for r, c in leftover:
            out[r, c] = cycle[0]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the original four spiral tracks, but color each visible branch with a repeating palette.

    Input structure:
    - Background is black (0).
    - One hollow teal cross is the geometric seed, exactly as in the base task.
    - The top-left corner contains a small 2x2-block legend listing the trajectory color cycle.

    Rule:
    - Each arm tip launches the same clockwise spiral turtle as before: right-turn priority,
      step lengths 2/1, and a one-cell orthogonal spacing constraint.
    - Geometry is computed purely from occupancy, so path growth ignores the palette colors.
    - After the final spiral occupancy is known, every branch is recolored from its tip outward
      using the legend colors in a repeating cycle.

    Key insight:
    - The variant changes only how the trajectory is painted, not how it is routed. Solving the
      monochrome spiral first and replaying a visible color cycle afterward preserves the original
      structure while making the new D3 rule explicit.
    """

    grid = np.asarray(grid, dtype=int)
    cycle, _ = _extract_cycle_legend(grid)
    cross = _find_cross(grid)
    if len(cycle) < 2:
        cycle = [cross["color"], 2 if cross["color"] != 2 else 1]

    occ = _solve_geometry(grid, cross)
    out = _color_geometry(occ, cross, cycle)
    return out


def _build_input(arm: int, size: int, cr: int, cc: int, cycle: list[int]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    _place_cycle_legend(grid, cycle)
    return grid


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE
    cycle = [CROSS_COLOR, random.choice(SECONDARY_COLORS)]

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [
        (r, c)
        for r in range(lo, hi + 1)
        for c in range(lo, hi + 1)
        if (r + c) % 2 == 1
    ]
    if not candidates:
        candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc, cycle)
        try:
            out = main(inp)
        except Exception:
            continue
        fill = float(np.mean(out != 0))
        secondary_count = int(np.count_nonzero(out == cycle[1]))
        if FILL_LO <= fill <= FILL_HI and secondary_count >= MIN_SECONDARY_PIXELS:
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

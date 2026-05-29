# Combo variant of da515329 (index 15)
# combo_spec: D3/0+D8/1
# Name: 中心点不同色标记 + 周期颜色循环

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


# Design analysis for combo D3/0 + D8/1 (kept in-code as requested):
# design:
#   concrete_behavior:
#     input:
#       - Keep the original large hollow teal cross as the geometric seed.
#       - Add a fixed top-left template with two 2x2 blocks for the trajectory cycle.
#       - Add a small 3x3 cross icon whose center pixel gives the output center-marker color.
#     output:
#       - Reuse the original spiral occupancy exactly: four clockwise self-avoiding tracks grow
#         from the arm tips with the same spacing and border cleanup.
#       - Recolor only the grown trajectory cells with a repeating 2-color cycle.
#       - Paint the formerly empty large-cross center with the marker color from the small icon.
#   behavior_trace:
#     example_arm1:
#       - Original task: input has a hollow plus; solver writes only color 8 and leaves center 0.
#       - Combo, case A: cycle blocks are [2, 6], center icon center is 2.
#         The geometry is unchanged, branch cells become 2,6,2,6,... from each tip outward,
#         and the big cross center becomes 2.
#       - Combo, case B: cycle blocks are [2, 6], center icon center is 4.
#         The same occupancy is built, branch cells still alternate 2/6, but the big center
#         becomes 4, so the solver must separate "trajectory palette" from "center marker".
#   complexity_calibration:
#     - Limit the trajectory cycle to exactly two colors; longer cycles would be harder to infer
#       from 2-4 ARC examples and would add low-value variation.
#     - Limit the center-color relation to two observable cases only:
#         1. center color == cycle[0]
#         2. center color is a third color distinct from both cycle colors
#       This preserves the fusion requirement without creating a large hidden mapping table.
#   implicit_assumptions_review:
#     - The original row/col argmax-on-color-8 center detection breaks because the template also
#       contains teal arms. Replace it with explicit hollow-cross detection.
#     - Any code that treated "8 means occupied" is no longer valid once trajectories become
#       multicolor; occupancy is solved on a boolean mask and recolored afterward.
#     - Fill checks in the generator must use out != 0, not out == 8.
#     - The center marker is written only after path coloring so it cannot interfere with branch
#       traversal or spacing checks.
#   placement_legality:
#     - The template occupies only a fixed top-left footprint.
#     - Candidate centers are filtered so the large cross never overlaps template cells.
#     - The original grid-size formula is retained, and solver validation remains in-the-loop.
#   observability:
#     - The two 2x2 blocks visibly repeat as the two path colors across long spiral branches.
#     - The small cross icon makes it visually obvious that its center pixel maps to the large
#       cross center, so humans can infer whether that color matches cycle[0] or is independent.
#   visual_cleanliness:
#     - The template is erased in the output; only the spiral and one center marker remain.
#     - The center key uses a compact icon rather than extra scattered markers, avoiding clutter.
#   inferability:
#     - The key color relation is structural and visible in input, not a hidden numeric rule.
#     - Every accepted sample contains many cells of both cycle colors, plus one visible center
#       marker, so the fused behavior is supported by repeated evidence instead of a corner case.


# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.53
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
CYCLE_COLORS = (1, 2, 3, 4, 6, 7, 9)
CENTER_EXTRA_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
MIN_SECONDARY_PIXELS = 10
LEGEND_BLOCK_STARTS = (0, 3)
LEGEND_ROWS = slice(0, 2)
CENTER_ICON_CENTER = (4, 2)
# --- end CFG -----------------------------------------------------------


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _add(pos, d, k=1):
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _template_cells(cycle: list[int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for idx, _ in enumerate(cycle):
        c0 = LEGEND_BLOCK_STARTS[idx]
        for r in range(LEGEND_ROWS.start, LEGEND_ROWS.stop):
            for c in range(c0, c0 + 2):
                cells.add((r, c))
    cr, cc = CENTER_ICON_CENTER
    cells.add((cr, cc))
    for dr, dc in _DIRS_URDL:
        cells.add((cr + dr, cc + dc))
    return cells


def _place_cycle_legend(grid: np.ndarray, cycle: list[int]) -> None:
    for idx, color in enumerate(cycle):
        c0 = LEGEND_BLOCK_STARTS[idx]
        grid[LEGEND_ROWS, c0 : c0 + 2] = color


def _place_center_icon(grid: np.ndarray, center_color: int, cross_color: int) -> None:
    cr, cc = CENTER_ICON_CENTER
    grid[cr, cc] = center_color
    for dr, dc in _DIRS_URDL:
        grid[cr + dr, cc + dc] = cross_color


def _extract_cycle_legend(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    for c0 in LEGEND_BLOCK_STARTS:
        block = grid[LEGEND_ROWS, c0 : c0 + 2]
        if block.shape != (2, 2):
            continue
        v = int(block[0, 0])
        if v != 0 and np.all(block == v):
            colors.append(v)
    return colors


def _extract_center_icon_color(grid: np.ndarray, cross_color: int) -> int | None:
    h, w = grid.shape
    cr, cc = CENTER_ICON_CENTER
    if not _inb(h, w, cr, cc):
        return None
    for dr, dc in _DIRS_URDL:
        rr, cc2 = cr + dr, cc + dc
        if not _inb(h, w, rr, cc2) or int(grid[rr, cc2]) != cross_color:
            return None
    color = int(grid[cr, cc])
    return color if color != 0 else None


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


def _solve_geometry(shape: tuple[int, int], cross: dict) -> np.ndarray:
    h, w = shape
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
            nxt = max(nxts, key=lambda cell: abs(cell[0] - start_tip[0]) + abs(cell[1] - start_tip[1]))
        state = (nxt, cur)
        if state in seen:
            break
        seen.add(state)
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _color_geometry(
    occ: np.ndarray,
    cross: dict,
    cycle: list[int],
    center_color: int,
) -> np.ndarray:
    out = np.zeros_like(occ, dtype=int)
    cr, cc = cross["center"]

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
    for r, c in leftover:
        out[int(r), int(c)] = cycle[0]

    out[cr, cc] = center_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the original spiral labyrinth, then separate its path palette from its center marker.

    Input structure:
    - Background is black (0).
    - One large hollow teal cross is the geometric seed, as in the base task.
    - The top-left template has two 2x2 blocks for the trajectory color cycle and a small cross
      icon whose center pixel specifies the output color of the large cross center.

    Rule:
    - Each arm tip launches the same clockwise spiral turtle as before: right-turn preference,
      2/1 step lengths, and a one-cell orthogonal spacing constraint.
    - Geometry is solved on occupancy only, so the route does not depend on visible colors.
    - After the final occupied spiral is known, every grown branch cell is recolored by repeating
      the two legend colors from the tip outward, and the large cross center is filled with the
      icon's center color.

    Key insight:
    - This fusion changes two visual layers while preserving one geometry layer: the spiral shape
      still comes from the base task, but the template separately tells you which colors belong to
      the trajectory cycle and which single color belongs only to the central marker.
    """

    grid = np.asarray(grid, dtype=int)
    cross = _find_cross(grid)
    cycle = _extract_cycle_legend(grid)
    if len(cycle) < 2:
        cycle = [2, 6]
    center_color = _extract_center_icon_color(grid, cross["color"])
    if center_color is None:
        center_color = cycle[0]

    occ = _solve_geometry(grid.shape, cross)
    return _color_geometry(occ, cross, cycle, center_color)


def _build_input(
    arm: int,
    size: int,
    cr: int,
    cc: int,
    cycle: list[int],
    center_color: int,
) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    _place_cycle_legend(grid, cycle)
    _place_center_icon(grid, center_color, CROSS_COLOR)
    return grid


def _cross_overlaps_template(arm: int, cr: int, cc: int, reserved: set[tuple[int, int]]) -> bool:
    coords = [(cr - k, cc) for k in range(1, arm + 1)]
    coords += [(cr + k, cc) for k in range(1, arm + 1)]
    coords += [(cr, cc - k) for k in range(1, arm + 1)]
    coords += [(cr, cc + k) for k in range(1, arm + 1)]
    return any(cell in reserved for cell in coords)


def _pick_center_color(cycle: list[int]) -> tuple[int, bool]:
    if random.random() < 0.5:
        return cycle[0], True
    choices = [c for c in CENTER_EXTRA_COLORS if c not in cycle and c != CROSS_COLOR]
    return random.choice(choices), False


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 4 * arm + 2 * math.ceil(arm / 2) + GRID_SIZE_BASE

    cycle = random.sample(CYCLE_COLORS, 2)
    center_color, _ = _pick_center_color(cycle)
    reserved = _template_cells(cycle)

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [
        (r, c)
        for r in range(lo, hi + 1)
        for c in range(lo, hi + 1)
        if not _cross_overlaps_template(arm, r, c, reserved)
    ]
    odd_candidates = [(r, c) for (r, c) in candidates if (r + c) % 2 == 1]
    even_candidates = [(r, c) for (r, c) in candidates if (r + c) % 2 == 0]
    random.shuffle(odd_candidates)
    random.shuffle(even_candidates)
    candidates = odd_candidates + even_candidates if odd_candidates else even_candidates

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc, cycle, center_color)
        try:
            out = main(inp)
        except Exception:
            continue
        fill = float(np.mean(out != 0))
        secondary_count = int(np.count_nonzero(out == cycle[1]))
        center_hits = int(np.count_nonzero(out == center_color))
        if FILL_LO <= fill <= FILL_HI and secondary_count >= MIN_SECONDARY_PIXELS and center_hits >= 1:
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

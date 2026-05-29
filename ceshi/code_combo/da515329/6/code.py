# Combo variant of da515329 (index 6)
# combo_spec: D3/0+D14/1
# Name: 周期颜色循环 + 四象限不同色

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
BG = 0
CROSS_COLOR = 8
BG_OWNER = -2
CROSS_OWNER = -1
_DIR_TO_CYCLE = {
    (-1, 0): (2, 7),  # up tip -> red / orange
    (0, 1): (1, 3),   # right tip -> blue / green
    (1, 0): (4, 6),   # down tip -> yellow / magenta
    (0, -1): (9, 5),  # left tip -> brown / gray
}


# Design analysis for combo D3/0 + D14/1:
# design:
#   concrete_behavior:
#     input:
#       - Keep the original input family unchanged: one hollow teal plus-sign cross on black.
#       - No extra legend is added; the learner must infer the new color logic from repeated
#         input/output pairs, not from an explicit key.
#     output:
#       - Preserve the original four clockwise self-avoiding rectangular spirals exactly.
#       - Replace monochrome painting with direction-owned 2-color cycles:
#         up=(2,7), right=(1,3), down=(4,6), left=(9,5).
#       - The cross itself remains teal; only grown spiral cells participate in the cycles.
#   behavior_trace:
#     - Original code wrote literal 8 inside do_move(), so geometry and paint were fused.
#     - In the combo version, movement still claims cells branch-by-branch, but those cells are
#       first stored as ownership ids rather than final colors.
#     - After all parity cleanup and border corrections finish, each visible branch is replayed
#       from its tip outward and colored base/accent/base/accent/... using that direction's pair.
#     - Example: the top tip's first grown cell becomes 2, the next visible top-branch cell
#       becomes 7, then 2 again; the right branch independently alternates 1 and 3.
#   complexity_calibration:
#     - Each branch uses exactly two colors. That is enough to make the D3 change obvious
#       without exploding the palette into hard-to-learn many-step sequences.
#     - Each direction has its own fixed pair, so D14 is also visible. The four base colors are
#       all different, which keeps the quadrant-to-palette mapping structurally readable.
#   implicit_assumptions_review:
#     - Row/column argmax on color 8 still works because the input remains cross-only.
#     - Collision checks can no longer test equality to 8; they must treat any owned cell as
#       occupied, regardless of its eventual display color.
#     - Post-processing can no longer add literal teal pixels. Any new border cell must first be
#       assigned to a spiral owner, then colored later by that owner's cycle.
#     - Coloring must happen after cleanup. If cells were colored during growth, later deletions
#       could destroy the visible alternation and accidentally leave same-color neighbors.
#   placement_legality:
#     - The generator keeps the original size formula and tip-to-border gap constraints, so the
#       spirals still fit without overlap or out-of-bounds moves.
#     - Candidate centers are filtered by solved outputs: every branch must be long enough to
#       show both colors in its cycle, so the fusion never degenerates into nearly monochrome arms.
#   visual_observability:
#     - The cross moves around between samples while the direction-to-cycle mapping stays fixed,
#       giving repeated evidence that "which tip" determines the palette.
#     - Alternation repeats many times within each branch, so the periodic cycle is visible from
#       local structure rather than from one isolated exception pixel.
#   visual_cleanliness:
#     - The original 1-cell spacing rule already separates branches, so the multicolor painting
#       does not create overlaps or layered clutter.
#     - The chosen color pairs are distinct enough that neighboring branches remain separable even
#       when they run in parallel around the outer rings.
#   inferability:
#     - The first grown cell after each tip is always the branch's base color, making the
#       quadrant-to-base mapping stable and human-visible.
#     - The second color recurs every other visible step, so the cycle rule has repeated support
#       inside a single example and across multiple examples.
#   new_main_assumptions_checked_against_generator:
#     - There is exactly one hollow cross, with four non-empty equal-length arms.
#     - Each spiral owner has at least a small visible path, and every branch shows both cycle
#       colors in the accepted generated samples.
#     - Rare disconnected post-added cells are allowed, but the renderer handles them explicitly
#       after tracing the main visible branch.


# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.48, 0.52
MAX_OUTER_RETRIES = 300
MIN_BRANCH_CELLS = 6
MIN_ACCENT_CELLS = 2
# --- end CFG -----------------------------------------------------------


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _add(pos, d, k=1):
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _collect_cross(grid: np.ndarray) -> dict:
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
        while _inb(h, w, r + dr, c + dc) and int(grid[r + dr, c + dc]) == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    if len(arms) != 4:
        raise ValueError("expected four cross arms")
    arm_len = len(arms[0][1])
    if arm_len == 0 or any(len(cells) != arm_len for _, cells in arms):
        raise ValueError("expected equal non-empty arm lengths")

    return {"center": (cr, cc), "arms": arms, "arm_len": arm_len}


def _solve_owned_geometry(grid: np.ndarray, cross: dict) -> dict:
    h, w = grid.shape
    cr, cc = cross["center"]
    arms = cross["arms"]
    arm_len = cross["arm_len"]
    even_center = (cr + cc) % 2 == 0

    owner = np.full((h, w), BG_OWNER, dtype=int)
    for _, cells in arms:
        for r, c in cells:
            owner[r, c] = CROSS_OWNER

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "id": len(agents),
                "tip": cells[-1],
                "inward": cells[-2] if len(cells) >= 2 else None,
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "cycle": _DIR_TO_CYCLE[outward],
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
            }
        )

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def occupied(r: int, c: int) -> bool:
        return int(owner[r, c]) != BG_OWNER

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if _inb(h, w, nr, nc) and occupied(nr, nc) and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = _add(cur, d, t)
            if not _inb(h, w, rr, cc2) or occupied(rr, cc2):
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def claim_cells(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            owner[rr, cc2] = agent["id"]
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    def erase_cell(r: int, c: int) -> None:
        owner[r, c] = BG_OWNER

    def choose_owner_for_new_cell(cell: tuple[int, int]) -> int:
        r, c = cell
        adjacent_ids = []
        for nr, nc in neighbors4(r, c):
            if _inb(h, w, nr, nc) and int(owner[nr, nc]) >= 0:
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
        if occupied(r, c):
            return
        owner[r, c] = choose_owner_for_new_cell((r, c))

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            if can_move(agent["cur"], agent["out_dir"], 1):
                claim_cells(agent, agent["out_dir"], 1)
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
                claim_cells(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                claim_cells(agent, nd, 1)
                agent["p"] = 0
                return True

        agent["done"] = True
        return False

    changed = True
    guard = 0
    while changed and guard < 200000:
        guard += 1
        changed = False
        for agent in agents:
            if step_agent(agent):
                changed = True
        if all(agent["done"] for agent in agents):
            break

    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        owner[(rr % 2 == rp) & (ccg % 2 == cp)] = BG_OWNER
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if not occupied(r, c) or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if _inb(h, w, nr, nc) and occupied(nr, nc):
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            erase_cell(r, c)

        br = (h - 1, w - 1)
        if not occupied(*br):
            if all(not (_inb(h, w, nr, nc) and occupied(nr, nc)) for nr, nc in neighbors4(br[0], br[1])):
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
                if occupied(h - 1, c) and not occupied(h - 2, c):
                    erase_cell(h - 1, c)

    return {"owner": owner, "cross": cross, "agents": agents}


def _follow_owned_path(
    owner: np.ndarray,
    agent_id: int,
    start_tip: tuple[int, int],
    inward: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    h, w = owner.shape
    cur = start_tip
    prev = inward
    order: list[tuple[int, int]] = []
    seen = {(cur, prev)}

    while True:
        nxts = []
        for nr, nc in neighbors4(cur[0], cur[1]):
            if not _inb(h, w, nr, nc) or int(owner[nr, nc]) != agent_id:
                continue
            if prev is not None and (nr, nc) == prev:
                continue
            nxts.append((nr, nc))
        if not nxts:
            break

        if len(nxts) == 1:
            nxt = nxts[0]
        else:
            nxt = max(
                nxts,
                key=lambda cell: (
                    abs(cell[0] - start_tip[0]) + abs(cell[1] - start_tip[1]),
                    cell[0],
                    cell[1],
                ),
            )
        state = (nxt, cur)
        if state in seen:
            break
        seen.add(state)
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _render_output(sol: dict) -> np.ndarray:
    owner = sol["owner"]
    cross = sol["cross"]
    agents = sol["agents"]
    h, w = owner.shape
    out = np.zeros((h, w), dtype=int)

    for _, cells in cross["arms"]:
        for r, c in cells:
            out[r, c] = CROSS_COLOR

    for agent in agents:
        cycle = agent["cycle"]
        visible_order = _follow_owned_path(owner, agent["id"], agent["tip"], agent["inward"])
        used = set(visible_order)
        for idx, (r, c) in enumerate(visible_order):
            out[r, c] = cycle[idx % len(cycle)]

        leftovers = [tuple(cell) for cell in np.argwhere(owner == agent["id"]) if tuple(cell) not in used]
        leftovers.sort(
            key=lambda cell: (
                -(abs(cell[0] - agent["tip"][0]) + abs(cell[1] - agent["tip"][1])),
                cell[0],
                cell[1],
            )
        )
        for offset, (r, c) in enumerate(leftovers, start=len(visible_order)):
            out[r, c] = cycle[offset % len(cycle)]

    return out


def _solve_with_metadata(grid: np.ndarray) -> dict:
    cross = _collect_cross(grid)
    sol = _solve_owned_geometry(grid, cross)
    sol["output"] = _render_output(sol)
    return sol


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the original four spiral tracks, but let each endpoint own its own 2-color rhythm.

    Input structure:
    - Background is black (0).
    - A single hollow teal cross is the only foreground object; there are no extra legends or markers.

    Rule:
    - The spiral routing is unchanged from the base task: every arm tip launches a clockwise turtle,
      with right-turn priority, 2/1 step lengths, and a mandatory 1-cell orthogonal gap.
    - The four arm directions now own four different alternating color pairs, while the teal cross
      itself stays fixed.
    - After the final spiral occupancy is settled, each branch is replayed from its tip outward and
      painted with its direction's repeating 2-color cycle.

    Key insight:
    - This fusion changes both "which branch owns a cell" and "how that branch is painted", but not
      the underlying geometry. Solving cleanly means separating occupancy from final color, then
      coloring each visible branch only after all cleanup is complete.
    """

    grid = np.asarray(grid, dtype=int)
    return _solve_with_metadata(grid)["output"]


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _branch_stats(sol: dict) -> list[dict]:
    owner = sol["owner"]
    out = sol["output"]
    stats = []
    for agent in sol["agents"]:
        branch_colors = out[owner == agent["id"]]
        path = _follow_owned_path(owner, agent["id"], agent["tip"], agent["inward"])
        stats.append(
            {
                "id": agent["id"],
                "cells": int(branch_colors.size),
                "path_cells": len(path),
                "base_count": int(np.count_nonzero(branch_colors == agent["cycle"][0])),
                "accent_count": int(np.count_nonzero(branch_colors == agent["cycle"][1])),
                "unique_colors": len(set(map(int, branch_colors.tolist()))),
            }
        )
    return stats


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
            sol = _solve_with_metadata(inp)
        except Exception:
            continue

        out = sol["output"]
        fill = float(np.mean(out != BG))
        if not (FILL_LO <= fill <= FILL_HI):
            continue

        branch_stats = _branch_stats(sol)
        if any(stat["cells"] < MIN_BRANCH_CELLS for stat in branch_stats):
            continue
        if any(stat["accent_count"] < MIN_ACCENT_CELLS for stat in branch_stats):
            continue
        if any(stat["unique_colors"] != 2 for stat in branch_stats):
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

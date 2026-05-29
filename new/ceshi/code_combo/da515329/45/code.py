# Combo variant of da515329 (index 45)
# combo_spec: D8/0+D9/0+D16/0
# Name: 中心保持青色 + 臂擦除 + 无回退直接前进

import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CROSS_COLOR = 8
BG_COLOR = 0


# Design analysis for the combo variant
# -------------------------------------
# spec_combo:
#   preserved_input:
#     grid: square black canvas
#     object: exactly one teal hollow-centered cross, same as the base task input
#     arm_len: generator keeps equal cardinal arms, but narrows the visible regime to
#       arm lengths 1-2 because larger arms plus erased stems become too sparse on the
#       base-size canvases and stop looking like a human-inferable spiral family
#   fused_output:
#     source_markers:
#       the center hole is explicitly filled with teal and becomes the only surviving
#       cross pixel in the output
#       every original arm pixel, including each tip cell, is erased before spiral
#       drawing starts
#     path_launch:
#       each tip still owns one turtle, but all turtles take the same mandatory first
#       step one cell farther in the arm's outward direction
#       only after that do they resume the clockwise wall-growing dynamics inherited
#       from the base task
#   concrete_behavior_trace:
#     example_arm_2:
#       input around the center is
#         row cr-2..cr+2, col cc-2..cc+2:
#           ..8..
#           ..8..
#           88.88
#           ..8..
#           ..8..
#       base task:
#         out starts as a copy of the cross, so arm cells remain occupied and the
#         visible pattern grows out of those teal stems while the center stays black
#       fused task:
#         out starts as all black except the center cell (cr, cc)=8
#         the four arm cells at distance 1 and 2 are forced back to black
#         the top tip at (cr-2, cc) first paints (cr-3, cc), the right tip first
#         paints (cr, cc+3), etc.; only then do the turtles begin right-turn / straight
#         / left fallback choices
#       visible difference:
#         the output now shows four black gaps where the stems used to be, one teal
#         center seed, and spiral walls that begin from detached launch points
#   complexity_calibration:
#     if we kept the original size formula for all arm lengths, D9/0 and D16/0 together
#     leave arm_len 3-4 samples with very low occupancy and weak spiral cues
#     the generator therefore samples only arm_len 1-2 and a small set of compact sizes
#     where multiple rectangular turns remain visible and the erased stems are obvious
#   implicit_assumption_audit:
#     broken_base_assumptions:
#       out = grid.copy() is no longer legal because D9/0 requires every arm pixel to
#       disappear from the final drawing
#       the base solver's special "short arm only" outward pre-step is too weak because
#       D16/0 requires the direct-forward launch for every tip, not just arm_len 1
#       parity cleanup may accidentally erase the restored center, so the combo must
#       restore the center after cleanup
#     surviving_assumptions:
#       the densest teal row and column still identify the hollow-cross center
#       the cross remains axis-aligned with one contiguous arm in each cardinal
#       direction; generator enforces that construction
#       spacing checks still work if the launch tip is treated as an unpainted anchor,
#       because the first painted cell is beyond the erased stem gap
#   placement_legality:
#     generator keeps every center far enough from the border that all four mandatory
#     outward launch cells are in bounds
#     candidate samples are accepted only if the output keeps center=8, clears all arm
#     pixels, and paints all four outward launch cells
#   observability:
#     across accepted samples, a human sees the same three structural cues every time:
#       the cross arms vanish completely
#       the center hole becomes a single teal pixel
#       each spiral starts detached one cell beyond a tip, leaving a visible black gap
#     no hidden ordering, numeric color sorting, or non-visual metadata is used
#   visual_cleanliness:
#     acceptance filtering removes samples whose density is too low or whose launches
#     fail, so the remaining boards keep obvious rectangular turns instead of noisy
#     fragments
#   fusion_metric:
#     define score = center_kept (0/1) + cleared_arm_cells + launched_cells
#       center_kept: output center is teal
#       cleared_arm_cells: number of original arm pixels that became black
#       launched_cells: number of first outward launch cells that are teal
#     accepted generator samples satisfy the maximum score 1 + 4*arm_len + 4


def _find_cross(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]], int]:
    """Return center row/col, directional arm cell lists, and the common arm length."""
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

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
        if cells:
            arms.append((d, cells))

    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return pos[0] + d[0] * k, pos[1] + d[1] * k


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _spacing_ok(
    out: np.ndarray, cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]
) -> bool:
    h, w = out.shape
    r, c = cell
    for nr, nc in neighbors4(r, c):
        if _in_bounds((h, w), nr, nc) and out[nr, nc] != BG_COLOR:
            if (nr, nc) != prev and (nr, nc) != start:
                return False
    return True


def _can_move(out: np.ndarray, cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
    prev = cur
    for t in range(1, length + 1):
        rr, cc = _add(cur, d, t)
        if not _in_bounds(out.shape, rr, cc) or out[rr, cc] != BG_COLOR:
            return False
        if not _spacing_ok(out, (rr, cc), prev, cur):
            return False
        prev = (rr, cc)
    return True


def _do_move(agent: dict, out: np.ndarray, d: tuple[int, int], length: int) -> None:
    r0, c0 = agent["cur"]
    for t in range(1, length + 1):
        rr = r0 + d[0] * t
        cc = c0 + d[1] * t
        out[rr, cc] = CROSS_COLOR
    agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)


def _step_agent(agent: dict, out: np.ndarray) -> bool:
    if agent["done"]:
        return False

    if agent["phase"] == "forward_first":
        if _can_move(out, agent["cur"], agent["out_dir"], 1):
            _do_move(agent, out, agent["out_dir"], 1)
        agent["phase"] = None
        agent["dir"] = _right(agent["out_dir"])
        agent["p"] = 0
        return True

    cur = agent["cur"]
    dirs = [_right(agent["dir"]), agent["dir"]]
    if not _is_border(out.shape, *cur):
        dirs.append(_left(agent["dir"]))

    for nd in dirs:
        turned = nd != agent["dir"]
        length = 1 if turned else (2 if agent["p"] == 0 else 1)
        if _can_move(out, agent["cur"], nd, length):
            _do_move(agent, out, nd, length)
            agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
            agent["dir"] = nd
            return True
        if not turned and length == 2 and _can_move(out, agent["cur"], nd, 1):
            _do_move(agent, out, nd, 1)
            agent["p"] = 0
            return True

    agent["done"] = True
    return False


def _postprocess(out: np.ndarray, cr: int, cc: int, arm_len: int) -> None:
    """Keep the base task's parity/border repairs, then restore the combo center pixel."""
    h, w = out.shape
    even_center = (cr + cc) % 2 == 0

    if not even_center:
        rp = 1 - (cr % 2)
        cp = 1 - (cc % 2)
        rr, ccg = np.indices((h, w))
        center_mask = (rr == cr) & (ccg == cc)
        out[(rr % 2 == rp) & (ccg % 2 == cp) & (~center_mask)] = BG_COLOR
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != CROSS_COLOR or not _is_border((h, w), r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if _in_bounds((h, w), nr, nc) and out[nr, nc] == CROSS_COLOR:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and _is_border((h, w), *nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = BG_COLOR

        br = (h - 1, w - 1)
        if out[br] == BG_COLOR:
            if all(
                not (_in_bounds((h, w), nr, nc) and out[nr, nc] == CROSS_COLOR)
                for nr, nc in neighbors4(br[0], br[1])
            ):
                out[br] = CROSS_COLOR
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = CROSS_COLOR
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = CROSS_COLOR
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == CROSS_COLOR and out[h - 2, c] == BG_COLOR:
                    out[h - 1, c] = BG_COLOR

    out[cr, cc] = CROSS_COLOR


def _fused_metrics(
    inp: np.ndarray, out: np.ndarray
) -> tuple[int, int, int, int, int]:
    """Return (score, center_kept, cleared_arm_cells, launch_cells, arm_len)."""
    cr, cc, arms, arm_len = _find_cross(inp)
    center_kept = int(out[cr, cc] == CROSS_COLOR)

    cleared_arm_cells = 0
    launch_cells = 0
    for outward, cells in arms:
        for r, c in cells:
            cleared_arm_cells += int(out[r, c] == BG_COLOR)
        lr, lc = _add(cells[-1], outward, 1)
        if _in_bounds(out.shape, lr, lc):
            launch_cells += int(out[lr, lc] == CROSS_COLOR)

    score = center_kept + cleared_arm_cells + launch_cells
    return score, center_kept, cleared_arm_cells, launch_cells, arm_len


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the teal cross arms, keep only the teal center seed, then launch detached clockwise spirals.

    Input structure:
    - Background is black (0). The only object is one teal (8) cardinal cross whose center cell is empty.
    - The four equal arms identify a center hole and four tip cells; there are no other foreground objects.

    Rule:
    - Recover the cross center from the densest teal row and column, then read the four arm tip cells.
    - Start the output from a blank grid, restore only the center cell in teal, and erase every original arm pixel.
    - Each tip owns one turtle: it first paints exactly one cell farther outward along its arm direction, then continues the inherited clockwise wall-growth rule with priority right turn > straight > left, length pattern 1-after-turn / otherwise 2, and the same one-cell orthogonal spacing constraint.
    - Finish with the base task's parity/border cleanup, but restore the teal center last so the center never disappears.

    Key insight: The combo keeps the original spiral mechanics but removes the supporting cross scaffold, so the visible clues become a single teal center plus four spirals that begin from detached launch cells beyond the erased stems.
    """

    grid = np.asarray(grid, dtype=int)
    cr, cc, arms, arm_len = _find_cross(grid)
    if arm_len == 0 or len(arms) != 4:
        return grid.copy()

    out = np.zeros_like(grid)
    out[cr, cc] = CROSS_COLOR

    agents = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "forward_first",
                "done": False,
                "p": 0,
            }
        )

    changed = True
    guard = 0
    while changed and guard < 200000:
        guard += 1
        changed = False
        for agent in agents:
            if _step_agent(agent, out):
                changed = True
        if all(agent["done"] for agent in agents):
            break

    _postprocess(out, cr, cc, arm_len)
    for _, cells in arms:
        for r, c in cells:
            out[r, c] = BG_COLOR
    out[cr, cc] = CROSS_COLOR
    return out


# --------------- generator (cx) ---------------

# The generator intentionally narrows the arm/size regime to compact boards where
# the fused behavior is easy to see and the detached starts remain dense enough.
_ARM_TO_SIZES = {
    1: (8, 9),
    2: (9, 10),
}
_FILL_LO = 0.40
_FILL_HI = 0.52
_CANDIDATES: list[np.ndarray] | None = None


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _candidate_centers(size: int, arm: int) -> list[tuple[int, int]]:
    centers = []
    for cr in range(arm + 1, size - arm - 1):
        for cc in range(arm + 1, size - arm - 1):
            centers.append((cr, cc))
    return centers


def _enumerate_candidates() -> list[np.ndarray]:
    candidates = []
    for arm, sizes in _ARM_TO_SIZES.items():
        for size in sizes:
            for cr, cc in _candidate_centers(size, arm):
                inp = _build_input(arm, size, cr, cc)
                out = main(inp)
                fill = float(np.mean(out == CROSS_COLOR))
                score, center_kept, cleared_arm_cells, launch_cells, arm_len = _fused_metrics(inp, out)
                max_score = 1 + 4 * arm_len + 4
                if not (_FILL_LO <= fill <= _FILL_HI):
                    continue
                if center_kept != 1:
                    continue
                if cleared_arm_cells != 4 * arm_len:
                    continue
                if launch_cells != 4:
                    continue
                if score != max_score:
                    continue
                if not np.array_equal(main(inp), out):
                    continue
                candidates.append(inp)
    return candidates


def generate_input() -> np.ndarray:
    global _CANDIDATES
    if _CANDIDATES is None:
        _CANDIDATES = _enumerate_candidates()
    if not _CANDIDATES:
        raise RuntimeError("generate_input: no valid fused samples found")
    return np.array(random.choice(_CANDIDATES), copy=True)

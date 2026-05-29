# Combo variant of da515329 (index 7)
# combo_spec: D3/1+D7/0
# Name: 每圈递增颜色 + 固定圈数终止

import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   preserved_core:
#     input is still a single hollow-centered teal cross on black
#     four agents still start from the arm tips and follow the original
#     right-turn > straight > left priority with 2/1 step lengths
#     spacing is still enforced on a boolean occupancy mask, not on colors
#   fused_change:
#     arm length now controls two linked facts at once:
#       1) how many concentric rectangular spiral bands are allowed to appear
#       2) which color the outermost occupied band receives
#     to make "outer color == total ring count" exact, the allowed Chebyshev
#     radius is set to 2 * arm_len rather than to the larger single-D7 box
#     used by the standalone stopping variant
#   concretization:
#     ring_count = arm_len
#     candidate moves must satisfy max(abs(r-cr), abs(c-cc)) <= 2 * ring_count
#     after occupancy is traced, every occupied cell is recolored by visible
#     rectangular band depth:
#       radii 1-2 -> color 1
#       radii 3-4 -> color 2
#       radii 5-6 -> color 3
#       ...
#     so the largest color is exactly ring_count
#   behavior_trace:
#     example arm_len = 3
#     original task:
#       the spiral keeps growing until it interacts with the true canvas border
#       and then uses parity/border cleanup to finish a near-50% tiling
#     combo variant:
#       the same local turtle logic runs, but any move leaving radius 6 is
#       rejected, so the drawing stops after exactly three visible bands
#       then occupied cells are recolored 1/2/3 by band depth, leaving a black
#       outer frame beyond the third band
#     difference:
#       termination and coloring are coupled through the same quantity rather
#       than being two unrelated post-hoc edits
#   complexity_calibration:
#     the generator uses arm lengths 2..4 only, so every accepted sample shows
#     at least two colors and a nontrivial blank border
#     ring counts stay within 2, 3, 4, giving repeated support without
#     exploding the color vocabulary
#   implicit_assumption_audit:
#     lines that used to test "out != 0" during growth must operate on a
#     monochrome occupancy mask, otherwise already-recolored cells would break
#     collision logic
#     the original parity and border post-processing is removed because it
#     assumes the outer spiral reaches the real canvas edge; in this combo that
#     assumption is false by construction and would destroy the intended buffer
#     colorization assumes the center remains a 1-cell hole, which generator
#     guarantees by drawing only the four cross arms
#   placement_legality:
#     generator chooses size = 4 * arm_len + 8 and places the center so the
#     radius limit box has at least a 2-cell black margin to every border
#     this guarantees no overlap, no out-of-bounds movement, and a visible
#     outer blank region after stopping
#   visual_observability:
#     human solvers can read the rule from few samples because the same band
#     structure simultaneously explains shape extent and color sequence
#     outermost color changes in lockstep with total band count, making the
#     fused dependency directly visible instead of hidden in metadata


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _rings_from_arm(arm_len: int) -> int:
    return int(arm_len)


def _find_center_and_arms(grid: np.ndarray) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]]]:
    h, w = grid.shape
    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

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

    return cr, cc, arms


def _solve_limited_occupancy(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, int]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    occ = grid == 8
    cr, cc, arms = _find_center_and_arms(grid)
    arm_len = max((len(cells) for _, cells in arms), default=0)
    ring_count = _rings_from_arm(arm_len)
    limit_radius = 2 * ring_count

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def within_limit(cell) -> bool:
        r, c = cell
        return max(abs(r - cr), abs(c - cc)) <= limit_radius

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
            if not within_limit((rr, cc2)):
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

    return occ, cr, cc, arm_len, ring_count


def _colorize_by_ring(occ: np.ndarray, cr: int, cc: int) -> np.ndarray:
    out = np.zeros_like(occ, dtype=int)
    for r, c in zip(*np.where(occ)):
        radius = max(abs(r - cr), abs(c - cc))
        out[r, c] = 1 + (radius - 1) // 2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral tracks from a teal cross, stop after a fixed number of bands, and color each band by its index.

    Input structure:
    - Background is black (0). A single teal (8) hollow-centered plus is the only object.
    - The four equal arm lengths provide the shared control value for both extent and coloring.

    Rule:
    - Each arm tip launches a turtle whose initial heading is a clockwise quarter-turn from the arm's outward direction.
    - Motion priority stays right turn > straight > left, with straight lengths 2/1 and the original 1-cell orthogonal spacing rule.
    - Arm length N means the drawing may occupy only Chebyshev radii up to 2N, so exactly N concentric rectangular bands can appear.
    - After tracing occupancy, recolor the occupied cells by band depth: radii 1-2 become color 1, 3-4 become color 2, and so on until color N at the outside.

    Key insight: The same hidden quantity drives both changes. Arm length tells you when the spiral must stop and, because band colors follow radius, it also determines the outermost color.
    """

    occ, cr, cc, _arm_len, _ring_count = _solve_limited_occupancy(grid)
    return _colorize_by_ring(occ, cr, cc)


# --------------- generator (cx) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 2, 4
GRID_MARGIN = 2
GRID_EXTRA = 8
MAX_OUTER_RETRIES = 200
CROSS_COLOR = 8
FILL_LO, FILL_HI = 0.18, 0.29


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _limit_radius_from_arm(arm: int) -> int:
    return 2 * _rings_from_arm(arm)


def _output_metrics(inp: np.ndarray, out: np.ndarray) -> tuple[float, int, int, int, int]:
    cr = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=1)))
    cc = int(np.argmax(np.sum(inp == CROSS_COLOR, axis=0)))
    nz = out[out != 0]
    ys, xs = np.where(out != 0)
    used_radius = int(max(np.max(np.abs(ys - cr)), np.max(np.abs(xs - cc))))
    border_ink = int(
        np.count_nonzero(out[0])
        + np.count_nonzero(out[-1])
        + np.count_nonzero(out[1:-1, 0])
        + np.count_nonzero(out[1:-1, -1])
    )
    return (
        float(np.mean(out != 0)),
        used_radius,
        border_ink,
        int(nz.max(initial=0)),
        int(len(np.unique(nz))),
    )


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    limit_radius = _limit_radius_from_arm(arm)
    size = 2 * limit_radius + GRID_EXTRA

    lo = limit_radius + GRID_MARGIN
    hi = size - limit_radius - GRID_MARGIN - 1
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

        fill, used_radius, border_ink, max_color, ncolors = _output_metrics(inp, out)
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if used_radius != limit_radius:
            continue
        if border_ink != 0:
            continue
        if max_color != arm or ncolors != arm:
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

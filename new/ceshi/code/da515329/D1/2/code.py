# Variant of da515329
# Dimension: D1 — 螺旋方向 (path)
# Change: 四个端点统一按顺时针方向螺旋展开... → 交替方向螺旋

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis
# ---------------
# The input object stays the same: one hollow teal cross on black. The new behavior is
# purely geometric in the output: the visible square-rings alternate handedness, with
# the innermost corridor following the original clockwise spiral, the next ring using
# the mirrored counter-clockwise geometry, and so on. The cleanest way to preserve the
# original step-length / spacing constraints is:
# 1. Solve the original task exactly once to get the clockwise occupancy.
# 2. Solve the horizontally mirrored input with the same solver, then mirror the output
#    back. This produces the counter-clockwise counterpart without having to re-fit the
#    original parity/border post-processing by hand.
# 3. Compute a visible square-ring index from the cross center and arm length, and pick
#    clockwise cells on even ring indices and mirrored cells on odd ring indices.
# This keeps the D1 change isolated, avoids hidden numeric tricks, and is easy for a
# human to infer because every example shows several alternating nested rectangular
# corridors in one color.

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
MAX_OUTER_RETRIES = 300
FILL_LO, FILL_HI = 0.48, 0.52
CROSS_COLOR = 8
MIN_ALT_DIFF = 18


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))
    arm = int(max(np.max(row_counts), np.max(col_counts)) // 2)
    return cr, cc, arm


def _base_solve(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

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

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = CROSS_COLOR
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
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = 0
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != CROSS_COLOR or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == CROSS_COLOR:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0 and all(
            not (inb(nr, nc) and out[nr, nc] == CROSS_COLOR)
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
                if out[h - 1, c] == CROSS_COLOR and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    return out


def _counterclockwise_solve(grid: np.ndarray) -> np.ndarray:
    mirrored_in = np.fliplr(grid)
    mirrored_out = _base_solve(mirrored_in)
    return np.fliplr(mirrored_out)


def _layer_index(shape: tuple[int, int], cr: int, cc: int, arm: int) -> np.ndarray:
    rr, ccg = np.indices(shape)
    cheb = np.maximum(np.abs(rr - cr), np.abs(ccg - cc))
    layer = np.maximum((cheb - arm + 1) // 2, 0)
    return layer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build an alternating-handed rectangular spiral from a teal cross.

    Input structure:
    - bg = 0 (black). One teal (8) cross with equal arms and a one-cell center hole.

    Rule:
    - Interpret the innermost ring around the cross using the original clockwise spiral rule.
    - The next visible square-ring switches to the mirrored counter-clockwise version, then the next switches back, and so on.
    - Ring boundaries are read from the same 1-cell-gap geometry as the seed task, so each extra ring starts 2 cells farther from the cross tips.

    Key insight: You do not need a new local turtle rule for every step. Solve the clockwise and mirrored counter-clockwise spirals on the same input, then alternate which one contributes each visible ring.
    """

    grid = np.asarray(grid, dtype=int)
    cr, cc, arm = _find_cross(grid)
    cw = _base_solve(grid)
    ccw = _counterclockwise_solve(grid)
    layer = _layer_index(grid.shape, cr, cc, arm)
    out = np.where(layer % 2 == 0, cw, ccw)
    out[grid == CROSS_COLOR] = CROSS_COLOR
    return out.astype(int)


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
        out = main(inp)
        fill = float(np.mean(out == CROSS_COLOR))
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        cw = _base_solve(inp)
        layer = _layer_index(inp.shape, cr, cc, arm)
        diff = int(np.sum((layer % 2 == 1) & (out != cw)))
        if diff < MIN_ALT_DIFF:
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

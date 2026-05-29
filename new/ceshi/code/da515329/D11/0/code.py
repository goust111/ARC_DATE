# Variant of da515329
# Dimension: D11 — 多端点的螺旋协调方式 (multi_step)
# Change: 四个端点的螺旋同时独立展开... → 端点按顺序依次展开

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ORDER = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


# Design analysis
# ---------------
# The visible change is not a new color or new object, but scheduling: instead of
# letting all four tips advance round-robin on the same frontier, we finish the upper
# spiral first, then the right, then the lower, then the left. Later spirals therefore
# meet already-painted corridors and must bend or terminate around them. The original
# parity/border repair stage is intentionally removed because it encoded assumptions
# about a nearly simultaneous, almost full tiling; for a sequential variant those
# patches would fabricate pixels no agent truly reached. Generator-side, we search a
# cache of placements that satisfy three human-visible conditions:
# - at least three spirals actually contribute visible cells,
# - at least one later spiral experiences blocking from an earlier trail,
# - fill remains moderate so the sequential truncations are easy to see.

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
FILL_LO, FILL_HI = 0.30, 0.55
MIN_ACTIVE = 3
MIN_LATER_AVOID = 2

_POOL: list[tuple[int, int, int, int]] | None = None


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


def _extract_arms(grid: np.ndarray):
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = {}
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms[d] = cells
    return cr, cc, arms


def _solve_seq(grid: np.ndarray, want_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    _cr, _cc, arms = _extract_arms(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def block_reason(cell, prev, start) -> str | None:
        r, c = cell
        if not inb(r, c):
            return "border"
        if out[r, c] != 0:
            return "occupied"
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return "spacing"
        return None

    def can_move(cur, d, length: int):
        prev = cur
        for t in range(1, length + 1):
            cell = add(cur, d, t)
            reason = block_reason(cell, prev, cur)
            if reason is not None:
                return False, reason
            prev = cell
        return True, None

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc] = CROSS_COLOR
            agent["moves"] += 1
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    agents = []
    for outward in _ORDER:
        cells = arms[outward]
        agents.append(
            {
                "out_dir": outward,
                "cur": cells[-1],
                "dir": _right(outward),
                "phase": "out_first" if len(cells) == 1 else None,
                "p": 0,
                "done": False,
                "moves": 0,
                "avoid": 0,
            }
        )

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False
        if agent["phase"] == "out_first":
            ok, reason = can_move(agent["cur"], agent["out_dir"], 1)
            if ok:
                do_move(agent, agent["out_dir"], 1)
            elif reason is not None:
                agent["avoid"] += 1
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        saw_block = False
        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            ok, reason = can_move(agent["cur"], nd, length)
            if ok:
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            if reason is not None:
                saw_block = True
            if not turned and length == 2:
                ok1, reason1 = can_move(agent["cur"], nd, 1)
                if ok1:
                    do_move(agent, nd, 1)
                    agent["p"] = 0
                    return True
                if reason1 is not None:
                    saw_block = True
        if saw_block:
            agent["avoid"] += 1
        agent["done"] = True
        return False

    for agent in agents:
        guard = 0
        while not agent["done"] and guard < 200000:
            guard += 1
            step_agent(agent)

    if want_stats:
        stats = {
            "active": sum(ag["moves"] > 0 for ag in agents),
            "later_avoid": sum(ag["avoid"] for ag in agents[1:]),
            "moves": [ag["moves"] for ag in agents],
            "avoid": [ag["avoid"] for ag in agents],
        }
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the four spiral sources one after another in top-right-bottom-left order.

    Input structure:
    - bg = 0 (black). One hollow teal cross with four equal cardinal arms.

    Rule:
    - The upper tip finishes its whole spiral first, then the right tip starts, then the lower, then the left.
    - Each individual spiral keeps the seed task's local movement logic: right turn > straight > left, with 2-cell straights and 1-cell post-turn steps.
    - Later spirals must respect already-painted cells from earlier spirals, so they may bend or stop sooner than they would under simultaneous growth.

    Key insight: The geometry still comes from the same spiral turtle, but the order of execution is now part of the rule. Reading-order priority makes earlier spirals act like visible barriers for later ones.
    """

    return _solve_seq(grid, want_stats=False)


def _build_pool() -> list[tuple[int, int, int, int]]:
    pool = []
    for arm in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        for size in range(max(8, 4 * arm), 4 * arm + 2 * math.ceil(arm / 2) + 10, 2):
            lo = arm + 1
            hi = size - arm - 2
            if lo > hi:
                continue
            for cr in range(lo, hi + 1):
                for cc in range(lo, hi + 1):
                    inp = _build_input(arm, size, cr, cc)
                    out, stats = _solve_seq(inp, want_stats=True)
                    fill = float(np.mean(out == CROSS_COLOR))
                    if not (FILL_LO <= fill <= FILL_HI):
                        continue
                    if stats["active"] < MIN_ACTIVE:
                        continue
                    if stats["later_avoid"] < MIN_LATER_AVOID:
                        continue
                    pool.append((arm, size, cr, cc))
    if not pool:
        raise RuntimeError("no valid sequential pool found")
    return pool


def generate_input() -> np.ndarray:
    global _POOL
    if _POOL is None:
        _POOL = _build_pool()
    for _ in range(MAX_OUTER_RETRIES):
        arm, size, cr, cc = random.choice(_POOL)
        grid = _build_input(arm, size, cr, cc)
        out, stats = _solve_seq(grid, want_stats=True)
        fill = float(np.mean(out == CROSS_COLOR))
        if stats["active"] >= MIN_ACTIVE and stats["later_avoid"] >= MIN_LATER_AVOID and FILL_LO <= fill <= FILL_HI:
            return grid
    raise RuntimeError("generate_input: failed after retries")

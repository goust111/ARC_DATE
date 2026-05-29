# Combo variant of da515329 (index 14)
# combo_spec: D8/2+D9/1
# Name: 十字完全保留 + 臂用不同颜色

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

BACKGROUND = 0
SPIRAL_COLOR = 8
SOURCE_COLOR = 2

# Design analysis for the fused variant:
# - Concrete behavior: the base task grows four clockwise teal spirals from the arm tips of
#   a teal cross with a center hole. Here the same tip geometry and spiral trajectories stay
#   intact, but the original cross pixels are protected and later recolored. On a 12x12 arm=1
#   example, the old output turns the arm-tip cells into part of the teal wall system; the new
#   output instead leaves the four original arm cells in place and starts the spiral from the
#   cells immediately beyond / beside them, so the solver can visually separate source and path.
# - Complexity calibration: only one extra structural cue is introduced, a single source color.
#   This keeps the transformation obvious in 2-3 examples: "same spiral, but cross stays and is
#   recolored". No per-arm palette, no parity-dependent recolor, and no hidden color ordering.
# - Hidden assumptions to revisit:
#   1) The original code assumes every nonzero cell can be treated as teal wall material.
#      That is safe during path growth, but no longer safe in the returned grid, so source
#      recoloring is deferred until every geometric decision is complete.
#   2) The parity cleanup can clear any parity class globally. That would accidentally erase
#      protected cross cells, so the source mask is restored after cleanup.
#   3) Generator checks based on "fraction of teal cells" no longer hold because some nonzero
#      cells become SOURCE_COLOR. The generator therefore validates non-background density and
#      explicit source/spiral counts instead.
# - Placement legality: the generator keeps the original size formula and center-gap rule, so the
#   cross remains fully in-bounds and the spiral still has room to develop without overlaps.
# - Visual observability / tidiness: only two active colors remain in the output: teal spiral and
#   red source cross. The source sits exactly on the original input object, so the recolor rule is
#   supported by strong positional and shape cues rather than by invisible numeric conventions.
# - Quantifiable fusion signal: for every generated sample, output must satisfy
#   source_color_count == 4 * arm_len and source_mask ∩ spiral_color_count == 0. Comparing against
#   the base solver should show at least those source cells differ in color, preventing regression
#   to the original task.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the original clockwise spiral field around a teal cross, but keep the cross visible as a separately recolored source object.

    Input structure:
    - bg = 0 (black).
    - A single teal (8) plus-sign cross is present, with equal arm lengths and a 1-cell center hole.

    Rule:
    - Detect the cross hole via the densest teal row/column, then find the four arm tips.
    - Launch one spiral walker from each tip using the base motion rule: prefer right turns, then straight, and only use left turns away from the border; straight steps are length 2 except immediately after a turn.
    - Treat the original cross cells as protected occupied cells during growth, run the same parity / border cleanup as the base task, then restore every original cross pixel and recolor it to red (2). Only non-source cells belong to the teal spiral.

    Key insight: The geometry of the spiral is unchanged; the fusion only separates "where the pattern came from" from "where the walkers travelled" by preserving the source mask and recoloring it after the spiral is finished.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    source_mask = grid == SPIRAL_COLOR

    row_counts = np.sum(source_mask, axis=1)
    col_counts = np.sum(source_mask, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def right(d: tuple[int, int]) -> tuple[int, int]:
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]

    def left(d: tuple[int, int]) -> tuple[int, int]:
        return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != BACKGROUND and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != BACKGROUND:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = SPIRAL_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == SPIRAL_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    if not arms:
        return grid.copy()

    even_center = (cr + cc) % 2 == 0
    arm_len = max((len(cells) for _, cells in arms), default=0)

    agents: list[dict[str, object]] = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "p": 0,
            }
        )

    def step_agent(agent: dict[str, object]) -> bool:
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
        out[(rr % 2 == rp) & (ccg % 2 == cp)] = BACKGROUND
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != SPIRAL_COLOR:
                    continue
                if not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == SPIRAL_COLOR:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = BACKGROUND

        br = (h - 1, w - 1)
        if out[br] == BACKGROUND:
            if all(
                not (inb(nr, nc) and out[nr, nc] == SPIRAL_COLOR)
                for nr, nc in neighbors4(br[0], br[1])
            ):
                out[br] = SPIRAL_COLOR
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = SPIRAL_COLOR
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = SPIRAL_COLOR
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == SPIRAL_COLOR and out[h - 2, c] == BACKGROUND:
                    out[h - 1, c] = BACKGROUND

    out[source_mask] = SOURCE_COLOR
    return out


# --------------- generator (cx) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
NONZERO_FILL_LO, NONZERO_FILL_HI = 0.48, 0.54
MAX_OUTER_RETRIES = 300


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = SPIRAL_COLOR
    grid[cr + offsets, cc] = SPIRAL_COLOR
    grid[cr, cc - offsets] = SPIRAL_COLOR
    grid[cr, cc + offsets] = SPIRAL_COLOR
    return grid


def _output_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, float]:
    arm = int(np.sum(inp == SPIRAL_COLOR) // 4)
    return {
        "arm": arm,
        "nonzero_fill": float(np.mean(out != BACKGROUND)),
        "spiral_count": int(np.sum(out == SPIRAL_COLOR)),
        "source_count": int(np.sum(out == SOURCE_COLOR)),
    }


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
        metrics = _output_metrics(inp, out)
        if not (NONZERO_FILL_LO <= metrics["nonzero_fill"] <= NONZERO_FILL_HI):
            continue
        if metrics["source_count"] != 4 * arm:
            continue
        if metrics["spiral_count"] <= metrics["source_count"]:
            continue
        if np.any((inp == SPIRAL_COLOR) & (out == SPIRAL_COLOR)):
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

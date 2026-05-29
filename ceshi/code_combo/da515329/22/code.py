# Combo variant of da515329 (index 22)
# combo_spec: D3/2+D14/2
# Name: 交替涂色棋盘格 + 间歇涂色留空

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

CROSS_COLOR = 8
ALT_COLOR = 2
SKIP_PERIOD = 3
SKIP_INDEX = 2

# Design analysis for the fused variant:
# - Concrete behavior: keep the original spiral geometry as a hidden dense wall,
#   then project it to a visible layer where each trajectory alternates two colors
#   while every third path cell is omitted. Example: starting from a tip, the first
#   four dense cells would conceptually be red, teal, blank, teal. This differs
#   from the original, where all four would be teal and none would be blank.
# - Complexity calibration: only two foreground colors are used, and the skip rule
#   is fixed to a 2-on / 1-off cadence. That is visually obvious without exploding
#   the hypothesis space for a human solver.
# - Hidden assumptions audit: the original movement code assumes that every painted
#   wall cell blocks future motion. That assumption breaks under intermittent
#   drawing, because visible blanks would incorrectly open corridors. The fix is to
#   simulate motion on a hidden occupancy mask and apply the fused coloring only at
#   the end.
# - Placement legality: the generator still builds a single centered cross with the
#   original size formula and border gaps, so the hidden spiral remains non-
#   overlapping and in-bounds before the visible coloring is applied.
# - Visual observability: output now has exactly three visible states in the spiral
#   area: teal, red, and black skips. The cross remains a stable seed object in the
#   input, while the output shows enough repeated cadence for both rules to be read
#   off from 2-4 examples.
# - Visual cleanliness: the two rules operate on one trajectory only; no extra
#   overlay objects are introduced, so regions do not become ambiguous.
# - Color inference: the alternating rule is structural rather than numeric. We map
#   parity-opposite-to-tip cells to red and same-parity cells to teal, so every
#   orthogonal step flips color exactly like a checkerboard.


def _find_cross_center(grid: np.ndarray) -> tuple[int, int]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    return int(np.argmax(row_counts)), int(np.argmax(col_counts))


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _build_dense_spiral(grid: np.ndarray):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    cr, cc = _find_cross_center(grid)
    wall = grid == CROSS_COLOR

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return pos[0] + d[0] * k, pos[1] + d[1] * k

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and wall[nr, nc] and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or wall[rr, cc2]:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            wall[rr, cc2] = True
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
        wall[(rr % 2 == rp) & (ccg % 2 == cp)] = False
    elif arm_len % 2 == 1:
        to_clear = []
        for r in range(h):
            for c in range(w):
                if not wall[r, c] or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and wall[nr, nc]:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            wall[r, c] = False

        br = (h - 1, w - 1)
        if not wall[br]:
            if all(not (inb(nr, nc) and wall[nr, nc]) for nr, nc in neighbors4(br[0], br[1])):
                wall[br] = True
    else:
        if arm_len >= 2:
            wall[0 : min(h, arm_len - 1), 0] = True
        if arm_len >= 1:
            wall[0, max(0, w - arm_len) : w] = True
        if h >= 2:
            for c in range(w):
                if wall[h - 1, c] and not wall[h - 2, c]:
                    wall[h - 1, c] = False

    return wall, (cr, cc), arms, arm_len


def _extract_paths(wall: np.ndarray, center: tuple[int, int], arms) -> list[list[tuple[int, int]]]:
    h, w = wall.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    paths = []
    for _outward, cells in arms:
        tip = cells[-1]
        prev = cells[-2] if len(cells) >= 2 else center
        cur = tip
        path = []
        seen = {tip}

        while True:
            nxts = []
            for nr, nc in neighbors4(*cur):
                if inb(nr, nc) and wall[nr, nc] and (nr, nc) != prev:
                    nxts.append((nr, nc))
            if not nxts:
                break
            if len(nxts) > 1:
                nxts.sort(key=lambda rc: (abs(rc[0] - center[0]) + abs(rc[1] - center[1]), rc[0], rc[1]), reverse=True)
            nxt = nxts[0]
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        paths.append(path)

    return paths


def _visible_color(cell: tuple[int, int], tip_parity: int) -> int:
    return CROSS_COLOR if (cell[0] + cell[1]) % 2 == tip_parity else ALT_COLOR


def _render_fused_output(grid: np.ndarray, wall: np.ndarray, center: tuple[int, int], arms, arm_len: int) -> np.ndarray:
    out = np.zeros_like(grid)
    out[grid == CROSS_COLOR] = CROSS_COLOR

    tip_parity = (center[0] + center[1] + arm_len) % 2
    visited = set()
    for path in _extract_paths(wall, center, arms):
        for idx, cell in enumerate(path):
            visited.add(cell)
            if idx % SKIP_PERIOD == SKIP_INDEX:
                continue
            out[cell] = _visible_color(cell, tip_parity)

    cross_mask = grid == CROSS_COLOR
    for r, c in np.argwhere(wall):
        cell = (int(r), int(c))
        if cross_mask[cell] or cell in visited:
            continue
        out[cell] = _visible_color(cell, tip_parity)

    return out


def _sample_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, float]:
    wall, center, arms, arm_len = _build_dense_spiral(inp)
    dense_new = wall & ~(inp == CROSS_COLOR)
    dense_new_count = int(np.sum(dense_new))
    visible_new = int(np.sum((out != 0) & dense_new))
    red_new = int(np.sum((out == ALT_COLOR) & dense_new))
    teal_new = int(np.sum((out == CROSS_COLOR) & dense_new))
    total_painted = red_new + teal_new
    skip_ratio = 0.0 if dense_new_count == 0 else 1.0 - (visible_new / dense_new_count)
    red_share = 0.0 if total_painted == 0 else red_new / total_painted
    return {
        "fill": float(np.mean(out != 0)),
        "dense_new": float(dense_new_count),
        "visible_new": float(visible_new),
        "skip_ratio": float(skip_ratio),
        "red_new": float(red_new),
        "teal_new": float(teal_new),
        "red_share": float(red_share),
        "arm_len": float(arm_len),
        "tip_parity": float((center[0] + center[1] + arm_len) % 2),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the original four clockwise spirals, then reveal them as a two-color dashed pattern.

    Input structure:
    - Background is black (0).
    - A single teal (8) plus-sign cross with a hollow center is the only object.

    Rule:
    - First reconstruct the same dense spiral skeleton as the base task: each arm tip launches a clockwise wall-following path with the original turn priority, 2/1 step schedule, and 1-cell spacing rule.
    - Then convert each tip-to-end trajectory into a fused visible pattern: advance cell by cell, leave every third cell black, and color the remaining cells in checkerboard alternation using teal and red.
    - The seed cross stays visible as teal, serving as the anchor from which the dashed bicolor spirals emanate.

    Key insight: The geometry is still the original self-avoiding spiral tiling; the new difficulty is that two independent appearance rules are layered on top of that hidden skeleton, so blanks must not be mistaken for free space and the two colors must be read as parity, not as separate objects.
    """

    wall, center, arms, arm_len = _build_dense_spiral(grid)
    return _render_fused_output(np.asarray(grid, dtype=int), wall, center, arms, arm_len)


# --------------- generator (fused variant) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.31, 0.37
SKIP_LO, SKIP_HI = 0.28, 0.39
RED_SHARE_LO, RED_SHARE_HI = 0.35, 0.65
MIN_NEW_VISIBLE = 18
MAX_OUTER_RETRIES = 300


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _valid_fused_sample(inp: np.ndarray, out: np.ndarray) -> bool:
    metrics = _sample_metrics(inp, out)
    if not (FILL_LO <= metrics["fill"] <= FILL_HI):
        return False
    if metrics["visible_new"] < MIN_NEW_VISIBLE:
        return False
    if not (SKIP_LO <= metrics["skip_ratio"] <= SKIP_HI):
        return False
    if metrics["red_new"] <= 0 or metrics["teal_new"] <= 0:
        return False
    if not (RED_SHARE_LO <= metrics["red_share"] <= RED_SHARE_HI):
        return False
    return True


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
        if _valid_fused_sample(inp, out) and np.array_equal(main(inp), out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

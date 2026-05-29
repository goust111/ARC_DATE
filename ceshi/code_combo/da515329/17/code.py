# Combo variant of da515329 (index 17)
# combo_spec: D7/1+D10/0
# Name: 裁剪到最小外接矩形 + 边缘停止

import math
import random

import numpy as np

from arc_utils import neighbors4

# Design analysis for the fused variant:
# - Concrete behavior trace: with arm=2 on a 16x16 grid and an off-center cross,
#   the base task keeps each spiral agent turning after it first reaches row 0 or
#   col 15, then uses border/parity cleanup to approximate a full-frame tiling.
#   In this combo, the border-touching step is still painted, but that agent is
#   frozen immediately afterward; other agents continue independently. The raw
#   result leaves one or more empty outer strips, and the final output is cropped
#   to the teal footprint's tight bounding box.
# - Complexity calibration: the recognizable part of da515329 stays intact
#   (same four launch points, same clockwise preference, same 2/1 stride rhythm,
#   same one-cell spacing rule). Only the stopping rule and output size change, so
#   the output still reads as a spiral fill rather than a trivial crop of the input.
# - Hidden-assumption review: the original parity/border post-processing relied on
#   "eventually fill almost the whole board", which is false once edge contact is
#   terminal, so that cleanup is removed entirely. The agent loop now treats
#   "current head is on the frame" as a terminal state before any further turns.
# - Generator legality/observability: the input remains a single hollow teal plus
#   with equal arm lengths and generous tip-to-border clearance. The generator
#   accepts only cases where cropping actually removes at least one border strip
#   and the cropped fill stays in a narrow range, making the edge-stop + tight
#   crop behavior visually obvious and keeping the pattern clean.

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _solve_core(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], dict[str, object]]:
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = 8
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == 8:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    agents: list[dict[str, object]] = []
    for outward, cells in arms:
        agents.append(
            {
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "done": False,
                "stopped_on_edge": False,
                "p": 0,
            }
        )

    def step_agent(agent: dict[str, object]) -> bool:
        if agent["done"]:
            return False

        if is_border(*agent["cur"]):
            agent["done"] = True
            agent["stopped_on_edge"] = True
            return False

        if agent["phase"] == "out_first":
            moved = False
            if can_move(agent["cur"], agent["out_dir"], 1):
                do_move(agent, agent["out_dir"], 1)
                moved = True
                if is_border(*agent["cur"]):
                    agent["done"] = True
                    agent["stopped_on_edge"] = True
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_right(agent["dir"]), agent["dir"]]
        dirs.append(_left(agent["dir"]))

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            if can_move(agent["cur"], nd, length):
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                if is_border(*agent["cur"]):
                    agent["done"] = True
                    agent["stopped_on_edge"] = True
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                do_move(agent, nd, 1)
                agent["p"] = 0
                if is_border(*agent["cur"]):
                    agent["done"] = True
                    agent["stopped_on_edge"] = True
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

    pts = np.argwhere(out == 8)
    if pts.size == 0:
        bbox = (0, 0, h - 1, w - 1)
    else:
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))

    r0, c0, r1, c1 = bbox
    meta = {
        "center": (cr, cc),
        "bbox": bbox,
        "margins": (r0, c0, h - 1 - r1, w - 1 - c1),
        "edge_stop_count": sum(bool(ag["stopped_on_edge"]) for ag in agents),
        "agent_done_count": sum(bool(ag["done"]) for ag in agents),
        "input_shape": (h, w),
        "crop_shape": (r1 - r0 + 1, c1 - c0 + 1),
    }
    return out, bbox, meta


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    """Original da515329 solver, kept here only as a generator-side anti-degeneracy oracle."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def spacing_ok(cell: tuple[int, int], prev: tuple[int, int], start: tuple[int, int]) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] != 0 and (nr, nc) != prev and (nr, nc) != start:
                return False
        return True

    def can_move(cur: tuple[int, int], d: tuple[int, int], length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            rr, cc2 = add(cur, d, t)
            if not inb(rr, cc2) or out[rr, cc2] != 0:
                return False
            if not spacing_ok((rr, cc2), prev, cur):
                return False
            prev = (rr, cc2)
        return True

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = 8
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] == 8:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    even_center = (cr + cc) % 2 == 0
    arm_len = max((len(cells) for _, cells in arms), default=0)

    agents: list[dict[str, object]] = []
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

    def step_agent(agent: dict[str, object]) -> bool:
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
        to_clear: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if out[r, c] != 8 or not is_border(r, c):
                    continue
                deg = 0
                nbr = None
                for nr, nc in neighbors4(r, c):
                    if inb(nr, nc) and out[nr, nc] == 8:
                        deg += 1
                        nbr = (nr, nc)
                if deg == 1 and nbr is not None and is_border(*nbr):
                    to_clear.append((r, c))
        for r, c in to_clear:
            out[r, c] = 0

        br = (h - 1, w - 1)
        if out[br] == 0:
            if all(
                not (inb(nr, nc) and out[nr, nc] == 8)
                for nr, nc in neighbors4(br[0], br[1])
            ):
                out[br] = 8
    else:
        if arm_len >= 2:
            out[0 : min(h, arm_len - 1), 0] = 8
        if arm_len >= 1:
            out[0, max(0, w - arm_len) : w] = 8
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == 8 and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four teal spiral paths from a hollow cross, stop each one at first border contact, then crop to the painted footprint.

    Input structure:
    - Background is black (0).
    - A single teal (8) plus sign appears with a one-cell hole at its center.
    - All four arms are cardinal and have equal length.

    Rule:
    - Each arm tip launches a turtle whose initial heading is the clockwise turn from that arm's outward direction.
    - Movement priority remains right turn, then straight, then left; a straight segment normally advances 2 cells, but a move taken immediately after a turn advances only 1 cell.
    - Every painted cell must keep a one-cell orthogonal gap from older teal cells except for the predecessor link that extends the path.
    - Unlike the base task, a turtle stops permanently as soon as its head lands on the outer border. After all turtles stop, crop the result to the smallest rectangle containing every teal cell.

    Key insight:
    - The local self-avoiding spiral mechanism is unchanged; the new behavior comes from truncating each spiral at its first frame contact. Tight cropping then reveals those missing outer bands as the actual output shape.
    """

    out, bbox, _meta = _solve_core(grid)
    r0, c0, r1, c1 = bbox
    return out[r0 : r1 + 1, c0 : c1 + 1]


# --------------- generator (combo) ---------------

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
CROSS_COLOR = 8
MAX_OUTER_RETRIES = 400
FILL_LO, FILL_HI = 0.44, 0.53
AREA_RATIO_LO, AREA_RATIO_HI = 0.65, 0.96


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _accept_instance(inp: np.ndarray, out: np.ndarray, meta: dict[str, object]) -> bool:
    h, w = inp.shape
    ch, cw = out.shape
    if (ch, cw) == (h, w):
        return False

    margins = meta["margins"]
    removed_sides = sum(int(m > 0) for m in margins)
    if removed_sides == 0:
        return False

    crop_area = ch * cw
    area_ratio = crop_area / float(h * w)
    fill = float(np.mean(out == CROSS_COLOR))

    if not (AREA_RATIO_LO <= area_ratio <= AREA_RATIO_HI):
        return False
    if not (FILL_LO <= fill <= FILL_HI):
        return False
    if int(meta["edge_stop_count"]) < 2:
        return False
    if sum(int(m) for m in margins) < 2:
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
            raw, bbox, meta = _solve_core(inp)
        except Exception:
            continue
        r0, c0, r1, c1 = bbox
        out = raw[r0 : r1 + 1, c0 : c1 + 1]
        if np.array_equal(raw, _baseline_main(inp)):
            continue
        if _accept_instance(inp, out, meta):
            if np.array_equal(main(inp), out):
                return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")

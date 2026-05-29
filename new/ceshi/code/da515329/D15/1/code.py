# Variant of da515329
# Dimension: D15 — 螺旋转弯规则 (path)
# Change: 螺旋在遇到边界或间距约束时固定转90度弯... → 转弯方向由障碍物决定

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

CROSS_COLOR = 8
LEFT_MARKER = 2
RIGHT_MARKER = 3
ARM_LEN_MIN, ARM_LEN_MAX = 3, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
MAX_OUTER_RETRIES = 400
FILL_LO, FILL_HI = 0.35, 0.55
MIN_MARKER_USE = 1


# Design analysis
# ---------------
# Instead of letting every bend be implied only by boundary/spacing pressure, this
# variant makes turn direction explicit on the grid: colored one-cell corner markers
# lie on the traversed route. A turtle may step onto marker cells just like background,
# but once it lands there the marker changes the heading for the next segment:
# red (2) means "next heading becomes left of the current move", green (3) means
# "next heading becomes right of the current move". The marker is then absorbed into
# the teal path in the output. This keeps the rule human-visible: examples show the
# same hollow cross plus repeated colored corner hints, and the resulting spiral bends
# differently depending on those colors. Generator-side, we sample candidate marker
# cells from the ordinary unmarked spiral path, then keep only inputs where both marker
# colors are actually used by the modified solver.


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
    return cr, cc, arms


def _solve(grid: np.ndarray, want_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = np.where(grid == CROSS_COLOR, CROSS_COLOR, 0)
    _cr, _cc, arms = _extract_arms(out)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def is_static_block(cell, prev, start) -> bool:
        r, c = cell
        if not inb(r, c):
            return True
        if out[r, c] == CROSS_COLOR:
            return True
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and out[nr, nc] == CROSS_COLOR and (nr, nc) != prev and (nr, nc) != start:
                return True
        return False

    def can_move(cur, d, length: int) -> bool:
        prev = cur
        for t in range(1, length + 1):
            cell = add(cur, d, t)
            if is_static_block(cell, prev, cur):
                return False
            prev = cell
        return True

    stats = {"used_left": 0, "used_right": 0}

    def do_move(agent, d, length: int) -> tuple[int, int]:
        r0, c0 = agent["cur"]
        next_dir = d
        for t in range(1, length + 1):
            rr, cc = r0 + d[0] * t, c0 + d[1] * t
            marker = int(grid[rr, cc])
            if marker == LEFT_MARKER:
                next_dir = _left(d)
                stats["used_left"] += 1
            elif marker == RIGHT_MARKER:
                next_dir = _right(d)
                stats["used_right"] += 1
            out[rr, cc] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)
        return next_dir

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
                agent["dir"] = do_move(agent, agent["out_dir"], 1)
            else:
                agent["dir"] = _right(agent["out_dir"])
            agent["phase"] = None
            agent["p"] = 0
            return True

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            if can_move(agent["cur"], nd, length):
                next_dir = do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = next_dir
                return True
            if not turned and length == 2 and can_move(agent["cur"], nd, 1):
                next_dir = do_move(agent, nd, 1)
                agent["p"] = 0
                agent["dir"] = next_dir
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

    return (out, stats) if want_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the spiral while colored corner tiles decide the next turn direction.

    Input structure:
    - bg = 0 (black). One hollow teal cross plus some colored path markers.
    - Red (2) and green (3) marker cells lie on future spiral routes; they are not final walls yet.

    Rule:
    - Turtles still launch from the four cross tips and still use the seed task's right-turn / straight / left candidate ordering with the same 2/1 step schedule.
    - Marker cells are traversable. When a turtle lands on red, its next heading becomes left of the current move; when it lands on green, its next heading becomes right of the current move.
    - After being traversed, marker cells become ordinary teal path cells in the output.

    Key insight: The colored tiles are explicit steering instructions embedded on the route. You follow the usual spiral motion until a marker is reached, then the marker tells you which way the path bends next.
    """

    return _solve(grid, want_stats=False)


def _base_path_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    out, _stats = _solve(grid, want_stats=True)
    cells = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if out[r, c] == CROSS_COLOR and grid[r, c] == 0:
                cells.append((r, c))
    return cells


def _place_markers(inp: np.ndarray) -> np.ndarray | None:
    cells = _base_path_cells(inp)
    if len(cells) < 10:
        return None
    random.shuffle(cells)
    marked = inp.copy()
    picks = cells[:10]
    for idx, (r, c) in enumerate(picks):
        marked[r, c] = LEFT_MARKER if idx % 2 == 0 else RIGHT_MARKER
    out, stats = _solve(marked, want_stats=True)
    fill = float(np.mean(out == CROSS_COLOR))
    if not (FILL_LO <= fill <= FILL_HI):
        return None
    if stats["used_left"] < MIN_MARKER_USE or stats["used_right"] < MIN_MARKER_USE:
        return None
    return marked


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
        base = _build_input(arm, size, cr, cc)
        for _ in range(20):
            marked = _place_markers(base)
            if marked is not None and np.array_equal(main(marked), _solve(marked, want_stats=False)):
                return marked
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")

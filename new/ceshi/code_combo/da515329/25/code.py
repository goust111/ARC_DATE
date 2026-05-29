# Combo variant of da515329 (index 25)
# combo_spec: D15/1+D17/0
# Name: 固定障碍物 + 转弯方向由障碍物决定

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

CROSS_COLOR = 8
LEFT_OBSTACLE = 2
RIGHT_OBSTACLE = 3
OBSTACLE_COLORS = {LEFT_OBSTACLE, RIGHT_OBSTACLE}

ARM_LEN_MIN, ARM_LEN_MAX = 2, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
MAX_OUTER_RETRIES = 450
OBSTACLE_TRIES_PER_CENTER = 40
TEAL_FILL_LO, TEAL_FILL_HI = 0.33, 0.50
MIN_PATH_DELTA = 6
MIN_SWAP_DELTA = 10


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace
#    - Input still starts from the original hollow teal cross on black.
#    - New elements: isolated red(2) and green(3) one-pixel obstacles are placed on cells that
#      would otherwise be traversed by the spiral. They remain visible in the output.
#    - Fusion rule: a turtle cannot enter an obstacle cell. If the very next cell in its current
#      heading is red, it reads that blocker as "turn left now"; if it is green, it reads it as
#      "turn right now". The obstacle therefore both blocks occupancy and actively chooses the bend.
#    - Small trace: suppose a turtle at (r,c) heading east would normally paint (r,c+1). In the
#      combo input that cell may be green. The base task would paint through it; the obstacle-only
#      variant would merely refuse that step and try the fixed candidate ordering; here the turtle
#      instead keeps the obstacle untouched, pivots south immediately, and paints (r+1,c). That is
#      the coupled behavior this combo needs: the reroute is triggered by a visible blocker, and the
#      direction of that reroute depends on the blocker's color.
# b) Complexity calibration
#    - Only two obstacle colors are used, each with one stable meaning repeated across samples.
#      This keeps the color-to-turn mapping inferable from 2-4 examples.
#    - Obstacles are single pixels and spaced apart, so the scene stays close to the original
#      spiral task rather than becoming a dense maze.
#    - Generator requires both colors to be hit at least once and also requires the output to change
#      if the two obstacle colors are swapped. This prevents a degenerate "obstacles matter only as
#      blockers" dataset.
# c) Hidden assumptions in the original solver
#    - Cross detection by counting teal cells still works because only the cross/path use color 8.
#    - The original movement test assumed any nonzero cell was an occupied wall. That remains valid
#      for occupancy, but spacing checks must only consider teal path cells; otherwise obstacles
#      would incorrectly forbid adjacent legal turns.
#    - The original post-processing freely cleared or painted cells. In the combo variant those edits
#      must only touch teal or black cells, never overwrite obstacle colors.
#    - The original turn candidate list encoded a fixed clockwise bias. Here that bias is kept only
#      for ordinary no-obstacle motion; obstacle-triggered bends bypass that list and use the color
#      instruction directly, otherwise D15 would not meaningfully fuse with D17.
# d) Placement legality
#    - Obstacles are chosen only from black input cells that become interior teal path cells in the
#      obstacle-free output, are far enough from the cross, and are well separated from each other.
#    - Each chosen obstacle is validated by running the combo solver and checking that the obstacle is
#      actually encountered and that the resulting output stays within the intended fill range.
# e) Visual observability
#    - A human sees the same seed cross plus repeated red/green blocker dots. In the outputs those
#      dots remain fixed while nearby teal corridors peel away in color-specific directions. Because
#      the same two colors recur across examples, the mapping from color to turn is observable.
# f) Visual cleanliness
#    - Obstacles are isolated single pixels, not clusters, and the generator keeps them away from the
#      crowded cross core. Outputs therefore contain three crisp layers: black background, teal path,
#      and red/green control blockers.
# g) Change inferability
#    - The only new color semantics are binary and structural: red means one turn sense, green means
#      the other. No hidden ordering or black-cell metadata is used. Generator-side swap-delta checks
#      ensure the color assignment itself, not just obstacle presence, changes the produced spiral.


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


def _extract_cross(grid: np.ndarray):
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
    out = grid.copy()
    cr, cc, arms = _extract_cross(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def is_teal(r: int, c: int) -> bool:
        return out[r, c] == CROSS_COLOR

    def is_obstacle(r: int, c: int) -> bool:
        return out[r, c] in OBSTACLE_COLORS

    def spacing_ok(cell, prev, start) -> bool:
        r, c = cell
        for nr, nc in neighbors4(r, c):
            if inb(nr, nc) and is_teal(nr, nc) and (nr, nc) != prev and (nr, nc) != start:
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

    stats = {
        "used_left": 0,
        "used_right": 0,
        "forced_turns": 0,
        "obstacle_hits": set(),
    }

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc2] = CROSS_COLOR
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

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

    def try_forced_obstacle_turn(agent) -> bool:
        nr, nc = add(agent["cur"], agent["dir"], 1)
        if not inb(nr, nc) or not is_obstacle(nr, nc):
            return False

        color = int(out[nr, nc])
        nd = _left(agent["dir"]) if color == LEFT_OBSTACLE else _right(agent["dir"])
        if not can_move(agent["cur"], nd, 1):
            agent["done"] = True
            return False

        do_move(agent, nd, 1)
        agent["p"] = 1 if agent["p"] == 0 else 0
        agent["dir"] = nd
        stats["forced_turns"] += 1
        stats["obstacle_hits"].add((nr, nc))
        if color == LEFT_OBSTACLE:
            stats["used_left"] += 1
        else:
            stats["used_right"] += 1
        return True

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

        if try_forced_obstacle_turn(agent):
            return True
        if agent["done"]:
            return False

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
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
        clear_mask = (rr % 2 == rp) & (ccg % 2 == cp) & (out == CROSS_COLOR)
        out[clear_mask] = 0
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
            for r in range(min(h, arm_len - 1)):
                if out[r, 0] == 0:
                    out[r, 0] = CROSS_COLOR
        if arm_len >= 1:
            for c in range(max(0, w - arm_len), w):
                if out[0, c] == 0:
                    out[0, c] = CROSS_COLOR
        if h >= 2:
            for c in range(w):
                if out[h - 1, c] == CROSS_COLOR and out[h - 2, c] == 0:
                    out[h - 1, c] = 0

    return (out, stats) if want_stats else out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the spiral while fixed blocker colors decide some of the bends.

    Input structure:
    - bg = 0 (black). One hollow teal cross plus isolated red and green obstacle pixels.
    - Red (2) and green (3) cells are permanent blockers; they are not part of the final teal path.

    Rule:
    - Each arm tip still launches a turtle with the original 2/1 step schedule and the same spacing constraint between teal path cells.
    - If the cell directly ahead is a red obstacle, the turtle does not enter it and instead turns left; if it is a green obstacle, it turns right. The turn step has length 1, matching the seed task's post-turn short step.
    - When there is no obstacle immediately ahead, motion falls back to the original candidate policy: prefer right turn, then straight, then left if not on the border.
    - Obstacles remain unchanged in the output and also block occupancy during the parity/border cleanup phase.

    Key insight: The colored blockers act like tiny steering signs attached to walls. They are visible in both input and output, and the nearby teal corridor bends differently depending on whether the blocking pixel is red or green.
    """

    return _solve(grid, want_stats=False)


def _teal_degree(grid: np.ndarray, r: int, c: int) -> int:
    h, w = grid.shape
    deg = 0
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == CROSS_COLOR:
            deg += 1
    return deg


def _candidate_obstacles(
    base_inp: np.ndarray, base_out: np.ndarray, cr: int, cc: int, arm: int
) -> list[tuple[int, int]]:
    h, w = base_out.shape
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if base_inp[r, c] != 0 or base_out[r, c] != CROSS_COLOR:
                continue
            if _teal_degree(base_out, r, c) < 1:
                continue
            if abs(r - cr) + abs(c - cc) < arm + 3:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _swap_obstacle_colors(inp: np.ndarray) -> np.ndarray:
    swapped = inp.copy()
    left_mask = swapped == LEFT_OBSTACLE
    right_mask = swapped == RIGHT_OBSTACLE
    swapped[left_mask] = RIGHT_OBSTACLE
    swapped[right_mask] = LEFT_OBSTACLE
    return swapped


def _count_path_delta(a: np.ndarray, b: np.ndarray) -> int:
    mask = (a == CROSS_COLOR) != (b == CROSS_COLOR)
    return int(np.count_nonzero(mask))


def _evaluate_single_obstacle(
    base_inp: np.ndarray,
    base_out: np.ndarray,
    cell: tuple[int, int],
    color: int,
) -> tuple[bool, int]:
    test = base_inp.copy()
    r, c = cell
    test[r, c] = color
    out, stats = _solve(test, want_stats=True)
    if (r, c) not in stats["obstacle_hits"]:
        return False, 0
    if out[r, c] != color:
        return False, 0
    delta = _count_path_delta(out, base_out)
    if delta < 2:
        return False, 0
    want = "used_left" if color == LEFT_OBSTACLE else "used_right"
    if stats[want] < 1:
        return False, 0
    return True, delta


def _validated_color_candidates(
    base_inp: np.ndarray,
    base_out: np.ndarray,
    cr: int,
    cc: int,
    arm: int,
) -> dict[int, list[tuple[int, int]]]:
    candidates = _candidate_obstacles(base_inp, base_out, cr, cc, arm)
    good = {LEFT_OBSTACLE: [], RIGHT_OBSTACLE: []}
    for cell in candidates:
        for color in (LEFT_OBSTACLE, RIGHT_OBSTACLE):
            ok, _delta = _evaluate_single_obstacle(base_inp, base_out, cell, color)
            if ok:
                good[color].append(cell)
    return good


def _select_combo_obstacles(
    good: dict[int, list[tuple[int, int]]], desired_count: int
) -> list[tuple[int, int, int]] | None:
    left_cells = list(good[LEFT_OBSTACLE])
    right_cells = list(good[RIGHT_OBSTACLE])
    random.shuffle(left_cells)
    random.shuffle(right_cells)

    pairs = [(LEFT_OBSTACLE, left_cells), (RIGHT_OBSTACLE, right_cells)]
    chosen: list[tuple[int, int, int]] = []
    used = set()

    for color, pool in pairs:
        picked = None
        for r, c in pool:
            if (r, c) in used:
                continue
            if all(abs(r - rr) + abs(c - cc) >= 5 for rr, cc, _ in chosen):
                picked = (r, c, color)
                break
        if picked is None:
            return None
        chosen.append(picked)
        used.add((picked[0], picked[1]))

    extras: list[tuple[int, int, int]] = []
    for color, pool in pairs:
        for r, c in pool:
            if (r, c) not in used:
                extras.append((r, c, color))
    random.shuffle(extras)
    for r, c, color in extras:
        if len(chosen) >= desired_count:
            break
        if all(abs(r - rr) + abs(c - cc) >= 5 for rr, cc, _ in chosen):
            chosen.append((r, c, color))
            used.add((r, c))

    if len(chosen) != desired_count:
        return None
    return chosen


def _metrics(base_out: np.ndarray, inp: np.ndarray) -> dict[str, int | float]:
    out, stats = _solve(inp, want_stats=True)
    swapped_out = main(_swap_obstacle_colors(inp))
    teal_fill = float(np.mean(out == CROSS_COLOR))
    return {
        "used_left": int(stats["used_left"]),
        "used_right": int(stats["used_right"]),
        "forced_turns": int(stats["forced_turns"]),
        "hit_count": len(stats["obstacle_hits"]),
        "path_delta_vs_base": _count_path_delta(out, base_out),
        "swap_delta": _count_path_delta(out, swapped_out),
        "teal_fill": teal_fill,
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
        base_inp = _build_input(arm, size, cr, cc)
        try:
            base_out = main(base_inp)
        except Exception:
            continue

        good = _validated_color_candidates(base_inp, base_out, cr, cc, arm)
        if not good[LEFT_OBSTACLE] or not good[RIGHT_OBSTACLE]:
            continue

        desired_count = 3 if arm <= 2 else 4
        for _ in range(OBSTACLE_TRIES_PER_CENTER):
            chosen = _select_combo_obstacles(good, desired_count)
            if chosen is None:
                break

            inp = base_inp.copy()
            for r, c, color in chosen:
                inp[r, c] = color

            try:
                out = main(inp)
            except Exception:
                continue

            metrics = _metrics(base_out, inp)
            if metrics["used_left"] < 1 or metrics["used_right"] < 1:
                continue
            if metrics["hit_count"] < 2:
                continue
            if metrics["path_delta_vs_base"] < MIN_PATH_DELTA:
                continue
            if metrics["swap_delta"] < MIN_SWAP_DELTA:
                continue
            if not (TEAL_FILL_LO <= metrics["teal_fill"] <= TEAL_FILL_HI):
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

# Combo variant of da515329 (index 13)
# combo_spec: D7/2+D11/0
# Name: 螺旋碰撞即停 + 端点按顺序展开

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ORDER = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


# Design analysis for the combo variant
# -------------------------------------
# variant:
#   concretization:
#     input: unchanged from the source task and both single-dimension references:
#       one hollow teal cross on a black square board.
#     output: still four clockwise, 1-cell-spaced spiral walls, but they are no
#       longer coordinated simultaneously. The upper tip finishes first, then the
#       right tip, then the lower tip, then the left tip.
#     fused_rule: earlier spirals become permanent territory. When a later spiral
#       would place its next pixel onto an earlier trail, or into the 1-cell
#       orthogonal safety halo around an earlier trail, that later spiral stops
#       immediately before the first conflicting pixel instead of searching for a
#       lower-priority detour around that earlier frontier.
#   behavior_trace:
#     example:
#       setup: on a medium board with arm length 2, the upper spiral already wraps
#         around the outside. The right spiral later reaches a 2-cell straight
#         segment whose first cell is safe but whose second cell would touch the
#         upper trail.
#       source task: round-robin growth lets all tips share territory; whichever
#         arm is stepped first in code can indirectly decide who wins contested
#         regions, and parity/border cleanup then repairs the nearly full tiling.
#       D11-only sequential version: the right spiral treats the upper trail as an
#         ordinary obstacle, so a blocked right-turn can still fall through to
#         straight or left and keep carving more cells.
#       combo version: the right spiral may draw that first safe cell, but as soon
#         as the next substep would collide with the already-finished upper trail,
#         it terminates. The lower and left spirals inherit even less free space,
#         so the launch order becomes visible in the final area split.
#   complexity_calibration:
#     preserved:
#       - same single-object input family
#       - same local turtle geometry: clockwise preference, 2-cell straights, and
#         1-cell post-turn steps
#       - same equal arm-length constraint across the four tips
#     adapted:
#       - only four fixed launch positions exist, so the order cue is low-entropy
#         and repeatedly supported across samples
#       - collision-stop is only applied against earlier spirals, not against the
#         seed cross or the current spiral itself; this keeps the visual change
#         interpretable instead of overloading every kind of blockage
#       - the original parity/border patch stage is removed because it assumed the
#         outer ring would be completed; here those writes would fabricate pixels
#         after a legitimate stop event
#   hidden_assumptions_review:
#     original_dependencies_broken:
#       - the round-robin loop assumed all four agents remain coequal frontiers;
#         after D11/0 this is false because later agents always see a larger static
#         wall set than earlier ones
#       - can_move() treated every failed candidate as a generic obstacle that still
#         allowed trying lower-priority directions; after D7/2 this is false when
#         the failure is specifically caused by an earlier spiral frontier
#       - the parity cleanup depended on almost-complete tilings and can erase the
#         deliberate black seams that now encode collision stopping
#     new_assumptions_and_guards:
#       - _extract_arms() assumes exactly one four-armed equal-color cross; the
#         generator constructs only that family
#       - owner tags distinguish seed-cross pixels from spiral pixels so only the
#         latter can trigger a collision-stop
#       - every loop branch that interprets a failure as "collision" is matched by
#         generator-side validation requiring visible later-arm collision stops and
#         a nonzero difference from the relaxed sequential baseline
#   placement_legality:
#     generator enumerates arm lengths, board sizes, and legal cross centers. Every
#     candidate is built in-bounds, solved by the combo solver, and filtered by fill,
#     activity, order-skew, and collision metrics before entering the reusable pool.
#   visual_observability:
#     humans can see two cues at once:
#       1. the top/right/down/left order because earlier spirals occupy visibly more
#          territory on average,
#       2. black seams where later spirals end abruptly against earlier walls.
#     To avoid hidden numeric cues, all ordering is tied to geometric arm position.
#   visual_cleanliness:
#     no new colors or markers are introduced. The board stays sparse enough that the
#     four spiral territories remain legible and do not collapse into noisy overlap.
#   inferability:
#     there is no color remapping dimension here. The only changing signal is shape:
#     repeated samples show the same launch order and the same "stop on first contact
#     with an earlier spiral" rule, which is inferable from 2-4 examples.

ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
FILL_LO, FILL_HI = 0.24, 0.46
MIN_ACTIVE = 3
MIN_LATER_COLLISIONS = 2
MIN_DIFF_FROM_RELAXED = 6
MIN_ORDER_SKEW = 4

_POOL: list[tuple[int, int, int, int]] | None = None


def _add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _extract_arms(
    grid: np.ndarray,
) -> tuple[int, int, dict[tuple[int, int], list[tuple[int, int]]]]:
    h, w = grid.shape
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while inb(r + dr, c + dc) and int(grid[r + dr, c + dc]) == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms[d] = cells

    assert len(arms) == 4, "expected one 4-armed cross"
    arm_lens = {len(cells) for cells in arms.values()}
    assert len(arm_lens) == 1, "expected equal arm lengths"
    return cr, cc, arms


def _solve_sequential(
    grid: np.ndarray,
    *,
    collision_stop: bool,
    want_stats: bool = False,
):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()
    _cr, _cc, arms = _extract_arms(grid)

    owner = np.full((h, w), -1, dtype=int)
    owner[grid == CROSS_COLOR] = -2

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    agents: list[dict[str, object]] = []
    for aid, outward in enumerate(_ORDER):
        cells = arms[outward]
        agents.append(
            {
                "id": aid,
                "out_dir": outward,
                "cur": cells[-1],
                "dir": _right(outward),
                "phase": "out_first" if len(cells) == 1 else None,
                "p": 0,
                "done": False,
                "moves": 0,
                "collision_stop": False,
                "collision_with": set(),
                "static_block_terminations": 0,
            }
        )

    def scan_segment(
        aid: int,
        cur: tuple[int, int],
        d: tuple[int, int],
        length: int,
    ) -> tuple[int, str | None, set[int]]:
        prev = cur
        colliders: set[int] = set()
        for t in range(1, length + 1):
            rr, cc = _add(cur, d, t)
            if not inb(rr, cc):
                return t - 1, "blocked", colliders

            occ_owner = int(owner[rr, cc])
            if occ_owner != -1:
                if collision_stop and 0 <= occ_owner < aid:
                    colliders.add(occ_owner)
                    return t - 1, "collision", colliders
                return t - 1, "blocked", colliders

            for nr, nc in neighbors4(rr, cc):
                if not inb(nr, nc) or (nr, nc) == prev or (nr, nc) == cur:
                    continue
                nbr_owner = int(owner[nr, nc])
                if nbr_owner == -1:
                    continue
                if collision_stop and 0 <= nbr_owner < aid:
                    colliders.add(nbr_owner)
                    return t - 1, "collision", colliders
                return t - 1, "blocked", colliders
            prev = (rr, cc)
        return length, None, colliders

    def do_move(agent: dict[str, object], d: tuple[int, int], length: int) -> None:
        r0, c0 = agent["cur"]
        aid = int(agent["id"])
        for t in range(1, length + 1):
            rr, cc = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc] = CROSS_COLOR
            owner[rr, cc] = aid
            agent["moves"] = int(agent["moves"]) + 1
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    def stop_on_collision(agent: dict[str, object], colliders: set[int]) -> None:
        agent["collision_stop"] = True
        agent["collision_with"] = set(colliders)
        agent["done"] = True

    def finish_normal_step(
        agent: dict[str, object],
        nd: tuple[int, int],
        turned: bool,
        consumed_length: int,
    ) -> None:
        if consumed_length > 0:
            do_move(agent, nd, consumed_length)
        agent["dir"] = nd
        agent["p"] = 1 if (int(agent["p"]) == 0 and turned and consumed_length > 0) else 0

    def step_agent(agent: dict[str, object]) -> None:
        if bool(agent["done"]):
            return

        aid = int(agent["id"])

        if agent["phase"] == "out_first":
            prefix, cause, colliders = scan_segment(aid, agent["cur"], agent["out_dir"], 1)
            if prefix == 1:
                do_move(agent, agent["out_dir"], 1)
            elif cause == "collision":
                stop_on_collision(agent, colliders)
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        saw_static_block = False
        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if int(agent["p"]) == 0 else 1)
            prefix, cause, colliders = scan_segment(aid, agent["cur"], nd, length)

            if prefix == length:
                finish_normal_step(agent, nd, turned, length)
                return

            if cause == "collision":
                if prefix > 0:
                    do_move(agent, nd, prefix)
                stop_on_collision(agent, colliders)
                return

            if not turned and length == 2 and prefix == 1:
                finish_normal_step(agent, nd, turned, 1)
                return

            saw_static_block = True

        if saw_static_block:
            agent["static_block_terminations"] = int(agent["static_block_terminations"]) + 1
        agent["done"] = True

    for agent in agents:
        guard = 0
        while not bool(agent["done"]) and guard < 200000:
            guard += 1
            step_agent(agent)

    if want_stats:
        stats = {
            "active": sum(int(ag["moves"]) > 0 for ag in agents),
            "moves": [int(ag["moves"]) for ag in agents],
            "collision_stop": [bool(ag["collision_stop"]) for ag in agents],
            "collision_count": sum(bool(ag["collision_stop"]) for ag in agents),
            "later_collision_count": sum(bool(ag["collision_stop"]) for ag in agents[1:]),
            "collision_with": [sorted(int(x) for x in ag["collision_with"]) for ag in agents],
            "static_block_terminations": [int(ag["static_block_terminations"]) for ag in agents],
        }
        return out, stats
    return out


def _solve_combo(grid: np.ndarray, want_stats: bool = False):
    return _solve_sequential(grid, collision_stop=True, want_stats=want_stats)


def _solve_relaxed_seq(grid: np.ndarray, want_stats: bool = False):
    return _solve_sequential(grid, collision_stop=False, want_stats=want_stats)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spirals in a fixed order, and stop later spirals when they hit earlier territory.

    Input structure:
    - bg = 0 (black). One teal (8) hollow cross appears with four equal cardinal arms and an empty center.

    Rule:
    - The arm tips launch in the order top, then right, then bottom, then left.
    - Each spiral keeps the seed turtle mechanics: start facing 90 degrees clockwise from the arm direction, prefer right turn > straight > left, use 2-cell straights and 1-cell post-turn steps, and preserve a 1-cell orthogonal gap.
    - Earlier spirals become fixed walls. If a later spiral's next substep would enter an earlier trail or its orthogonal safety halo, that later spiral stops immediately before the collision point instead of detouring around that earlier frontier.

    Key insight:
    - The final picture is controlled by the interaction of schedule and collision. Earlier spirals claim broad territory first, while later spirals end in clean black seams exactly where they would have touched previously drawn spiral walls.
    """

    return _solve_combo(grid, want_stats=False)


def _build_pool() -> list[tuple[int, int, int, int]]:
    pool: list[tuple[int, int, int, int]] = []
    for arm in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        size_lo = max(8, 4 * arm)
        size_hi = 4 * arm + 2 * math.ceil(arm / 2) + 8
        for size in range(size_lo, size_hi + 1, 2):
            lo = arm + 1
            hi = size - arm - 2
            if lo > hi:
                continue
            for cr in range(lo, hi + 1):
                for cc in range(lo, hi + 1):
                    inp = _build_input(arm, size, cr, cc)
                    combo_out, combo_stats = _solve_combo(inp, want_stats=True)
                    relaxed_out, _relaxed_stats = _solve_relaxed_seq(inp, want_stats=True)
                    fill = float(np.mean(combo_out == CROSS_COLOR))
                    diff = int(np.count_nonzero(combo_out != relaxed_out))
                    moves = combo_stats["moves"]
                    if not (FILL_LO <= fill <= FILL_HI):
                        continue
                    if combo_stats["active"] < MIN_ACTIVE:
                        continue
                    if combo_stats["later_collision_count"] < MIN_LATER_COLLISIONS:
                        continue
                    if diff < MIN_DIFF_FROM_RELAXED:
                        continue
                    if max(moves) - min(moves) < MIN_ORDER_SKEW:
                        continue
                    if moves[0] < moves[-1]:
                        continue
                    pool.append((arm, size, cr, cc))
    if not pool:
        raise RuntimeError("no valid combo pool found")
    return pool


def generate_input() -> np.ndarray:
    global _POOL
    if _POOL is None:
        _POOL = _build_pool()
    for _ in range(MAX_OUTER_RETRIES):
        arm, size, cr, cc = random.choice(_POOL)
        inp = _build_input(arm, size, cr, cc)
        combo_out, combo_stats = _solve_combo(inp, want_stats=True)
        relaxed_out, _relaxed_stats = _solve_relaxed_seq(inp, want_stats=True)
        fill = float(np.mean(combo_out == CROSS_COLOR))
        diff = int(np.count_nonzero(combo_out != relaxed_out))
        moves = combo_stats["moves"]
        if combo_stats["active"] < MIN_ACTIVE:
            continue
        if combo_stats["later_collision_count"] < MIN_LATER_COLLISIONS:
            continue
        if diff < MIN_DIFF_FROM_RELAXED:
            continue
        if max(moves) - min(moves) < MIN_ORDER_SKEW:
            continue
        if moves[0] < moves[-1]:
            continue
        if FILL_LO <= fill <= FILL_HI and np.array_equal(main(inp), combo_out):
            return inp
    raise RuntimeError("generate_input: failed after retries")

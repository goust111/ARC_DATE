# Variant of da515329
# Dimension: D7 — 终止条件 (dst)
# Change: 螺旋一直展开直到网格空间被完全占据（无空白可继续画）... → 螺旋之间碰撞即停止

import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the D7 variant
# ----------------------------------
# variant:
#   core_change:
#     old_behavior: spiral arms only stopped when the current grid state blocked them.
#       Because agents were stepped sequentially, one arm could occupy a frontier cell
#       first and the others merely reacted to a static obstacle.
#     new_behavior: when two still-growing arms are about to meet, both terminate before
#       the first conflicting pixel is drawn. The visible result is a stable seam of black
#       cells between opposing spiral fronts.
#   concretization:
#     input: unchanged from the source task: one teal cross with a center hole on black.
#     output: still four clockwise, 1-cell-spaced spiral walls, but they do not continue
#       to wrap around once a rival arm reaches the same frontier. Collision can occur
#       by proposing the same next cell, or by trying to step into the 1-cell safety
#       halo of another still-growing arm.
#   behavior_trace:
#     example:
#       setup: two opposite arms have expanded so that their next legal step would place
#         new cells in adjacent positions across a one-pixel corridor.
#       old_solver: the earlier agent paints first; the later one sees that fresh pixel
#         as an ordinary obstacle and may try another turn later, so the region keeps
#         getting tiled.
#       new_solver: both agents choose their next step from the same frozen frontier, the
#         simultaneous proposal is recognized as a collision, and neither pixel is drawn.
#         The corridor remains as the arm boundary.
#   complexity_calibration:
#     preserved:
#       - same single cross object
#       - same right-turn preference and 2/1 stride schedule
#       - same arm-length and center-search regime
#     adapted:
#       - termination is resolved on synchronous one-cell substeps so collision is visible
#         and not overwritten by update order
#       - the old parity/border patch-up stage is removed because it depended on complete
#         fill reaching the outer ring; keeping it would erase the new stopping seams
#       - generator uses a tighter board than the source task, because early termination
#         would look too sparse on the old 12-26 side-length formula
#   hidden_assumptions_review:
#     original_dependencies:
#       - can_move() assumed all blocking pixels were already final; this is false once
#         rival arms may reach the same frontier in the same round.
#       - the main loop relied on fixed agent order; that order accidentally decided who
#         "won" contested cells. The variant must remove this bias.
#       - parity cleanup and border caps assumed the outermost spiral always finishes its
#         tour. After collision-stop, those writes would fabricate pixels that no arm
#         actually reached.
#     new_assumptions:
#       - the input still contains exactly one 4-armed cross with equal-length arms.
#       - a collision-relevant rival pixel is any cell already owned by an arm whose
#         endpoint is still active; once that rival arm has already terminated, its trail
#         becomes an ordinary static wall.
#       - generate_input() must guarantee at least one collision event, otherwise the
#         sample degenerates to a near-source-task instance.
#   placement_legality:
#     generator keeps the cross fully in bounds on a compact even board and searches only
#     centers with enough tip-to-border clearance. Each candidate is validated by the
#     solver itself, and only samples with at least one collision seam are accepted.
#   visual_observability:
#     the change is visible as unfinished corridors and mirrored stopping fronts. The
#     input stays minimal, so the seam pattern in the output is the only new cue humans
#     need to infer.
#   visual_cleanliness:
#     no extra markers or colors are introduced. Removing the old end-game repair step
#     avoids isolated caps that would look unrelated to the collision rule.


def _add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _extract_arms(grid: np.ndarray) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]]]:
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    row_counts = np.sum(grid == 8, axis=1)
    col_counts = np.sum(grid == 8, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while inb(r + dr, c + dc) and int(grid[r + dr, c + dc]) == 8:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))
    return cr, cc, arms


def _solve(grid: np.ndarray, return_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    cr, cc, arms = _extract_arms(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    agents: list[dict[str, object]] = []
    for aid, (outward, cells) in enumerate(arms):
        agents.append(
            {
                "id": aid,
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "p": 0,
                "done": False,
                "plan_dir": None,
                "plan_remaining": 0,
                "plan_after_dir": None,
                "plan_after_p": 0,
                "plan_after_phase": None,
            }
        )

    owner = np.full((h, w), -1, dtype=int)

    stats = {
        "collision_rounds": 0,
        "collision_agents": 0,
        "front_collision_pairs": 0,
        "proposal_collision_pairs": 0,
        "arm_len": max((len(cells) for _, cells in arms), default=0),
        "center": (cr, cc),
    }

    def clear_plan(agent: dict[str, object]) -> None:
        agent["plan_dir"] = None
        agent["plan_remaining"] = 0
        agent["plan_after_dir"] = None
        agent["plan_after_p"] = 0
        agent["plan_after_phase"] = None

    def set_plan(
        agent: dict[str, object],
        d: tuple[int, int],
        remaining: int,
        after_dir: tuple[int, int],
        after_p: int,
        after_phase,
    ) -> None:
        agent["plan_dir"] = d
        agent["plan_remaining"] = remaining
        agent["plan_after_dir"] = after_dir
        agent["plan_after_p"] = after_p
        agent["plan_after_phase"] = after_phase

    def probe_step(
        agent: dict[str, object],
        d: tuple[int, int],
    ) -> tuple[str, tuple[int, int] | set[int] | None]:
        cur = agent["cur"]
        aid = int(agent["id"])
        rr, cc2 = _add(cur, d)
        if not inb(rr, cc2):
            return ("blocked", None)

        frontier_hits: set[int] = set()
        if int(out[rr, cc2]) != 0:
            other = int(owner[rr, cc2])
            if other != -1 and other != aid and not bool(agents[other]["done"]):
                frontier_hits.add(other)
            else:
                return ("blocked", None)

        for nr, nc in neighbors4(rr, cc2):
            if not inb(nr, nc) or (nr, nc) == cur or int(out[nr, nc]) == 0:
                continue
            other = int(owner[nr, nc])
            if other != -1 and other != aid and not bool(agents[other]["done"]):
                frontier_hits.add(other)
            else:
                return ("blocked", None)

        if frontier_hits:
            return ("front_collision", frontier_hits)
        return ("ok", (rr, cc2))

    def choose_plan(
        agent: dict[str, object],
    ) -> tuple[str, set[int] | None]:
        if agent["phase"] == "out_first":
            status, payload = probe_step(agent, agent["out_dir"])
            if status == "ok":
                set_plan(agent, agent["out_dir"], 1, _right(agent["out_dir"]), 0, None)
                return ("planned", None)
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            if status == "front_collision":
                return ("front_collision", payload)
            return ("skip", None)

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            status, payload = probe_step(agent, nd)
            if status == "front_collision":
                return ("front_collision", payload)
            if status != "ok":
                continue
            turned = nd != agent["dir"]
            remaining = 1 if (turned or int(agent["p"]) == 1) else 2
            after_p = 1 if (int(agent["p"]) == 0 and turned) else 0
            set_plan(agent, nd, remaining, nd, after_p, None)
            return ("planned", None)

        return ("blocked", None)

    guard = 0
    while guard < 200000:
        guard += 1
        if all(bool(agent["done"]) for agent in agents):
            break

        progressed = False
        to_stop: set[int] = set()
        front_pairs: set[frozenset[int]] = set()
        proposal_pairs: set[frozenset[int]] = set()
        proposals: dict[int, tuple[int, int]] = {}

        for agent in agents:
            if bool(agent["done"]):
                continue
            if int(agent["plan_remaining"]) == 0:
                status, payload = choose_plan(agent)
                if status == "blocked":
                    agent["done"] = True
                    progressed = True
                    continue
                if status == "skip":
                    progressed = True
                    continue
                if status == "front_collision":
                    aid = int(agent["id"])
                    to_stop.add(aid)
                    for oid in payload or set():
                        to_stop.add(int(oid))
                        front_pairs.add(frozenset((aid, int(oid))))
                    continue

            if bool(agent["done"]) or int(agent["plan_remaining"]) == 0:
                continue

            status, payload = probe_step(agent, agent["plan_dir"])
            if status == "blocked":
                clear_plan(agent)
                progressed = True
                continue
            if status == "front_collision":
                aid = int(agent["id"])
                to_stop.add(aid)
                for oid in payload or set():
                    to_stop.add(int(oid))
                    front_pairs.add(frozenset((aid, int(oid))))
                continue
            proposals[int(agent["id"])] = payload  # type: ignore[assignment]

        by_cell: dict[tuple[int, int], list[int]] = {}
        for aid, cell in proposals.items():
            by_cell.setdefault(cell, []).append(aid)
        for ids in by_cell.values():
            if len(ids) <= 1:
                continue
            for aid in ids:
                to_stop.add(aid)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    proposal_pairs.add(frozenset((ids[i], ids[j])))

        if to_stop:
            stats["collision_rounds"] += 1
            stats["collision_agents"] += len(to_stop)
            stats["front_collision_pairs"] += len(front_pairs)
            stats["proposal_collision_pairs"] += len(proposal_pairs)
            progressed = True
            for aid in to_stop:
                agent = agents[aid]
                agent["done"] = True
                clear_plan(agent)

        for agent in agents:
            aid = int(agent["id"])
            if aid in to_stop or aid not in proposals or bool(agent["done"]):
                continue
            rr, cc2 = proposals[aid]
            out[rr, cc2] = 8
            owner[rr, cc2] = aid
            agent["cur"] = (rr, cc2)
            agent["plan_remaining"] = int(agent["plan_remaining"]) - 1
            if int(agent["plan_remaining"]) == 0:
                agent["dir"] = agent["plan_after_dir"]
                agent["p"] = int(agent["plan_after_p"])
                agent["phase"] = agent["plan_after_phase"]
                clear_plan(agent)
            progressed = True

        if not progressed:
            break

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spiral walls from the tips of a teal cross, but stop rival fronts at impact.

    Input structure:
    - bg = 0 (black). A single teal (8) plus-sign cross with an empty center sits in the grid.

    Rule:
    - Each arm tip launches a turtle heading 90 degrees clockwise from its outward direction.
    - Motion priority stays right-turn > straight > left (left only away from the border), with the same 2/1 stride rhythm as the source task.
    - Growth is resolved synchronously one pixel at a time: if two active fronts would claim the same next cell, or if one front would enter the safety halo around another still-growing spiral, both fronts stop before drawing the conflicting pixel.

    Key insight:
    - The visible pattern is still made from clockwise self-avoiding spirals, but the real discriminator is the simultaneous collision check: contested frontier pixels remain black, creating permanent seams between competing arms.
    """

    return _solve(grid, return_stats=False)


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_BASE = 6
TIP_GAP_MIN = 2
FILL_LO, FILL_HI = 0.24, 0.41
MIN_COLLISION_ROUNDS = 1
MAX_OUTER_RETRIES = 400
CROSS_COLOR = 8
# --- end CFG -----------------------------------------------------------


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 2 * arm + GRID_SIZE_BASE

    lo = arm + TIP_GAP_MIN
    hi = size - arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    for cr, cc in candidates:
        inp = _build_input(arm, size, cr, cc)
        try:
            out, stats = _solve(inp, return_stats=True)
        except Exception:
            continue
        fill = float(np.mean(out == CROSS_COLOR))
        if stats["collision_rounds"] < MIN_COLLISION_ROUNDS:
            continue
        if not (FILL_LO <= fill <= FILL_HI):
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

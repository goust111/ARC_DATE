# Combo variant of da515329 (index 19)
# combo_spec: D7/2+D11/2
# Name: 链式激活 + 碰撞停止

import math
import random

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ORDER = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left


# Design analysis for the combo variant
# -------------------------------------
# variant:
#   core_change:
#     old_behavior: all four spiral tips expanded in round-robin fashion until the board
#       was essentially tiled, after which parity-specific repair code patched the outer
#       ring.
#     new_behavior: only the upper tip starts. Whenever an active spiral first paints a
#       border cell, the next dormant tip in clockwise order wakes up. Active spirals are
#       then advanced synchronously one pixel at a time. If two active fronts would meet,
#       both terminate before the conflict pixel; if a later spiral would enter a finished
#       earlier trail or its 1-cell halo, only the later spiral terminates.
#   concretization:
#     input: unchanged source input, namely one hollow teal cross on black.
#     output: still built from the same clockwise, 1-cell-spaced turtle motion, but the
#       border is now a trigger rather than just a hard limit. Earlier spirals obtain a
#       head start, so later arms are visibly truncated by seams or by dead ends against
#       already-established corridors.
#   behavior_trace:
#     example:
#       setup: on an 8x8 board with arm length 2 and an off-center cross, the upper
#         spiral reaches the top border after several substeps. That single border touch
#         activates the right spiral while the upper one may continue. When the right
#         spiral later tries to step into the upper spiral's safety halo, the right arm
#         stops before painting the contested pixel. If two currently active heads try to
#         enter the same frontier cell, both heads stop and the shared seam stays black.
#       difference_from_source: the source solver would keep all four tips live from the
#         start and let update order decide contested cells; the combo instead exposes the
#         schedule directly via delayed starts and explicit stop lines.
#   complexity_calibration:
#     preserved:
#       - identical single-object input grammar: one equal-armed teal cross with center
#         hole on a square black board
#       - same local motion logic per spiral: right > straight > left, with the original
#         2/1 stride rhythm
#     adapted:
#       - execution is now staged, so the visible effect is stronger than a mere tie-break
#         change: samples show both delayed activation and collision seams
#       - the old parity/border patch stage is removed because it depended on eventual
#         near-complete tiling; keeping it would fabricate pixels never reached under the
#         new stop protocol
#       - generator searches only compact boards where all four activations occur and at
#         least one collision stop is observable, avoiding ambiguous "almost original"
#         cases
#   hidden_assumptions_review:
#     original_dependencies_broken:
#       - fixed agent order was previously harmless because all growth was implicit and
#         long-lived; here contested frontiers must be resolved from a frozen round state
#       - can_move() used only the current painted board, assuming every obstacle was
#         final. After chaining, some obstacles belong to still-active rivals and must be
#         treated as symmetric collisions instead of static walls
#       - parity cleanup assumed the outer spiral ring eventually closed against the
#         border; the combo can terminate much earlier, so those writes would be wrong
#     new_assumptions_added:
#       - activation order is human-visible and fixed as top -> right -> bottom -> left
#       - owner ids are internal only; generator therefore enforces geometry where the
#         staged starts are readable from shape, not from any hidden color ordering
#       - later-vs-earlier collision asymmetry only applies when the earlier trail is
#         already inactive; active rival collisions remain symmetric
#   placement_legality:
#     generator enumerates arm length, board size, and center position. Every candidate is
#     validated by the combo solver itself, so the produced cross is always in bounds and
#     the resulting output always matches the solver's hidden assumptions.
#   visual_observability:
#     the input stays minimal, so the rule must be inferable from output geometry alone.
#     Requiring four activations plus at least one collision produces samples with clear
#     "first arm reaches edge, next arm appears later, final arms get truncated" evidence.
#   visual_cleanliness:
#     no extra colors or markers are added. Compact boards keep the cascade legible, and
#     synchronous collision handling leaves crisp black seams instead of ragged overwrite
#     artifacts.


ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
SIZE_EXTRA_MIN, SIZE_EXTRA_MAX = 4, 12
MAX_OUTER_RETRIES = 400
CROSS_COLOR = 8
FILL_LO, FILL_HI = 0.22, 0.55
MIN_ACTIVATIONS = 4
MIN_BORDER_TRIGGERS = 3
MIN_COLLISION_SCORE = 1

_POOL: list[tuple[int, int, int, int]] | None = None


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
    return cr, cc, arms


def _solve(grid: np.ndarray, return_stats: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    _cr, _cc, arms = _extract_arms(grid)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    owner = np.full((h, w), -1, dtype=int)
    agents: list[dict[str, object]] = []
    for aid, outward in enumerate(_ORDER):
        cells = arms[outward]
        agents.append(
            {
                "id": aid,
                "cur": cells[-1],
                "dir": _right(outward),
                "out_dir": outward,
                "phase": "out_first" if len(cells) == 1 else None,
                "p": 0,
                "done": False,
                "active": False,
                "triggered": False,
                "plan_dir": None,
                "plan_remaining": 0,
                "plan_after_dir": None,
                "plan_after_p": 0,
                "plan_after_phase": None,
                "stop_reason": None,
            }
        )
    agents[0]["active"] = True
    next_to_activate = 1

    stats = {
        "activations": 1,
        "border_triggers": 0,
        "trail_collision_stops": 0,
        "active_collision_rounds": 0,
        "front_collision_pairs": 0,
        "proposal_collision_pairs": 0,
        "max_active": 1,
        "moves": [0, 0, 0, 0],
        "stop_reason": [None, None, None, None],
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

    def classify_owner(other: int, aid: int) -> str:
        if other == aid:
            return "self"
        if other == -1:
            return "wall"
        if bool(agents[other]["active"]) and not bool(agents[other]["done"]):
            return "active"
        if other < aid:
            return "earlier"
        return "wall"

    def probe_step(
        agent: dict[str, object],
        d: tuple[int, int],
    ) -> tuple[str, tuple[int, int] | set[int] | None]:
        aid = int(agent["id"])
        cur = agent["cur"]
        rr = int(cur[0] + d[0])
        cc2 = int(cur[1] + d[1])
        if not inb(rr, cc2):
            return ("blocked", None)

        active_hits: set[int] = set()
        earlier_hits: set[int] = set()

        if int(out[rr, cc2]) != 0:
            kind = classify_owner(int(owner[rr, cc2]), aid)
            if kind == "active":
                active_hits.add(int(owner[rr, cc2]))
            elif kind == "earlier":
                earlier_hits.add(int(owner[rr, cc2]))
            else:
                return ("blocked", None)

        for nr, nc in neighbors4(rr, cc2):
            if not inb(nr, nc) or (nr, nc) == cur or int(out[nr, nc]) == 0:
                continue
            kind = classify_owner(int(owner[nr, nc]), aid)
            if kind == "active":
                active_hits.add(int(owner[nr, nc]))
            elif kind == "earlier":
                earlier_hits.add(int(owner[nr, nc]))
            else:
                return ("blocked", None)

        if active_hits:
            return ("active_collision", active_hits)
        if earlier_hits:
            return ("trail_collision", earlier_hits)
        return ("ok", (rr, cc2))

    def choose_plan(agent: dict[str, object]) -> tuple[str, set[int] | None]:
        if agent["phase"] == "out_first":
            status, payload = probe_step(agent, agent["out_dir"])
            if status == "ok":
                set_plan(agent, agent["out_dir"], 1, _right(agent["out_dir"]), 0, None)
                return ("planned", None)
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            if status in ("active_collision", "trail_collision"):
                return (status, payload)
            return ("skip", None)

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        for nd in dirs:
            status, payload = probe_step(agent, nd)
            if status in ("active_collision", "trail_collision"):
                return (status, payload)
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
        active_agents = [agent for agent in agents if bool(agent["active"]) and not bool(agent["done"])]
        stats["max_active"] = max(stats["max_active"], len(active_agents))
        if not active_agents:
            break

        progressed = False
        to_stop: dict[int, str] = {}
        front_pairs: set[frozenset[int]] = set()
        proposal_pairs: set[frozenset[int]] = set()
        proposals: dict[int, tuple[int, int]] = {}

        for agent in active_agents:
            if int(agent["plan_remaining"]) == 0:
                status, payload = choose_plan(agent)
                if status == "blocked":
                    agent["done"] = True
                    agent["active"] = False
                    agent["stop_reason"] = "blocked"
                    progressed = True
                    continue
                if status == "skip":
                    progressed = True
                    continue
                if status == "trail_collision":
                    to_stop[int(agent["id"])] = "trail_collision"
                    progressed = True
                    continue
                if status == "active_collision":
                    aid = int(agent["id"])
                    to_stop[aid] = "active_collision"
                    for oid in payload or set():
                        to_stop[int(oid)] = "active_collision"
                        front_pairs.add(frozenset((aid, int(oid))))
                    continue

            if bool(agent["done"]) or int(agent["plan_remaining"]) == 0:
                continue

            status, payload = probe_step(agent, agent["plan_dir"])
            if status == "blocked":
                clear_plan(agent)
                progressed = True
                continue
            if status == "trail_collision":
                to_stop[int(agent["id"])] = "trail_collision"
                progressed = True
                continue
            if status == "active_collision":
                aid = int(agent["id"])
                to_stop[aid] = "active_collision"
                for oid in payload or set():
                    to_stop[int(oid)] = "active_collision"
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
                to_stop[aid] = "active_collision"
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    proposal_pairs.add(frozenset((ids[i], ids[j])))

        if to_stop:
            progressed = True
            if any(reason == "active_collision" for reason in to_stop.values()):
                stats["active_collision_rounds"] += 1
            stats["trail_collision_stops"] += sum(
                reason == "trail_collision" for reason in to_stop.values()
            )
            stats["front_collision_pairs"] += len(front_pairs)
            stats["proposal_collision_pairs"] += len(proposal_pairs)
            for aid, reason in to_stop.items():
                agent = agents[aid]
                agent["done"] = True
                agent["active"] = False
                agent["stop_reason"] = reason
                clear_plan(agent)

        border_hits: list[int] = []
        for agent in agents:
            aid = int(agent["id"])
            if aid in to_stop or aid not in proposals or bool(agent["done"]) or not bool(agent["active"]):
                continue
            rr, cc2 = proposals[aid]
            out[rr, cc2] = CROSS_COLOR
            owner[rr, cc2] = aid
            agent["cur"] = (rr, cc2)
            agent["plan_remaining"] = int(agent["plan_remaining"]) - 1
            stats["moves"][aid] += 1
            if is_border(rr, cc2) and not bool(agent["triggered"]):
                agent["triggered"] = True
                border_hits.append(aid)
            if int(agent["plan_remaining"]) == 0:
                agent["dir"] = agent["plan_after_dir"]
                agent["p"] = int(agent["plan_after_p"])
                agent["phase"] = agent["plan_after_phase"]
                clear_plan(agent)
            progressed = True

        for aid in sorted(border_hits):
            _ = aid
            stats["border_triggers"] += 1
            if next_to_activate < len(agents):
                agents[next_to_activate]["active"] = True
                stats["activations"] += 1
                next_to_activate += 1

        if not progressed:
            break

    stats["stop_reason"] = [agent["stop_reason"] for agent in agents]
    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow teal spiral walls under a border-triggered chain reaction with collision stops.

    Input structure:
    - bg = 0 (black). One teal (8) hollow cross with four equal cardinal arms.

    Rule:
    - Only the upper tip starts. When an active spiral first reaches any border cell, it activates the next tip in clockwise order: top, right, bottom, left.
    - Each active spiral keeps the source task's local turtle rule: prefer right turn, then straight, then left (left only away from the border), with 2-cell straights and 1-cell post-turn steps.
    - Growth is resolved synchronously one pixel at a time. If two active fronts would meet, both stop before the conflict pixel. If a later spiral would touch a finished earlier trail or its 1-cell halo, the later spiral stops there.

    Key insight:
    - The output is no longer a uniform full tiling. Border contact acts as the activation trigger, so earlier spirals carve out territory first and later spirals visibly terminate against those pre-existing corridors.
    """

    return _solve(grid, return_stats=False)


def _build_pool() -> list[tuple[int, int, int, int]]:
    pool: list[tuple[int, int, int, int]] = []
    for arm in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        size_lo = max(8, 2 * arm + SIZE_EXTRA_MIN)
        size_hi = 2 * arm + SIZE_EXTRA_MAX
        for size in range(size_lo, size_hi + 1, 2):
            lo = arm + 1
            hi = size - arm - 2
            if lo > hi:
                continue
            for cr in range(lo, hi + 1):
                for cc in range(lo, hi + 1):
                    inp = _build_input(arm, size, cr, cc)
                    out, stats = _solve(inp, return_stats=True)
                    fill = float(np.mean(out == CROSS_COLOR))
                    collision_score = (
                        int(stats["trail_collision_stops"])
                        + int(stats["front_collision_pairs"])
                        + int(stats["proposal_collision_pairs"])
                    )
                    visible_arms = sum(moves > 0 for moves in stats["moves"])
                    if stats["activations"] < MIN_ACTIVATIONS:
                        continue
                    if stats["border_triggers"] < MIN_BORDER_TRIGGERS:
                        continue
                    if collision_score < MIN_COLLISION_SCORE:
                        continue
                    if visible_arms < 3:
                        continue
                    if stats["max_active"] < 2:
                        continue
                    if not (FILL_LO <= fill <= FILL_HI):
                        continue
                    pool.append((arm, size, cr, cc))
    if not pool:
        raise RuntimeError("no valid chain-collision pool found")
    return pool


def generate_input() -> np.ndarray:
    global _POOL
    if _POOL is None:
        _POOL = _build_pool()

    for _ in range(MAX_OUTER_RETRIES):
        arm, size, cr, cc = random.choice(_POOL)
        inp = _build_input(arm, size, cr, cc)
        out, stats = _solve(inp, return_stats=True)
        fill = float(np.mean(out == CROSS_COLOR))
        collision_score = (
            int(stats["trail_collision_stops"])
            + int(stats["front_collision_pairs"])
            + int(stats["proposal_collision_pairs"])
        )
        visible_arms = sum(moves > 0 for moves in stats["moves"])
        if (
            stats["activations"] >= MIN_ACTIVATIONS
            and stats["border_triggers"] >= MIN_BORDER_TRIGGERS
            and collision_score >= MIN_COLLISION_SCORE
            and visible_arms >= 3
            and stats["max_active"] >= 2
            and FILL_LO <= fill <= FILL_HI
            and np.array_equal(main(inp), out)
        ):
            return inp
    raise RuntimeError("generate_input: failed after retries")

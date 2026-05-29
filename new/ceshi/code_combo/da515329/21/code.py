# Combo variant of da515329 (index 21)
# combo_spec: D7/2+D13/1
# Name: 多个十字 + 碰撞停止

import random
from collections import Counter

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for the combo variant
# -------------------------------------
# variant:
#   fused_rule:
#     input:
#       - the black square grid now contains exactly two teal plus-sign crosses
#       - each cross still has a black center hole and four equal-length cardinal arms
#       - both crosses use the same arm length in the generator to keep the repeated
#         visual cue simple: "same object type, duplicated twice"
#     output:
#       - every arm tip launches the source spiral rule: heading starts as the right
#         turn from the arm's outward direction, then each walker prefers right > straight
#         > left, with straight lengths following the original 2/1 rhythm
#       - all walkers advance on a shared canvas synchronously one pixel at a time
#       - if a walker from cross A would claim the same next cell as a walker from
#         cross B, or would step into the 1-cell moat around a still-growing walker
#         from cross B, the colliding walkers stop before the contested pixel is drawn
#       - same-cross interactions are not treated as "rival collisions"; they remain
#         ordinary self-blocking walls, so the Voronoi-like partition is between crosses
#   behavior_trace:
#     example:
#       setup: two arm-2 crosses sit on opposite sides of the board. In the source task,
#         each cross would have filled its neighborhood independently until borders or
#         older walls blocked it.
#       original_single-cross assumptions:
#         - only one center hole exists, so row/column argmax can locate it
#         - obstacles are static because walkers update in a fixed order
#         - parity/border cleanup can fabricate the final outer ring because the spiral
#           is expected to keep expanding until the board is almost fully tiled
#       combo behavior:
#         - the solver first detects both center holes explicitly
#         - it spawns eight walkers and stores which visible cross each trail belongs to
#         - walkers choose moves from the same frozen frontier; if opposite sources try
#           to close the same corridor in the same round, both fronts terminate and a
#           black seam remains between their territories
#       visible difference:
#         - compared with the source task, the output is no longer a near-50% full
#           tiling from one origin; instead it is two unfinished spiral basins whose
#           boundary is made from collision seams
#   complexity_calibration:
#     - using exactly two crosses keeps the main novelty obvious in 2-4 ARC examples:
#       humans can see repeated source objects and a single large boundary between them
#     - keeping one shared arm length per sample avoids extra latent variables while
#       still allowing arm=1..4 across the dataset
#     - the synchronous collision rule is the only way to make "both stop" observable;
#       a sequential update would secretly encode an order-based winner and would weaken
#       the intended Voronoi-style partition
#   hidden_assumptions_review:
#     broken_from_source:
#       - row/column argmax no longer identifies the unique center hole
#       - one global arm_len / center parity no longer exists
#       - post-hoc border caps and parity erasures would invent pixels that no source
#         actually reached after an early collision stop
#       - can_move() can no longer assume every blocker is already final; rival fronts
#         may arrive simultaneously
#     new_assumptions_and_guards:
#       - each source is a clean plus sign with equal arm lengths; _find_crosses checks
#         local geometry and the generator avoids overlapping or row/column-sharing crosses
#       - a dynamic collision only matters across different visible crosses; ownership is
#         therefore tracked per walker and per source
#       - the generator validates that at least one inter-source collision pair occurs
#         and that both sources claim nontrivial output area, preventing degeneration into
#         two disconnected copies of the original task
#   placement_legality:
#     - centers stay far enough from borders to keep the entire input crosses in-bounds
#     - the two crosses never overlap, never share a row or column, and remain at a
#       medium separation so their spirals are likely to meet before both merely hit borders
#     - every candidate is solver-validated before acceptance
#   visual_observability:
#     - the only visible objects in the input are duplicated teal crosses, so humans can
#       infer "one spiral system per cross"
#     - the output seam is geometric, not hidden-id-based: black corridors appear exactly
#       where two source regions meet
#   visual_cleanliness:
#     - no extra colors or markers are introduced
#     - restricting to two sources avoids unreadable over-fragmentation while still making
#       the fusion qualitatively different from both the source task and the single-axis variants


# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_MIN, GRID_SIZE_MAX = 14, 24
NUM_CROSSES = 2
FILL_LO, FILL_HI = 0.18, 0.42
MIN_COLLISION_PAIRS = 2
MIN_TERMINATED_BY_OTHER = 2
MIN_SOURCE_CELLS_EXTRA = 5
MAX_SOURCE_DOMINANCE = 0.72
MAX_OUTER_RETRIES = 500
MAX_PLACEMENT_ATTEMPTS = 300
CROSS_COLOR = 8
# --- end CFG -----------------------------------------------------------


def _add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _find_crosses(grid: np.ndarray) -> list[dict]:
    """Detect every teal plus sign with a black center hole."""
    h, w = grid.shape
    crosses: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != 0:
                continue
            arms: dict[tuple[int, int], list[tuple[int, int]]] = {}
            lengths: list[int] = []
            ok = True
            for d in _DIRS_URDL:
                dr, dc = d
                rr, cc = r + dr, c + dc
                if int(grid[rr, cc]) != CROSS_COLOR:
                    ok = False
                    break
                cells: list[tuple[int, int]] = []
                while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == CROSS_COLOR:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                arms[d] = cells
                lengths.append(len(cells))
            if ok and min(lengths) > 0 and len(set(lengths)) == 1:
                crosses.append({"center": (r, c), "arm_len": lengths[0], "arms": arms})
    crosses.sort(key=lambda cross: cross["center"])
    return crosses


def _solve(grid: np.ndarray, want_meta: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    crosses = _find_crosses(grid)
    if len(crosses) < 2:
        raise ValueError("expected at least two crosses")

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    agents: list[dict[str, object]] = []
    for src, cross in enumerate(crosses):
        for outward in _DIRS_URDL:
            cells = cross["arms"][outward]
            agents.append(
                {
                    "id": len(agents),
                    "src": src,
                    "arm_cells": list(cells),
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

    owner_agent = np.full((h, w), -1, dtype=int)
    owner_src = np.full((h, w), -1, dtype=int)
    source_counts = Counter()
    for agent in agents:
        aid = int(agent["id"])
        src = int(agent["src"])
        for r, c in agent["arm_cells"]:
            owner_agent[r, c] = aid
            owner_src[r, c] = src
            source_counts[src] += 1

    stats = {
        "num_crosses": len(crosses),
        "collision_rounds": 0,
        "front_collision_pairs": 0,
        "proposal_collision_pairs": 0,
        "terminated_by_other": 0,
        "guard": 0,
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
        agent: dict[str, object], d: tuple[int, int]
    ) -> tuple[str, tuple[int, int] | set[int] | None]:
        cur = agent["cur"]
        aid = int(agent["id"])
        src = int(agent["src"])
        rr, cc = _add(cur, d)
        if not inb(rr, cc):
            return ("blocked", None)

        frontier_hits: set[int] = set()
        if int(out[rr, cc]) != 0:
            other_aid = int(owner_agent[rr, cc])
            other_src = int(owner_src[rr, cc])
            if other_aid != -1 and other_src != src and not bool(agents[other_aid]["done"]):
                frontier_hits.add(other_aid)
            else:
                return ("blocked", None)

        for nr, nc in neighbors4(rr, cc):
            if not inb(nr, nc) or (nr, nc) == cur or int(out[nr, nc]) == 0:
                continue
            other_aid = int(owner_agent[nr, nc])
            other_src = int(owner_src[nr, nc])
            if other_aid != -1 and other_src != src and not bool(agents[other_aid]["done"]):
                frontier_hits.add(other_aid)
            else:
                return ("blocked", None)

        if frontier_hits:
            return ("front_collision", frontier_hits)
        return ("ok", (rr, cc))

    def choose_plan(agent: dict[str, object]) -> tuple[str, set[int] | None]:
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

    def register_pair(
        pair_set: set[frozenset[int]], a: int, b: int, require_other_src: bool = True
    ) -> None:
        if a == b:
            return
        if require_other_src and int(agents[a]["src"]) == int(agents[b]["src"]):
            return
        pair_set.add(frozenset((a, b)))

    guard = 0
    while guard < 200000:
        guard += 1
        stats["guard"] = guard
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
                        register_pair(front_pairs, aid, int(oid))
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
                    register_pair(front_pairs, aid, int(oid))
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
                    register_pair(proposal_pairs, ids[i], ids[j])

        if to_stop:
            stats["collision_rounds"] += 1
            stats["front_collision_pairs"] += len(front_pairs)
            stats["proposal_collision_pairs"] += len(proposal_pairs)
            stats["terminated_by_other"] += sum(
                1
                for aid in to_stop
                if any(
                    aid in pair and int(agents[next(iter(pair - {aid}))]["src"]) != int(agents[aid]["src"])
                    for pair in front_pairs | proposal_pairs
                    if aid in pair
                )
            )
            progressed = True
            for aid in to_stop:
                agent = agents[aid]
                agent["done"] = True
                clear_plan(agent)

        for agent in agents:
            aid = int(agent["id"])
            if aid in to_stop or aid not in proposals or bool(agent["done"]):
                continue
            rr, cc = proposals[aid]
            out[rr, cc] = CROSS_COLOR
            owner_agent[rr, cc] = aid
            owner_src[rr, cc] = int(agent["src"])
            source_counts[int(agent["src"])] += 1
            agent["cur"] = (rr, cc)
            agent["plan_remaining"] = int(agent["plan_remaining"]) - 1
            if int(agent["plan_remaining"]) == 0:
                agent["dir"] = agent["plan_after_dir"]
                agent["p"] = int(agent["plan_after_p"])
                agent["phase"] = agent["plan_after_phase"]
                clear_plan(agent)
            progressed = True

        if not progressed:
            break

    if not want_meta:
        return out

    meta = {
        "crosses": crosses,
        "source_counts": source_counts,
        "stats": stats,
        "fill": float(np.mean(out == CROSS_COLOR)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow clockwise spirals from every teal cross and stop rival fronts where their regions meet.

    Input structure:
    - bg = 0 (black). The grid contains multiple teal (8) plus-sign crosses, each with a black center hole and equal-length arms.

    Rule:
    - Detect every cross and launch one spiral walker from each arm tip, keeping the source task's heading rule, right > straight > left priority, and 2/1 straight-step rhythm.
    - All walkers advance on one shared canvas in synchronous single-pixel substeps so simultaneous frontier checks are resolved fairly.
    - A proposed pixel is legal only if it stays empty and keeps the usual 1-cell orthogonal moat; if the violating frontier belongs to a still-growing different cross, every colliding walker stops before drawing that pixel, leaving a black boundary.

    Key insight:
    - This is no longer one cross tiling the whole board. The important pattern is the multi-source partition: each cross grows the familiar spiral texture only inside the region it reaches before colliding with another cross.
    """

    return _solve(grid, want_meta=False)


def _build_input(size: int, centers: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for cr, cc, arm in centers:
        offsets = np.arange(1, arm + 1)
        grid[cr - offsets, cc] = CROSS_COLOR
        grid[cr + offsets, cc] = CROSS_COLOR
        grid[cr, cc - offsets] = CROSS_COLOR
        grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _choose_size(arm: int) -> int:
    lo = max(GRID_SIZE_MIN, 4 * arm + 8)
    hi = min(GRID_SIZE_MAX, 4 * arm + 12)
    return random.choice(list(range(lo, hi + 1, 2)))


def _placement_ok(
    existing: list[tuple[int, int, int]], r: int, c: int, arm: int, size: int
) -> bool:
    for r0, c0, arm0 in existing:
        reach = max(arm, arm0)
        cheb = max(abs(r - r0), abs(c - c0))
        if r == r0 or c == c0:
            return False
        if cheb < 2 * reach + 3:
            return False
        if cheb > min(size - 4, 3 * reach + 8):
            return False
    return True


def _variant_metrics(inp: np.ndarray) -> dict:
    out, meta = _solve(inp, want_meta=True)
    source_counts = [meta["source_counts"][i] for i in range(len(meta["crosses"]))]
    total = sum(source_counts)
    return {
        "num_crosses": len(meta["crosses"]),
        "arm_lengths": [cross["arm_len"] for cross in meta["crosses"]],
        "fill": float(meta["fill"]),
        "collision_rounds": int(meta["stats"]["collision_rounds"]),
        "front_collision_pairs": int(meta["stats"]["front_collision_pairs"]),
        "proposal_collision_pairs": int(meta["stats"]["proposal_collision_pairs"]),
        "terminated_by_other": int(meta["stats"]["terminated_by_other"]),
        "source_counts": source_counts,
        "dominance": max(source_counts) / total,
        "out": out,
    }


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = _choose_size(arm)

    lo = arm + 2
    hi = size - arm - 3
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        random.shuffle(candidates)
        centers: list[tuple[int, int, int]] = []
        for r, c in candidates:
            if not _placement_ok(centers, r, c, arm, size):
                continue
            centers.append((r, c, arm))
            if len(centers) == NUM_CROSSES:
                break
        if len(centers) != NUM_CROSSES:
            continue

        inp = _build_input(size, centers)
        out, meta = _solve(inp, want_meta=True)
        source_counts = [meta["source_counts"][i] for i in range(len(meta["crosses"]))]
        total = sum(source_counts)
        if len(meta["crosses"]) != NUM_CROSSES:
            continue
        if meta["stats"]["front_collision_pairs"] + meta["stats"]["proposal_collision_pairs"] < MIN_COLLISION_PAIRS:
            continue
        if meta["stats"]["terminated_by_other"] < MIN_TERMINATED_BY_OTHER:
            continue
        if not (FILL_LO <= meta["fill"] <= FILL_HI):
            continue
        if min(source_counts) < 4 * arm + MIN_SOURCE_CELLS_EXTRA:
            continue
        if max(source_counts) / total > MAX_SOURCE_DOMINANCE:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")

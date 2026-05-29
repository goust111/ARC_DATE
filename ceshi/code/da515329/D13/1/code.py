# Variant of da515329
# Dimension: D13 — 十字在网格中的位置 (meta)
# Change: 十字可位于网格中的任意位置（非必须居中），螺旋从实际位置展开并被网格边界截断... → 多个十字同时存在

import random
from collections import Counter

import numpy as np

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Design analysis:
# spec_variant:
#   input:
#     grid: square, even side length 14-26 in the generator below
#     background: black (0)
#     objects:
#       - two or three teal plus-sign crosses, each with a black center hole
#       - every cross keeps the original equal arm-length constraint
#   output:
#     transformation:
#       - every arm tip still launches the original clockwise spiral walker
#       - all walkers share one canvas, so later moves must respect already-drawn
#         walls from every source, not just from their own source
#       - walkers are stepped in visible reading order of cross centers; this makes
#         collision priority observable without adding colors
#   concretization:
#     - In the single-cross task, the empty center hole is unique and the solver can
#       recover it by taking the row/column with the most 8s.
#     - In this variant, there are multiple center holes. The solver must instead
#       detect every local plus-sign pattern and spawn four walkers per cross.
#     - Example: with two arm-2 crosses, the original logic would pick one dominant
#       row/column pair and ignore the second cross entirely; the new logic finds
#       both holes, sorts them top-to-bottom then left-to-right, and alternates all
#       eight walkers on a shared occupancy map.
#   complexity_calibration:
#     - Cross count is limited to 2 or 3 so the new behavior is visible but still
#       inferable from a few ARC examples.
#     - Generator validation requires genuine inter-source blocking so samples do not
#       degenerate into several disconnected copies of the old task.
#   old_assumptions_that_break:
#     - argmax(row_counts) / argmax(col_counts) assumed one cross; now it misses the
#       remaining sources.
#     - the parity-based border cleanup assumed one global center parity; applying it
#       unchanged would let one cross erase or add cells that belong to another.
#     - using a single arm_len variable assumed a single object; now arm length is
#       stored per detected cross.
#   placement_legality:
#     - Generator places centers far enough apart that input crosses never overlap,
#       never share a row or column, and do not create ambiguous fake center holes.
#     - Validation also checks that each source contributes a substantial number of
#       output cells, so no cross is reduced to noise.
#   observability_and_cleanliness:
#     - Priority is based on reading order of the visible cross holes, not on hidden
#       ids or colors.
#     - Shared-color overlap stays legible because the evidence is geometric: one
#       spiral bends or truncates where another earlier spiral has already claimed
#       the corridor.

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
GRID_SIZE_MIN, GRID_SIZE_MAX = 14, 26
NUM_CROSSES_CHOICES = (2, 3)
FILL_LO, FILL_HI = 0.30, 0.60
INTERACTION_MIN = 4
MAX_OWNER_SHARE = 0.72
MAX_OUTER_RETRIES = 400
MAX_PLACEMENT_ATTEMPTS = 250
CROSS_COLOR = 8
# --- end CFG -----------------------------------------------------------


def _turn_right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _turn_left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _find_crosses(grid: np.ndarray) -> list[dict]:
    """Detect every plus-sign center hole and record its four equal arms."""
    h, w = grid.shape
    crosses = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != 0:
                continue
            arms = {}
            lengths = []
            ok = True
            for d in _DIRS_URDL:
                dr, dc = d
                rr, cc = r + dr, c + dc
                if grid[rr, cc] != CROSS_COLOR:
                    ok = False
                    break
                cells = []
                while 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == CROSS_COLOR:
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

    owner = -np.ones((h, w), dtype=int)
    owner_counts = Counter()
    for src, cross in enumerate(crosses):
        for cells in cross["arms"].values():
            for r, c in cells:
                owner[r, c] = src
                owner_counts[src] += 1

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def add(pos, d, k=1):
        return (pos[0] + d[0] * k, pos[1] + d[1] * k)

    def conflict_reason(cell, prev, start, src: int) -> str | None:
        r, c = cell
        occ = owner[r, c]
        if occ != -1:
            if occ != src:
                return "occupied_other"
            if (r, c) != prev and (r, c) != start:
                return "occupied_self"
        for nr, nc in neighbors4(r, c):
            if not inb(nr, nc) or (nr, nc) == prev or (nr, nc) == start:
                continue
            occ2 = owner[nr, nc]
            if occ2 != -1:
                return "adj_other" if occ2 != src else "adj_self"
        return None

    events = Counter()
    terminations = Counter()
    agents = []
    for src, cross in enumerate(crosses):
        for outward in _DIRS_URDL:
            cells = cross["arms"][outward]
            agents.append(
                {
                    "src": src,
                    "cur": cells[-1],
                    "dir": _turn_right(outward),
                    "out_dir": outward,
                    "phase": "out_first" if len(cells) == 1 else None,
                    "done": False,
                    "p": 0,
                }
            )

    def can_move(agent, d, length: int) -> tuple[bool, str | None]:
        prev = agent["cur"]
        for t in range(1, length + 1):
            rr, cc = add(agent["cur"], d, t)
            if not inb(rr, cc):
                return False, "border"
            reason = conflict_reason((rr, cc), prev, agent["cur"], agent["src"])
            if reason is not None:
                return False, reason
            prev = (rr, cc)
        return True, None

    def do_move(agent, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc = r0 + d[0] * t, c0 + d[1] * t
            out[rr, cc] = CROSS_COLOR
            owner[rr, cc] = agent["src"]
            owner_counts[agent["src"]] += 1
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    def note_other_block(reason: str | None) -> None:
        if reason in {"occupied_other", "adj_other"}:
            events["other_blocks"] += 1

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            ok, reason = can_move(agent, agent["out_dir"], 1)
            if ok:
                do_move(agent, agent["out_dir"], 1)
            else:
                note_other_block(reason)
            agent["phase"] = None
            agent["dir"] = _turn_right(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_turn_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_turn_left(agent["dir"]))

        terminal_reason = "stuck"
        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            ok, reason = can_move(agent, nd, length)
            if ok:
                do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                return True
            note_other_block(reason)
            if reason is not None:
                terminal_reason = reason
            if not turned and length == 2:
                ok1, reason1 = can_move(agent, nd, 1)
                if ok1:
                    do_move(agent, nd, 1)
                    agent["p"] = 0
                    return True
                note_other_block(reason1)
                if reason1 is not None:
                    terminal_reason = reason1

        if terminal_reason in {"occupied_other", "adj_other"}:
            terminations["other"] += 1
        elif terminal_reason == "border":
            terminations["border"] += 1
        else:
            terminations["self"] += 1
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

    if not want_meta:
        return out

    meta = {
        "crosses": crosses,
        "owner_counts": owner_counts,
        "events": events,
        "terminations": terminations,
        "interaction_score": events["other_blocks"] + terminations["other"],
        "guard": guard,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch clockwise spiral walls from every teal cross and let them compete for space.

    Input structure:
    - bg = 0 (black). Two or more teal (8) plus-sign crosses appear in the same square grid.
    - Each cross has a black center hole and four equal cardinal arms.

    Rule:
    - Detect every cross, sort the center holes in reading order, and start one agent from each arm tip.
    - Each agent keeps the original motion policy: turn right > go straight > turn left, with left forbidden on the border; straight moves are length 2 except immediately after a turn when they shrink to 1; arm-length-1 crosses still take the initial outward step.
    - All agents draw onto one shared teal wall map. A move is legal only when every new cell stays empty and keeps a 1-cell orthogonal moat from existing teal cells except for the predecessor/start cell, so earlier sources block later ones at collisions.

    Key insight:
    - The local spiral rule is unchanged, but multiple visible source holes create a deterministic multi-source packing process where reading-order priority decides which spiral owns a contested corridor.
    """

    return _solve(grid, want_meta=False)


def _build_input(size: int, centers: list[tuple[int, int, int]]) -> np.ndarray:
    """Build a grid containing multiple plus-sign crosses with center holes."""
    grid = np.zeros((size, size), dtype=int)
    for cr, cc, arm in centers:
        offsets = np.arange(1, arm + 1)
        grid[cr - offsets, cc] = CROSS_COLOR
        grid[cr + offsets, cc] = CROSS_COLOR
        grid[cr, cc - offsets] = CROSS_COLOR
        grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _choose_size(arm: int) -> int:
    base = max(GRID_SIZE_MIN, min(GRID_SIZE_MAX, 4 * arm + 10))
    hi = min(GRID_SIZE_MAX, base + 2)
    return random.choice(list(range(base, hi + 1, 2)))


def _placement_ok(existing: list[tuple[int, int, int]], r: int, c: int, arm: int) -> bool:
    for r0, c0, arm0 in existing:
        reach = max(arm, arm0)
        if r == r0 or c == c0:
            return False
        if max(abs(r - r0), abs(c - c0)) < 2 * reach + 3:
            return False
    return True


def _variant_metrics(inp: np.ndarray) -> dict:
    out, meta = _solve(inp, want_meta=True)
    owner_counts = [meta["owner_counts"][i] for i in range(len(meta["crosses"]))]
    return {
        "num_crosses": len(meta["crosses"]),
        "arm_lengths": [cross["arm_len"] for cross in meta["crosses"]],
        "fill": float(np.mean(out == CROSS_COLOR)),
        "interaction_score": int(meta["interaction_score"]),
        "other_terminations": int(meta["terminations"]["other"]),
        "owner_counts": owner_counts,
        "dominance": max(owner_counts) / sum(owner_counts),
    }


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = _choose_size(arm)
    num_crosses = 2 if arm == ARM_LEN_MAX else random.choice(NUM_CROSSES_CHOICES)

    lo = arm + 1
    hi = size - arm - 2
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        random.shuffle(candidates)
        centers = []
        for r, c in candidates:
            if not _placement_ok(centers, r, c, arm):
                continue
            centers.append((r, c, arm))
            if len(centers) == num_crosses:
                break
        if len(centers) < num_crosses:
            continue

        inp = _build_input(size, centers)
        out, meta = _solve(inp, want_meta=True)
        owner_counts = [meta["owner_counts"][i] for i in range(len(meta["crosses"]))]
        fill = float(np.mean(out == CROSS_COLOR))

        if len(meta["crosses"]) != num_crosses:
            continue
        if meta["interaction_score"] < INTERACTION_MIN:
            continue
        if meta["terminations"]["other"] < 1:
            continue
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if min(owner_counts) < 4 * arm + 4:
            continue
        if max(owner_counts) / sum(owner_counts) > MAX_OWNER_SHARE:
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

# Combo variant of da515329 (index 42)
# combo_spec: D3/0+D11/2+D13/1
# Name: 多个十字 + 链式激活 + 颜色循环

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_UTILS) not in sys.path:
    sys.path.append(str(_SOLVER_UTILS))

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for combo D3/0 + D11/2 + D13/1:
# design:
#   concretization_and_trace:
#     input:
#       - Keep the base family cue: hollow teal crosses on a black square grid.
#       - Upgrade D13 visibly by placing at least two separate crosses; the generator
#         uses two because that already satisfies "multiple sources" while keeping the
#         chain order and collisions easy to read from few-shot examples.
#       - Upgrade D3 visibly by adding a compact top-left 2x2-block legend that lists a
#         short repeating trajectory palette, always [8, secondary].
#     output:
#       - A cross still means "launch four clockwise spiral walkers from the arm tips".
#       - The D11 fusion is applied at the cross level: only the first visible cross
#         (reading order of center holes) is active at the start. Its four arm-tip walkers
#         expand together until all four are stuck; only then does the next cross start.
#       - Each branch records its own path order and is recolored afterward with the legend
#         cycle, restarting from cycle[0] at that branch's first grown cell.
#     small_example_trace:
#       - Original single-cross logic: detect one hole by row/col argmax, launch four teal
#         walkers immediately, and every new cell is written as literal 8.
#       - Multi-cross simultaneous variant: detect several holes, but all crosses compete at
#         once; ownership depends on the interleaving schedule of all agents.
#       - Fused combo: detect several holes, sort them by (row, col), exhaust cross 0 first,
#         then cross 1, then cross 2. Earlier phases can seal corridors before later phases
#         are even allowed to move. After geometry is frozen, branch traces are repainted
#         teal/secondary/teal/secondary... from each tip outward.
#   complexity_calibration:
#     - Use only a 2-color cycle. That is enough to make D3 obvious without turning the task
#       into arbitrary palette memorization.
#     - Keep the generated cross count at 2 and give both input crosses the same arm length.
#       The solver can handle more, but two phases are the cleanest observable chain.
#     - Reading order is a human-visible ordering rule, unlike hidden numeric ids.
#   implicit_assumptions_review:
#     - Base assumption "there is one cross" breaks: row/column argmax would miss the extra
#       holes, so the solver now explicitly detects every local hollow plus-sign.
#     - Base assumption "trajectory color == occupancy marker == 8" breaks: geometry is solved
#       on a boolean occupancy map and a per-source owner map, then colored afterward.
#     - Base assumption "all spirals are active together" breaks: the schedule now has phases,
#       so the solver must run one cross group to completion before spawning the next phase.
#     - Base post-processing based on one global center parity is no longer valid. With multiple
#       sources, such cleanup could erase cells that belong to a different cross, so it is
#       intentionally removed from the combo.
#     - New assumption from the legend: top-left 2x2 blocks should not be mistaken for crosses.
#       Cross detection therefore requires a zero center hole plus four cardinal teal arms.
#   placement_legality:
#     - The legend occupies only rows 0-1 and cols 0-4.
#     - Cross centers are generated far enough from borders and from each other that the input
#       crosses never overlap, never share a row or column, and never touch the legend.
#   observability:
#     - D13 is visible in the input because there are multiple hollow crosses.
#     - D3 is visible in the input because the palette appears as a small explicit legend.
#     - D11 is visible in the output because the first cross gets first claim on shared
#       corridors; the generator explicitly rejects layouts where sequential activation would
#       produce the same output as simultaneous activation.
#   visual_cleanliness:
#     - The output removes the legend entirely and keeps only the cross seeds plus colored
#       trajectories, so the visible structure is not cluttered by template markers.
#     - Generator layouts alternate left/right bands by row to create interaction while still
#       leaving clear empty lanes between seed objects.
#   combo_metric:
#     - A sample is accepted only if the sequential combo output differs from a simultaneous
#       multi-cross baseline by at least MIN_CHAIN_DIFF cells. This guards against degenerate
#       inputs where the chain schedule has no effect.


# --- CFG ---------------------------------------------------------------
CROSS_COLOR = 8
SECONDARY_COLORS = (1, 2, 3, 4, 6, 7)
ARM_LEN_CHOICES = (1, 2, 3)
NUM_CROSSES_CHOICES = (2,)
GRID_JITTER_CHOICES = (0, 2)
MAX_OUTER_RETRIES = 400
MAX_LAYOUT_RETRIES = 120
FILL_LO, FILL_HI = 0.22, 0.56
MIN_CHAIN_DIFF = 8
MIN_SECONDARY_PIXELS = 10
MIN_INTERACTION_SCORE = 3
MAX_DOMINANCE = 0.72
LEGEND_TOP = 0
LEGEND_LEFT = 0
LEGEND_GAP = 1
# --- end CFG -----------------------------------------------------------


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _add(pos, d, k=1):
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _legend_block_cols(idx: int) -> tuple[int, int]:
    c0 = LEGEND_LEFT + idx * (2 + LEGEND_GAP)
    return c0, c0 + 1


def _place_cycle_legend(grid: np.ndarray, cycle: list[int]) -> None:
    for idx, color in enumerate(cycle):
        c0, c1 = _legend_block_cols(idx)
        grid[LEGEND_TOP : LEGEND_TOP + 2, c0 : c1 + 1] = color


def _extract_cycle_legend(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    colors: list[int] = []
    seen_cols: set[int] = set()
    if h < 2:
        return colors
    for c in range(0, max(0, w - 1)):
        if c in seen_cols:
            continue
        block = grid[LEGEND_TOP : LEGEND_TOP + 2, c : c + 2]
        if block.shape != (2, 2):
            continue
        v = int(block[0, 0])
        if v == 0:
            continue
        if np.all(block == v):
            colors.append(v)
            seen_cols.update({c, c + 1})
    return colors


def _find_crosses(grid: np.ndarray) -> list[dict]:
    """Detect every hollow teal plus-sign and return them in reading order."""
    h, w = grid.shape
    crosses = []

    for cr in range(1, h - 1):
        for cc in range(1, w - 1):
            if int(grid[cr, cc]) != 0:
                continue

            arms = {}
            lengths = []
            ok = True
            for d in _DIRS_URDL:
                dr, dc = d
                rr, cc2 = cr + dr, cc + dc
                if not _inb(h, w, rr, cc2) or int(grid[rr, cc2]) != CROSS_COLOR:
                    ok = False
                    break
                cells = []
                while _inb(h, w, rr, cc2) and int(grid[rr, cc2]) == CROSS_COLOR:
                    cells.append((rr, cc2))
                    rr += dr
                    cc2 += dc
                arms[d] = cells
                lengths.append(len(cells))
            if not ok or min(lengths) <= 0 or len(set(lengths)) != 1:
                continue

            arm_len = lengths[0]
            # Reject merged teal shapes by enforcing path-like degree along each arm.
            for cells in arms.values():
                for i, (r, c) in enumerate(cells):
                    allowed = {(cr, cc)}
                    if i > 0:
                        allowed.add(cells[i - 1])
                    if i + 1 < len(cells):
                        allowed.add(cells[i + 1])
                    for nr, nc in neighbors4(r, c):
                        if _inb(h, w, nr, nc) and int(grid[nr, nc]) == CROSS_COLOR:
                            if (nr, nc) not in allowed:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue

            crosses.append({"center": (cr, cc), "arm_len": arm_len, "arms": arms})

    crosses.sort(key=lambda cross: cross["center"])
    return crosses


def _make_agents(crosses: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for src, cross in enumerate(crosses):
        group = []
        for outward in _DIRS_URDL:
            cells = cross["arms"][outward]
            group.append(
                {
                    "src": src,
                    "cur": cells[-1],
                    "dir": _right(outward),
                    "out_dir": outward,
                    "phase": "out_first" if len(cells) == 1 else None,
                    "done": False,
                    "p": 0,
                    "trace": [],
                }
            )
        groups.append(group)
    return groups


def _color_output(shape: tuple[int, int], crosses: list[dict], groups: list[list[dict]], cycle: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for cross in crosses:
        for cells in cross["arms"].values():
            for r, c in cells:
                out[r, c] = CROSS_COLOR
    for group in groups:
        for agent in group:
            for idx, (r, c) in enumerate(agent["trace"]):
                out[r, c] = cycle[idx % len(cycle)]
    return out


def _solve(grid: np.ndarray, sequential_crosses: bool = True, want_meta: bool = False):
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    cycle = _extract_cycle_legend(grid)
    if len(cycle) < 2:
        cycle = [CROSS_COLOR, 2 if CROSS_COLOR != 2 else 1]

    crosses = _find_crosses(grid)
    if len(crosses) < 2:
        raise ValueError("expected at least two hollow crosses")

    occ = np.zeros((h, w), dtype=bool)
    owner = -np.ones((h, w), dtype=int)
    owner_counts = [0 for _ in crosses]

    for src, cross in enumerate(crosses):
        for cells in cross["arms"].values():
            for r, c in cells:
                if occ[r, c]:
                    raise ValueError("overlapping input crosses")
                occ[r, c] = True
                owner[r, c] = src
                owner_counts[src] += 1

    groups = _make_agents(crosses)
    events = Counter()
    terminations = Counter()

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def conflict_reason(cell, prev, start, src: int) -> str | None:
        r, c = cell
        occ_src = int(owner[r, c])
        if occ_src != -1:
            if occ_src != src:
                return "occupied_other"
            if (r, c) != prev and (r, c) != start:
                return "occupied_self"
        for nr, nc in neighbors4(r, c):
            if not _inb(h, w, nr, nc) or (nr, nc) == prev or (nr, nc) == start:
                continue
            occ2 = int(owner[nr, nc])
            if occ2 != -1:
                return "adj_other" if occ2 != src else "adj_self"
        return None

    def note_other_block(reason: str | None) -> None:
        if reason in {"occupied_other", "adj_other"}:
            events["other_blocks"] += 1

    def can_move(agent: dict, d, length: int) -> tuple[bool, str | None]:
        prev = agent["cur"]
        for t in range(1, length + 1):
            rr, cc = _add(agent["cur"], d, t)
            if not _inb(h, w, rr, cc):
                return False, "border"
            reason = conflict_reason((rr, cc), prev, agent["cur"], agent["src"])
            if reason is not None:
                return False, reason
            prev = (rr, cc)
        return True, None

    def do_move(agent: dict, d, length: int) -> None:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc = r0 + d[0] * t, c0 + d[1] * t
            occ[rr, cc] = True
            owner[rr, cc] = agent["src"]
            owner_counts[agent["src"]] += 1
            agent["trace"].append((rr, cc))
        agent["cur"] = (r0 + d[0] * length, c0 + d[1] * length)

    def step_agent(agent: dict) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            ok, reason = can_move(agent, agent["out_dir"], 1)
            if ok:
                do_move(agent, agent["out_dir"], 1)
            else:
                note_other_block(reason)
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

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

    def exhaust(agents: list[dict]) -> int:
        changed = True
        guard = 0
        steps = 0
        while changed and guard < 200000:
            guard += 1
            changed = False
            for agent in agents:
                if step_agent(agent):
                    changed = True
                    steps += 1
            if all(agent["done"] for agent in agents):
                break
        return steps

    phase_steps = []
    if sequential_crosses:
        for group in groups:
            phase_steps.append(exhaust(group))
    else:
        phase_steps.append(exhaust([agent for group in groups for agent in group]))

    out = _color_output((h, w), crosses, groups, cycle)

    if not want_meta:
        return out

    grown_counts = []
    for src, cross in enumerate(crosses):
        seed_count = sum(len(cells) for cells in cross["arms"].values())
        grown_counts.append(owner_counts[src] - seed_count)

    meta = {
        "crosses": crosses,
        "cycle": cycle,
        "owner_counts": owner_counts,
        "grown_counts": grown_counts,
        "events": events,
        "terminations": terminations,
        "interaction_score": int(events["other_blocks"] + terminations["other"]),
        "phase_steps": phase_steps,
        "sequential": sequential_crosses,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow multi-cross spiral phases in sequence and repaint each branch with a repeating palette.

    Input structure:
    - Background is black (0).
    - Two or three hollow teal crosses appear in the same square grid.
    - The top-left corner contains a short 2x2-block legend that lists the trajectory color cycle.

    Rule:
    - Detect every hollow cross and sort the center holes in reading order.
    - For one cross at a time, launch the original four arm-tip spiral walkers together:
      each walker prefers turn-right, then straight, then turn-left away from borders; straight
      motion uses length 2 except immediately after a turn when it shrinks to 1; arm-length-1
      crosses still take the initial outward step.
    - All phases share one occupancy map, so cells claimed by earlier crosses block later crosses.
      After the geometry is finished, recolor each branch from its tip outward with the legend's
      repeating cycle while keeping the input cross arms teal.

    Key insight:
    - The fused task changes three different assumptions at once: there are multiple visible
      sources, they activate one phase at a time instead of all together, and path occupancy is
      separate from visible color so the same spiral mechanics can be replayed with a palette cycle.
    """

    return _solve(grid, sequential_crosses=True, want_meta=False)


def _build_input(size: int, centers: list[tuple[int, int, int]], cycle: list[int]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    _place_cycle_legend(grid, cycle)
    for cr, cc, arm in centers:
        offsets = np.arange(1, arm + 1)
        grid[cr - offsets, cc] = CROSS_COLOR
        grid[cr + offsets, cc] = CROSS_COLOR
        grid[cr, cc - offsets] = CROSS_COLOR
        grid[cr, cc + offsets] = CROSS_COLOR
    return grid


def _choose_size(arm: int, num_crosses: int) -> int:
    base = 4 * arm + 10 + 2 * (num_crosses - 2)
    return base + random.choice(GRID_JITTER_CHOICES)


def _placement_ok(existing: list[tuple[int, int, int]], r: int, c: int, arm: int, size: int) -> bool:
    if r - arm <= 2 or c - arm <= 5:
        return False
    if r + arm >= size - 1 or c + arm >= size - 1:
        return False
    for r0, c0, arm0 in existing:
        reach = max(arm, arm0)
        if r == r0 or c == c0:
            return False
        if max(abs(r - r0), abs(c - c0)) < 2 * reach + 3:
            return False
    return True


def _sample_centers(size: int, arm: int, num_crosses: int) -> list[tuple[int, int, int]] | None:
    lo_r = arm + 4
    hi_r = size - arm - 4
    lo_c = arm + 6
    hi_c = size - arm - 4
    if lo_r > hi_r or lo_c > hi_c:
        return None

    for _ in range(MAX_LAYOUT_RETRIES):
        rows = sorted(random.sample(range(lo_r, hi_r + 1), num_crosses))
        if any(rows[i + 1] - rows[i] < 2 * arm + 2 for i in range(len(rows) - 1)):
            continue

        center_c = (lo_c + hi_c) // 2
        left_band = range(lo_c, max(lo_c + 1, center_c - 1))
        right_band = range(min(center_c + 1, hi_c), hi_c + 1)
        if not left_band or not right_band:
            continue

        cols = []
        for idx in range(num_crosses):
            if idx % 2 == 0:
                band = right_band
            else:
                band = left_band
            cols.append(random.choice(list(band)))

        if len(set(cols)) != len(cols):
            continue

        centers = [(rows[i], cols[i], arm) for i in range(num_crosses)]
        centers.sort()
        if all(_placement_ok(centers[:i], centers[i][0], centers[i][1], arm, size) for i in range(len(centers))):
            return centers
    return None


def _variant_metrics(inp: np.ndarray) -> dict:
    seq_out, seq_meta = _solve(inp, sequential_crosses=True, want_meta=True)
    sim_out, sim_meta = _solve(inp, sequential_crosses=False, want_meta=True)
    chain_diff = int(np.count_nonzero(seq_out != sim_out))
    return {
        "num_crosses": len(seq_meta["crosses"]),
        "fill": float(np.mean(seq_out != 0)),
        "secondary_count": int(np.count_nonzero(seq_out == seq_meta["cycle"][1])),
        "interaction_score": int(seq_meta["interaction_score"]),
        "chain_diff": chain_diff,
        "owner_counts": list(seq_meta["owner_counts"]),
        "grown_counts": list(seq_meta["grown_counts"]),
        "dominance": max(seq_meta["owner_counts"]) / sum(seq_meta["owner_counts"]),
        "phase_steps": list(seq_meta["phase_steps"]),
        "simultaneous_interaction": int(sim_meta["interaction_score"]),
    }


def _gen_attempt() -> np.ndarray | None:
    arm = random.choice(ARM_LEN_CHOICES)
    num_crosses = random.choice(NUM_CROSSES_CHOICES)
    size = _choose_size(arm, num_crosses)
    cycle = [CROSS_COLOR, random.choice(SECONDARY_COLORS)]

    centers = _sample_centers(size, arm, num_crosses)
    if centers is None:
        return None

    inp = _build_input(size, centers, cycle)
    try:
        out = main(inp)
        metrics = _variant_metrics(inp)
    except Exception:
        return None

    if len(_find_crosses(inp)) != num_crosses:
        return None
    if not (FILL_LO <= metrics["fill"] <= FILL_HI):
        return None
    if metrics["secondary_count"] < MIN_SECONDARY_PIXELS:
        return None
    if metrics["interaction_score"] < MIN_INTERACTION_SCORE:
        return None
    if metrics["chain_diff"] < MIN_CHAIN_DIFF:
        return None
    if min(metrics["grown_counts"]) <= 0:
        return None
    if metrics["dominance"] > MAX_DOMINANCE:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")

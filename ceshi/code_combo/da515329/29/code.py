# Combo variant of da515329 (index 29)
# combo_spec: D3/0+D7/0+D11/0
# Name: 周期颜色循环 + 固定圈数 + 顺序展开

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_UTILS) not in sys.path:
    sys.path.append(str(_SOLVER_UTILS))

from arc_utils import neighbors4

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ORDER = [(-1, 0), (0, 1), (1, 0), (0, -1)]


# Design analysis for combo D3/0 + D7/0 + D11/0:
# design:
#   concrete_behavior_and_trace:
#     input:
#       - Keep one hollow cross as the geometric seed.
#       - Add a compact 2x2-block legend in the top-left corner that lists a 2-color cycle.
#       - Arm length still encodes the stopping radius: rings = arm_len.
#     output:
#       - Four spiral agents are still launched from the arm tips, but only one agent runs at a time:
#         every tip first emits one opening move so all four sources are visible, then the ordered
#         expansion continues up first, then right, then down, then left.
#       - Every agent uses the original local turtle rule, but cannot paint outside the arm-length
#         limit box, and it halts once it reaches the arm-length-th return point on its own launch
#         axis. That makes the ring budget branch-local instead of letting the first branch occupy
#         the whole shared box.
#       - New trajectory cells are recolored afterward with a repeating 2-color cycle, restarting
#         from cycle[0] at the first painted cell of each branch.
#     example_trace:
#       - Suppose arm_len = 2. The limit radius becomes 6, so no branch may expand beyond the
#         13x13 box centered on the cross hole.
#       - Original task: all four agents advance round-robin and eventually tile almost half of
#         the whole canvas in one color.
#       - Combo task: the upper branch consumes its allowed 2 rings first, then the right branch
#         starts and must route around the upper branch, and so on; each branch shows an 8/x/8/x
#         color alternation along its own trail.
#       - Visible difference: there is an outer black frame from D7, asymmetric blocking from D11,
#         and repeated local color alternation from D3.
#   complexity_calibration:
#     - Use only a 2-color cycle. That keeps the color rule easy to read while still making D3
#       unambiguously different from the base monochrome task.
#     - Keep rings in the small discrete set 1..4 via arm length, so D7 is visible but not noisy.
#     - In practice, arm lengths 2 and 3 give the clearest balance: all four sources become visible,
#       later branches still show blocking, and the ring stop changes the geometry versus an
#       unlimited ordered rollout. The generator restricts itself to that stable subset.
#   implicit_assumptions_review:
#     - The base solver found the center via argmax on color 8. That breaks once a legend may also
#       contain color 8, so the combo solver must explicitly detect the hollow equal-arm cross.
#     - The base solver also assumed "nonzero means wall written by the main process". In the combo
#       input, nonzero legend pixels must not obstruct geometry, so the solver reconstructs the
#       cross first and solves occupancy from scratch on a boolean mask.
#     - The original parity/border cleanup encoded the assumption that spirals reach the real
#       canvas border under near-simultaneous growth. D7 and D11 both invalidate that assumption,
#       so the combo solver removes those repairs entirely.
#     - Color values must never drive movement legality. Geometry is solved independently from
#       the palette, then replayed with cycle colors. This avoids hidden color-dependent bugs.
#   placement_legality:
#     - The legend occupies rows 0-1 and cols 0-4 only.
#     - The generator chooses the cross center so the fixed ring-limit box starts below row 2 and
#       to the right of col 4, which keeps the legend disjoint from the drawable spiral region.
#     - The same placement also leaves a true black margin on the other sides, so limit stopping is
#       visible and no path ever needs to rely on clipping at the real grid boundary.
#   observability_and_inference:
#     - D3 is explicit because the legend shows the exact cycle and the alternation repeats many
#       times on every branch.
#     - D7 is explicit because the path stops well before the canvas border, leaving a consistent
#       outer black frame whose thickness depends on the arm length.
#     - D11 is explicit because later branches are visibly shorter or bent around earlier ones, and
#       generator filtering rejects cases where sequential and simultaneous outputs would coincide.
#   visual_cleanliness:
#     - The legend disappears in the output, so the final image is only the cross plus spirals.
#     - Limiting the palette to two colors and reserving teal as cycle[0] keeps the seed object and
#       trajectory visually related instead of producing an over-busy multicolor maze.
#   main_assumptions_checked_against_generator:
#     - main() assumes there is exactly one detectable hollow equal-arm cross and at least two
#       legend colors; generate_input() constructs exactly that.
#     - main() assumes the spiral limit box is fully inside the canvas; generate_input() enforces
#       the required top/left/right/bottom margins from the derived radius.
#     - main() assumes D11 is meaningful on generated samples; generate_input() keeps only samples
#       with nonzero order_diff and later-branch interference.


ARM_LEN_MIN, ARM_LEN_MAX = 2, 3
MAX_OUTER_RETRIES = 300
CROSS_COLOR = 8
SECONDARY_COLORS = (1, 2, 3, 4, 6, 7)
LEGEND_TOP = 0
LEGEND_LEFT = 0
LEGEND_GAP = 1
LEGEND_WIDTH = 5
LEGEND_HEIGHT = 2
TOP_MARGIN = 2
LEFT_MARGIN = 5
RIGHT_MARGIN = 2
BOTTOM_MARGIN = 2
FILL_LO, FILL_HI = 0.08, 0.30
MIN_ACTIVE_AGENTS = 4
MIN_LATER_AVOID = 1
MIN_LIMIT_STOPS = 0
MIN_ORDER_DIFF = 10
MIN_LIMIT_DIFF = 5
MIN_SECONDARY_PIXELS = 8

OWNER_EMPTY = -1
OWNER_CROSS = -2

_POOL = None


def _inb(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _right(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d):
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _add(pos, d, k=1):
    return (pos[0] + d[0] * k, pos[1] + d[1] * k)


def _rings_from_arm(arm_len: int) -> int:
    return int(arm_len)


def _limit_radius_from_arm(arm_len: int) -> int:
    return arm_len + 2 * _rings_from_arm(arm_len)


def _legend_block_cols(idx: int) -> tuple[int, int]:
    c0 = LEGEND_LEFT + idx * (2 + LEGEND_GAP)
    return c0, c0 + 1


def _place_cycle_legend(grid: np.ndarray, cycle: list[int]) -> None:
    for idx, color in enumerate(cycle):
        c0, c1 = _legend_block_cols(idx)
        grid[LEGEND_TOP : LEGEND_TOP + 2, c0 : c1 + 1] = color


def _extract_cycle_legend(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    colors: list[int] = []
    cells: set[tuple[int, int]] = set()
    seen_cols: set[int] = set()
    if h < 2:
        return colors, cells

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
            for rr in range(LEGEND_TOP, LEGEND_TOP + 2):
                for cc in range(c, c + 2):
                    cells.add((rr, cc))
    return colors, cells


def _find_cross(grid: np.ndarray) -> dict:
    h, w = grid.shape
    best = None

    for cr in range(1, h - 1):
        for cc in range(1, w - 1):
            if int(grid[cr, cc]) != 0:
                continue

            color = None
            arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
            ok = True
            for d in _DIRS_URDL:
                dr, dc = d
                r, c = cr, cc
                cells = []
                while _inb(h, w, r + dr, c + dc) and int(grid[r + dr, c + dc]) != 0:
                    r += dr
                    c += dc
                    v = int(grid[r, c])
                    if color is None:
                        color = v
                    elif v != color:
                        ok = False
                        break
                    cells.append((r, c))
                if not ok or not cells:
                    ok = False
                    break
                arms.append((d, cells))
            if not ok or color is None:
                continue

            arm_len = len(arms[0][1])
            if any(len(cells) != arm_len for _, cells in arms):
                continue

            for _, cells in arms:
                for i, (r, c) in enumerate(cells):
                    allowed = set()
                    if i > 0:
                        allowed.add(cells[i - 1])
                    if i + 1 < len(cells):
                        allowed.add(cells[i + 1])
                    for nr, nc in neighbors4(r, c):
                        if _inb(h, w, nr, nc) and int(grid[nr, nc]) == color:
                            if (nr, nc) not in allowed:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break

            if not ok:
                continue

            cand = {
                "center": (cr, cc),
                "color": color,
                "arms": arms,
                "arm_len": arm_len,
            }
            if best is None or arm_len > best["arm_len"]:
                best = cand

    if best is None:
        raise ValueError("failed to detect hollow cross")
    return best


def _build_input(arm: int, size: int, cr: int, cc: int, cycle: list[int]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    offsets = np.arange(1, arm + 1)
    grid[cr - offsets, cc] = CROSS_COLOR
    grid[cr + offsets, cc] = CROSS_COLOR
    grid[cr, cc - offsets] = CROSS_COLOR
    grid[cr, cc + offsets] = CROSS_COLOR
    _place_cycle_legend(grid, cycle)
    return grid


def _solve_geometry(
    shape: tuple[int, int],
    cross: dict,
    *,
    sequential: bool,
    use_limit: bool,
    stop_on_rings: bool = True,
    want_stats: bool = False,
):
    h, w = shape
    cr, cc = cross["center"]
    arm_len = cross["arm_len"]
    arm_map = {d: cells for d, cells in cross["arms"]}

    occ = np.zeros((h, w), dtype=bool)
    owner = np.full((h, w), OWNER_EMPTY, dtype=int)
    for _, cells in cross["arms"]:
        for r, c in cells:
            occ[r, c] = True
            owner[r, c] = OWNER_CROSS

    limit_radius = _limit_radius_from_arm(arm_len) if use_limit else None
    if limit_radius is None:
        limit_r0 = limit_r1 = limit_c0 = limit_c1 = 0
    else:
        limit_r0 = cr - limit_radius
        limit_r1 = cr + limit_radius
        limit_c0 = cc - limit_radius
        limit_c1 = cc + limit_radius

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def within_limit(cell) -> bool:
        if limit_radius is None:
            return True
        r, c = cell
        return limit_r0 <= r <= limit_r1 and limit_c0 <= c <= limit_c1

    def block_reason(cell, prev, start, agent_idx: int) -> str | None:
        r, c = cell
        if not _inb(h, w, r, c):
            return "border"
        if not within_limit(cell):
            return "limit"
        if occ[r, c]:
            if owner[r, c] == agent_idx:
                return "occupied_self"
            if owner[r, c] == OWNER_CROSS:
                return "occupied_cross"
            return "occupied_other"
        for nr, nc in neighbors4(r, c):
            if not _inb(h, w, nr, nc) or not occ[nr, nc]:
                continue
            if (nr, nc) == prev or (nr, nc) == start:
                continue
            if owner[nr, nc] == agent_idx:
                return "spacing_self"
            if owner[nr, nc] == OWNER_CROSS:
                return "spacing_cross"
            return "spacing_other"
        return None

    def can_move(cur, d, length: int, agent_idx: int):
        prev = cur
        for t in range(1, length + 1):
            cell = _add(cur, d, t)
            reason = block_reason(cell, prev, cur, agent_idx)
            if reason is not None:
                return False, reason
            prev = cell
        return True, None

    agents = []
    for idx, outward in enumerate(_ORDER):
        cells = arm_map[outward]
        agents.append(
            {
                "idx": idx,
                "out_dir": outward,
                "cur": cells[-1],
                "dir": _right(outward),
                "start_dir": _right(outward),
                "phase": "out_first" if len(cells) == 1 else None,
                "p": 0,
                "done": False,
                "moves": 0,
                "trail": [],
                "avoid_other": 0,
                "hit_limit": 0,
                "rings_goal": _rings_from_arm(arm_len),
                "rings_done": 0,
            }
        )

    def note_reason(agent, reason: str | None) -> None:
        if reason == "limit":
            agent["hit_limit"] += 1
        elif reason is not None and reason.endswith("_other"):
            agent["avoid_other"] += 1

    def advance_ring_if_needed(agent) -> bool:
        if not stop_on_rings:
            return False
        next_ring = agent["rings_done"] + 1
        if next_ring > agent["rings_goal"]:
            return False
        target_radius = arm_len + 2 * next_ring
        target = (
            cr + agent["out_dir"][0] * target_radius,
            cc + agent["out_dir"][1] * target_radius,
        )
        if agent["dir"] == agent["start_dir"] and agent["cur"] == target:
            agent["rings_done"] = next_ring
            if agent["rings_done"] >= agent["rings_goal"]:
                agent["done"] = True
                return True
        return False

    def do_move(agent, d, length: int) -> bool:
        r0, c0 = agent["cur"]
        for t in range(1, length + 1):
            rr, cc2 = r0 + d[0] * t, c0 + d[1] * t
            occ[rr, cc2] = True
            owner[rr, cc2] = agent["idx"]
            agent["trail"].append((rr, cc2))
            agent["moves"] += 1
            agent["cur"] = (rr, cc2)
            agent["dir"] = d
            if advance_ring_if_needed(agent):
                return True
        return False

    def step_agent(agent) -> bool:
        if agent["done"]:
            return False

        if agent["phase"] == "out_first":
            ok, reason = can_move(agent["cur"], agent["out_dir"], 1, agent["idx"])
            if ok:
                do_move(agent, agent["out_dir"], 1)
            else:
                note_reason(agent, reason)
            agent["phase"] = None
            agent["dir"] = _right(agent["out_dir"])
            agent["p"] = 0
            return True

        dirs = [_right(agent["dir"]), agent["dir"]]
        if not is_border(*agent["cur"]):
            dirs.append(_left(agent["dir"]))

        saw_other = False
        saw_limit = False
        for nd in dirs:
            turned = nd != agent["dir"]
            length = 1 if turned else (2 if agent["p"] == 0 else 1)
            ok, reason = can_move(agent["cur"], nd, length, agent["idx"])
            if ok:
                reached_goal = do_move(agent, nd, length)
                agent["p"] = 1 if (agent["p"] == 0 and turned) else 0
                agent["dir"] = nd
                if reached_goal:
                    agent["done"] = True
                return True
            if reason == "limit":
                saw_limit = True
            elif reason is not None and reason.endswith("_other"):
                saw_other = True

            if not turned and length == 2:
                ok1, reason1 = can_move(agent["cur"], nd, 1, agent["idx"])
                if ok1:
                    reached_goal = do_move(agent, nd, 1)
                    agent["p"] = 0
                    agent["dir"] = nd
                    if reached_goal:
                        agent["done"] = True
                    return True
                if reason1 == "limit":
                    saw_limit = True
                elif reason1 is not None and reason1.endswith("_other"):
                    saw_other = True

        if saw_limit:
            agent["hit_limit"] += 1
        if saw_other:
            agent["avoid_other"] += 1
        agent["done"] = True
        return False

    def bootstrap_openings() -> None:
        for agent in agents:
            if agent["done"]:
                continue
            step_agent(agent)

    bootstrap_openings()

    if sequential:
        for agent in agents:
            guard = 0
            while not agent["done"] and guard < 200000:
                guard += 1
                step_agent(agent)
    else:
        changed = True
        guard = 0
        while changed and guard < 200000:
            guard += 1
            changed = False
            for agent in agents:
                if step_agent(agent):
                    changed = True
            if all(agent["done"] for agent in agents):
                break

    trails = [list(agent["trail"]) for agent in agents]
    if want_stats:
        stats = {
            "active": sum(agent["moves"] > 0 for agent in agents),
            "later_avoid": sum(agent["avoid_other"] for agent in agents[1:]),
            "limit_stops": sum(agent["hit_limit"] for agent in agents),
            "moves": [agent["moves"] for agent in agents],
            "avoid_other": [agent["avoid_other"] for agent in agents],
            "limit_radius": limit_radius,
            "trajectory_cells": sum(len(trail) for trail in trails),
        }
        return occ, trails, stats
    return occ, trails


def _color_solution(
    shape: tuple[int, int],
    cross: dict,
    trails: list[list[tuple[int, int]]],
    cycle: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for _, cells in cross["arms"]:
        for r, c in cells:
            out[r, c] = cross["color"]

    for trail in trails:
        for idx, (r, c) in enumerate(trail):
            out[r, c] = cycle[idx % len(cycle)]
    return out


def _used_radius(mask: np.ndarray, center: tuple[int, int]) -> int:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0
    cr, cc = center
    return int(max(np.max(np.abs(ys - cr)), np.max(np.abs(xs - cc))))


def _border_ink(mask: np.ndarray) -> int:
    return int(
        np.count_nonzero(mask[0])
        + np.count_nonzero(mask[-1])
        + np.count_nonzero(mask[1:-1, 0])
        + np.count_nonzero(mask[1:-1, -1])
    )


def behavior_metrics(grid: np.ndarray) -> dict:
    grid = np.asarray(grid, dtype=int)
    cycle, _ = _extract_cycle_legend(grid)
    cross = _find_cross(grid)
    if len(cycle) < 2:
        cycle = [cross["color"], 1 if cross["color"] != 1 else 2]

    seq_occ, seq_trails, seq_stats = _solve_geometry(
        grid.shape, cross, sequential=True, use_limit=True, stop_on_rings=True, want_stats=True
    )
    sim_occ, _, _ = _solve_geometry(
        grid.shape, cross, sequential=False, use_limit=True, stop_on_rings=True, want_stats=True
    )
    free_occ, _, _ = _solve_geometry(
        grid.shape, cross, sequential=True, use_limit=True, stop_on_rings=False, want_stats=True
    )
    out = _color_solution(grid.shape, cross, seq_trails, cycle)
    mask = out != 0

    metrics = {
        "fill": float(np.mean(mask)),
        "secondary_count": int(np.count_nonzero(out == cycle[1])),
        "used_radius": _used_radius(mask, cross["center"]),
        "border_ink": _border_ink(mask),
        "order_diff": int(np.count_nonzero(seq_occ != sim_occ)),
        "limit_diff": int(np.count_nonzero(seq_occ != free_occ)),
        "active_agents": int(seq_stats["active"]),
        "later_avoid": int(seq_stats["later_avoid"]),
        "limit_stops": int(seq_stats["limit_stops"]),
        "trajectory_cells": int(seq_stats["trajectory_cells"]),
        "limit_radius": int(seq_stats["limit_radius"]),
        "arm_len": int(cross["arm_len"]),
    }
    return metrics


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a hollow cross into four clockwise spirals, but do it sequentially, stop after a fixed number of rings, and repaint each branch with a visible color cycle.

    Input structure:
    - Background is black (0).
    - One hollow cross is the geometric seed; all four arms have the same length and keep the original teal color.
    - A small top-left legend lists the repeating trajectory palette as two solid 2x2 blocks.

    Rule:
    - Read the arm length from the cross and convert it to a fixed ring budget: rings = arm_len.
    - First let every tip emit one opening move so all four sources are established. After that, continue in strict top, right, bottom, left order until each branch finishes or gets blocked.
    - Each source keeps the base turtle logic: right turn > straight > left, with 2-cell straights, 1-cell post-turn steps, a 1-cell orthogonal spacing rule, and a one-step fallback when a 2-step straight is partially blocked.
    - A move is legal only if it stays inside the ring-limit box implied by the arm length; a branch stops once it reaches its own arm-length-th return point on the launch axis. After geometry is finished, recolor each branch from its first new cell outward by cycling through the legend colors.

    Key insight:
    - The puzzle no longer depends on one dimension alone. The solver must separate geometry from color, infer the branch-local ring budget from arm length, and also respect the top-right-bottom-left execution order, because later branches visibly inherit tighter space after the shared opening launch.
    """

    grid = np.asarray(grid, dtype=int)
    cycle, _ = _extract_cycle_legend(grid)
    cross = _find_cross(grid)
    if len(cycle) < 2:
        cycle = [cross["color"], 1 if cross["color"] != 1 else 2]

    _, trails = _solve_geometry(
        grid.shape, cross, sequential=True, use_limit=True, stop_on_rings=True
    )
    return _color_solution(grid.shape, cross, trails, cycle)


def _candidate_size(arm: int) -> int:
    return 2 * _limit_radius_from_arm(arm) + LEFT_MARGIN + RIGHT_MARGIN + 1


def _center_bounds(size: int, arm: int) -> tuple[int, int, int, int]:
    radius = _limit_radius_from_arm(arm)
    row_lo = radius + TOP_MARGIN
    row_hi = size - radius - BOTTOM_MARGIN - 1
    col_lo = radius + LEFT_MARGIN
    col_hi = size - radius - RIGHT_MARGIN - 1
    return row_lo, row_hi, col_lo, col_hi


def _build_pool():
    pool = []
    probe_cycle = [CROSS_COLOR, 1]
    for arm in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        size = _candidate_size(arm)
        row_lo, row_hi, col_lo, col_hi = _center_bounds(size, arm)
        if row_lo > row_hi or col_lo > col_hi:
            continue
        for cr in range(row_lo, row_hi + 1):
            for cc in range(col_lo, col_hi + 1):
                grid = _build_input(arm, size, cr, cc, probe_cycle)
                try:
                    metrics = behavior_metrics(grid)
                except Exception:
                    continue
                if not (FILL_LO <= metrics["fill"] <= FILL_HI):
                    continue
                if metrics["secondary_count"] < MIN_SECONDARY_PIXELS:
                    continue
                if metrics["border_ink"] != 0:
                    continue
                if metrics["active_agents"] < MIN_ACTIVE_AGENTS:
                    continue
                if metrics["later_avoid"] < MIN_LATER_AVOID:
                    continue
                if metrics["order_diff"] < MIN_ORDER_DIFF:
                    continue
                if metrics["limit_diff"] < MIN_LIMIT_DIFF:
                    continue
                if metrics["used_radius"] < metrics["limit_radius"] - 1:
                    continue
                pool.append(
                    {
                        "arm": arm,
                        "size": size,
                        "cr": cr,
                        "cc": cc,
                        "metrics": metrics,
                    }
                )
    if not pool:
        raise RuntimeError("no valid combo pool found")
    return pool


def generate_input() -> np.ndarray:
    global _POOL
    if _POOL is None:
        _POOL = _build_pool()

    for _ in range(MAX_OUTER_RETRIES):
        choice = random.choice(_POOL)
        cycle = [CROSS_COLOR, random.choice(SECONDARY_COLORS)]
        grid = _build_input(choice["arm"], choice["size"], choice["cr"], choice["cc"], cycle)
        metrics = behavior_metrics(grid)
        if (
            FILL_LO <= metrics["fill"] <= FILL_HI
            and metrics["secondary_count"] >= MIN_SECONDARY_PIXELS
            and metrics["border_ink"] == 0
            and metrics["active_agents"] >= MIN_ACTIVE_AGENTS
            and metrics["later_avoid"] >= MIN_LATER_AVOID
            and metrics["order_diff"] >= MIN_ORDER_DIFF
            and metrics["limit_diff"] >= MIN_LIMIT_DIFF
        ):
            out = main(grid)
            if np.array_equal(main(grid), out):
                return grid
    raise RuntimeError("generate_input: failed after retries")

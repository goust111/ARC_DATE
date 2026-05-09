# Combo variant of 8b7bacbf (index 11)
# combo_spec: D1/2+D5/1+D8/0
# Name: 有限步追踪+多色边界腔体+迭代传播
#
# Design analysis
# ---------------
# fused_variant:
#   concretization:
#     - Keep the original corner seed + 8-connected line idea, but each seed now
#       also has a straight border meter bar. The meter length N is the trace
#       budget for one round.
#     - Fillable cavities are no longer single-color loops. Each cavity boundary
#       uses exactly 2 colors, with one "entry side" in color A and the other
#       three sides in color B. The cavity fills with A only when the current
#       round's traced region reaches that entry side.
#     - At least one chain of cavities is generated. The first cavity in the
#       chain is near the line and can be reached within N steps. Later cavities
#       do not touch the line; they share a wall with the previous cavity, so
#       they become reachable only after earlier cavities were filled.
#     - A separate far bicolor cavity is attached later on the same line, beyond
#       the round-1 budget. It is fillable for the no-step-limit ablation but is
#       intentionally missed by the fused solver.
#   behavior_trace:
#     - Original task: trace the full non-bg component once, then fill every
#       single-color cavity touched by that component using the seed color.
#     - D1 change: tracing is budgeted. The solver may only advance N graph steps
#       per round, where N is read from the border meter.
#     - D5 change: valid targets are multi-color cavities, and the fill color is
#       the uniquely touched boundary color, not the seed color.
#     - D8 change: after a cavity is filled, its interior becomes passable for
#       the next round, so the trace can continue through it and activate the
#       next cavity in the chain.
#     - Concrete example: if N=4, cavity A touches the line after 3 steps, cavity
#       B only touches A, and cavity C is attached far down the line after 9
#       steps, then the fused solver fills A in round 1, B in round 2, and never
#       fills C. A no-propagation solver would miss B; a no-step-limit solver
#       would also fill C.
#   complexity_calibration:
#     - Meter lengths stay in a small visible range (3-5).
#     - Bicolor cavities always use exactly 2 colors with a repeated side-based
#       pattern, so the touched-side rule is legible from a few examples.
#     - Chain length is only 2-3, enough to force propagation without turning
#       the grid into clutter.
#   implicit_assumptions_review:
#     - Original assumption "the seed's non-bg neighbor is just the line" fails
#       because the seed also touches the meter. The solver must distinguish the
#       longer straight border run as the meter and the other branch as the line.
#     - Original assumption "full connected-component tracing is always correct"
#       fails because the trace must stop after N steps in each round.
#     - Original assumption "reachable cavity -> fill with seed color" fails
#       twice: only multi-color cavities are targets, and the output color comes
#       from the uniquely contacted boundary color.
#     - Original assumption "order does not matter" fails because later cavities
#       are only activated after earlier filled cavities are added as new passable
#       frontier.
#   placement_legality:
#     - The generator keeps the meter on the border and the line on the other
#       corner-adjacent branch, making the decoding unambiguous.
#     - Frames stay at least one cell away from the outer border, cavities stay
#       empty before solving, and line pixels are forbidden from touching cavity
#       interiors directly.
#     - Only the first cavity in a chain may touch the line, and only on its
#       designated entry side; later cavities may touch only earlier cavities.
#   visual_observability_and_cleanliness:
#     - The input visibly shows three cues at once: a countable border bar, a
#       line that reaches one nearby loop but not a farther one, and a chain of
#       cavities where only the first touches the line.
#     - The output visibly separates the three effects: fills use boundary colors
#       rather than seed color, they spread round by round along the chain, and
#       the farther line-attached cavity stays blank under the finite-step rule.

from __future__ import annotations

from collections import deque
import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _border_run_length(g: np.ndarray, seed: tuple[int, int], nbr: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed
    nr, nc = nbr
    col = int(g[nr, nc])
    if col == bg:
        return 0
    dr = nr - sr
    dc = nc - sc
    run = 0
    r, c = nr, nc
    while _in_bounds(h, w, r, c) and int(g[r, c]) == col and (r == 0 or c == 0 or r == h - 1 or c == w - 1):
        run += 1
        r += dr
        c += dc
    return run


def _seed_info(g: np.ndarray, seed: tuple[int, int], bg: int) -> dict[str, int | tuple[int, int]]:
    h, w = g.shape
    sr, sc = seed
    nonbg_neighbors = []
    for nr, nc in neighbors4(sr, sc):
        if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) != bg:
            nonbg_neighbors.append((nr, nc))
    assert nonbg_neighbors, "seed pixel must have at least one non-bg neighbor"

    runs = [(_border_run_length(g, seed, nbr, bg), nbr) for nbr in nonbg_neighbors]
    runs.sort(key=lambda x: x[0], reverse=True)
    best_run, meter_pt = runs[0]
    assert best_run >= 2, "combo generator requires a visible border meter"

    line_candidates = [nbr for nbr in nonbg_neighbors if nbr != meter_pt]
    assert line_candidates, "seed must have a non-meter branch"
    line_pt = line_candidates[0]
    return {
        "seed_col": int(g[sr, sc]),
        "line_col": int(g[line_pt]),
        "line_pt": line_pt,
        "step_limit": int(best_run),
    }


def _compute_cavities(g: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities: list[dict[str, object]] = []
    for pts in bg_comps:
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if not boundary:
            continue
        cavities.append(
            {
                "region": region,
                "boundary": boundary,
                "bcols": {int(g[p]) for p in boundary},
            }
        )
    return cavities


def _limited_reach(
    g: np.ndarray,
    starts: set[tuple[int, int]],
    bg: int,
    extra_passable: set[tuple[int, int]],
    step_limit: int | None,
) -> set[tuple[int, int]]:
    h, w = g.shape

    def passable(r: int, c: int) -> bool:
        return int(g[r, c]) != bg or (r, c) in extra_passable

    q = deque()
    dist: dict[tuple[int, int], int] = {}
    for p in starts:
        r, c = p
        if _in_bounds(h, w, r, c) and passable(r, c):
            dist[p] = 0
            q.append((p, 0))

    while q:
        (r, c), d = q.popleft()
        if step_limit is not None and d >= step_limit:
            continue
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in dist or not passable(nr, nc):
                continue
            dist[(nr, nc)] = d + 1
            q.append(((nr, nc), d + 1))
    return set(dist)


def _boundary_contact_colors(
    g: np.ndarray,
    boundary: set[tuple[int, int]],
    reached: set[tuple[int, int]],
) -> set[int]:
    h, w = g.shape
    touched: set[int] = set()
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) in reached and (nr, nc) not in boundary:
                touched.add(int(g[r, c]))
                break
    return touched


def _seed_fill_plan(
    g: np.ndarray,
    cavities: list[dict[str, object]],
    seed: tuple[int, int],
    bg: int,
    use_step_limit: bool,
    iterative: bool,
) -> dict[int, dict[str, int]]:
    info = _seed_info(g, seed, bg)
    frontier = {seed}
    extra_passable: set[tuple[int, int]] = set()
    filled: dict[int, dict[str, int]] = {}
    round_id = 0
    step_limit = int(info["step_limit"]) if use_step_limit else None

    while frontier:
        reached = _limited_reach(g, frontier, bg, extra_passable, step_limit)
        new_hits: list[tuple[int, int]] = []
        for idx, cav in enumerate(cavities):
            if idx in filled:
                continue
            boundary = cav["boundary"]
            bcols = cav["bcols"]
            assert isinstance(boundary, set)
            assert isinstance(bcols, set)
            if len(bcols) < 2 or not (boundary & reached):
                continue
            contact_cols = _boundary_contact_colors(g, boundary, reached)
            if len(contact_cols) != 1:
                continue
            new_hits.append((idx, next(iter(contact_cols))))
        if not new_hits:
            break
        round_id += 1
        next_frontier: set[tuple[int, int]] = set()
        for idx, fill_col in new_hits:
            filled[idx] = {"round": round_id, "fill_col": fill_col}
            region = cavities[idx]["region"]
            assert isinstance(region, set)
            next_frontier.update(region)
        if not iterative:
            break
        extra_passable.update(next_frontier)
        frontier = next_frontier
    return filled


def _solve_core(grid: np.ndarray, use_step_limit: bool, iterative: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _compute_cavities(g, bg)
    out = g.copy()

    for seed in _seed_pixels(g, bg):
        plan = _seed_fill_plan(g, cavities, seed, bg, use_step_limit=use_step_limit, iterative=iterative)
        for idx, info in plan.items():
            fill_col = int(info["fill_col"])
            region = cavities[idx]["region"]
            assert isinstance(region, set)
            for r, c in region:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a budgeted seed trace through filled cavities and copy boundary colors inward.

    Input structure:
    - The background is the most frequent color.
    - Each active system has one unique border seed pixel at a corner.
    - Next to the seed, a straight border bar of another color encodes a step limit N.
    - A separate 8-connected line leaves the seed into the grid and touches a set of
      closed background cavities whose boundaries use multiple colors.

    Rule:
    - For each seed, trace only N 8-neighbor steps in the current round.
    - Any enclosed cavity whose boundary is reached in that round and whose reached
      contact color is unique gets flood-filled with that touched boundary color.
    - The newly filled cavity interior becomes passable for the next round, so the
      trace can continue through it for another N-step burst.
    - Stop when no new cavity can be activated.

    Key insight:
    The border bar limits how far the trace may travel at once, but filled cavities
    act like temporary tunnels. The output therefore depends on the interaction of
    step budgeting, multi-color boundary selection, and iterative propagation.
    """
    return _solve_core(grid, use_step_limit=True, iterative=True)


# ========================== CFG ==========================
GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
BG_COLORS = [0, 5]

METER_LEN_RANGE = (3, 5)
LINE_LEN_RANGE = (16, 30)
MIN_LINE_LEN = 14

CHAIN_LEN_CHOICES = [2, 2, 3]
CHAIN_CAVITY_SIZES = [3, 3, 4]
SOLO_CAVITY_SIZES = [3, 3, 4]

FRAME_PLACEMENT_PAD = 1
CHAIN_PLACEMENT_TRIES = 100
FRAME_ATTACH_TRIES = 80
DISTRACTOR_FRAME_TRIES = 40
DISTRACTOR_SOLO_MAX = 2

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.3

GEN_RETRIES = 600
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "right", "bottom", "left"]
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DIR_TO_SIDE = {"R": "right", "L": "left", "D": "bottom", "U": "top"}


def _weighted_dirs(dr_bias: int, dc_bias: int) -> list[tuple[int, int]]:
    dirs = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))
    return dirs


def _grow_line_from_prefix(
    grid: np.ndarray,
    prefix: list[tuple[int, int]],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = list(prefix)
    cur = path[-1]
    path_set = set(path)
    dr_bias = 1 if cur[0] < h // 2 else -1
    dc_bias = 1 if cur[1] < w // 2 else -1

    while len(path) < target_len:
        dirs = _weighted_dirs(dr_bias, dc_bias)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            foreign_touch = False
            for nnr, nnc in neighbors8(nr, nc):
                if not _in_bounds(h, w, nnr, nnc) or (nnr, nnc) == cur:
                    continue
                if grid[nnr, nnc] != -1:
                    foreign_touch = True
                    break
            if foreign_touch:
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = line_col
            placed = True
            break
        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break
    return path


def _corner_layout(
    corner_kind: str,
    h: int,
    w: int,
    meter_len: int,
    meter_horizontal: bool,
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    if corner_kind == "TL":
        seed = (0, 0)
        if meter_horizontal:
            meter = [(0, c) for c in range(1, meter_len + 1)]
            prefix = [(1, 0), (2, 1)]
        else:
            meter = [(r, 0) for r in range(1, meter_len + 1)]
            prefix = [(0, 1), (1, 2)]
    elif corner_kind == "TR":
        seed = (0, w - 1)
        if meter_horizontal:
            meter = [(0, c) for c in range(w - 2, w - 2 - meter_len, -1)]
            prefix = [(1, w - 1), (2, w - 2)]
        else:
            meter = [(r, w - 1) for r in range(1, meter_len + 1)]
            prefix = [(0, w - 2), (1, w - 3)]
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        if meter_horizontal:
            meter = [(h - 1, c) for c in range(1, meter_len + 1)]
            prefix = [(h - 2, 0), (h - 3, 1)]
        else:
            meter = [(r, 0) for r in range(h - 2, h - 2 - meter_len, -1)]
            prefix = [(h - 1, 1), (h - 2, 2)]
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        if meter_horizontal:
            meter = [(h - 1, c) for c in range(w - 2, w - 2 - meter_len, -1)]
            prefix = [(h - 2, w - 1), (h - 3, w - 2)]
        else:
            meter = [(r, w - 1) for r in range(h - 2, h - 2 - meter_len, -1)]
            prefix = [(h - 1, w - 2), (h - 2, w - 3)]
    else:
        raise ValueError(corner_kind)
    return seed, meter, prefix


def _frame_ring_layout(r0: int, c0: int, ch: int, cw: int):
    sides = {
        "top": [(r0, c0 + dc) for dc in range(1, cw + 1)],
        "bottom": [(r0 + ch + 1, c0 + dc) for dc in range(1, cw + 1)],
        "left": [(r0 + dr, c0) for dr in range(1, ch + 1)],
        "right": [(r0 + dr, c0 + cw + 1) for dr in range(1, ch + 1)],
    }
    ring = [pt for side in SIDES for pt in sides[side]]
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return sides, ring, cavity


def _side_slots(ch: int, cw: int, side: str, strict_middle: bool) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(2, cw)) if strict_middle else list(range(1, cw + 1))
    return list(range(2, ch)) if strict_middle else list(range(1, ch + 1))


def _placement_from_touch(lp: tuple[int, int], ch: int, cw: int, side: str, slot: int) -> tuple[int, int]:
    if side == "top":
        return lp[0] + 1, lp[1] - slot
    if side == "bottom":
        return lp[0] - (ch + 2), lp[1] - slot
    if side == "left":
        return lp[0] - slot, lp[1] + 1
    if side == "right":
        return lp[0] - slot, lp[1] - (cw + 2)
    raise ValueError(side)


def _validate_painting(
    canvas: np.ndarray,
    paint: dict[tuple[int, int], int],
    cavity_cells: set[tuple[int, int]],
    line_col: int | None,
    allowed_touch: set[tuple[int, int]] | None,
) -> bool:
    h, w = canvas.shape
    ring_cells = set(paint)
    all_internal = ring_cells | cavity_cells

    for rr, cc in ring_cells:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity_cells:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False

    if line_col is not None:
        for cr, cc in cavity_cells:
            for nr, nc in neighbors8(cr, cc):
                if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                    return False

    touched = set()
    for rr, cc in ring_cells:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in all_internal:
                continue
            v = int(canvas[nr, nc])
            if v == -1:
                continue
            if line_col is not None and v == line_col:
                touched.add((rr, cc))
                continue
            return False

    if line_col is None:
        return True
    if not touched:
        return False
    if allowed_touch is not None and not touched <= allowed_touch:
        return False
    return True


def _paint(canvas: np.ndarray, paint: dict[tuple[int, int], int]) -> None:
    for (r, c), col in paint.items():
        canvas[r, c] = col


def _chain_layout(
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    length: int,
    direction: str,
) -> list[tuple[dict[str, list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]]:
    layout = []
    if direction in ("R", "L"):
        step = cw + 1
        sign = 1 if direction == "R" else -1
        for i in range(length):
            layout.append(_frame_ring_layout(r0, c0 + sign * i * step, ch, cw))
    else:
        step = ch + 1
        sign = 1 if direction == "D" else -1
        for i in range(length):
            layout.append(_frame_ring_layout(r0 + sign * i * step, c0, ch, cw))
    return layout


def _try_place_bicolor_chain(
    canvas: np.ndarray,
    line_points: list[tuple[int, int]],
    line_col: int,
    fill_seq: list[int],
    terminal_alt: int,
    rng: random.Random,
) -> bool:
    if not line_points:
        return False
    h, w = canvas.shape
    length = len(fill_seq)
    for _ in range(CHAIN_PLACEMENT_TRIES):
        ch = rng.choice(CHAIN_CAVITY_SIZES)
        cw = rng.choice(CHAIN_CAVITY_SIZES)
        direction = rng.choice(["R", "L", "D", "U"])
        entry_side = OPPOSITE[DIR_TO_SIDE[direction]]
        slots = _side_slots(ch, cw, entry_side, strict_middle=True)
        if not slots:
            continue
        lp = rng.choice(line_points)
        slot = rng.choice(slots)
        r0, c0 = _placement_from_touch(lp, ch, cw, entry_side, slot)
        layout = _chain_layout(r0, c0, ch, cw, length, direction)

        all_pts = []
        for sides, ring, cavity in layout:
            all_pts.extend(ring)
            all_pts.extend(cavity)
        if not all_pts:
            continue
        rs = [r for r, _ in all_pts]
        cs = [c for _, c in all_pts]
        if min(rs) < FRAME_PLACEMENT_PAD or min(cs) < FRAME_PLACEMENT_PAD:
            continue
        if max(rs) > h - 1 - FRAME_PLACEMENT_PAD or max(cs) > w - 1 - FRAME_PLACEMENT_PAD:
            continue

        paint: dict[tuple[int, int], int] = {}
        cavity_cells: set[tuple[int, int]] = set()
        conflict = False
        for i, (sides, _ring, cavity) in enumerate(layout):
            fill_col = fill_seq[i]
            other_col = fill_seq[i + 1] if i + 1 < length else terminal_alt
            side_colors = {side: other_col for side in SIDES}
            side_colors[entry_side] = fill_col
            for side, pts in sides.items():
                col = int(side_colors[side])
                for pt in pts:
                    prev = paint.get(pt)
                    if prev is not None and prev != col:
                        conflict = True
                        break
                    paint[pt] = col
                if conflict:
                    break
            if conflict:
                break
            cavity_cells.update(cavity)
        if conflict:
            continue

        allowed_touch = set(layout[0][0][entry_side])
        if not _validate_painting(canvas, paint, cavity_cells, line_col=line_col, allowed_touch=allowed_touch):
            continue
        _paint(canvas, paint)
        return True
    return False


def _try_attach_bicolor_frame(
    canvas: np.ndarray,
    line_points: list[tuple[int, int]],
    line_col: int,
    entry_color: int,
    other_color: int,
    rng: random.Random,
) -> bool:
    if not line_points:
        return False
    h, w = canvas.shape
    for _ in range(FRAME_ATTACH_TRIES):
        ch = rng.choice(SOLO_CAVITY_SIZES)
        cw = rng.choice(SOLO_CAVITY_SIZES)
        contact_side = rng.choice(SIDES)
        slots = _side_slots(ch, cw, contact_side, strict_middle=True)
        if not slots:
            continue
        lp = rng.choice(line_points)
        slot = rng.choice(slots)
        r0, c0 = _placement_from_touch(lp, ch, cw, contact_side, slot)
        fh, fw = ch + 2, cw + 2
        if r0 < FRAME_PLACEMENT_PAD or c0 < FRAME_PLACEMENT_PAD:
            continue
        if r0 + fh > h - FRAME_PLACEMENT_PAD or c0 + fw > w - FRAME_PLACEMENT_PAD:
            continue
        sides, _ring, cavity = _frame_ring_layout(r0, c0, ch, cw)
        paint: dict[tuple[int, int], int] = {}
        side_colors = {side: other_color for side in SIDES}
        side_colors[contact_side] = entry_color
        for side, pts in sides.items():
            for pt in pts:
                paint[pt] = int(side_colors[side])
        if not _validate_painting(canvas, paint, set(cavity), line_col=line_col, allowed_touch=set(sides[contact_side])):
            continue
        _paint(canvas, paint)
        return True
    return False


def _try_place_solo_frame(canvas: np.ndarray, side_colors: dict[str, int], rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(SOLO_CAVITY_SIZES)
        cw = rng.choice(SOLO_CAVITY_SIZES)
        fh, fw = ch + 2, cw + 2
        if h - fh - 2 < 1 or w - fw - 2 < 1:
            continue
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        sides, _ring, cavity = _frame_ring_layout(r0, c0, ch, cw)
        paint: dict[tuple[int, int], int] = {}
        for side, pts in sides.items():
            for pt in pts:
                paint[pt] = int(side_colors[side])
        if not _validate_painting(canvas, paint, set(cavity), line_col=None, allowed_touch=None):
            continue
        _paint(canvas, paint)
        return True
    return False


def _combo_profile(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _compute_cavities(g, bg)
    seeds = _seed_pixels(g, bg)

    propagated_only = 0
    step_pruned = 0
    max_round = 0
    seed_colors = {int(g[p]) for p in seeds}

    for seed in seeds:
        combo_plan = _seed_fill_plan(g, cavities, seed, bg, use_step_limit=True, iterative=True)
        one_round_plan = _seed_fill_plan(g, cavities, seed, bg, use_step_limit=True, iterative=False)
        no_limit_plan = _seed_fill_plan(g, cavities, seed, bg, use_step_limit=False, iterative=True)
        propagated_only += len(set(combo_plan) - set(one_round_plan))
        step_pruned += len(set(no_limit_plan) - set(combo_plan))
        if combo_plan:
            max_round = max(max_round, max(info["round"] for info in combo_plan.values()))

    combo_out = _solve_core(g, use_step_limit=True, iterative=True)
    filled_vals = {int(v) for v in combo_out[(g == bg) & (combo_out != bg)]}
    nonseed_fill_colors = len(filled_vals - seed_colors)

    return {
        "max_round": max_round,
        "propagated_only": propagated_only,
        "step_pruned": step_pruned,
        "nonseed_fill_colors": nonseed_fill_colors,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)
    if len(palette) < 7:
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_col = palette.pop()
    line_col = palette.pop()
    meter_col = palette.pop()

    meter_len = rng.randint(*METER_LEN_RANGE)
    corner = rng.choice(["TL", "TR", "BL", "BR"])
    seed_rc, meter_cells, prefix = _corner_layout(corner, h, w, meter_len, rng.choice([False, True]))
    all_cells = [seed_rc] + meter_cells + prefix
    if any(not _in_bounds(h, w, r, c) for r, c in all_cells):
        return None
    if any(canvas[r, c] != -1 for r, c in all_cells):
        return None

    canvas[seed_rc] = seed_col
    for r, c in meter_cells:
        canvas[r, c] = meter_col
    for r, c in prefix:
        canvas[r, c] = line_col

    lo = max(LINE_LEN_RANGE[0], meter_len + 12)
    hi = min(LINE_LEN_RANGE[1], h + w)
    if lo > hi:
        return None
    target_len = rng.randint(lo, hi)
    path = _grow_line_from_prefix(canvas, prefix, line_col, target_len, rng)
    if len(path) < MIN_LINE_LEN:
        return None

    near_points = path[1 : min(len(path), meter_len)]
    far_start = meter_len + 5
    far_points = path[far_start:] if far_start < len(path) else []
    if not near_points or len(far_points) < 3:
        return None

    chain_len = rng.choice(CHAIN_LEN_CHOICES)
    if len(palette) < chain_len + 1:
        return None
    chain_colors = rng.sample(palette, chain_len + 1)
    fill_seq = chain_colors[:chain_len]
    terminal_alt = chain_colors[-1]
    if not _try_place_bicolor_chain(canvas, near_points, line_col, fill_seq, terminal_alt, rng):
        return None

    far_entry, far_other = rng.sample(chain_colors, 2) if len(chain_colors) >= 2 else rng.sample(palette, 2)
    if not _try_attach_bicolor_frame(canvas, far_points, line_col, far_entry, far_other, rng):
        return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        if rng.random() < 0.5 and len(chain_colors) >= 2:
            a, b = rng.sample(chain_colors, 2)
            contact_side = rng.choice(SIDES)
            side_colors = {side: b for side in SIDES}
            side_colors[contact_side] = a
        else:
            fc = rng.choice(chain_colors)
            side_colors = {side: fc for side in SIDES}
        _try_place_solo_frame(canvas, side_colors, rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    if int((grid == seed_col).sum()) != 1:
        return None

    try:
        combo_out = main(grid)
        one_round = _solve_core(grid, use_step_limit=True, iterative=False)
        no_limit = _solve_core(grid, use_step_limit=False, iterative=True)
    except Exception:
        return None

    if np.array_equal(combo_out, grid):
        return None
    if np.array_equal(combo_out, one_round):
        return None
    if np.array_equal(combo_out, no_limit):
        return None

    profile = _combo_profile(grid)
    if profile["max_round"] < 2:
        return None
    if profile["propagated_only"] < 1:
        return None
    if profile["step_pruned"] < 1:
        return None
    if profile["nonseed_fill_colors"] < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")

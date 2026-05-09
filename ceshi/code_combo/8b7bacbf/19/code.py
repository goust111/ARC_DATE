# Combo variant of 8b7bacbf (index 19)
# combo_spec: D2/1+D5/1+D8/0
# Name: 分支路径+迭代传播+多色边界腔体

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis
# ---------------
# concretization:
#   - Keep the original seed-on-border idea, but each seed now grows an 8-connected tree
#     rather than a single chain. The trunk bends inward, and at least one branch sprouts
#     from a bend node.
#   - Each trunk/branch endpoint region carries a multicolor cavity chain. The first cavity
#     in a chain is 8-adjacent to the line tree; later cavities are not line-adjacent and
#     are reachable only through already filled earlier cavities.
#   - Every fillable cavity uses exactly two visible boundary colors. For a horizontal chain
#     extending right, the left wall of cavity i has color A/B/A/... and all three other
#     sides use the opposite color. The shared wall to cavity i+1 is therefore the color
#     that becomes cavity i+1's entry color.
#
# behavior_trace:
#   - Small example: seed S touches line color L at the border. The line branches into a
#     trunk segment T and a branch segment B. A 2-color cavity C1 touches T on its left wall,
#     so round 1 fills C1 with that touched wall color. A second cavity C2 shares C1's right
#     wall but does not touch the line at all, so the original solver would either skip it
#     or overfill it immediately if it only looked at traced-component membership.
#   - In this combo solver, the active frontier starts as the seed's own line pixels. After
#     round 1, the interior of C1 is added to the frontier. Now C2 has exactly one boundary
#     color adjacent to the active frontier, so round 2 fills C2 with that newly touched
#     boundary color. The same can happen independently on multiple branches.
#   - Difference from the source task: path-following is no longer single-threaded, filling
#     is no longer one-shot, and multicolor boundary handling is no longer reducible to
#     "fill with seed color".
#
# complexity_calibration:
#   - Boundary colors are limited to two per fillable cavity. This keeps the touched-color
#     rule visible across 2-4 ARC examples.
#   - Branch count is capped at one per seed, so the tree geometry is obvious without turning
#     the grid into noise. Each actual branch gets its own attached chain, guaranteeing the
#     branch matters in the output.
#   - Propagation depth is limited to 2-3 steps per chain. That is enough to visibly differ
#     from one-step filling while keeping layouts readable.
#
# implicit_assumption_audit_for_main:
#   - `_seed_pixels` still assumes seed colors are unique border singletons. The generator
#     preserves that exactly.
#   - `_seed_line_info` assumes an orthogonal non-bg neighbor of the seed belongs to the
#     intended line color. The generator always places the first line pixel orthogonally.
#   - The original `boundary & traced` check is no longer sufficient for activation because
#     later cavities are in the same traced component but should wait for propagation. The
#     solver therefore separates "eligible for this seed" from "active this round".
#   - The original `len(boundary_colors) == 1` assumption is invalid. The solver now requires
#     at least two boundary colors and computes the unique boundary color touched by the
#     current frontier each round.
#   - The new frontier logic assumes a successor cavity contacts the previous filled cavity by
#     a visible shared wall/edge, not by hidden metadata. The chain generator enforces that.
#
# placement_legality:
#   - Tree pixels are placed on empty cells and may not touch any occupied 8-neighbor except
#     their parent. Branches only sprout from bend nodes to keep the tree acyclic in 8-neigh.
#   - Chain placement requires empty ring/cavity cells, forbids cavity interiors from touching
#     any line pixel, forces only the first cavity's entry side to touch the line, and forbids
#     later cavities from touching the line at all.
#   - Solo distractors are surrounded by a 1-cell empty halo so they cannot accidentally join
#     a seeded network or start a fake propagation path.
#
# observability_and_cleanliness:
#   - The branch effect is visible because distinct branches have their own directly activated
#     head cavities. The propagation effect is visible because some filled cavities appear one
#     step farther from the line and can only be explained by earlier fills acting as relays.
#   - The color effect is visible because cavity fills copy one side color of a 2-color frame,
#     not the seed color. The generator also rejects samples whose filled colors collapse back
#     to seed colors or whose multi-step solver matches a one-step baseline.
#   - Layout clutter is controlled by moderate branch counts, short chains, and strict no-touch
#     checks between different structures.
#
# post_change_generator_guarantees:
#   - At least one seeded line graph is genuinely branched.
#   - At least one branch-attached chain has length >= 2, so propagation changes the output.
#   - The final sample differs from a one-step multicolor fill baseline.
#   - Every solver-side assumption used by activation order and touched-color disambiguation is
#     enforced geometrically by construction.


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


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


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _same_color_component8_from(g: np.ndarray, starts: list[tuple[int, int]], color: int) -> set[tuple[int, int]]:
    h, w = g.shape
    vis = {p for p in starts if 0 <= p[0] < h and 0 <= p[1] < w and int(g[p]) == color}
    st = list(vis)
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == color:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _collect_cavities(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities: list[dict] = []
    for pts in bg_comps:
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
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


def _seed_line_info(g: np.ndarray, seed: tuple[int, int], bg: int) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = g.shape
    sr, sc = seed
    neigh_cols = []
    starts = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
            neigh_cols.append(int(g[nr, nc]))
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    line_col = Counter(neigh_cols).most_common(1)[0][0]
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
            starts.append((nr, nc))
    line_pixels = _same_color_component8_from(g, starts, int(line_col))
    traced = _nonbg_component8_from(g, seed, bg)
    return int(line_col), line_pixels, traced


def _boundary_contact_colors(
    g: np.ndarray,
    boundary: set[tuple[int, int]],
    active: set[tuple[int, int]],
) -> set[int]:
    h, w = g.shape
    touched: set[int] = set()
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in active:
                touched.add(int(g[r, c]))
                break
    return touched


def _seed_activation_plan(g: np.ndarray, cavities: list[dict], seed: tuple[int, int], bg: int) -> dict:
    line_col, line_pixels, traced = _seed_line_info(g, seed, bg)
    active = set(line_pixels)
    rounds: dict[int, int] = {}
    fill_cols: dict[int, int] = {}
    eligible = [i for i, cav in enumerate(cavities) if len(cav["bcols"]) >= 2 and bool(cav["boundary"] & traced)]

    round_id = 0
    while True:
        new_hits: list[tuple[int, int]] = []
        for idx in eligible:
            if idx in rounds:
                continue
            contact_cols = _boundary_contact_colors(g, cavities[idx]["boundary"], active)
            if len(contact_cols) != 1:
                continue
            new_hits.append((idx, next(iter(contact_cols))))
        if not new_hits:
            break
        round_id += 1
        for idx, col in new_hits:
            rounds[idx] = round_id
            fill_cols[idx] = int(col)
        for idx, _ in new_hits:
            active.update(cavities[idx]["region"])

    return {
        "seed": seed,
        "line_col": line_col,
        "line_pixels": line_pixels,
        "traced": traced,
        "rounds": rounds,
        "fill_cols": fill_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate cavity filling from each border seed through a branched line-and-frame network.

    Input structure:
    - The background is the most common color.
    - One or two unique single-pixel seed colors appear on the border.
    - Each seed touches its own 8-connected line tree, and multicolor frame rings are attached
      to the tree or to other frame rings deeper in the network.

    Rule:
    - For each seed, trace its whole non-background component but use only the seed's line
      pixels as the initial active frontier.
    - A cavity is eligible only if its boundary has at least two colors and belongs to that
      seed's traced component.
    - In each round, fill every eligible cavity whose boundary is touched by the current
      frontier in exactly one boundary color; use that touched boundary color as the fill.
    - Newly filled cavities become frontier for the next round, and the process repeats until
      no new cavity activates.

    Key insight:
    Membership in the seed's connected drawing is not enough anymore. The solver must combine
    branch-aware tracing, per-round frontier expansion, and re-evaluation of which boundary
    color is touched at the moment a cavity activates.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _collect_cavities(g, bg)
    out = g.copy()

    for seed in _seed_pixels(g, bg):
        plan = _seed_activation_plan(g, cavities, seed, bg)
        for idx, round_id in plan["rounds"].items():
            _ = round_id
            fill_col = plan["fill_cols"][idx]
            for r, c in cavities[idx]["region"]:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col

    return out


def _one_step_multicolor_fill(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _collect_cavities(g, bg)
    out = g.copy()
    for seed in _seed_pixels(g, bg):
        line_col, line_pixels, traced = _seed_line_info(g, seed, bg)
        _ = line_col
        for cav in cavities:
            if len(cav["bcols"]) < 2 or not (cav["boundary"] & traced):
                continue
            contact_cols = _boundary_contact_colors(g, cav["boundary"], line_pixels)
            if len(contact_cols) != 1:
                continue
            fill_col = next(iter(contact_cols))
            for r, c in cav["region"]:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col
    return out


def _activation_profile(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _collect_cavities(g, bg)
    per_seed = []
    max_round = 0
    propagated = 0
    branchy = 0
    for seed in _seed_pixels(g, bg):
        plan = _seed_activation_plan(g, cavities, seed, bg)
        line_pixels = plan["line_pixels"]
        degrees = {
            pt: sum((nr, nc) in line_pixels for nr, nc in neighbors8(*pt))
            for pt in line_pixels
        }
        is_branchy = any(deg >= 3 for deg in degrees.values())
        branchy += int(is_branchy)
        if plan["rounds"]:
            seed_max = max(plan["rounds"].values())
            seed_prop = sum(v >= 2 for v in plan["rounds"].values())
        else:
            seed_max = 0
            seed_prop = 0
        max_round = max(max_round, seed_max)
        propagated += seed_prop
        per_seed.append(
            {
                "seed": seed,
                "max_round": seed_max,
                "propagated": seed_prop,
                "branchy": is_branchy,
                "filled": len(plan["rounds"]),
            }
        )
    return {
        "max_round": max_round,
        "propagated_cavities": propagated,
        "branchy_seed_count": branchy,
        "per_seed": per_seed,
    }


def fusion_profile(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    seeds = _seed_pixels(g, bg)
    seed_cols = {int(g[p]) for p in seeds}
    out = main(g)
    one_step = _one_step_multicolor_fill(g)
    filled_mask = (g == bg) & (out != bg)
    fill_colors = {int(v) for v in np.unique(out[filled_mask])} if filled_mask.any() else set()
    profile = _activation_profile(g)
    profile.update(
        {
            "seed_count": len(seeds),
            "filled_cells": int(filled_mask.sum()),
            "non_seed_fill_colors": sorted(fill_colors - seed_cols),
            "one_step_delta_cells": int(np.sum(out != one_step)),
        }
    )
    return profile


# ----------------------------- generator -----------------------------

GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 1, 2]

MIN_TRUNK_LEN = 8
TRUNK_LEN_RANGE = (10, 19)
MIN_BRANCH_LEN = 4
BRANCH_LEN_RANGE = (4, 8)
MAX_BRANCHES_PER_SEED = 1
TOTAL_BRANCHES_ONE_SEED = (1, 1)
TOTAL_BRANCHES_TWO_SEEDS = (1, 2)
BRANCH_ATTACH_TRIES = 40

FRAME_COLORS_MAX = 4
CHAIN_CAVITY_SIZES = [2, 2, 3]
SOLO_CAVITY_SIZES = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
CHAIN_PLACEMENT_TRIES = 90
SOLO_FRAME_TRIES = 50
DISTRACTOR_SOLO_MAX = 2

TRUNK_CHAIN_LENGTHS = [1, 1, 2]
BRANCH_CHAIN_LENGTHS = [1, 2, 2, 3]

DIAG_WEIGHT_ALIGNED = 4
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
TURN_BONUS = 2
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.35
PATH_STALL_LIMIT = 5

GEN_RETRIES = 500

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "right", "bottom", "left"]
DIRECTION_SPECS = {
    "R": {"entry": "left", "dr": 0, "dc": 1},
    "L": {"entry": "right", "dr": 0, "dc": -1},
    "D": {"entry": "top", "dr": 1, "dc": 0},
    "U": {"entry": "bottom", "dr": -1, "dc": 0},
}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_layout(r0: int, c0: int, ch: int, cw: int) -> tuple[dict[str, list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
    sides = {
        "top": [(r0, c0 + dc) for dc in range(1, cw + 1)],
        "bottom": [(r0 + ch + 1, c0 + dc) for dc in range(1, cw + 1)],
        "left": [(r0 + dr, c0) for dr in range(1, ch + 1)],
        "right": [(r0 + dr, c0 + cw + 1) for dr in range(1, ch + 1)],
    }
    ring = [pt for side in SIDES for pt in sides[side]]
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return sides, ring, cavity


def _side_slots(ch: int, cw: int, side: str) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(1, cw + 1))
    return list(range(1, ch + 1))


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


def _choose_two_distinct(colors: list[int], rng: random.Random) -> tuple[int, int]:
    a, b = rng.sample(colors, 2)
    return int(a), int(b)


def _paint_cells(canvas: np.ndarray, paint_map: dict[tuple[int, int], int]) -> None:
    for (r, c), col in paint_map.items():
        canvas[r, c] = col


def _halo_clear(canvas: np.ndarray, pts: set[tuple[int, int]]) -> bool:
    h, w = canvas.shape
    for r, c in pts:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not _in_bounds(h, w, nr, nc):
                    continue
                if (nr, nc) in pts:
                    continue
                if int(canvas[nr, nc]) != -1:
                    return False
    return True


def _build_chain_meta(
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    length: int,
    direction: str,
    colors: tuple[int, int],
) -> tuple[list[dict], dict[tuple[int, int], int]] | None:
    spec = DIRECTION_SPECS[direction]
    entry_side = spec["entry"]
    step_r = spec["dr"] * (ch + 1)
    step_c = spec["dc"] * (cw + 1)
    chain: list[dict] = []
    paint_map: dict[tuple[int, int], int] = {}

    for i in range(length):
        rr = r0 + i * step_r
        cc = c0 + i * step_c
        sides, ring, cavity = _frame_ring_layout(rr, cc, ch, cw)
        entry_col = int(colors[i % 2])
        other_col = int(colors[(i + 1) % 2])
        side_colors = {side: other_col for side in SIDES}
        side_colors[entry_side] = entry_col
        for side, pts in sides.items():
            col = int(side_colors[side])
            for pt in pts:
                old = paint_map.get(pt)
                if old is not None and old != col:
                    return None
                paint_map[pt] = col
        chain.append(
            {
                "sides": sides,
                "ring": ring,
                "cavity": cavity,
                "entry_side": entry_side,
                "entry_color": entry_col,
                "boundary_colors": {entry_col, other_col},
            }
        )
    return chain, paint_map


def _validate_chain_placement(
    canvas: np.ndarray,
    chain: list[dict],
    paint_map: dict[tuple[int, int], int],
    all_line_pixels: set[tuple[int, int]],
    anchor_pixels: set[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = canvas.shape
    all_ring = set(paint_map)
    all_cavity = set().union(*[set(item["cavity"]) for item in chain])
    occupied = all_ring | all_cavity

    for rr, cc in occupied:
        if not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != -1:
            return False

    for rr, cc in all_cavity:
        for nr, nc in neighbors8(rr, cc):
            if (nr, nc) in all_line_pixels:
                return False

    touched_by_line = set()
    for rr, cc in all_ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in occupied:
                continue
            v = int(canvas[nr, nc])
            if v == -1:
                continue
            if v != line_col:
                return False
            touched_by_line.add((rr, cc))

    first_entry_side = set(chain[0]["sides"][chain[0]["entry_side"]])
    if not touched_by_line or not touched_by_line <= first_entry_side:
        return False

    if not any(any((nr, nc) in anchor_pixels for nr, nc in neighbors8(rr, cc)) for rr, cc in first_entry_side):
        return False

    return True


def _place_multicolor_chain(
    canvas: np.ndarray,
    anchor_pixels: list[tuple[int, int]],
    all_line_pixels: set[tuple[int, int]],
    line_col: int,
    frame_colors_pool: list[int],
    length: int,
    rng: random.Random,
) -> list[dict] | None:
    if len(frame_colors_pool) < 2 or not anchor_pixels:
        return None
    h, w = canvas.shape
    anchor_set = set(anchor_pixels)
    for _ in range(CHAIN_PLACEMENT_TRIES):
        ch = rng.choice(CHAIN_CAVITY_SIZES)
        cw = rng.choice(CHAIN_CAVITY_SIZES)
        direction = rng.choice(list(DIRECTION_SPECS))
        entry_side = DIRECTION_SPECS[direction]["entry"]
        slots = _side_slots(ch, cw, entry_side)
        if not slots:
            continue
        lp = rng.choice(anchor_pixels)
        slot = rng.choice(slots)
        r0, c0 = _placement_from_touch(lp, ch, cw, entry_side, slot)
        colors = _choose_two_distinct(frame_colors_pool, rng)
        built = _build_chain_meta(r0, c0, ch, cw, length, direction, colors)
        if built is None:
            continue
        chain, paint_map = built
        all_pts = set(paint_map) | set().union(*[set(item["cavity"]) for item in chain])
        rs = [r for r, _ in all_pts]
        cs = [c for _, c in all_pts]
        if min(rs) < FRAME_PLACEMENT_PAD or min(cs) < FRAME_PLACEMENT_PAD:
            continue
        if max(rs) > h - 1 - FRAME_PLACEMENT_PAD or max(cs) > w - 1 - FRAME_PLACEMENT_PAD:
            continue
        if not _validate_chain_placement(canvas, chain, paint_map, all_line_pixels, anchor_set, line_col):
            continue
        _paint_cells(canvas, paint_map)
        return chain
    return None


def _place_solo_pattern_frame(canvas: np.ndarray, side_colors: dict[str, int], rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(SOLO_FRAME_TRIES):
        ch = rng.choice(SOLO_CAVITY_SIZES)
        cw = rng.choice(SOLO_CAVITY_SIZES)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        sides, ring, cavity = _frame_ring_layout(r0, c0, ch, cw)
        occupied = set(ring) | set(cavity)
        if not _halo_clear(canvas, occupied):
            continue
        paint_map = {}
        for side, pts in sides.items():
            for pt in pts:
                paint_map[pt] = int(side_colors[side])
        _paint_cells(canvas, paint_map)
        return True
    return False


def _try_place_solo_multicolor_frame(canvas: np.ndarray, frame_colors_pool: list[int], rng: random.Random) -> bool:
    if len(frame_colors_pool) < 2:
        return False
    a, b = _choose_two_distinct(frame_colors_pool, rng)
    contact_side = rng.choice(SIDES)
    side_colors = {side: b for side in SIDES}
    side_colors[contact_side] = a
    return _place_solo_pattern_frame(canvas, side_colors, rng)


def _try_place_solo_single_frame(canvas: np.ndarray, frame_colors_pool: list[int], rng: random.Random) -> bool:
    fc = int(rng.choice(frame_colors_pool))
    return _place_solo_pattern_frame(canvas, {side: fc for side in SIDES}, rng)


def _bias_for(pt: tuple[int, int], h: int, w: int, rng: random.Random) -> tuple[int, int]:
    dr_bias = 1 if pt[0] < h // 2 else -1
    dc_bias = 1 if pt[1] < w // 2 else -1
    if rng.random() < 0.25:
        dr_bias *= -1
    if rng.random() < 0.25:
        dc_bias *= -1
    return dr_bias, dc_bias


def _weighted_dirs(
    dr_bias: int,
    dc_bias: int,
    prev_dir: tuple[int, int] | None,
    rng: random.Random,
) -> list[tuple[int, int]]:
    dirs = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
        if prev_dir is not None and (ddr, ddc) != prev_dir:
            weight += TURN_BONUS
        dirs.extend([(ddr, ddc)] * max(1, weight))
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
        if prev_dir is not None and (ddr, ddc) != prev_dir:
            weight += TURN_BONUS
        dirs.extend([(ddr, ddc)] * max(1, weight))
    rng.shuffle(dirs)
    return dirs


def _can_place_line_pixel(canvas: np.ndarray, parent: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = canvas.shape
    r, c = cell
    if not _in_bounds(h, w, r, c) or int(canvas[r, c]) != -1:
        return False
    for nr, nc in neighbors8(r, c):
        if not _in_bounds(h, w, nr, nc) or (nr, nc) == parent:
            continue
        if int(canvas[nr, nc]) != -1:
            return False
    return True


def _add_line_edge(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    parent: tuple[int, int],
    child: tuple[int, int],
    line_col: int,
) -> None:
    canvas[child] = line_col
    tree_set.add(child)
    degrees.setdefault(parent, 0)
    degrees.setdefault(child, 0)
    degrees[parent] += 1
    degrees[child] += 1


def _extend_tip_path_inplace(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    tip: tuple[int, int],
    target_extra: int,
    line_col: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = canvas.shape
    cur = tip
    prev_dir = None
    dr_bias, dc_bias = _bias_for(tip, h, w, rng)
    added: list[tuple[int, int]] = []
    stalls = 0
    while len(added) < target_extra and stalls < PATH_STALL_LIMIT:
        dirs = _weighted_dirs(dr_bias, dc_bias, prev_dir, rng)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            if not _can_place_line_pixel(canvas, cur, (nr, nc)):
                continue
            _add_line_edge(canvas, tree_set, degrees, cur, (nr, nc), line_col)
            cur = (nr, nc)
            added.append(cur)
            prev_dir = (ddr, ddc)
            placed = True
            break
        if placed:
            stalls = 0
            if rng.random() < 0.15:
                dr_bias, dc_bias = _bias_for(cur, h, w, rng)
        else:
            stalls += 1
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
    return added


def _line_neighbors_of(tree_set: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    return [(nr, nc) for nr, nc in neighbors8(*pt) if (nr, nc) in tree_set]


def _is_bend_node(tree_set: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    neighs = _line_neighbors_of(tree_set, pt)
    if len(neighs) != 2:
        return False
    (r1, c1), (r2, c2) = neighs
    v1 = (r1 - pt[0], c1 - pt[1])
    v2 = (r2 - pt[0], c2 - pt[1])
    return not (v1[0] == -v2[0] and v1[1] == -v2[1])


def _spawn_branch_trial(
    canvas: np.ndarray,
    tree_set: set[tuple[int, int]],
    degrees: dict[tuple[int, int], int],
    base: tuple[int, int],
    target_len: int,
    line_col: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    trial_canvas = canvas.copy()
    trial_tree = set(tree_set)
    trial_degrees = dict(degrees)
    h, w = canvas.shape
    dr_bias, dc_bias = _bias_for(base, h, w, rng)
    for ddr, ddc in _weighted_dirs(dr_bias, dc_bias, None, rng):
        first = (base[0] + ddr, base[1] + ddc)
        if not _can_place_line_pixel(trial_canvas, base, first):
            continue
        _add_line_edge(trial_canvas, trial_tree, trial_degrees, base, first, line_col)
        path = [first]
        if target_len > 1:
            path.extend(
                _extend_tip_path_inplace(
                    trial_canvas,
                    trial_tree,
                    trial_degrees,
                    first,
                    target_len - 1,
                    line_col,
                    rng,
                )
            )
        if len(path) < MIN_BRANCH_LEN:
            continue
        canvas[:, :] = trial_canvas
        tree_set.clear()
        tree_set.update(trial_tree)
        degrees.clear()
        degrees.update(trial_degrees)
        return path
    return []


def _grow_branching_tree(
    canvas: np.ndarray,
    first_rc: tuple[int, int],
    line_col: int,
    branch_count: int,
    rng: random.Random,
) -> dict | None:
    h, w = canvas.shape
    tree_set = {first_rc}
    degrees = {first_rc: 0}

    trunk_hi = min(TRUNK_LEN_RANGE[1], max(TRUNK_LEN_RANGE[0], h + w - 8))
    trunk_target = rng.randint(TRUNK_LEN_RANGE[0], trunk_hi)
    trunk_path = [first_rc]
    trunk_path.extend(_extend_tip_path_inplace(canvas, tree_set, degrees, first_rc, trunk_target - 1, line_col, rng))
    if len(trunk_path) < MIN_TRUNK_LEN:
        return None

    segments = [trunk_path]
    for _ in range(branch_count):
        branch_hi = min(BRANCH_LEN_RANGE[1], max(BRANCH_LEN_RANGE[0], (h + w) // 3))
        branch_target = rng.randint(BRANCH_LEN_RANGE[0], branch_hi)
        bend_bases = [pt for pt in trunk_path if pt != first_rc and _is_bend_node(tree_set, pt)]
        rng.shuffle(bend_bases)
        placed: list[tuple[int, int]] = []
        for base in bend_bases[:BRANCH_ATTACH_TRIES]:
            placed = _spawn_branch_trial(canvas, tree_set, degrees, base, branch_target, line_col, rng)
            if placed:
                break
        if not placed:
            return None
        segments.append(placed)

    branch_points = [pt for pt, deg in degrees.items() if deg >= 3]
    if branch_count > 0 and not branch_points:
        return None

    return {
        "tree_set": tree_set,
        "degrees": degrees,
        "segments": segments,
        "branch_count": len(segments) - 1,
    }


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        return (0, 0), rng.choice([(0, 1), (1, 0)])
    if corner_kind == "TR":
        return (0, w - 1), rng.choice([(0, w - 2), (1, w - 1)])
    if corner_kind == "BL":
        return (h - 1, 0), rng.choice([(h - 1, 1), (h - 2, 0)])
    if corner_kind == "BR":
        return (h - 1, w - 1), rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    raise ValueError(corner_kind)


def _distribute_branch_budget(num_seeds: int, rng: random.Random) -> list[int]:
    total = rng.randint(*(TOTAL_BRANCHES_ONE_SEED if num_seeds == 1 else TOTAL_BRANCHES_TWO_SEEDS))
    out = [0] * num_seeds
    order = list(range(num_seeds))
    rng.shuffle(order)
    for _ in range(total):
        choices = [i for i in order if out[i] < MAX_BRANCHES_PER_SEED]
        if not choices:
            break
        out[rng.choice(choices)] += 1
    if sum(out) == 0:
        out[rng.choice(order)] = 1
    return out


def _segment_chain_lengths(lines_info: list[dict], rng: random.Random) -> None:
    branch_slots: list[tuple[dict, int]] = []
    for info in lines_info:
        lengths = [rng.choice(TRUNK_CHAIN_LENGTHS)]
        for _ in info["tree_info"]["segments"][1:]:
            lengths.append(rng.choice(BRANCH_CHAIN_LENGTHS))
            branch_slots.append((info, len(lengths) - 1))
        info["chain_lengths"] = lengths
    if branch_slots and not any(info["chain_lengths"][idx] >= 2 for info, idx in branch_slots):
        info, idx = rng.choice(branch_slots)
        info["chain_lengths"][idx] = rng.choice([2, 3])


def _place_seed_chains(canvas: np.ndarray, lines_info: list[dict], frame_colors_pool: list[int], rng: random.Random) -> bool:
    plans = []
    for seed_idx, info in enumerate(lines_info):
        segments = info["tree_info"]["segments"]
        chain_lengths = info["chain_lengths"]
        for seg_idx in range(1, len(segments)):
            plans.append((0, -chain_lengths[seg_idx], seed_idx, seg_idx))
        plans.append((1, -chain_lengths[0], seed_idx, 0))
    plans.sort()

    for _prio, _neg_len, seed_idx, seg_idx in plans:
        info = lines_info[seed_idx]
        length = info["chain_lengths"][seg_idx]
        placed = _place_multicolor_chain(
            canvas,
            info["tree_info"]["segments"][seg_idx],
            info["tree_info"]["tree_set"],
            info["line_col"],
            frame_colors_pool,
            length,
            rng,
        )
        if placed is None:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if len(palette) < num_seeds * 2 + 2:
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    num_frame_cols = rng.randint(2, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    corners_used = corners[:num_seeds]

    per_seed_branches = _distribute_branch_budget(num_seeds, rng)
    lines_info: list[dict] = []

    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        tree_info = _grow_branching_tree(canvas, first_rc, line_colors[i], per_seed_branches[i], rng)
        if tree_info is None:
            return None
        lines_info.append(
            {
                "seed_col": seed_colors[i],
                "seed_rc": seed_rc,
                "line_col": line_colors[i],
                "tree_info": tree_info,
            }
        )

    if not any(info["tree_info"]["branch_count"] > 0 for info in lines_info):
        return None

    _segment_chain_lengths(lines_info, rng)
    if not _place_seed_chains(canvas, lines_info, frame_colors_pool, rng):
        return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        if rng.random() < 0.5:
            _try_place_solo_multicolor_frame(canvas, frame_colors_pool, rng)
        else:
            _try_place_solo_single_frame(canvas, frame_colors_pool, rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    for info in lines_info:
        if int((grid == info["seed_col"]).sum()) != 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    one_step = _one_step_multicolor_fill(grid)
    profile = fusion_profile(grid)

    if profile["branchy_seed_count"] < 1:
        return None
    if profile["max_round"] < 2 or profile["propagated_cavities"] < 1:
        return None
    if profile["one_step_delta_cells"] <= 0 or np.array_equal(out, one_step):
        return None
    if not profile["non_seed_fill_colors"]:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")

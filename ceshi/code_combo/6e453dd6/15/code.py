# Combo variant of 6e453dd6 (index 15)
# combo_spec: D2/0+D3/0+D7/0
# Name: 对角周期射线遇障碍停止

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and step trace
# - The left pane keeps the seed task's slide-first logic: every black component moves right
#   until its right edge is flush with the grey separator.
# - The trigger is still an enclosed background hole that touches column sep-2 after the slide,
#   but the generator now uses emitters with a single mouth row so one hole produces one nozzle row.
# - The emitted signal combines all three changes on the same branch:
#   1. path: each mouth row launches two 45-degree branches, one up-right and one down-right;
#   2. trajectory: each branch paints by a 2-step cycle derived from the hole width
#      (width 1 -> [red, bg], width 2 -> [red, blue]);
#   3. dst: the branch stops immediately before the first pre-existing colored target on that diagonal.
# - Small trace with sep=8 and a 2-wide hole on mouth row 9:
#   old task: row 9 right pane -> red red red red ...
#   combo task:
#     up branch path   : (8,9), (7,10), (6,11), ...
#     down branch path : (10,9), (11,10), (12,11), ...
#     cycle            : red, blue, red, blue, ...
#     blockers         : if a target already sits at (6,11), the up branch paints (8,9)=red,
#                        (7,10)=blue and stops before (6,11); the down branch continues until
#                        its own first target. The output therefore differs simultaneously in
#                        geometry, color rhythm, and stopping point.
#
# b) Complexity calibration
# - The right pane is widened to 5-7 columns. This is enough room for diagonal travel, visible
#   periodicity, and an interior blocker before the border.
# - Emitters use only hole widths 1 or 2. That keeps the periodic mapping small and learnable.
# - Within one sample all emitters share the same triggering hole width. This prevents ambiguous
#   collisions between branches with incompatible color cycles while still letting the task-level
#   dataset show both width classes across different examples.
#
# c) Implicit-assumption audit of the original main()
# - Separator detection and border-based background inference still work, but only if right-side
#   target pixels stay off the outer border; the generator enforces that.
# - Component sliding still assumes left-side sprites do not overlap after unsliding. Row-band
#   placement preserves that invariant.
# - Hole detection by "bg component not touching the left-pane boundary" remains valid because
#   enclosure semantics are unchanged.
# - The original row-level write `out[r, sep+1:] = red` is invalid in three ways here:
#   rays are diagonal rather than horizontal, colors vary along a branch, and targets already
#   occupy some right-side cells. The solver therefore walks each branch cell-by-cell, checks the
#   original input grid for blockers, and paints only non-blocked steps.
# - Because branches may cross, the solver needs collision-safe coloring. The generator guarantees
#   that all emitters in one sample use the same cycle, so any geometric overlap asks for the same
#   color on the same step index.
#
# d) Placement legality
# - The generator is output-first: place flush sprites, infer trigger rows, then place right-side
#   blockers on those exact future branch paths, then derive the input by unsliding sprites and
#   erasing emitted colors.
# - Targets are chosen only from interior right-pane cells, so they never overwrite the separator,
#   the border, or the left pane.
# - Every branch receives at least one blocker candidate on its own path; branches without a legal
#   interior blocker are rejected and regenerated.
#
# e) Visual observability
# - A learner can see three linked clues at once: the rays are diagonal, wide-hole examples produce
#   red/blue diagonals while narrow-hole examples produce dashed red diagonals, and both kinds stop
#   at colored right-side pixels rather than at the edge.
# - The generator rejects samples where all branches are too short or the periodic signature is too
#   faint, so the combo does not collapse into "some red near the right side".
#
# f) Visual cleanliness
# - The grey separator still isolates the moving left objects from the right-side emission logic.
# - Target colors are drawn from a palette that excludes red and blue, so blockers stay visually
#   distinct from the emitted branch colors.
# - Decoy targets may appear away from active branch paths, but they do not overlap the branches'
#   first reachable cells, which avoids clutter at the critical interaction points.
#
# g) Color inferability
# - The only trajectory-dependent colors are red and blue. Blue appears only on every second step
#   of 2-wide emitters; 1-wide emitters instead leave every second diagonal step as background.
# - Blocker colors carry no extra semantics: any non-background target on a branch path blocks it.
#
# h) New solver assumptions checked against the generator
# - Triggering holes have width 1 or 2, and all triggering holes within one sample share that width.
# - Each trigger row has two diagonal branches, and each branch has at least one interior blocker.
# - Cells before the first blocker on a branch contain only background in the input.
# - Decoy targets never accidentally become the first blocker of an intended branch.


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = 5) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if c != sep_col and g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if c != sep_col and g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    assert border, "expected non-separator border pixels to infer the background"
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must be unique on the border, got {winners.tolist()}"
    return int(winners[0])


def _hole_width(pts) -> int:
    cols = [c for _, c in pts]
    return int(max(cols) - min(cols) + 1)


def _ray_cycle(hole_w: int, bg: int, red: int = 2, blue: int = 1) -> list[int]:
    assert hole_w in (1, 2), f"expected triggering hole width 1 or 2, got {hole_w}"
    if hole_w == 1:
        return [red, bg]
    return [red, blue]


def _collect_trigger_specs(left: np.ndarray, sep: int, bg: int) -> list[tuple[list[int], int]]:
    h, _ = left.shape
    trigger_col = sep - 2
    specs = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        rows = sorted({int(r) for r, c in pts if int(c) == trigger_col})
        if not rows:
            continue
        hole_w = _hole_width(pts)
        specs.append((rows, hole_w))

    trigger_rows = [r for rows, _ in specs for r in rows]
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return specs


def _branch_cells(start_r: int, start_c: int, dr: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    rr = int(start_r)
    cc = int(start_c)
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += int(dr)
        cc += 1
    return cells


def _paint_periodic_branch(
    out: np.ndarray,
    blockers: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    cycle: list[int],
    bg: int,
) -> dict:
    h, w = out.shape
    rr = int(start_r)
    cc = int(start_c)
    step = 0
    visible = 0
    suppressed = 0
    hit_target = False

    while 0 <= rr < h and 0 <= cc < w:
        if int(blockers[rr, cc]) != int(bg):
            hit_target = True
            break
        color = int(cycle[step % len(cycle)])
        if color == int(bg):
            assert int(out[rr, cc]) == int(bg), f"unexpected prepainted dashed cell at {(rr, cc)}"
            suppressed += 1
        else:
            assert int(out[rr, cc]) in (int(bg), color), f"conflicting branch color at {(rr, cc)}"
            out[rr, cc] = color
            visible += 1
        rr += int(dr)
        cc += 1
        step += 1

    return {
        "steps": int(step),
        "visible": int(visible),
        "suppressed": int(suppressed),
        "hit_target": bool(hit_target),
    }


def _emit_all_branches(
    out: np.ndarray,
    blockers: np.ndarray,
    specs: list[tuple[list[int], int]],
    sep: int,
    bg: int,
) -> dict:
    h, w = out.shape
    stats = {
        "branch_count": 0,
        "stopped_branches": 0,
        "max_steps": 0,
        "visible": 0,
        "suppressed": 0,
    }
    for rows, hole_w in specs:
        cycle = _ray_cycle(hole_w, bg=bg, red=RED, blue=BLUE)
        for r in rows:
            for dr in (-1, +1):
                branch = _paint_periodic_branch(out, blockers, r + dr, sep + 1, dr, cycle, bg)
                stats["branch_count"] += 1
                stats["stopped_branches"] += int(branch["hit_target"])
                stats["max_steps"] = max(stats["max_steps"], int(branch["steps"]))
                stats["visible"] += int(branch["visible"])
                stats["suppressed"] += int(branch["suppressed"])
                assert _branch_cells(r + dr, sep + 1, dr, h, w), "branch path must exist"
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the grey divider, then launch periodic diagonal rays that stop at blockers.

    Input structure:
    - A single full-height grey (5) separator column splits the board into a left sprite pane and a right target pane.
    - Black components appear only on the left, while the right side contains background plus isolated colored blocker pixels.

    Rule:
    - Shift every black 4-connected component rightward until its right edge is flush with the separator.
    - In that slid arrangement, find enclosed background holes whose mouth touches column sep-2; those mouth rows become emitters, and the hole width chooses the branch cycle.
    - From each mouth row, cast one up-right and one down-right diagonal branch. Along each branch, paint by a repeating pattern (1-wide hole -> red/background, 2-wide hole -> red/blue) and stop just before the first blocker pixel already present on that diagonal.

    Key insight: The left side still tells you which enclosed mouths activate, but the final right-side pattern is a fused code: hole width chooses the diagonal rhythm, and the pre-existing target pixels decide where each diagonal branch terminates.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()

    # Step 1: slide black components flush to the separator.
    left = out[:, :sep].copy()
    comps = _components4(left == BLACK)
    left[left == BLACK] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = BLACK

    out[:, :sep] = left

    # Step 2: detect separator-adjacent enclosed holes and read their width.
    specs = _collect_trigger_specs(left, sep, bg)
    trigger_widths = {hole_w for _, hole_w in specs}
    assert trigger_widths.issubset({1, 2}), f"unexpected hole widths: {sorted(trigger_widths)}"

    # Step 3: emit periodic diagonal branches, checking blockers on the original input grid.
    _emit_all_branches(out, g, specs, sep, bg)
    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
BLUE = 1
RED = 2
SEP_COLOR = 5
TARGET_COLORS = [3, 4, 7, 8, 9]

# ── Grid dimensions ──
H_MIN, H_MAX = 18, 25
W_MIN, W_MAX = 12, 19
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 5, 7

# ── Layout constraints ──
N_SPRITES_MIN, N_SPRITES_MAX = 2, 3
SPRITE_W_MAX = 6
EDGE_PAD = 3
MIN_BAND_GAP = 1

# ── Probabilities ──
EXTRA_EMITTERS_PROB = 0.35
FLUSH_PROB = 0.3
SECOND_TARGET_PROB = 0.35
DECOY_TARGET_PROB = 0.6

# ── Retry budget ──
MAX_RETRIES = 500


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n):
    return (3 * n + 4) // 5


def _sprite_height(is_emitter: bool) -> int:
    if is_emitter:
        return 3
    return random.randint(2, 4)


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_emitter_flags(h):
    valid = []
    for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1):
        min_emitters = _min_emitters_for(n)
        for n_emitters in range(min_emitters, n + 1):
            min_rows = 2 * EDGE_PAD + (n - 1) * MIN_BAND_GAP + n_emitters * 3 + (n - n_emitters) * 2
            if min_rows <= h:
                valid.append((n, n_emitters))
    n, n_emitters = random.choice(valid)
    if n_emitters < n:
        max_emitters = max(k for k in range(n_emitters, n + 1)
                           if 2 * EDGE_PAD + (n - 1) * MIN_BAND_GAP + k * 3 + (n - k) * 2 <= h)
        if max_emitters > n_emitters and random.random() < EXTRA_EMITTERS_PROB:
            n_emitters = random.randint(n_emitters + 1, max_emitters)
    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _sprite_bands(h, flags):
    heights = [_sprite_height(is_emitter) for is_emitter in flags]
    n = len(heights)
    min_total = 2 * EDGE_PAD + (n - 1) * MIN_BAND_GAP + sum(heights)
    if min_total > h:
        return None

    slack = h - min_total
    gap_extras = _scatter(slack, n + 1)
    gaps = [EDGE_PAD + gap_extras[0]]
    for i in range(n - 1):
        gaps.append(MIN_BAND_GAP + gap_extras[i + 1])
    gaps.append(EDGE_PAD + gap_extras[-1])

    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh
        if i < n - 1:
            r += gaps[i + 1]
    return bands


def _emitter_patch(sh, max_sw, hole_w):
    assert sh == 3, f"emitters are designed as 3-row nozzle shells, got {sh}"
    assert hole_w in (1, 2), f"unexpected hole width {hole_w}"
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    left_wall = sw - hole_w - 2
    right_wall = sw - 1
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1, left_wall] = BLACK
    patch[1, right_wall] = BLACK
    return patch


def _distractor_patch(sh, max_sw):
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), BLACK, dtype=int)

    min_sw = 3 if kind.startswith("c_") or sh == 2 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    if kind == "c_left":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, -1] = BLACK
    elif kind == "c_right":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, 0] = BLACK
    else:
        patch[:, -1] = BLACK
        if random.random() < 0.5:
            patch[0, :] = BLACK
        else:
            patch[-1, :] = BLACK
    return patch


def _render_left_output(h, w, sep, bands, flags, hole_w):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw, hole_w) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0))
    return out, sprites


def _branch_records_from_specs(specs, sep, h, w):
    records = []
    for rows, hole_w in specs:
        for r in rows:
            for dr in (-1, +1):
                cells = _branch_cells(r + dr, sep + 1, dr, h, w)
                records.append({
                    "row": int(r),
                    "dr": int(dr),
                    "hole_w": int(hole_w),
                    "cells": cells,
                })
    return records


def _interior_blocker_candidates(cells, h, w):
    preferred = []
    fallback = []
    for step, (r, c) in enumerate(cells):
        if not (1 <= r <= h - 2 and 1 <= c <= w - 2):
            continue
        if step >= 3:
            preferred.append((step, r, c))
        elif step >= 2:
            fallback.append((step, r, c))
    return preferred or fallback


def _all_branch_path_cells(branches):
    cells = set()
    for branch in branches:
        for r, c in branch["cells"]:
            cells.add((int(r), int(c)))
    return cells


def _stamp_targets(base: np.ndarray, sep: int, branches: list[dict]) -> dict | None:
    h, w = base.shape
    info = {
        "branch_targets": [],
        "decoys": 0,
    }
    path_cells = _all_branch_path_cells(branches)
    path_membership = {}
    for branch in branches:
        for cell in branch["cells"]:
            path_membership[cell] = path_membership.get(cell, 0) + 1

    for branch in branches:
        candidates = _interior_blocker_candidates(branch["cells"], h, w)
        if not candidates:
            return None
        exclusive = [(step, r, c) for step, r, c in candidates if path_membership[(r, c)] == 1]
        if not exclusive:
            return None
        step, r, c = random.choice(exclusive)
        if int(base[r, c]) == BG:
            base[r, c] = random.choice(TARGET_COLORS)
        info["branch_targets"].append((r, c, step))

        farther = [(s, rr, cc) for s, rr, cc in exclusive if s > step]
        if farther and random.random() < SECOND_TARGET_PROB:
            _s2, rr2, cc2 = random.choice(farther)
            if int(base[rr2, cc2]) == BG:
                base[rr2, cc2] = random.choice(TARGET_COLORS)

    if random.random() < DECOY_TARGET_PROB:
        free = []
        for r in range(1, h - 1):
            for c in range(sep + 1, w - 1):
                if (r, c) in path_cells:
                    continue
                if int(base[r, c]) != BG:
                    continue
                free.append((r, c))
        if free:
            n_decoys = random.randint(1, min(3, len(free)))
            for r, c in random.sample(free, n_decoys):
                base[r, c] = random.choice(TARGET_COLORS)
                info["decoys"] += 1

    return info


def _render_output(h, w, sep, bands, flags, hole_w):
    out, sprites = _render_left_output(h, w, sep, bands, flags, hole_w)
    specs = _collect_trigger_specs(out[:, :sep], sep, BG)
    assert specs, "at least one emitter hole is required"
    assert {spec_hole_w for _, spec_hole_w in specs} == {hole_w}, "sample should use one hole width"

    branches = _branch_records_from_specs(specs, sep, h, w)
    target_info = _stamp_targets(out, sep, branches)
    if target_info is None:
        return None

    blockers = out.copy()
    stats = _emit_all_branches(out, blockers, specs, sep, BG)
    stats["hole_w"] = int(hole_w)
    stats["blue_cells"] = int(np.sum(out[:, sep + 1 :] == BLUE))
    stats["red_cells"] = int(np.sum(out[:, sep + 1 :] == RED))
    stats["targets"] = len(target_info["branch_targets"])
    stats["decoys"] = int(target_info["decoys"])
    return out, sprites, stats


def _unslide_sprites(h, w, sep, sprites, right_template):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    right = right_template[:, sep + 1 :].copy()
    right[np.isin(right, [RED, BLUE])] = BG
    inp[:, sep + 1 :] = right

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _visible_enough(stats):
    if stats["branch_count"] == 0:
        return False
    if stats["stopped_branches"] != stats["branch_count"]:
        return False
    if stats["max_steps"] < 4:
        return False
    if stats["visible"] < stats["branch_count"] + 2:
        return False
    if stats["hole_w"] == 1:
        return stats["suppressed"] >= stats["branch_count"]
    return stats["blue_cells"] >= stats["branch_count"]


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    hole_w = random.choice([1, 2])
    rendered = _render_output(h, w, sep, bands, flags, hole_w)
    if rendered is None:
        return None
    out, sprites, stats = rendered
    if not _visible_enough(stats):
        return None

    inp = _unslide_sprites(h, w, sep, sprites, out)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo variant input")

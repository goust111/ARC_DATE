# Combo variant of 6e453dd6 (index 17)
# combo_spec: D5/3+D10/0
# Name: 洞行数决定射线数+按距离排序处理

# Design analysis
# source_task: 6e453dd6
# combo: D5/3 + D10/0
#
# a) concretization + trace
#   - The original task clears all black sprites, slides every component flush to the separator,
#     then fires one red ray per triggering-hole row.
#   - This combo changes two links in that chain:
#       1. D10/0: sprites are processed from smaller initial gap to larger initial gap, and earlier
#          landed sprites remain in the left pane as obstacles for later sprites.
#       2. D5/3: even when a hole triggers, it contributes only one ray from its center row.
#   - Concrete cluster used by the generator:
#       near blocker: an L-shape whose right wall is closer to the separator
#       rear emitter: a closed shell one column farther left
#       free emitter: another closed shell in a disjoint row band
#   - Counterfactual trace on the blocked pair:
#       input gaps = blocker 1, rear emitter 4
#       original-style independent flush: both would reach sep-1, so the rear hole would touch sep-2
#       combo rule: blocker lands first; its wall at sep-1 stops the rear shell at right wall = sep-2,
#       so that shell's hole ends at sep-3 and no longer triggers
#   - The free emitter still reaches sep-1. If its hole spans rows {10,11,12}, the original task would
#     paint rows 10/11/12, while this combo paints only row 11.
#
# b) complexity calibration
#   - If every emitter hole had height 1, D5/3 would collapse back to the source task. The generator
#     therefore uses odd emitter heights 5 or 7, yielding hole heights 3 or 5 and a visible center row.
#   - If no sprites shared rows, D10/0 would be inert. The generator always includes one overlapping
#     blocker/emitter pair so the processing order changes the left-side final placement.
#   - The object vocabulary stays small: closed shell emitters, one L-shaped blocker, and at most no
#     extra decoration. This keeps the rule inferable from 2-3 examples.
#
# c) implicit-assumption audit for main()
#   - The source code assumed "clear all black, then move every component independently to sep-1".
#     That assumption is no longer valid, so sliding is rewritten as an ordered placement routine.
#   - Hole detection still assumes enclosed bg components are 4-connected and do not touch the outer
#     boundary of the left pane. The generator preserves this.
#   - The source task asserted unique trigger rows because it emitted every hole row. Under D5/3 we only
#     need hole centers, so duplicate centers are deduplicated with a set.
#   - Ordered placement introduces a new assumption: the sort key must be deterministic when distances
#     tie. The solver uses (distance, top row, left col). The generator keeps the interacting pair at
#     distinct distances so the critical behavior does not depend on tie-breaking.
#
# d) placement legality
#   - Input sprites are stamped as three separate 4-connected components.
#   - The blocker sits one row higher than the blocked emitter, so its horizontal bar never overlaps the
#     emitter hole rows in the independent-flush counterfactual.
#   - Row bands for the blocked cluster and the free emitter are separated by at least one full bg row.
#   - Widths, separator positions, and initial gaps are chosen so every component fits strictly inside
#     the left pane before any motion.
#
# e) visual observability
#   - The blocked shell and the free shell look like the same emitter family, but only the one that can
#     actually reach the separator produces a red line. This makes the order-dependent blocking visible.
#   - The emitted ray is a single line centered inside a tall cavity, so D5/3 is visible on the right.
#   - The ordering cue is geometric and human-visible: distance to the separator, not color id or index.
#
# f) visual cleanliness
#   - Only one right-side mark type exists: full-width red lines.
#   - The interacting pair is confined to one cluster, while the free emitter occupies a separate band.
#     That avoids clutter and makes the obstacle relationship readable.
#
# g) color inference
#   - Color roles stay unchanged from the source task: black sprites, grey separator, magenta background,
#     red rays. The combo is inferred entirely from placement and motion.
#
# h) combo metric used after implementation
#   - For each generated sample:
#       independent_emitters = number of center-row triggers after counterfactual independent flush
#       ordered_emitters = number of center-row triggers after combo-ordered placement
#       ordered_full_rows = total triggering rows under the old row-per-hole emission map
#   - A valid non-degenerate sample must satisfy:
#       independent_emitters > ordered_emitters
#       ordered_full_rows > ordered_emitters
#     The first inequality proves D10/0 matters; the second proves D5/3 matters.
#
# i) new solver assumptions checked against generator
#   - The interacting blocker/emitter pair has distinct initial distances. Guaranteed by construction.
#   - The free emitter remains non-interacting because its row band is disjoint from the blocked cluster.
#   - The right pane is blank before emission. Guaranteed directly by the generator.

import random

import numpy as np

from arc_utils import components_mask, paste


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


def _center_row_from_points(pts) -> int:
    rows = sorted({int(r) for r, _ in pts})
    assert rows, "hole must contain at least one row"
    return rows[(len(rows) - 1) // 2]


def _component_records(left: np.ndarray, black: int = 0) -> list[dict]:
    comps = []
    for pts in _components4(left == black):
        pts = [(int(r), int(c)) for r, c in pts]
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        comps.append(
            {
                "pts": pts,
                "max_c": max(cols),
                "min_r": min(rows),
                "min_c": min(cols),
            }
        )
    return comps


def _sorted_components_for_motion(left: np.ndarray, sep: int, black: int = 0) -> list[dict]:
    target_right = sep - 1
    comps = _component_records(left, black=black)
    comps.sort(key=lambda comp: (target_right - comp["max_c"], comp["min_r"], comp["min_c"]))
    return comps


def _max_noncolliding_delta(pts, occupied: np.ndarray, target_right: int) -> int:
    max_c = max(c for _, c in pts)
    max_delta = target_right - max_c
    assert max_delta >= 0, "component must start strictly inside the left pane"
    for delta in range(max_delta, -1, -1):
        if all(not occupied[r, c + delta] for r, c in pts):
            return delta
    raise AssertionError("component should always be placeable at delta 0")


def _slide_left_pane(left: np.ndarray, sep: int, bg: int, black: int = 0, *, ordered: bool) -> np.ndarray:
    target_right = sep - 1
    comps = _sorted_components_for_motion(left, sep, black=black)
    moved = left.copy()
    moved[moved == black] = bg
    occupied = moved == black

    for comp in comps:
        pts = comp["pts"]
        delta = _max_noncolliding_delta(pts, occupied, target_right) if ordered else target_right - comp["max_c"]
        for r, c in pts:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            moved[r, nc] = black
            occupied[r, nc] = True
    return moved


def _trigger_holes(left: np.ndarray, sep: int, bg: int):
    h, _ = left.shape
    trigger_col = sep - 2
    holes = []
    for pts in _components4(left == bg):
        pts = [(int(r), int(c)) for r, c in pts]
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if any(c == trigger_col for _, c in pts):
            holes.append(pts)
    return holes


def _trigger_rows(left: np.ndarray, sep: int, bg: int, *, center_only: bool) -> list[int]:
    rows = set()
    for pts in _trigger_holes(left, sep, bg):
        if center_only:
            rows.add(_center_row_from_points(pts))
        else:
            rows.update(int(r) for r, _ in pts)
    return sorted(rows)


def _solve(grid: np.ndarray, *, ordered: bool, center_only: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = _slide_left_pane(g[:, :sep].copy(), sep, bg, black=black, ordered=ordered)
    out[:, :sep] = left

    for r in _trigger_rows(left, sep, bg, center_only=center_only):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move black sprites toward the grey separator in distance order, then emit one center-row ray per valid hole.

    Input structure:
    - A single full-height grey (5) separator column splits the board into a left sprite pane and a right empty pane.
    - Several black 4-connected sprites live on the left over one uniform background color.

    Rule:
    - Sort left-side sprites by their initial horizontal gap to the separator, from nearest to farthest.
    - Process them in that order: each sprite shifts right as far as possible, but already-landed sprites stay put and can block later ones.
    - After all motion, find enclosed background holes whose rightmost cells touch the trigger column just inside the separator.
    - Each triggering hole paints exactly one red ray, using the hole's center row only, across the entire right pane.

    Key insight: The output depends on two coupled effects: motion is no longer simultaneous, so nearer sprites can prevent farther shells from reaching the separator, and even the shells that do trigger collapse their whole cavity into a single central ray.
    """
    return _solve(grid, ordered=True, center_only=True)


BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

SPRITE_W_MAX = 6
EMITTER_HEIGHTS = (5, 7)
BLOCKER_WIDTHS = (2, 3)
MIN_BAND_GAP = 1

MAX_RETRIES = 400


def _emitter_patch(sh: int, sw: int) -> np.ndarray:
    """Closed shell with a hole that reaches the column just left of the right wall when flush."""
    assert sh in EMITTER_HEIGHTS, f"unexpected emitter height {sh}"
    assert 3 <= sw <= SPRITE_W_MAX, f"unexpected emitter width {sw}"
    hole_w = 2 if sw >= 5 and random.random() < 0.5 else 1
    assert sw >= hole_w + 2

    patch = np.full((sh, sw), BG, dtype=int)
    left_wall = sw - hole_w - 2
    right_wall = sw - 1
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, left_wall] = BLACK
    patch[1:-1, right_wall] = BLACK
    return patch


def _blocker_patch(sh: int, sw: int) -> np.ndarray:
    """L-shape with a full right wall and a top bar placed outside the emitter's overlap rows."""
    assert sw in BLOCKER_WIDTHS, f"unexpected blocker width {sw}"
    patch = np.full((sh, sw), BG, dtype=int)
    patch[:, -1] = BLACK
    patch[0, :] = BLACK
    return patch


def _choose_sep_and_width(blocked_patch: np.ndarray, free_patch: np.ndarray, blocker_patch: np.ndarray):
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    blocker_gap = random.randint(1, 2)
    blocked_gap = blocker_gap + random.randint(2, 4)
    free_gap = random.randint(1, 4)

    need_sep = max(
        blocked_patch.shape[1] + blocked_gap,
        free_patch.shape[1] + free_gap,
        blocker_patch.shape[1] + blocker_gap,
    )
    valid_seps = [
        sep
        for sep in range(max(SEP_MIN, need_sep), SEP_MAX + 1)
        if W_MIN <= sep + 1 + right_w <= W_MAX
    ]
    if not valid_seps:
        return None

    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return sep, w, blocker_gap, blocked_gap, free_gap


def _choose_band_rows(h: int, cluster_h: int, free_h: int):
    min_total = cluster_h + free_h + 2 + MIN_BAND_GAP
    if min_total > h:
        return None
    spare = h - min_total
    top_extra = random.randint(0, spare)
    spare -= top_extra
    middle_extra = random.randint(0, spare)
    bottom_extra = spare - middle_extra

    cluster_first = random.random() < 0.5
    if cluster_first:
        cluster_r0 = 1 + top_extra
        free_r0 = cluster_r0 + cluster_h + MIN_BAND_GAP + middle_extra
    else:
        free_r0 = 1 + top_extra
        cluster_r0 = free_r0 + free_h + MIN_BAND_GAP + middle_extra

    assert cluster_r0 >= 1 and free_r0 >= 1
    assert max(cluster_r0 + cluster_h, free_r0 + free_h) <= h - 1 - bottom_extra
    return cluster_r0, free_r0


def _stamp_input(
    h: int,
    w: int,
    sep: int,
    blocked_patch: np.ndarray,
    blocker_patch: np.ndarray,
    free_patch: np.ndarray,
    blocked_gap: int,
    blocker_gap: int,
    free_gap: int,
    cluster_r0: int,
    free_r0: int,
) -> np.ndarray:
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    blocked_c0 = sep - blocked_patch.shape[1] - blocked_gap
    blocker_c0 = sep - blocker_patch.shape[1] - blocker_gap
    free_c0 = sep - free_patch.shape[1] - free_gap

    paste(inp, blocker_patch, cluster_r0, blocker_c0, transparent=BG)
    paste(inp, blocked_patch, cluster_r0 + 1, blocked_c0, transparent=BG)
    paste(inp, free_patch, free_r0, free_c0, transparent=BG)
    return inp


def _combo_metrics(inp: np.ndarray) -> tuple[int, int, int]:
    sep = _find_separator_col(inp, sep_color=SEP_COLOR)
    bg = _background_color(inp, sep_col=sep, sep_color=SEP_COLOR)
    left = inp[:, :sep]

    ordered_left = _slide_left_pane(left, sep, bg, black=BLACK, ordered=True)
    independent_left = _slide_left_pane(left, sep, bg, black=BLACK, ordered=False)

    ordered_emitters = len(_trigger_rows(ordered_left, sep, bg, center_only=True))
    independent_emitters = len(_trigger_rows(independent_left, sep, bg, center_only=True))
    ordered_full_rows = len(_trigger_rows(ordered_left, sep, bg, center_only=False))
    return independent_emitters, ordered_emitters, ordered_full_rows


def _gen_attempt() -> np.ndarray | None:
    blocked_h = random.choice(EMITTER_HEIGHTS)
    free_h = random.choice(EMITTER_HEIGHTS)
    blocked_w = random.randint(3, 5)
    free_w = random.randint(3, 5)
    blocker_w = random.choice(BLOCKER_WIDTHS)

    blocked_patch = _emitter_patch(blocked_h, blocked_w)
    free_patch = _emitter_patch(free_h, free_w)
    blocker_patch = _blocker_patch(blocked_h + 1, blocker_w)

    dims = _choose_sep_and_width(blocked_patch, free_patch, blocker_patch)
    if dims is None:
        return None
    sep, w, blocker_gap, blocked_gap, free_gap = dims

    cluster_h = blocker_patch.shape[0]
    min_h = max(H_MIN, cluster_h + free_h + 2 + MIN_BAND_GAP)
    if min_h > H_MAX:
        return None
    h = random.randint(min_h, H_MAX)

    bands = _choose_band_rows(h, cluster_h, free_h)
    if bands is None:
        return None
    cluster_r0, free_r0 = bands

    inp = _stamp_input(
        h,
        w,
        sep,
        blocked_patch,
        blocker_patch,
        free_patch,
        blocked_gap,
        blocker_gap,
        free_gap,
        cluster_r0,
        free_r0,
    )

    left_comps = _components4(inp[:, :sep] == BLACK)
    if len(left_comps) != 3:
        return None

    try:
        out = main(inp)
        baseline = _solve(inp, ordered=False, center_only=True)
        independent_emitters, ordered_emitters, ordered_full_rows = _combo_metrics(inp)
    except Exception:
        return None

    if ordered_emitters < 1:
        return None
    if independent_emitters <= ordered_emitters:
        return None
    if ordered_full_rows <= ordered_emitters:
        return None
    if np.array_equal(out, baseline):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

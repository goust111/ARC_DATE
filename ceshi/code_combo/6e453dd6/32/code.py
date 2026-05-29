# Combo variant of 6e453dd6 (index 32)
# combo_spec: D11/0+D21/0
# Name: 精灵已紧贴分隔线+全发射

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# source_task: original 6e453dd6, fused with D11/0 + D21/0.
#
# concreteization_and_trace:
# - Original task: black sprites sit on the left of a full grey separator. Most, but not
#   all, sprites are emitters, and many inputs start left-shifted. The solver first slides
#   each black component right until it becomes flush with the separator, then it searches
#   for enclosed bg pockets whose rightmost cells end up at sep-2, and finally paints red
#   rays on the right for the rows spanned by those pockets.
# - This combo removes both moving parts at once:
#   1. D21/0: every sprite is already flush in the input, so there is no horizontal motion.
#   2. D11/0: every sprite is an emitter, so there are no C/L/solid distractors.
# - Small trace:
#   original input bands: [left-shifted emitter, flush distractor, left-shifted emitter]
#   -> slide all three -> only emitters create enclosed holes next to sep -> only some
#   rows on the right become red.
#   combo input bands: [flush emitter, flush emitter, flush emitter]
#   -> no slide step -> each band already contains a sealed hole touching the inner face
#   next to the separator -> every sprite contributes at least one red row.
# - The visible difference is therefore structural, not chromatic: the left pane is already
#   aligned and every sprite visibly behaves like a ray source.
#
# complexity_calibration:
# - The combo is simpler than the original, but it is not trivialized into a single
#   rectangle-copy operation. The solver must still identify enclosed holes rather than any
#   bg next to a wall, and it must project exactly the rows occupied by the triggering hole.
# - Visual variety is preserved through 2-5 sprites, different heights, different widths,
#   and optional split emitters that can contain a secondary non-triggering cavity.
# - Repetition remains strong enough for human inference: in each sample all sprite bands
#   are flush and all of them emit, so the fused rule is supported multiple times.
#
# implicit_assumption_review_for_main:
# - _find_separator_col still assumes exactly one full-height grey column. Generator keeps
#   this unchanged.
# - _background_color still assumes the non-separator border is dominated by the true bg.
#   Generator leaves the border magenta except where a flush sprite naturally touches it.
# - main now assumes the left-side black components are already flush with the separator.
#   Generator enforces max_col == sep-1 for every component and main asserts this.
# - Hole detection treats any bg component touching the region boundary as outside space.
#   This remains valid because every emitter is a closed shell and no distractor openings
#   are generated.
# - Trigger rows must be unique when aggregated. Disjoint row bands guarantee that two
#   sprites never claim the same row.
#
# placement_legality:
# - Since the flush layout is now the actual input state, generation can synthesize inputs
#   directly instead of producing a shifted input from an output-first scaffold.
# - Sprites are still placed in non-overlapping vertical bands separated by at least one bg
#   row, so they cannot merge into one component.
# - Width is capped by sep-1, so every sprite fits strictly inside the left pane while its
#   right wall lands on sep-1.
#
# visual_observability_and_cleanliness:
# - The fused behavior is easy to see from a few examples: every black object already hugs
#   the divider and every one of them produces red rays. There is no need to infer which
#   objects are exceptions.
# - The right pane stays plain magenta before solving and only acquires horizontal red
#   stripes, so the transformation remains visually clean.
# - No hidden ordering by color id, component index, or scan order is used.
#
# change_inferability_and_new_main_assumptions:
# - Colors remain fixed from the base task: black=sprites, grey=separator, magenta=bg,
#   red=rays. The combo changes geometry only, not a color mapping.
# - New main assumptions introduced by removing the slide step:
#   1. every black component already touches sep-1,
#   2. the right side starts as pure bg,
#   3. each component contains at least one enclosed hole that reaches sep-2.
#   Generator explicitly verifies all three properties before returning a sample.


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


def _enclosed_bg_components(region: np.ndarray, bg: int):
    h, w = region.shape
    holes = []
    for pts in _components4(region == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _trigger_rows_from_region(region: np.ndarray, bg: int, trigger_col: int) -> list[int]:
    if trigger_col < 0 or trigger_col >= region.shape[1]:
        return []
    trigger_rows = []
    for pts in _enclosed_bg_components(region, bg):
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted({r for r, _ in pts}))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return trigger_rows


def _extract_component_patch(region: np.ndarray, comp) -> np.ndarray:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return region[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]


def _component_is_emitter(region: np.ndarray, comp, bg: int) -> bool:
    patch = _extract_component_patch(region, comp)
    return bool(_trigger_rows_from_region(patch, bg, patch.shape[1] - 2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read already-aligned black shells beside the grey divider and fire red rays from their sealed outlets.

    Input structure:
    - A single full-height grey (5) separator column splits the grid into a left sprite pane and a right empty pane.
    - Every black connected component on the left is already flush with the separator and sits on the magenta background.

    Rule:
    - Do not move the sprites: inspect the left pane exactly as given.
    - Find enclosed magenta regions inside the black shells; whenever a hole reaches the inner column immediately left of the separator, all rows of that hole become ray rows.
    - Paint those rows red across the whole right pane, while leaving the left pane and the separator unchanged.

    Key insight: The fusion removes both distractors and pre-alignment offsets, so the task reduces to pure hole detection: separator-adjacent enclosed bg rows are the only rows that project rightward.
    """
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = out[:, :sep]
    right = out[:, sep + 1 :]
    comps = _components4(left == black)

    assert comps, "expected at least one black sprite on the left"
    assert np.all(right == bg), "right pane must start as background only"
    for comp in comps:
        max_c = max(c for _, c in comp)
        assert max_c == sep - 1, "all sprites must already be flush with the separator"
        assert _component_is_emitter(left, comp, bg), "combo variant expects every sprite to be an emitter"

    trigger_rows = _trigger_rows_from_region(left, bg, sep - 2)
    for r in trigger_rows:
        out[r, sep + 1 :] = red

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

SPLIT_PROB = 0.4

MAX_RETRIES = 300


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_rows_needed(n_sprites):
    return 4 * n_sprites + 1


def _sprite_bands(h, n_sprites):
    min_heights = [3] * n_sprites
    base_gaps = [MIN_BAND_GAP] * (n_sprites + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    extra_cap = sum(SPRITE_H_MAX - sh for sh in heights)
    extra = random.randint(0, min(extra_cap, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i, sh in enumerate(heights) if sh < SPRITE_H_MAX]
        heights[random.choice(choices)] += 1

    slack = h - sum(heights) - sum(base_gaps)
    gaps = [base + add for base, add in zip(base_gaps, _scatter(slack, n_sprites + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w):
    place_top = random.random() < 0.5
    mini_h = random.randint(3, min(4, sh - 2))
    main_h = sh - mini_h + 1
    if place_top:
        patch[0, :mini_w] = BLACK
        for r in range(mini_h):
            patch[r, 0] = BLACK
            patch[r, mini_w - 1] = BLACK
        patch[mini_h - 1, :] = BLACK
        patch[-1, :] = BLACK
        for r in range(mini_h, sh):
            patch[r, left_wall] = BLACK
            patch[r, right_wall] = BLACK
    else:
        patch[0, :] = BLACK
        for r in range(main_h - 1):
            patch[r, left_wall] = BLACK
            patch[r, right_wall] = BLACK
        patch[main_h - 1, :] = BLACK
        patch[-1, :mini_w] = BLACK
        for r in range(main_h, sh):
            patch[r, 0] = BLACK
            patch[r, mini_w - 1] = BLACK


def _emitter_patch(sh, max_sw):
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK

    assert _trigger_rows_from_region(patch, BG, patch.shape[1] - 2), "emitter patch must trigger"
    return patch


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_sprite_count(h):
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(n) <= h]
    return random.choice(valid_counts)


def _render_input(h, w, sep, bands):
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    max_sw = min(SPRITE_W_MAX, sep - 1)
    for r0, sh in bands:
        patch = _emitter_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(inp, patch, r0, c0, transparent=BG)
    return inp


def _sample_metrics(inp: np.ndarray):
    sep = _find_separator_col(inp, sep_color=SEP_COLOR)
    bg = _background_color(inp, sep_col=sep, sep_color=SEP_COLOR)
    left = inp[:, :sep]
    comps = _components4(left == BLACK)
    flush_flags = [max(c for _, c in comp) == sep - 1 for comp in comps]
    emitter_flags = [_component_is_emitter(left, comp, bg) for comp in comps]
    trigger_row_counts = []
    for comp in comps:
        patch = _extract_component_patch(left, comp)
        trigger_row_counts.append(len(_trigger_rows_from_region(patch, bg, patch.shape[1] - 2)))
    return {
        "n_sprites": len(comps),
        "flush_flags": flush_flags,
        "emitter_flags": emitter_flags,
        "trigger_row_counts": trigger_row_counts,
    }


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    n_sprites = _choose_sprite_count(h)
    bands = _sprite_bands(h, n_sprites)
    if bands is None:
        return None

    inp = _render_input(h, w, sep, bands)

    try:
        if not np.all(inp[:, sep + 1 :] == BG):
            return None
        metrics = _sample_metrics(inp)
        if metrics["n_sprites"] != n_sprites:
            return None
        if not all(metrics["flush_flags"]):
            return None
        if not all(metrics["emitter_flags"]):
            return None
        if not all(count > 0 for count in metrics["trigger_row_counts"]):
            return None
        out = main(inp)
        if not np.any(out[:, sep + 1 :] == RED):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

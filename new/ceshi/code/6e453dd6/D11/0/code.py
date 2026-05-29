# Variant of 6e453dd6
# Dimension: D11 — 发射精灵与干扰精灵的比例 (meta)
# Change: >=60%的精灵为射线发射体（emitter），其余为干扰项（distractor）。干扰精灵形状类似但不满足封闭洞条件... → 所有精灵均为发射体（无干扰项）

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D11 variant
# -----------------------------------
# source_task: None, so the baseline is only the original 6e453dd6 code/spec.
#
# concreteization_and_trace:
# - The separator is still one full grey column. All black sprites still live on the left.
# - In the original task, some sprites were closed shells that produced rays after sliding,
#   while others were visually similar distractors such as C/L shapes or solid blocks.
# - In this variant, every sprite is a closed shell emitter. After sliding flush to the
#   separator, each sprite contains at least one enclosed bg pocket whose rightmost cells
#   sit at col sep-2, so each sprite paints at least one red row on the right.
# - Small trace:
#   original generation: [emitter, distractor, emitter] -> after sliding only sprites 1
#   and 3 have enclosed holes touching sep-2 -> only their rows become red.
#   variant generation: [emitter, emitter, emitter] -> after sliding all three have such
#   holes -> every sprite band contributes red rows.
#   The solver pipeline itself is unchanged: slide components, find enclosed bg regions,
#   and emit rays from hole rows touching sep-2. The change is therefore in what inputs
#   are generated, not in the geometric rule used to solve them.
#
# complexity_calibration:
# - This does simplify the discrimination subproblem, but not the whole task: the solver
#   still has to infer per-component translation and post-translation hole-to-ray logic.
# - Visual variety is kept via different sprite heights/widths and optional split emitters,
#   so the dataset does not collapse into identical rectangles.
# - The number of sprites remains 2-5, giving repeated evidence that "every sprite emits"
#   rather than a single accidental example.
#
# implicit_assumption_review_for_main:
# - _find_separator_col assumes exactly one full-height separator column. Generator keeps
#   that unchanged.
# - _background_color assumes the border majority is the true background. Generator keeps
#   borders as bg except the separator column, so this still holds.
# - Step 1 in main shifts each connected black component independently. This depends on
#   components being disjoint before and after shifting. Generator preserves pairwise
#   disjoint row bands, so sprites cannot merge during the slide.
# - Step 2 treats any bg component not touching the left-pane boundary as a hole. This
#   requires emitter cavities to stay fully enclosed after sliding. Generator builds only
#   closed shells, so the triggering cavity is always a true hole.
# - Step 3 assumes trigger rows are unique. Because sprite row bands are disjoint, hole
#   rows from different sprites cannot overlap; split emitters may contain extra holes,
#   but only one hole per sprite reaches sep-2.
#
# placement_legality:
# - Output-first generation is still the safest strategy because the constrained state is
#   the slid configuration. Each sprite is first stamped flush against the separator, then
#   unslid left by an independent delta.
# - Bands are separated by at least one full bg row, which avoids overlap and preserves
#   component identity both before and after sliding.
# - Sprite width is capped by sep-1, so unsliding always stays inside the left pane.
#
# visual_observability_and_cleanliness:
# - The D11 change is easy to see: every object in the output now has matching red rays,
#   instead of only a subset doing so.
# - The right side remains plain bg except for red horizontal rays, so there is no visual
#   clutter or ambiguous overpainting.
# - No hidden ordering based on color ids or other non-visible metadata is introduced.
#
# color_inference:
# - Colors are unchanged from the original task: black=sprites, grey=separator, red=rays,
#   magenta=background. The variant changes frequency of ray-producing shapes only, not any
#   color mapping.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide every black emitter against the grey divider, then fire red rays from its sealed interior channel.

    Input structure:
    - A single full-height grey (5) separator column splits the grid into a left sprite area and a right empty area.
    - Every black connected component on the left is an emitter-type sprite placed on a magenta background.

    Rule:
    - Move each black component horizontally right until its right edge becomes flush with the column just left of the separator.
    - In the shifted left pane, locate enclosed magenta regions; whenever such a hole reaches the inner face next to the separator, use all rows of that hole as ray sources.
    - Paint those source rows red across the entire right side, while keeping the shifted black sprites and the separator in place.

    Key insight: The task no longer asks which sprites are real emitters; every sprite is built to become a nozzle after sliding, so the only reasoning is "slide first, then read hole rows to the right of each shell."
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
    comps = _components4(left == black)

    # Clear old black pixels (leave other colors intact).
    left[left == black] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = black

    out[:, :sep] = left

    # Step 2: find enclosed background regions (holes) on the left.
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    # Step 3: holes touching the inner column next to the separator emit rays on their rows.
    trigger_col = sep - 2
    trigger_rows = []
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted(set(r for r, _ in pts)))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    return out


# --------------- generator ---------------

# Colors
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# Grid dimensions
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

# Sprite constraints
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# Shape variation
SPLIT_PROB = 0.4
FLUSH_PROB = 0.35

# Retry budget
MAX_RETRIES = 300


def _scatter(total, slots):
    """Distribute *total* units randomly into *slots* bins."""
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_rows_needed(n_sprites):
    """All sprites are emitters, so each needs height >=3 plus one-row gaps."""
    return 4 * n_sprites + 1


def _trigger_rows_from_patch(patch: np.ndarray) -> list[int]:
    """Rows whose cavity touches the shell's right wall and will reach sep-2 when flush."""
    sh, sw = patch.shape
    if sw < 2:
        return []
    rows = [r for r in range(sh) if patch[r, sw - 1] == BLACK and patch[r, sw - 2] == BG]
    return rows


def _sprite_bands(h, n_sprites):
    """Partition grid rows into disjoint vertical bands for emitter sprites."""
    min_heights = [3] * n_sprites
    base_gaps = [MIN_BAND_GAP] * (n_sprites + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    extra_cap = sum(SPRITE_H_MAX - x for x in heights)
    extra = random.randint(0, min(extra_cap, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i, x in enumerate(heights) if x < SPRITE_H_MAX]
        heights[random.choice(choices)] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n_sprites + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _emitter_patch(sh, max_sw):
    """Build an emitter sprite patch: a closed shell with a hole that touches the right wall."""
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

    assert _trigger_rows_from_patch(patch), "emitter patch must contain at least one triggering hole row"
    return patch


def _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w):
    """Fill *patch* in-place with a split emitter (mini section + main section)."""
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


def _choose_grid_dims():
    """Pick grid height, width, and separator column satisfying all constraints."""
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_sprite_count(h):
    """Choose 2-5 sprites, all emitters, subject to vertical fit constraints."""
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(n) <= h]
    return random.choice(valid_counts)


def _render_output(h, w, sep, bands):
    """Stamp emitter sprites flush against the separator and paint their red rays."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for r0, sh in bands:
        patch = _emitter_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        for lr in _trigger_rows_from_patch(patch):
            out[r0 + lr, sep + 1 :] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each sprite leftward by a random amount to produce the input grid."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    n_sprites = _choose_sprite_count(h)
    bands = _sprite_bands(h, n_sprites)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, bands)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    try:
        if len(_components4(inp[:, :sep] == BLACK)) != n_sprites:
            return None
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D11 input")

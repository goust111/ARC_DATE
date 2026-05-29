# Combo variant of 6e453dd6 (index 27)
# combo_spec: D4/2+D11/0+D13/1
# Name: 大量空心框架精灵全发射

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# concreteization_and_trace:
# - Baseline 6e453dd6 keeps a single full grey separator, slides black left-side
#   components flush to it, and then emits red rays from enclosed bg holes that
#   touch the column just inside the separator.
# - This combo keeps that causal chain but changes the generated object family:
#   every sprite is now a regular 1-pixel-thick hollow rectangle, and there are
#   many more of them (6-10 instead of 2-5). There are no distractors at all.
# - Small trace:
#   original-style scene: [emitter irregular shell, distractor C-shape, emitter]
#   -> after sliding, only some objects have enclosed holes at sep-2, so only a
#   subset of row bands becomes red.
#   combo scene: [frame, frame, frame, frame, frame, frame, ...]
#   -> after sliding, each frame's interior touches sep-2 along every interior
#   row, so all bands contribute rays and the right pane becomes densely striped.
# - The observable difference is therefore not a new post-slide rule; it is the
#   input distribution and output density: regular frames, more objects, all emit.
#
# complexity_calibration:
# - Making shapes into rectangles simplifies local shape parsing, but increasing
#   the count to 6-10 restores reasoning load: the solver must still detect many
#   independent translations and aggregate many ray rows.
# - To avoid over-simplifying into identical copies, frame heights and widths are
#   varied within small ranges. This keeps a repeated but still nontrivial motif.
# - Dense red output is deliberate: it makes the fused change visually obvious in
#   2-3 examples, unlike a mild distribution shift that might look accidental.
#
# implicit_assumption_review_for_main:
# - _find_separator_col still assumes exactly one full-height grey column; the
#   generator preserves that exactly.
# - _background_color assumes border majority identifies bg; the generator keeps
#   separator aside and leaves the outer border mostly magenta, so the winner is
#   unique even with many sprites.
# - Step 1 in main moves each left-side 4-connected black component separately.
#   That requires components to stay disjoint before and after sliding. The
#   generator uses non-overlapping row bands with >=1 bg row gaps, so frames do
#   not merge when shifted flush.
# - Step 2 treats a bg component as a hole only if it avoids the left-pane outer
#   boundary. Rectangular frames satisfy this cleanly: the interior cavity is one
#   enclosed component, while the exterior bg still touches the pane boundary.
# - Step 3 expects trigger rows from different holes to be unique. Disjoint row
#   bands guarantee that two frames never share a trigger row.
#
# placement_legality:
# - Output-first generation remains the safest construction. We first place each
#   frame flush with the separator (the constrained state), paint the rays, then
#   derive the input by shifting each frame left by an independent legal delta.
# - Width is capped by the left-pane width, and height is allocated by explicit
#   vertical band partitioning, so no object can overlap or leave the grid.
#
# visual_observability_and_cleanliness:
# - The left pane shows repeated hollow rectangles rather than mixed irregular
#   silhouettes, so the D4 change is visible without ambiguity.
# - All components emit, so the output shows red stripes adjacent to every frame
#   band; this makes the D11 change directly inspectable.
# - The larger sprite count produces noticeably denser outputs, which exposes the
#   D13 change without relying on hidden metadata or statistical inference.
#
# color_inference:
# - Colors stay the original black/grey/red/magenta mapping. No extra colors or
#   invisible color-order logic are introduced, so the rule remains inferable.


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
    Slide a crowd of hollow black frames onto the grey divider, then let every sealed cavity project red rays.

    Input structure:
    - One full-height grey (5) separator column splits the grid into a left sprite pane and a right empty pane.
    - The left pane contains 6-10 black connected components, each drawn as a regular hollow rectangle on a magenta background.

    Rule:
    - Shift every black component horizontally right until its right edge is flush with the column immediately left of the separator.
    - In that shifted left pane, find enclosed magenta regions; any enclosed region that reaches the inner column next to the separator acts as a nozzle, and all of its rows become ray rows.
    - Paint each ray row solid red across the right pane while keeping the shifted black frames and the grey separator unchanged.

    Key insight: Because every object is a hollow rectangular frame, the solver does not need to distinguish emitters from distractors; the whole task is to align all frames first and then read the interior rows that now press against the separator.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    left = out[:, :sep].copy()
    comps = _components4(left == black)

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

    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

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

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 25, 45
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 6, 10
SPRITE_H_MIN, SPRITE_H_MAX = 3, 5
SPRITE_W_MIN, SPRITE_W_MAX = 3, 6
MIN_BAND_GAP = 1
HEIGHT_SLACK_MAX = 8

FLUSH_PROB = 0.25
MAX_RETRIES = 400


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_rows_needed(n_sprites: int) -> int:
    return n_sprites * SPRITE_H_MIN + (n_sprites + 1) * MIN_BAND_GAP


def _trigger_rows_from_patch(patch: np.ndarray) -> list[int]:
    sh, sw = patch.shape
    if sh < 3 or sw < 2:
        return []
    return [r for r in range(sh) if patch[r, sw - 1] == BLACK and patch[r, sw - 2] == BG]


def _is_frame_patch(patch: np.ndarray) -> bool:
    sh, sw = patch.shape
    if sh < 3 or sw < 3:
        return False
    if not np.all(patch[0, :] == BLACK):
        return False
    if not np.all(patch[-1, :] == BLACK):
        return False
    if not np.all(patch[:, 0] == BLACK):
        return False
    if not np.all(patch[:, -1] == BLACK):
        return False
    if not np.all(patch[1:-1, 1:-1] == BG):
        return False

    expected = np.zeros_like(patch, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(patch == BLACK, expected)


def _is_frame_component(mask: np.ndarray) -> bool:
    sh, sw = mask.shape
    if sh < 3 or sw < 3:
        return False
    if not np.all(mask[0, :]):
        return False
    if not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]):
        return False
    if not np.all(mask[:, -1]):
        return False
    return not np.any(mask[1:-1, 1:-1])


def _choose_sprite_count() -> int:
    return random.randint(N_SPRITES_MIN, N_SPRITES_MAX)


def _choose_grid_dims(n_sprites: int) -> tuple[int, int, int]:
    min_h = max(H_MIN, _min_rows_needed(n_sprites))
    assert min_h <= H_MAX, "height range must be able to fit the chosen sprite count"
    # Keep the scene compact enough that the "many frames, many rays" pattern is obvious.
    h = random.randint(min_h, min(H_MAX, min_h + HEIGHT_SLACK_MAX))
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _sprite_bands(h: int, n_sprites: int):
    min_heights = [SPRITE_H_MIN] * n_sprites
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


def _frame_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(SPRITE_W_MIN, min(SPRITE_W_MAX, max_sw))
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    assert _is_frame_patch(patch), "frame patch must be a pure hollow rectangle"
    assert _trigger_rows_from_patch(patch), "frame patch must contain triggering rows"
    return patch


def _enough_trigger_rows(bands, n_sprites: int) -> bool:
    total_trigger_rows = sum(sh - 2 for _, sh in bands)
    return total_trigger_rows >= n_sprites + 2


def _render_output(h: int, w: int, sep: int, bands):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    if max_sw < SPRITE_W_MIN:
        return None, None

    for r0, sh in bands:
        patch = _frame_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        for lr in _trigger_rows_from_patch(patch):
            out[r0 + lr, sep + 1 :] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h: int, w: int, sep: int, sprites):
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


def _verify_input_structure(inp: np.ndarray, sep: int, n_sprites: int) -> bool:
    comps = _components4(inp[:, :sep] == BLACK)
    if len(comps) != n_sprites:
        return False

    for comp in comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            submask[r - r0, c - c0] = True
        if not _is_frame_component(submask):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_sprites = _choose_sprite_count()
    h, w, sep = _choose_grid_dims(n_sprites)
    bands = _sprite_bands(h, n_sprites)
    if bands is None:
        return None
    if not _enough_trigger_rows(bands, n_sprites):
        return None

    out, sprites = _render_output(h, w, sep, bands)
    if out is None or sprites is None:
        return None

    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    try:
        if not _verify_input_structure(inp, sep, n_sprites):
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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

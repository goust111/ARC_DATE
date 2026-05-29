# Variant of 6e453dd6
# Dimension: D3 — 射线轨迹涂色模式 (trajectory)
# Change: 连续涂红色——射线经过的每个右侧格子统一涂为红色(2)... → 射线长度受限（非到边缘）

import numpy as np

from arc_utils import components_mask, paste
import random


# Design analysis
# a_concretization_and_trace:
#   retained_structure:
#     - The grey separator is still a single full-height column.
#     - Black components still slide right until they become flush with the separator.
#     - Only enclosed background holes that touch the inner trigger column still emit.
#   new_core_behavior:
#     - Each emitting row no longer fills the whole right pane.
#     - Ray length = width of the enclosing slid sprite (its black-component bbox width).
#   trace_example:
#     - Suppose right pane width is 6 and an emitter becomes 4 cells wide after sliding.
#     - Original task: a triggered row paints all 6 right-side cells red.
#     - Variant: the same row paints only the first 4 cells red, leaving the last 2 as background.
#     - Therefore the rule change only touches the final painting step, but generation must make
#       the right pane wider than the sprite so the shorter ray is visible.
# b_complexity_calibration:
#   - Width takes a small visible range (typically 3..5), so examples repeat counts and remain learnable.
#   - The trigger condition is unchanged, preventing the task from becoming a pure size-measurement puzzle.
#   - Right-pane width is increased to 5..7 so "shorter than border" is common instead of rare.
# c_hidden_assumptions_from_original_main:
#   - Clearing and sliding black components is still valid because only the ray span changed.
#   - Hole detection by enclosed background components is still valid.
#   - The old line `out[r, sep + 1 :] = red` relied on the invariant "every ray reaches the border";
#     this is false now and must be replaced by a per-hole bounded segment.
#   - To compute the bound safely, the solver now keeps each shifted component's width and matches
#     every triggering hole to its enclosing shifted component.
# d_placement_legality:
#   - Row bands remain disjoint, so shifted sprites do not overlap and each hole belongs to one sprite.
#   - Generator caps emitter width at `right_w - 1`, guaranteeing rays stop before the border.
#   - The same cap also keeps output-first rendering aligned with the solver's width-based rule.
# e_visual_observability:
#   - Human-solvable cue: rows are selected by the hole near the separator, while red extent mirrors
#     the visible width of that black sprite.
#   - Multiple emitters with different widths appear in one sample family, making the bounded-ray rule
#     easier to infer than a single constant cutoff.
# f_visual_cleanliness:
#   - Only one layer is drawn on the right: red segments on otherwise uniform background.
#   - Because sprite row bands are separated, rays from different sprites never collide.
# g_color_inference:
#   - Colors are unchanged from the original task: black sprite, grey separator, red ray, magenta bg.
#   - No hidden color ordering or color-id arithmetic is introduced.
# h_new_solver_assumptions_checked_against_generator:
#   - The solver assumes each triggering hole has a unique enclosing shifted component.
#     Generator guarantees this via disjoint vertical bands and single-component sprites.
#   - The solver allows clipping if a user-made input violates the width budget, but the generator
#     guarantees strict non-clipping by constructing emitters with width < right-pane width.


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


def _component_info_after_slide(comp, delta: int):
    pts = [(int(r), int(c + delta)) for r, c in comp]
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return {
        "pts": set(pts),
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
        "width": max(cols) - min(cols) + 1,
    }


def _component_width_for_hole(hole, comp_infos) -> int:
    hole_rows = [int(r) for r, _ in hole]
    hole_cols = [int(c) for _, c in hole]
    hr0, hc0, hr1, hc1 = min(hole_rows), min(hole_cols), max(hole_rows), max(hole_cols)

    matches = []
    for info in comp_infos:
        r0, c0, r1, c1 = info["bbox"]
        if not (r0 <= hr0 <= hr1 <= r1 and c0 <= hc0 <= hc1 <= c1):
            continue
        if any(
            (int(r + dr), int(c + dc)) in info["pts"]
            for r, c in hole
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            matches.append(info)
    assert len(matches) == 1, f"expected unique enclosing component for hole, got {len(matches)}"
    return int(matches[0]["width"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites against a grey separator, then emit short red rays whose lengths encode sprite width.

    Input structure:
    - A single grey (5) separator column splits the grid into a left sprite field and an empty right pane.
    - Black connected components sit on the left over one background color, and some contain enclosed bg holes.

    Rule:
    - Shift every black component right until its rightmost pixel is adjacent to the separator.
    - After sliding, locate enclosed bg regions whose cells touch the trigger column just inside the separator.
    - For each row occupied by such a hole, paint a red ray on the right, but only for as many cells as the enclosing sprite is wide.

    Key insight: The hole still decides which rows fire, but the enclosing sprite's width decides how far the ray travels, so the right pane encodes a count instead of always filling to the border.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
    left_mask = left == black
    comps = _components4(left_mask)

    # Clear old black pixels (leave other colors intact).
    left[left == black] = bg

    target_right = sep - 1  # last column in left side, adjacent to separator
    comp_infos = []
    for comp in comps:
        max_c = max(c for _, c in comp)
        min_c = min(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        info = _component_info_after_slide(comp, delta)
        assert info["width"] == max_c - min_c + 1
        for r, c in info["pts"]:
            assert 0 <= c < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, c] = black
        comp_infos.append(info)

    out[:, :sep] = left

    # Step 2: find enclosed background regions (holes) on the left.
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    # Step 3: if a hole touches column sep-2, emit a bounded red ray on the right.
    trigger_col = sep - 2
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        ray_len = _component_width_for_hole(pts, comp_infos)
        ray_end = min(w, sep + 1 + ray_len)
        trigger_rows = sorted(set(int(r) for r, _ in pts))
        for r in trigger_rows:
            assert np.all(out[r, sep + 1 : ray_end] == bg), f"target ray segment on row {r} must be empty"
            out[r, sep + 1 : ray_end] = red

    return out


# --------------- generator (D3 width-limited rays) ---------------

# Colors
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# Grid dimensions
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 11, 16
SEP_MIN, SEP_MAX = 5, 10
RIGHT_W_MIN, RIGHT_W_MAX = 5, 7

# Sprite constraints
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# Probabilities
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

# Retry budget
MAX_RETRIES = 300


def _scatter(total, slots):
    """Distribute *total* units randomly into *slots* bins."""
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n):
    """Minimum emitter count to satisfy >=60% emitter fraction for *n* sprites."""
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters, n_total):
    """Minimum grid rows to fit *n_total* sprites with *n_emitters* emitters."""
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(h, is_emitter):
    """Partition grid rows into disjoint vertical bands for each sprite."""
    n = len(is_emitter)
    min_heights = [3 if e else 2 for e in is_emitter]
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    extra_cap = sum(SPRITE_H_MAX - x for x in heights)
    extra = random.randint(0, min(extra_cap, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i, x in enumerate(heights) if x < SPRITE_H_MAX]
        heights[random.choice(choices)] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _emitter_patch(sh, max_sw):
    """Build an emitter sprite patch with a triggering hole near the right wall."""
    assert sh >= 3 and max_sw >= 3
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


def _distractor_patch(sh, max_sw):
    """Build a non-emitter distractor sprite (C-shape, L-shape, or solid block)."""
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


def _choose_grid_dims():
    """Pick grid height, width, separator column, and right-pane width."""
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep, right_w


def _choose_emitter_flags(h):
    """Decide sprite count and which sprites are emitters."""
    valid_counts = [
        n
        for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
        if _min_rows_needed(_min_emitters_for(n), n) <= h
    ]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = min_em
    for k in range(min_em + 1, n + 1):
        if _min_rows_needed(k, n) <= h:
            max_em = k
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _trigger_rows_from_patch(patch):
    rows = []
    h, w = patch.shape
    for r in range(h):
        if w >= 2 and patch[r, -1] == BLACK and patch[r, -2] == BG:
            rows.append(r)
    return rows


def _render_output(h, w, sep, right_w, bands, flags):
    """Stamp slid sprites and paint bounded rays based on sprite width."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    shared_max_sw = min(SPRITE_W_MAX, sep - 1, right_w - 1)
    assert shared_max_sw >= 3, "need enough width for emitters and visible short rays"
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, shared_max_sw) if is_emitter else _distractor_patch(sh, shared_max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            ray_end = sep + 1 + sw
            assert ray_end <= sep + right_w, "emitter ray must stop before the border in this variant"
            for lr in _trigger_rows_from_patch(patch):
                out[r0 + lr, sep + 1 : ray_end] = RED
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
    h, w, sep, right_w = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, right_w, bands, flags)
    inp = _unslide_sprites(h, w, sep, sprites)
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D3 input")

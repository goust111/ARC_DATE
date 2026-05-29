# Variant of 6e453dd6
# Dimension: D3 — 射线轨迹涂色模式 (trajectory)
# Change: 连续涂红色——射线经过的每个右侧格子统一涂为红色(2) → 射线颜色由精灵属性决定

import random
import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D3 variant
# ----------------------------------
# variant_spec:
#   source_task_baseline:
#     reference_task: 16de56c4
#     takeaway: the changed color should be recoverable from a small, low-cardinality
#       structural cue, not from hidden ids. The original geometry can stay mostly intact
#       while the painted trajectory color becomes conditional.
#   concrete_rule:
#     unchanged_geometry:
#       - one full grey separator column
#       - 2..5 black sprites on the left
#       - every sprite slides right until its right edge touches the separator
#       - enclosed background holes that touch the inner face of the right wall emit rays
#     changed_trajectory:
#       - original behavior: every triggered row on the right becomes red(2)
#       - new behavior: ray color depends on the visible width of the triggering hole
#       - mapping:
#           width_1_hole: red(2)
#           width_2_hole: blue(1)
#     why_hole_width:
#       - it is visible in the input before sliding because the hole is part of the sprite
#       - it survives sliding unchanged
#       - it is binary and easy to support with repeated examples
#   behavior_trace:
#     example:
#       - suppose two emitters both slide flush to the separator
#       - emitter A has a 1-cell-wide enclosed channel touching sep-2
#       - emitter B has a 2-cell-wide enclosed channel touching sep-2
#       - original solver paints both corresponding row spans red
#       - variant paints A red and B blue
#     changed_stage:
#       - only the "emit ray on the right" stage changes
#       - left-half translation and hole detection stay the same
#   complexity_calibration:
#     - using exactly two ray colors keeps the mapping inferable from 2-4 examples
#     - every generated sample contains at least one narrow-hole emitter and one wide-hole emitter
#       so the color rule is visible inside a single grid
#     - non-emitters are retained to preserve the original discrimination task
#   implicit_assumption_audit:
#     original_main_lines:
#       - `red = 2` assumed all triggered holes share one output color
#       - `out[r, sep + 1:] = red` was safe only because row color never varied
#     variant_fix:
#       - compute a color per triggered hole before painting
#       - collect a row->color map and assert consistency if a row is ever requested twice
#     surviving_assumptions:
#       - sprites occupy disjoint row bands, so trigger rows from different sprites do not overlap
#       - triggered holes remain fully enclosed after sliding
#       - trigger holes have width 1 or 2; the generator guarantees this
#   placement_legality:
#     - row bands are still disjoint with >=1 background row gaps
#     - emitter widths 1 and 2 require sprite widths >=3 and >=4 respectively; sep>=5 keeps both legal
#     - unsliding translates whole sprites left without changing connectivity or hole geometry
#   visual_observability:
#     - the right half stays clean: only full-row rays, with at most two colors
#     - left sprites are black, so hole size is the only cue; repeating both sizes in each sample
#       gives enough evidence for the mapping
#   cleanliness:
#     - row-band separation prevents ray overlap ambiguity
#     - the separator and background colors are unchanged, so the new effect is isolated to trajectories
#   generator_guardrails:
#     - every sample includes both red and blue rays
#     - output-first generation is retained and validated by re-running main()
#   new_main_assumptions_checked_against_generator:
#     - if a hole touches sep-2, its width is either 1 or 2: ensured by _emitter_patch()
#     - if a row is triggered, the right side is background before painting: ensured by disjoint bands
#     - hole components not touching the left-pane border are true enclosed regions: ensured by sprite shapes


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


def _hole_width(pts: np.ndarray) -> int:
    cols = [c for _, c in pts]
    return int(max(cols) - min(cols) + 1)


def _ray_color_from_hole_width(width: int) -> int:
    if width <= 1:
        return RED
    return BLUE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the separator, then project color-coded rays from sealed holes.

    Input structure:
    - A single grey (5) separator column splits the board into a left sprite area and an empty right area.
    - Black sprites occupy only the left side on top of one shared background color.

    Rule:
    - Shift every black connected component horizontally until its rightmost pixel sits next to the separator.
    - In the shifted arrangement, locate enclosed background holes whose rightmost cells touch the inner face of the sprite wall.
    - For each row covered by such a hole, paint the full right-side trajectory; 1-cell-wide holes emit red rays, wider holes emit blue rays.

    Key insight: the translation step still reveals which cavities behave like nozzles, but now the nozzle's visible size determines the ray color instead of every trajectory sharing one fixed paint.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()

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

    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_col = sep - 2
    row_to_color = {}
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        width = _hole_width(pts)
        color = _ray_color_from_hole_width(width)
        for r in sorted(set(int(r) for r, _ in pts)):
            if r in row_to_color:
                assert row_to_color[r] == color, f"conflicting ray colors on row {r}"
            else:
                row_to_color[r] = color

    for r, color in sorted(row_to_color.items()):
        assert np.all(out[r, sep + 1:] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1:] = color

    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
BLUE = 1
RED = 2
SEP_COLOR = 5

RAY_COLORS = {1: RED, 2: BLUE}

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

# ── Sprite constraints ──
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# ── Probabilities ──
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

# ── Retry budget ──
MAX_RETRIES = 300


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n):
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters, n_total):
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(h, is_emitter):
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


def _emitter_patch(sh, max_sw, hole_w):
    """Build a closed shell whose triggering hole has width 1 or 2."""
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
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_sprite_roles(h):
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
                    if _min_rows_needed(_min_emitters_for(n), n) <= h]
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

    assert n_emitters >= 2, "variant needs both emitter width classes"
    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)

    emitter_widths = [1, 2] + [random.choice([1, 2]) for _ in range(n_emitters - 2)]
    random.shuffle(emitter_widths)

    hole_widths = []
    idx = 0
    for is_emitter in flags:
        if is_emitter:
            hole_widths.append(emitter_widths[idx])
            idx += 1
        else:
            hole_widths.append(None)
    return flags, hole_widths


def _trigger_rows_from_patch(patch):
    rows = []
    for lr in range(patch.shape[0]):
        if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
            rows.append(lr)
    return rows


def _render_output(h, w, sep, bands, flags, hole_widths):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter, hole_w in zip(bands, flags, hole_widths):
        if is_emitter:
            patch = _emitter_patch(sh, max_sw, hole_w)
        else:
            patch = _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            ray_color = RAY_COLORS[hole_w]
            for lr in _trigger_rows_from_patch(patch):
                out[r0 + lr, sep + 1:] = ray_color
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
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
    h, w, sep = _choose_grid_dims()
    flags, hole_widths = _choose_sprite_roles(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, bands, flags, hole_widths)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    right = out[:, sep + 1:]
    if not (np.any(right == RED) and np.any(right == BLUE)):
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

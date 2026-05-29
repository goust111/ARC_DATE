# Variant of 6e453dd6
# Dimension: D8 — 精灵平移后的残影处理 (transform)
# Change: erase——精灵平移后原位置被擦除为背景色，仅在新位置（紧贴分隔线）显示精灵... → 原位置变为特殊颜色
#
# Design analysis for the D8 variant
# spec_variant:
#   preserved:
#     - grid size, separator placement, sprite count/bands, emitter-vs-distractor mix
#     - slide target: every black sprite still becomes flush with the separator
#     - hole logic: only enclosed bg holes that touch sep-2 after sliding emit red rays
#     - right pane: still starts as pure background and only receives red rays
#   changed:
#     - old black pixels no longer clear to bg after the slide
#     - vacated sprite pixels recolor to blue(1), producing a visible afterimage
#     - moved sprite pixels stay black(0), so overlap is resolved by black-over-blue
#
# a) Concrete behavior and trace
#    Example: a 3x4 hollow sprite at cols 1..4 slides right by 2 so its new bbox is 3..6.
#    Original task:
#      1. erase old black cells to bg
#      2. draw black sprite at cols 3..6
#      3. evaluate holes in the moved black sprite and emit red rays
#    D8 task:
#      1. recolor old black cells to blue
#      2. draw the moved black sprite at cols 3..6
#      3. evaluate holes using only the moved black sprite and emit the same red rays
#    Difference: only the vacated cells in cols 1..2 stay visibly marked blue; any overlap
#    between old and new positions ends black because the moved sprite is the foreground result.
#
# b) Complexity calibration
#    The variant keeps a single deterministic recolor rule: "moved-away black -> blue".
#    That is simple enough to infer from 2-3 examples, but still visually obvious because
#    every non-flush sprite leaves a blue trail on the left half. No extra color mapping or
#    sprite-dependent case split is introduced.
#
# c) Old implicit assumptions that must change
#    - Original line `left[left == black] = bg` assumed vacated cells should disappear.
#      This is the main D8-dependent line and must become blue recoloring.
#    - Hole detection must still inspect bg components, so blue must *not* be treated as bg.
#      The solver therefore detects holes from `left == bg`, unchanged.
#    - The redraw order matters now: if blue is painted after black, it would corrupt the
#      moved sprite. The correct order is recolor old black first, then stamp moved black.
#
# d) Placement legality
#    Sprites stay in disjoint row bands exactly as in the base task. That guarantees:
#    - no two sprites overlap before sliding
#    - no two moved sprites fight for the same rows after sliding
#    - blue afterimages from different sprites do not collide with each other
#    The generator still constrains every unslid sprite to fit strictly inside cols 0..sep-1.
#
# e) Visual observability
#    Input still contains only bg, separator, and black sprites. Output introduces a new blue
#    color exactly where motion occurred, while black flush sprites and red rays preserve the
#    original causal structure. A human can compare input/output and see:
#    - black shapes shift right
#    - the previous black footprint becomes blue instead of disappearing
#    - ray rows are unchanged relative to the base puzzle logic
#
# f) Visual cleanliness
#    Blue is confined to the left pane, red is confined to the right pane, grey is a fixed
#    separator, and black remains the active sprite color. This keeps the regions visually
#    separated instead of mixing several overlays in one area.
#
# g) Color-rule inferability
#    The only new color is blue(1), and it has a structural cue: it always occupies the old
#    footprint of a black sprite after that sprite slides right. Because the mapping is fixed
#    and repeated across multiple sprites per grid, the recolor rule is learnable.

import numpy as np

from arc_utils import components_mask, paste
import random


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
    Slide black shapes rightward to a grey separator, leaving blue afterimages, then shoot red rays.

    Input structure:
    - A single full-height grey (5) separator column splits the grid into a wide left pane and a narrow right pane.
    - The left pane contains black sprites on one background color, while the right pane is still untouched background.

    Rule:
    - Translate each black connected component horizontally so its right edge becomes flush with the separator.
    - Recolor the component's old black footprint blue, then draw the translated sprite in black at its new flush position.
    - After the move, find enclosed background holes inside the black sprites; any such hole touching the column just left of the separator emits a red ray across the entire right pane on those rows.

    Key insight: The ray condition still depends only on holes in the slid black sprite, but the movement itself is now visible because vacated black pixels turn into a blue trail instead of vanishing.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    blue = 1
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: mark original black pixels as blue, then stamp each sprite flush against the separator.
    left = out[:, :sep].copy()
    comps = _components4(left == black)
    left[left == black] = blue

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

    # Step 2: find enclosed background regions (holes) in the moved black layout.
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    # Step 3: holes that touch sep-2 act as nozzles and emit red rays to the right edge.
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

# ── Colors ──
BG = 6
BLACK = 0
BLUE = 1
RED = 2
SEP_COLOR = 5

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
    """Build an emitter sprite patch: a closed shell with an interior hole at the right edge."""
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
    """Pick grid height, width, and separator column satisfying all constraints."""
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_emitter_flags(h):
    """Decide sprite count and which sprites are emitters."""
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

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _emitter_rows(patch):
    rows = []
    for lr in range(patch.shape[0]):
        if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
            rows.append(lr)
    return rows


def _trail_patch(patch):
    return np.where(patch == BLACK, BLUE, BG)


def _render_flush_sprites(h, w, sep, bands, flags):
    """Create the canonical flush sprite set used as the moved end-state."""
    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        sprites.append((patch, r0, c0))
    return sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each sprite leftward by a random amount to produce the input grid."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    deltas = []
    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None, None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
        deltas.append(delta)
    return inp, deltas


def _render_expected_output(h, w, sep, sprites, flags, deltas):
    """Render the D8 output manually: blue trails first, then flush black sprites, then red rays."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    for (patch, r0, c0), delta in zip(sprites, deltas):
        paste(out, _trail_patch(patch), r0, c0 - delta, transparent=BG)

    for (patch, r0, c0), is_emitter in zip(sprites, flags):
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lr in _emitter_rows(patch):
                out[r0 + lr, sep + 1 :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build a valid input and verify the D8 output manually."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    sprites = _render_flush_sprites(h, w, sep, bands, flags)
    inp, deltas = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    expected = _render_expected_output(h, w, sep, sprites, flags, deltas)

    try:
        if BLUE in set(map(int, inp.flatten())):
            return None
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 D8 input")

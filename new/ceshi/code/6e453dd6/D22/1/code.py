# Variant of 6e453dd6
# Dimension: D22 — 射线涂色范围 (trajectory)
# Change: 射线涂满整行右侧（从sep+1到w-1），同一行所有列统一涂红色... → 射线涂色高度扩展（不仅当前行）

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for D22
# a) Concrete behavior and flow trace:
#    The left half still contains black sprites on a magenta background, split from
#    the right half by one full grey separator column. Each black component still
#    slides right until its outer wall touches the separator. The changed behavior
#    starts only after the slide: a closed hole that reaches the inner nozzle column
#    (sep-2) still triggers a red projection, but the projection is no longer a
#    one-row line. If a trigger hole spans rows 5..7 and has width 2, the old rule
#    painted rows 5,6,7 on the right. The new rule paints a rectangle covering rows
#    3..9 because the hole width 2 is reused as the up/down expansion amount.
# b) Complexity calibration:
#    Hole widths are kept in a small visible set {1,2,3}. That is enough to produce
#    noticeably different block heights without creating too many hard-to-track cases.
#    The generator forces at least one width >= 2 emitter so the variant does not
#    collapse into outputs that look like the original thin rays across all samples.
# c) Implicit assumptions audit:
#    The original Step 3 assumed each emitting row was painted once and therefore
#    asserted that the whole right slice of that row was still background before
#    assignment. D22 breaks that assumption because expanded rectangles from nearby
#    holes may overlap vertically. The solver now computes per-hole blocks and writes
#    red without requiring untouched rows.
# d) Placement legality:
#    Sprites still occupy disjoint vertical bands, so sliding cannot merge them on
#    the left. The right half is reserved for ray blocks only, and overlap between
#    red rectangles is legal because all overlaps keep the same color.
# e) Visual observability:
#    A human can see that the red output is thicker than the hole rows themselves,
#    and that thicker holes create taller red blocks. Using only 1-3 width levels
#    keeps the mapping inferable from 2-4 examples.
# f) Visual cleanliness:
#    The right side remains a single-color projection area. Even when two expanded
#    blocks touch or merge, there is no color conflict with the sprites or separator.
# g) Color inference:
#    No new color mapping is introduced. Red still means "emitted projection"; only
#    the geometric extent changes, which is visible from the relation between the
#    hole width and the block height.


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


def _trigger_hole_specs(left: np.ndarray, sep: int, bg: int) -> list[tuple[int, int, int]]:
    """Return [(row_min, row_max, hole_width), ...] for enclosed holes that touch sep-2."""
    h, _ = left.shape
    trigger_col = sep - 2
    specs = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if not any(c == trigger_col for _, c in pts):
            continue
        rows = [int(r) for r, _ in pts]
        cols = {int(c) for _, c in pts}
        hole_w = len(cols)
        assert hole_w >= 1, "trigger hole must occupy at least one column"
        specs.append((min(rows), max(rows), hole_w))
    return specs


def _paint_expanded_rays(out: np.ndarray, sep: int, specs: list[tuple[int, int, int]], red: int) -> None:
    """Paint rectangular red blocks on the right, expanding up/down by hole width."""
    h, _ = out.shape
    for row_min, row_max, hole_w in specs:
        top = max(0, row_min - hole_w)
        bottom = min(h - 1, row_max + hole_w)
        out[top : bottom + 1, sep + 1 :] = red


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to the grey divider, then emit widened red projections from enclosed holes.

    Input structure:
    - A single full-height grey (5) column splits the grid into a left sprite pane and a right projection pane.
    - Black components live on the left over one background color, while the right side starts empty.

    Rule:
    - Move every black connected component horizontally until its rightmost black cell sits against the separator.
    - On the slid left pane, find enclosed background holes whose cells reach the inner nozzle column just before the separator.
    - Each such hole still projects red to the full right half, but now the painted region expands above and below the hole by an amount equal to that hole's width, turning a line into a rectangle.

    Key insight: The separator-facing hole still acts like the trigger, but the new output size depends on a visible property of the hole itself: wider holes create taller red blocks.
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
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = int(c) + delta
            assert 0 <= nc < sep, f"shifted pixel {(int(r), int(c))} left the pane"
            left[int(r), nc] = black

    out[:, :sep] = left

    # Step 2 + 3: enclosed holes touching sep-2 emit rightward blocks whose vertical padding equals hole width.
    specs = _trigger_hole_specs(left, sep=sep, bg=bg)
    _paint_expanded_rays(out, sep=sep, specs=specs, red=red)

    return out


# --------------- generator (D22 variant) ---------------

# ── Colors ──
BG = 6
BLACK = 0
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

# ── Variant controls ──
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35
WIDE_HOLE_PROB = 0.7
MAX_HOLE_WIDTH = 3

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


def _choose_hole_width(max_sw: int, force_wide: bool) -> int:
    valid = [w for w in range(1, min(MAX_HOLE_WIDTH, max_sw - 2) + 1)]
    assert valid, "emitter needs room for a hole and two walls"
    wide = [w for w in valid if w >= 2]
    if force_wide and wide:
        return random.choice(wide)
    if wide and random.random() < WIDE_HOLE_PROB:
        return random.choice(wide)
    return random.choice(valid)


def _emitter_patch(sh, max_sw, force_wide=False):
    """Build an emitter sprite patch and return local trigger-hole specs."""
    hole_w = _choose_hole_width(max_sw, force_wide=force_wide)
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        trigger_specs = _build_split_emitter(patch, sh, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK
        trigger_specs = [(1, sh - 2, hole_w)]
    return patch, trigger_specs


def _build_split_emitter(patch, sh, left_wall, right_wall, mini_w):
    """Fill *patch* with a split emitter and return the triggering hole spec."""
    hole_w = right_wall - left_wall - 1
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
        return [(mini_h, sh - 2, hole_w)]

    patch[0, :] = BLACK
    for r in range(main_h - 1):
        patch[r, left_wall] = BLACK
        patch[r, right_wall] = BLACK
    patch[main_h - 1, :] = BLACK
    patch[-1, :mini_w] = BLACK
    for r in range(main_h, sh):
        patch[r, 0] = BLACK
        patch[r, mini_w - 1] = BLACK
    return [(1, main_h - 2, hole_w)]


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


def _render_output(h, w, sep, bands, flags):
    """Stamp flush sprites and paint D22 expanded ray blocks."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    trigger_specs = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    emitter_indices = [i for i, flag in enumerate(flags) if flag]
    wide_anchor = random.choice(emitter_indices)
    for i, ((r0, sh), is_emitter) in enumerate(zip(bands, flags)):
        if is_emitter:
            patch, local_specs = _emitter_patch(sh, max_sw, force_wide=(i == wide_anchor))
        else:
            patch = _distractor_patch(sh, max_sw)
            local_specs = []
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        for lr0, lr1, hole_w in local_specs:
            trigger_specs.append((r0 + lr0, r0 + lr1, hole_w))
        sprites.append((patch, r0, c0))

    _paint_expanded_rays(out, sep=sep, specs=trigger_specs, red=RED)
    return out, sprites, trigger_specs


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


def _baseline_red_area(trigger_specs, right_w):
    """Red area under the original thin-ray rule for the same trigger holes."""
    return sum((row_max - row_min + 1) * right_w for row_min, row_max, _ in trigger_specs)


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites, trigger_specs = _render_output(h, w, sep, bands, flags)
    if not trigger_specs or max(hole_w for _, _, hole_w in trigger_specs) < 2:
        return None
    if int(np.sum(out[:, sep + 1 :] == RED)) <= _baseline_red_area(trigger_specs, w - sep - 1):
        return None

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
    raise RuntimeError("failed to generate a valid 6e453dd6 D22 input")

# Combo variant of 6e453dd6 (index 26)
# combo_spec: D8/0+D22/1
# Name: 保留原位+射线高度扩展

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for combo D8/0 + D22/1
# a) Concrete behavior and flow trace:
#    Start from the original task: each black component on the left slides right until its
#    right wall reaches the grey separator, then any enclosed background hole that reaches
#    the nozzle column sep-2 paints a thin red row-block on the right half.
#    In this combo, the slide is reinterpreted as "copy to flush position" rather than
#    "move then erase", so the output left pane contains both the source sprite and its
#    separator-flush copy. The ray trigger must still be read from the flush copy only.
#    Then D22 changes the ray geometry: if a trigger hole spans rows 8..10 and has width 2,
#    the original task paints only rows 8..10 on the right, while the combo paints rows
#    6..12 across the full right width. The visible difference is therefore dual: duplicated
#    black sprites on the left and thicker rectangular red blocks on the right.
# b) Complexity calibration:
#    Hole widths are restricted to a small visible set {1,2,3}. That gives a compact mapping
#    from hole width to block height, and the generator forces at least one width>=2 emitter.
#    To make keep_source legible, the generator also forces at least one nonzero left shift
#    and usually one shift of at least 2 columns when space allows.
# c) Implicit assumptions audit:
#    - The original `left[left == black] = bg` assumed erase semantics; that is invalid now.
#    - Hole detection cannot run on the displayed union of source+copy, because preserved source
#      pixels can create extra enclosed background pockets. The solver therefore builds both a
#      display layer and a flush-only analysis layer from the same input components.
#    - The original thin-ray code asserted that each painted row was untouched beforehand.
#      Expanded rectangles can overlap vertically, so painting must allow repeated writes.
#    - Components must still be extracted from the input pane before copying, otherwise source
#      and copied pixels could merge and break per-component shift computation.
# d) Placement legality:
#    Sprites remain in disjoint vertical bands, so retaining sources does not cause inter-sprite
#    ambiguity. Every copied pixel stays within columns 0..sep-1 because each shift is
#    `sep - 1 - max_c`. The right half is reserved for red rectangles only; overlaps are safe
#    because all ray blocks use the same red color.
# e) Visual observability:
#    Across 2-4 samples a human can compare input/output and see that left objects do not vanish
#    after the flush copy appears. On the right, the red region is visibly taller than the hole
#    rows themselves, and wider holes repeatedly produce taller rectangles, which exposes the
#    width->height rule without any hidden numeric cue.
# f) Visual cleanliness:
#    The separator still cleanly partitions black geometry from red projections. The left pane
#    shows at most two aligned copies of each sprite, while the right pane uses only one solid
#    color, so even overlapping rectangles remain readable rather than noisy.
# g) Change inferability:
#    No new colors are introduced. Red keeps its original semantic role as the emitted projection;
#    only its vertical extent changes, and that extent is anchored to a visible hole width in the
#    flush copy next to the separator.


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


def _build_flush_and_display_left(
    left: np.ndarray,
    *,
    bg: int,
    black: int,
    sep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (display_left, flush_only_left) for the keep-source combo."""
    comps = _components4(left == black)
    display_left = left.copy()
    flush_only_left = left.copy()
    flush_only_left[flush_only_left == black] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            rr = int(r)
            cc = int(c)
            nc = cc + delta
            assert 0 <= nc < sep, f"shifted pixel {(rr, cc)} left the pane"
            display_left[rr, nc] = black
            flush_only_left[rr, nc] = black

    return display_left, flush_only_left


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
    """Paint right-side red rectangles, expanding above/below by hole width."""
    h, _ = out.shape
    for row_min, row_max, hole_w in specs:
        top = max(0, row_min - hole_w)
        bottom = min(h - 1, row_max + hole_w)
        out[top : bottom + 1, sep + 1 :] = red


def main(grid: np.ndarray) -> np.ndarray:
    """
    Duplicate black shapes against the grey divider, then emit vertically expanded red blocks from the copied holes.

    Input structure:
    - One full-height grey (5) separator column splits the board into a left sprite pane and a right projection pane.
    - Black connected components appear only on the left over a single background color, and the right pane starts empty.

    Rule:
    - For each black component, create a right-shifted copy whose right edge becomes flush with the separator, while keeping the original component visible in place.
    - Detect enclosed background holes only inside the flush-copy layer; preserved source pixels do not participate in the trigger logic.
    - Whenever such a hole reaches the inner nozzle column just left of the separator, paint the full right half red not only over the hole rows but also above and below them by an amount equal to that hole's width, producing a rectangle.

    Key insight: The output combines two coordinated cues: the left pane reveals that sprites were copied rather than erased, and the right pane reveals that separator-facing hole width controls the vertical thickness of the emitted red block.
    """
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = g[:, :sep].copy()
    display_left, flush_left = _build_flush_and_display_left(left, bg=bg, black=black, sep=sep)
    out[:, :sep] = display_left

    specs = _trigger_hole_specs(flush_left, sep=sep, bg=bg)
    _paint_expanded_rays(out, sep=sep, specs=specs, red=red)
    return out


# --------------- generator (combo D8/0 + D22/1) ---------------

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
FLUSH_PROB = 0.2
WIDE_HOLE_PROB = 0.7
MAX_HOLE_WIDTH = 3

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
    """Build an emitter patch and return local trigger-hole specs."""
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


def _choose_emitter_flags(h):
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


def _render_flush_output(h, w, sep, bands, flags):
    """Build the flush-copy state used for ray logic and copied sprites."""
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
    """Shift flush sprites leftward to produce the input and record deltas."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    prominent_choices = [i for i, (_patch, _r0, c0) in enumerate(sprites) if i != flush_i and c0 >= 2]
    prominent_i = random.choice(prominent_choices) if prominent_choices else -1

    deltas = []
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None, None
        if i == flush_i:
            delta = 0
        elif i == prominent_i:
            delta = random.randint(2, c0)
        else:
            delta = random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
        deltas.append(delta)
    return inp, deltas


def _compose_combo_output(inp, flush_out, sprites, sep):
    out = inp.copy()
    for patch, r0, c0 in sprites:
        paste(out, patch, r0, c0, transparent=BG)
    out[:, sep + 1 :] = flush_out[:, sep + 1 :]
    return out


def _baseline_red_area(trigger_specs, right_w):
    return sum((row_max - row_min + 1) * right_w for row_min, row_max, _ in trigger_specs)


def _source_trace_cells(inp, expected, sep):
    """Count black cells that remain from the source layer only."""
    actual_flush = np.zeros_like(inp[:, :sep], dtype=bool)
    for comp in _components4(inp[:, :sep] == BLACK):
        max_c = max(int(c) for _, c in comp)
        delta = (sep - 1) - max_c
        for r, c in comp:
            actual_flush[int(r), int(c) + delta] = True
    return int(np.sum((expected[:, :sep] == BLACK) & (~actual_flush)))


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    flush_out, sprites, trigger_specs = _render_flush_output(h, w, sep, bands, flags)
    if not trigger_specs or max(hole_w for _, _, hole_w in trigger_specs) < 2:
        return None
    if int(np.sum(flush_out[:, sep + 1 :] == RED)) <= _baseline_red_area(trigger_specs, w - sep - 1):
        return None

    inp, deltas = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None
    if max(deltas) <= 0:
        return None

    expected = _compose_combo_output(inp, flush_out, sprites, sep)
    if _source_trace_cells(inp, expected, sep) <= 0:
        return None

    try:
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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

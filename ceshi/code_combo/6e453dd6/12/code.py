# Combo variant of 6e453dd6 (index 12)
# combo_spec: D1/1+D6/2
# Name: 垂直平移配水平分隔线
#
# Design analysis
# ===============
# a) Concretization + behavior trace
#    - Original task:
#      * one full grey separator column splits the grid into left and right panes;
#      * black sprites live on the left;
#      * each sprite slides right until flush with the separator;
#      * a fully enclosed background hole whose rightmost cells land on the inner face
#        of that sprite fires a horizontal red ray into the right pane.
#    - Fused variant:
#      * rotate the whole mechanism by 90 degrees;
#      * the separator is now one full grey row, splitting the grid into upper and lower panes;
#      * sprites live above the separator and fall straight downward;
#      * a closed hole that ends up touching the separator-adjacent inner bottom face
#        fires a vertical red ray into the lower pane.
#    - Small concrete trace:
#      * Take a 3x3 ring:
#          000
#          060
#          000
#      * Original: after rightward sliding, the hole cell sits at column sep-2, so the
#        output paints that hole row red across the right pane.
#      * Here: after downward sliding, the hole cell sits at row sep-2, so the output
#        paints that hole column red through the lower pane.
#      * Difference: both the motion axis and the ray axis rotate together; this is not
#        just a separator swap, but a full spatial reorientation of cause and effect.
# b) Complexity calibration
#    - Keeping the old 10-16 width budget after rotation would make 2-5 side-by-side
#      upper-pane sprites too cramped, so the generator swaps the aspect bias:
#      height becomes 10-16, width becomes 16-25, and the lower receiving pane stays
#      narrow (2-4 rows) just like the original right receiving pane stayed narrow.
#    - The sprite families remain comparable to the source task:
#      * >=60% emitters with true enclosed holes,
#      * distractors that look locally similar but are open or solid,
#      * multiple disjoint objects per grid.
#      This preserves visual ambiguity without adding extra symbolic machinery.
# c) Implicit-assumption audit of the original solver
#    - `_find_separator_col` assumed a unique full separator column. The fused rule needs
#      a unique full separator row instead.
#    - `_background_color(..., sep_col)` excluded one separator column from border voting.
#      After rotation it must exclude one separator row.
#    - `left = out[:, :sep]` assumed the active pane is left of the divider. The fused
#      active pane is `top = out[:sep, :]`.
#    - `delta = target_right - max_c` assumed horizontal translation. Now motion is
#      vertical, so the shift is `target_bottom - max_r`.
#    - Hole filtering originally rejected bg components touching rows `0/h-1` or columns
#      `0/sep-1`. After rotation, the relevant upper-pane boundary is rows `0/sep-1`
#      and columns `0/w-1`.
#    - Trigger collection used inner column `sep-2` and emitted by row. Here it uses
#      inner row `sep-2` and emits by column.
#    - The original paint assert depended on the right pane being blank. The rotated
#      version similarly depends on the lower pane being blank before red columns appear.
# d) Placement legality
#    - The generator allocates sprites to disjoint column bands with at least one full
#      background column between neighboring bands, so downward motion cannot merge them.
#    - Output sprites are rendered flush to the separator while leaving at least one
#      spare row above (`sep - sprite_h >= 1`), guaranteeing legal upward unsliding.
#    - The separator row is chosen so the lower pane has 2-4 rows and the upper pane is
#      tall enough to host all sprites plus gaps.
# e) Visual observability
#    - A human can infer the fusion from a few examples because several cues move in lockstep:
#      * the divider is horizontal, not vertical;
#      * sprites occupy the upper pane, not the left pane;
#      * motion is downward, not rightward;
#      * red marks are vertical columns in the lower pane, not horizontal streaks.
#    - Emitter vs distractor status is readable from geometry: closed downward-facing
#      shells produce rays, open C/L/solid shapes do not.
# f) Visual cleanliness
#    - Only four colors appear: bg, black sprites, grey separator, red rays.
#    - Red is confined to the lower pane and black to the upper pane, so the two effects
#      never visually compete in the same area.
#    - Band separation prevents overlapping trigger columns, keeping each red ray legible.
# g) Behavior non-degeneracy
#    - To avoid accidental reversion to the source task, generated instances always have
#      a full separator row, upper-pane sprites, and lower-pane vertical rays when
#      emitters are present. On the same inputs, the original horizontal solver would not
#      even parse the separator structure correctly.
# h) New solver assumptions and generator guarantees
#    - Assumption: every black component lies strictly above the separator row.
#      Guarantee: the generator only pastes sprites into rows `0..sep-1`.
#    - Assumption: triggered columns are unique.
#      Guarantee: sprites occupy disjoint column bands, so holes from different sprites
#      cannot share a column.
#    - Assumption: the lower pane is pure background before ray emission.
#      Guarantee: output-first rendering paints red only after stamping upper-pane sprites.

import random

import numpy as np

from arc_utils import components_mask, paste


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_row(g: np.ndarray, sep_color: int = 5) -> int:
    h, _ = g.shape
    rows = [r for r in range(h) if np.all(g[r, :] == sep_color)]
    assert len(rows) == 1, f"expected a unique full separator row, got {rows}"
    return int(rows[0])


def _background_color(g: np.ndarray, sep_row: int, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if r != sep_row and g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if r != sep_row and g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    assert border, "expected non-separator border pixels to infer the background"
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must be unique on the border, got {winners.tolist()}"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop black sprites onto a grey horizontal separator, then send red rays downward from closed holes.

    Input structure:
    - A full-width grey (5) separator row divides the grid into an upper sprite pane and a lower receiving pane.
    - Black connected components appear only above the separator on a uniform background color.

    Rule:
    - Move each black component straight downward until its bottom edge is flush with the row directly above the separator.
    - In that settled arrangement, find enclosed background regions inside the black shapes that touch the separator-adjacent inner row.
    - For every column occupied by one of those hole cells, paint a red ray through the lower pane down to the bottom edge.

    Key insight: The decisive feature is not every hole, but only the holes that become pressed against the separator's inner face after the downward fall; those columns act like vertical nozzles.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_row(g, sep_color=sep_color)
    bg = _background_color(g, sep_row=sep, sep_color=sep_color)

    out = g.copy()

    # Move upper-pane components downward until their bottom edge reaches the separator.
    top = out[:sep, :].copy()
    comps = _components4(top == black)
    top[top == black] = bg

    target_bottom = sep - 1
    for comp in comps:
        max_r = max(r for r, _ in comp)
        delta = target_bottom - max_r
        assert delta >= 0, "upper-pane components must stay above the separator"
        for r, c in comp:
            nr = r + delta
            assert 0 <= nr < sep, f"shifted pixel {(r, c)} left the upper pane"
            top[nr, c] = black

    out[:sep, :] = top

    # Holes are background components in the upper pane that do not touch its outer boundary.
    holes = []
    for pts in _components4(top == bg):
        if any(r == 0 or r == sep - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        holes.append(pts)

    # A hole triggers iff it touches the row just above the touching black floor.
    trigger_row = sep - 2
    trigger_cols = []
    for pts in holes:
        if not any(r == trigger_row for r, _ in pts):
            continue
        trigger_cols.extend(sorted({c for _, c in pts}))

    assert len(trigger_cols) == len(set(trigger_cols)), f"trigger columns must be unique, got {trigger_cols}"
    for c in trigger_cols:
        assert np.all(out[sep + 1 :, c] == bg), f"lower pane of column {c} must be background before ray emission"
        out[sep + 1 :, c] = red

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 10, 16
W_MIN, W_MAX = 16, 25
SEP_MIN, SEP_MAX = 5, 11
BOTTOM_H_MIN, BOTTOM_H_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 6
SPRITE_W_MAX = 8
MIN_BAND_GAP = 1

SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

MAX_RETRIES = 300


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n):
    return (3 * n + 4) // 5


def _min_cols_needed(n_emitters, n_total):
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(w, is_emitter):
    """Partition upper-pane columns into disjoint sprite bands."""
    n = len(is_emitter)
    min_widths = [3 if e else 2 for e in is_emitter]
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    width_cap = w - sum(base_gaps)
    if sum(min_widths) > width_cap:
        return None

    widths = min_widths[:]
    extra_cap = sum(SPRITE_W_MAX - x for x in widths)
    extra = random.randint(0, min(extra_cap, width_cap - sum(widths)))
    for _ in range(extra):
        choices = [i for i, x in enumerate(widths) if x < SPRITE_W_MAX]
        widths[random.choice(choices)] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(w - sum(widths) - sum(base_gaps), n + 1))]
    bands = []
    c = gaps[0]
    for i, sw in enumerate(widths):
        bands.append((c, sw))
        c += sw + gaps[i + 1]
    return bands


def _emitter_patch_horizontal(sh, max_sw):
    """Right-facing emitter template used before transposition."""
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter_horizontal(patch, sh, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK
    return patch


def _build_split_emitter_horizontal(patch, sh, left_wall, right_wall, mini_w):
    """Fill a right-facing emitter template with one main vent and an optional mini shell."""
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


def _distractor_patch_horizontal(sh, max_sw):
    """Open or solid non-emitter template used before transposition."""
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


def _emitter_patch_vertical(max_sh, sw):
    """Build a downward-facing emitter by transposing the source-style template."""
    return _emitter_patch_horizontal(sw, max_sh).T.copy()


def _distractor_patch_vertical(max_sh, sw):
    """Build a downward-facing distractor by transposing the source-style template."""
    return _distractor_patch_horizontal(sw, max_sh).T.copy()


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if BOTTOM_H_MIN <= h - s - 1 <= BOTTOM_H_MAX]
    sep = random.choice(valid_seps)
    w = random.randint(W_MIN, W_MAX)
    return h, w, sep


def _choose_emitter_flags(w):
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_cols_needed(_min_emitters_for(n), n) <= w]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = min_em
    for k in range(min_em + 1, n + 1):
        if _min_cols_needed(k, n) <= w:
            max_em = k
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _render_output(h, w, sep, bands, flags):
    """Render the solved state first, with settled sprites and lower-pane rays."""
    out = np.full((h, w), BG, dtype=int)
    out[sep, :] = SEP_COLOR

    sprites = []
    max_sh = min(SPRITE_H_MAX, sep - 1)
    if max_sh < 3:
        return None

    for (c0, sw), is_emitter in zip(bands, flags):
        patch = _emitter_patch_vertical(max_sh, sw) if is_emitter else _distractor_patch_vertical(max_sh, sw)
        sh = patch.shape[0]
        r0 = sep - sh
        assert r0 >= 1, "settled sprites must leave room for upward unsliding"
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lc in range(sw):
                if patch[-1, lc] == BLACK and patch[-2, lc] == BG:
                    assert np.all(out[sep + 1 :, c0 + lc] == BG)
                    out[sep + 1 :, c0 + lc] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Move each settled sprite upward to produce the input grid."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[sep, :] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        delta = 0 if i == flush_i else random.randint(1, r0)
        paste(inp, patch, r0 - delta, c0, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(w)
    bands = _sprite_bands(w, flags)
    if bands is None:
        return None

    rendered = _render_output(h, w, sep, bands, flags)
    if rendered is None:
        return None
    out, sprites = rendered
    inp = _unslide_sprites(h, w, sep, sprites)

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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

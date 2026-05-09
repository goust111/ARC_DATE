# Variant of 6e453dd6
# Dimension: D1 — 精灵平移路径方向 (path)
# Change: gravity（重力吸引）——黑色精灵沿水平方向向右平移，直到右边缘紧贴灰色分隔线（sep-1列）... → 垂直方向平移（向下重力）
#
# Design analysis
# ===============
# a) Concretization + behavior trace
#    - The original task has a full-height grey separator column, black sprites on the left,
#      horizontal sliding, and red rays painted across the right side by row.
#    - This variant rotates the causal axis:
#      * the separator becomes a full-width grey row,
#      * all sprites start above it,
#      * each black component drops straight downward until its bottom edge is flush with
#        the row just above the separator,
#      * enclosed background holes that end up touching that inner bottom face trigger red
#        vertical rays into the lower pane.
#    - Small trace:
#      * Original: a 3x3 ring like
#          000
#          060
#          000
#        shifts right until its right wall is at sep-1. Its hole touches sep-2, so the output
#        paints the entire matching row red on the right.
#      * Variant: the same ring is effectively transposed into a bottom-facing shell. After
#        dropping, its hole touches row sep-2, so the output paints the entire matching column
#        red below the separator.
#      * Difference: the moved coordinate is row instead of column, and the emitted structure
#        is a vertical column instead of a horizontal row.
# b) Complexity calibration
#    - A pure row/column swap would make the grid too narrow for 2-5 side-by-side sprites if
#      we kept the old 10-16 width budget. To keep the multi-object ambiguity level comparable,
#      the generator swaps the aspect bias too: height becomes 10-16 while width becomes 16-25.
#    - Sprites are still 4-connected, still include emitter and distractor families, and still
#      require >=60% emitters. This keeps the task visually rich without introducing a new rule.
# c) Implicit-assumption audit of the original main()
#    - `_find_separator_col` assumed one full separator column; now it must detect one full row.
#    - `_background_color(..., sep_col)` excluded one separator column from border counting;
#      now it excludes one separator row.
#    - `left = out[:, :sep]` assumed the active pane is left of the separator; now the active
#      pane is `top = out[:sep, :]`.
#    - `delta = target_right - max_c` assumed horizontal motion; now `delta = target_bottom - max_r`.
#    - Hole filtering originally rejected bg components touching subgrid boundary rows 0/h-1 and
#      columns 0/sep-1. In the variant the relevant boundary is rows 0/sep-1 and columns 0/w-1.
#    - Trigger collection used the inner column `sep-2` and emitted by row. The variant uses the
#      inner row `sep-2` and emits by column.
#    - The old row-paint assert was harmless because the right pane was guaranteed blank. The new
#      column-paint assert is only safe because the generator guarantees the lower pane starts as bg.
# d) Placement legality
#    - The generator places sprites in disjoint column bands separated by at least one full bg
#      column, so vertical unsliding cannot cause overlaps.
#    - Output sprites are rendered flush to the separator but with at least one spare row above
#      (`sep - sprite_h >= 1`), ensuring almost every sprite can visibly move upward in the input.
#    - Bottom pane height is kept in 2-4 rows, mirroring the original narrow "receiving" side and
#      ensuring rays have a compact, legible destination region.
# e) Visual observability
#    - Human-solvers can see the changed axis directly:
#      * separator is horizontal rather than vertical,
#      * sprites stack in the upper pane rather than the left pane,
#      * movement is downward rather than rightward,
#      * red output marks are vertical columns rather than horizontal streaks.
#    - Emitter vs distractor structure is still visible from shape, not from hidden ids or ordering.
# f) Visual cleanliness + color inference
#    - Only four colors are used, matching the source task. Red appears only in the output lower
#      pane, so there is no confusing overdraw.
#    - Because bands are separated and the lower pane is otherwise empty, each red column reads as
#      a clean consequence of one hole touching the separator-adjacent inner row.
# g) New main() assumptions and generator guarantees
#    - Assumption: every black component lies strictly above the separator. Guaranteed by generator.
#    - Assumption: the separator row is unique and full width. Guaranteed by generator.
#    - Assumption: triggered columns are unique. Guaranteed because sprite column bands are disjoint.
#    - Assumption: the lower pane is blank before ray emission. Guaranteed by output-first rendering.

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
    Drop black shapes toward a grey horizontal separator, then fire red rays downward from enclosed holes.

    Input structure:
    - A horizontal grey (5) separator row splits the grid into an upper sprite pane and a lower empty pane.
    - Black shapes sit above the separator on a single background color (bg).

    Rule:
    - Translate each black connected component straight downward until its bottom edge is flush with the separator-adjacent row.
    - After dropping, locate enclosed bg regions inside the upper-pane shapes that touch the row just above the separator.
    - For each column occupied by such a hole, paint a red ray through the entire lower pane.

    Key insight: Only holes that end up pressed against the separator's inner face can "vent" into the lower pane, so the separator-adjacent hole columns determine the output rays.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_row(g, sep_color=sep_color)
    bg = _background_color(g, sep_row=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components above the separator so that their bottom edge touches it.
    top = out[:sep, :].copy()
    top_mask = top == black
    comps = _components4(top_mask)

    # Clear old black pixels (leave other colors intact).
    top[top == black] = bg

    target_bottom = sep - 1  # last row in upper pane, adjacent to separator
    for comp in comps:
        max_r = max(r for r, _ in comp)
        delta = target_bottom - max_r
        assert delta >= 0, "upper-pane components must stay above the separator"
        for r, c in comp:
            nr = r + delta
            assert 0 <= nr < sep, f"shifted pixel {(r, c)} left the pane"
            top[nr, c] = black

    out[:sep, :] = top

    # Step 2: find enclosed background regions (holes) in the upper pane.
    # A hole is a bg component not touching the upper subgrid boundary.
    holes = []
    for pts in _components4(top == bg):
        if any(r == 0 or r == sep - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        holes.append(pts)

    # Step 3: if a hole touches row sep-2 (one cell above the black border that touches the
    # separator), emit a red ray in the lower pane for each column containing hole pixels.
    trigger_row = sep - 2
    trigger_cols = []
    for pts in holes:
        if not any(r == trigger_row for r, _ in pts):
            continue
        trigger_cols.extend(sorted(set(c for _, c in pts)))
    assert len(trigger_cols) == len(set(trigger_cols)), f"trigger columns must be unique, got {trigger_cols}"
    for c in trigger_cols:
        assert np.all(out[sep + 1 :, c] == bg), f"lower pane of column {c} must be background before ray emission"
        out[sep + 1 :, c] = red

    return out


# --------------- generator ---------------

# Colors
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# Grid dimensions
H_MIN, H_MAX = 10, 16
W_MIN, W_MAX = 16, 25
SEP_MIN, SEP_MAX = 5, 11
BOTTOM_H_MIN, BOTTOM_H_MAX = 2, 4

# Sprite constraints
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 6
SPRITE_W_MAX = 8
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


def _min_cols_needed(n_emitters, n_total):
    """Minimum grid columns to fit *n_total* sprites with *n_emitters* emitters."""
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(w, is_emitter):
    """Partition grid columns into disjoint vertical bands for each sprite."""
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
        i = random.choice(choices)
        widths[i] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(w - sum(widths) - sum(base_gaps), n + 1))]
    bands = []
    c = gaps[0]
    for i, sw in enumerate(widths):
        bands.append((c, sw))
        c += sw + gaps[i + 1]
    return bands


def _emitter_patch_horizontal(sh, max_sw):
    """Original-style right-facing emitter patch used as a template before transposition."""
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter_horizontal(patch, sh, sw, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK
    return patch


def _build_split_emitter_horizontal(patch, sh, sw, left_wall, right_wall, mini_w):
    """Fill *patch* in-place with a split emitter template before transposition."""
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
    """Original-style non-emitter template before transposition."""
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
    """Build a downward-facing emitter by transposing the original right-facing template."""
    return _emitter_patch_horizontal(sw, max_sh).T.copy()


def _distractor_patch_vertical(max_sh, sw):
    """Build a non-emitter for the vertical variant by transposing the original template."""
    return _distractor_patch_horizontal(sw, max_sh).T.copy()


def _choose_grid_dims():
    """Pick grid height, width, and separator row satisfying all constraints."""
    h = random.randint(H_MIN, H_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if BOTTOM_H_MIN <= h - s - 1 <= BOTTOM_H_MAX]
    sep = random.choice(valid_seps)
    w = random.randint(W_MIN, W_MAX)
    return h, w, sep


def _choose_emitter_flags(w):
    """Decide sprite count and which sprites are emitters. Returns flags list."""
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
    """Stamp sprites flush against the separator and paint red rays for emitters."""
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
        assert r0 >= 1, "output sprites must leave room for upward unsliding"
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lc in range(sw):
                if patch[-1, lc] == BLACK and patch[-2, lc] == BG:
                    assert np.all(out[sep + 1 :, c0 + lc] == BG)
                    out[sep + 1 :, c0 + lc] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each sprite upward by a random amount to produce the input grid."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[sep, :] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        assert r0 >= 1, "rendered sprites must have room to move upward"
        delta = 0 if i == flush_i else random.randint(1, r0)
        paste(inp, patch, r0 - delta, c0, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, verify with oracle."""
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D1 input")

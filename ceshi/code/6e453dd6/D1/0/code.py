# Variant of 6e453dd6
# Dimension: D1 — 精灵平移路径方向 (path)
# Change: gravity（重力吸引）——黑色精灵沿水平方向向右平移，直到右边缘紧贴灰色分隔线（sep-1列）... → 向左平移（反向重力）

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D1 mirror variant
# -----------------------------------------
# source_task: None, so the baseline is the original 6e453dd6 solver/generator only.
#
# a) Concrete behavior and trace
# - Original: sprites live on the left, slide right until their right wall touches the
#   separator, then holes touching the inner-right side emit rays into the right pane.
# - Variant: everything is mirrored horizontally. Sprites now live on the right, slide
#   left until their left wall touches the separator, and holes touching the inner-left
#   side emit rays into the left pane.
# - Small trace:
#   original row fragment: [shape ..][sep][bg bg bg] -> slide right -> [.. shape][sep][red red red]
#   variant row fragment:  [bg bg bg][sep][.. shape] -> slide left  -> [red red red][sep][shape ..]
#   The triggering hole is still the hole adjacent to the separator-facing wall, but the
#   relevant inner column changes from sep-2 to sep+2.
#
# b) Complexity calibration
# - The logical rule stays the same object-count / hole-count class: >=60% of sprites are
#   emitters, distractors remain C/L/solid variants, and split emitters are preserved.
# - A pure solver-only mirror would be invalid because the original generator reserves only
#   2-4 columns on the right; that would collapse the sprite vocabulary. The pane widths are
#   therefore mirrored too: left becomes the narrow ray zone, right becomes the wide sprite
#   workspace. This is the minimum compatibility change needed to keep D1 observable.
#
# c) Old-assumption audit
# - `_find_separator_col` and `_background_color` remain valid.
# - In `main`, every place that assumed "active pane is left" must flip:
#   `out[:, :sep]` -> `out[:, sep+1:]`,
#   `target_right = sep-1` -> `target_left = 0` in right-pane coordinates,
#   hole boundary test uses pane columns `0` and `right_w-1`,
#   trigger column changes from pane-local `sep-2` / global `sep-2` to pane-local `1`
#   / global `sep+2`,
#   rays fill `:sep` instead of `sep+1:`.
# - The original generator also assumed `max_sw <= sep-1` so every sprite could be unslid
#   left. The mirror analogue is `max_sw <= right_w-1` so every sprite can be unslid right.
#
# d) Placement legality
# - Row bands remain disjoint, so sprites never overlap even if their horizontal shifts vary.
# - Output sprites are stamped flush at `sep+1`; generator restricts width to at most
#   `right_w-1`, guaranteeing in-bounds placement and at least one column of possible
#   rightward unslide.
# - Left pane stays background-only in the input, which matches solver expectations before
#   red rays are painted.
#
# e) Visual observability
# - The change is visible in every example: black sprites start on the right and end near the
#   separator; red rays appear on the left. This cannot be confused with the original task's
#   direction because both moving objects and beam destination are mirrored.
#
# f) Visual cleanliness
# - Separator still isolates motion and rays into different panes.
# - Only one overlay happens on the left pane, and only after motion, so the output remains
#   easy to parse: red marks beam rows, black marks settled sprite shapes.
#
# g) Color inference
# - No new color semantics are introduced. Grey remains the separator, black remains the
#   moving sprite, magenta remains background, red remains the beam. The variant is inferred
#   entirely from spatial direction, not hidden color-id logic.
#
# h) New solver assumptions and generator guarantees
# - `main()` now assumes all black components are on the right of the separator.
# - It assumes the left pane is pure background before beam emission.
# - It assumes hole-trigger rows are unique because row bands are vertically disjoint.
# - `generate_input()` enforces all three: sprites are built only in the right pane, the left
#   pane is untouched in inputs, and sprite bands never overlap in rows.


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
    Slide black shapes leftward toward a grey separator, then fire red rays from mirrored holes.

    Input structure:
    - A full-height grey (5) separator column splits a narrow left pane from a wider right pane.
    - Black shapes occupy only the right pane on a uniform background color (bg).

    Rule:
    - Translate each black connected component leftward until its left edge is flush with the separator.
    - After sliding, find enclosed bg regions (holes) inside those black shapes that touch the column just right of the separator-facing wall.
    - For every row covered by such a hole, paint a red ray across the entire left pane.

    Key insight: The task is the horizontal mirror of the original one: only holes pressed against the separator-facing side become nozzles, and they now project left instead of right.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the right so that their left edge touches the separator.
    right = out[:, sep + 1 :].copy()
    right_mask = right == black
    comps = _components4(right_mask)

    # Clear old black pixels (leave other colors intact).
    right[right == black] = bg

    for comp in comps:
        min_c = min(c for _, c in comp)
        delta = min_c
        assert delta >= 0, "right-side components must stay right of the separator"
        for r, c in comp:
            nc = c - delta
            assert 0 <= nc < right.shape[1], f"shifted pixel {(r, c)} left the pane"
            right[r, nc] = black

    out[:, sep + 1 :] = right

    # Step 2: find enclosed background regions (holes) on the right.
    # A hole is a bg component not touching the right subgrid boundary.
    holes = []
    pane_w = w - sep - 1
    for pts in _components4(right == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == pane_w - 1 for r, c in pts):
            continue
        holes.append(pts)

    # Step 3: if a hole touches pane-local column 1 (global sep+2), emit a red ray on the left.
    trigger_rows = []
    for pts in holes:
        if not any(c == 1 for _, c in pts):
            continue
        trigger_rows.extend(sorted(set(r for r, _ in pts)))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    for r in trigger_rows:
        assert np.all(out[r, :sep] == bg), f"left half of row {r} must be background before ray emission"
        out[r, :sep] = red

    return out


# --------------- generator (D1 mirror) ---------------

# Colors
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# Grid dimensions
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
LEFT_W_MIN, LEFT_W_MAX = 2, 4
RIGHT_W_MIN, RIGHT_W_MAX = 5, 11

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
        i = random.choice(choices)
        heights[i] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _mirror_patch(patch: np.ndarray) -> np.ndarray:
    """Horizontal mirror used to convert the original right-facing sprite logic into left-facing logic."""
    return patch[:, ::-1].copy()


def _emitter_patch_core(sh, max_sw):
    """Original orientation: closed shell with an interior hole at the right edge."""
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
    """Fill *patch* in-place with a split emitter in the original orientation."""
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
    """Mirror the original emitter so the triggering hole sits at the left edge."""
    return _mirror_patch(_emitter_patch_core(sh, max_sw))


def _distractor_patch_core(sh, max_sw):
    """Original orientation distractor: C-shape, L-shape, or solid block."""
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


def _distractor_patch(sh, max_sw):
    """Mirror distractors so their separator-facing side matches the leftward-slide regime."""
    return _mirror_patch(_distractor_patch_core(sh, max_sw))


def _choose_grid_dims():
    """Pick grid height, width, and separator column for the mirrored layout."""
    h = random.randint(H_MIN, H_MAX)
    left_w = random.randint(LEFT_W_MIN, LEFT_W_MAX)
    valid_right_ws = [
        right_w
        for right_w in range(RIGHT_W_MIN, RIGHT_W_MAX + 1)
        if W_MIN <= left_w + 1 + right_w <= W_MAX
    ]
    right_w = random.choice(valid_right_ws)
    sep = left_w
    w = left_w + 1 + right_w
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
    """Stamp sprites flush against the separator and paint leftward red rays for emitters."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    right_w = w - sep - 1
    max_sw = min(SPRITE_W_MAX, right_w - 1)
    assert max_sw >= 3, "mirrored right pane must be wide enough for emitters"

    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep + 1
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lr in range(sh):
                if patch[lr, 0] == BLACK and patch[lr, 1] == BG:
                    out[r0 + lr, :sep] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each sprite rightward by a random amount to produce the input grid."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        sw = patch.shape[1]
        max_shift = w - (c0 + sw)
        if max_shift < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, max_shift)
        paste(inp, patch, r0, c0 + delta, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, bands, flags)
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D1 input")

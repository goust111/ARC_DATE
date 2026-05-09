# Variant of 6e453dd6
# Dimension: D3 — 射线轨迹涂色模式 (trajectory)
# Change: 连续涂红色——射线经过的每个右侧格子统一涂为红色(2)... → 周期性颜色模式

import numpy as np

from arc_utils import components_mask, paste
import random


# Design analysis
# a) Concrete behavior and trace:
#    The left half keeps the original mechanic: black sprites slide right until
#    they are flush with the grey separator, then enclosed bg holes touching the
#    separator's inner face become emitters. The only changed stage is the ray
#    painting on the right.
#    Original: every triggered row became solid red from sep+1 to the right edge.
#    Variant: the right half is painted by a repeating cycle chosen from the
#    visible width of the triggering hole after sliding:
#      - 1-cell-wide hole -> [red, bg, red, bg, ...]
#      - 2-cell-wide hole -> [red, blue, red, blue, ...]
#    Example with right width 4:
#      old  : 2222
#      new w=1 hole: 2626 (red/background dashed)
#      new w=2 hole: 2121 (red/blue alternation)
#    The changed processing point is therefore "emit the ray", not movement or
#    hole detection.
# b) Complexity calibration:
#    Only two periodic branches are used. That is enough to make the D3 change
#    visible while keeping the rule learnable from a few examples. The generator
#    deliberately includes both widths among emitters whenever possible so the
#    mapping from hole width to ray pattern has repeated support.
# c) Old assumptions audited:
#    - The original `out[r, sep + 1:] = red` assumed one fixed ray color.
#      That is no longer valid; row painting must happen cell-by-cell by cycle.
#    - The original solver only needed trigger rows. The variant also needs hole
#      width, so each triggering hole must carry row span + width metadata.
#    - The original assertion that the right half is all bg before emission
#      remains valid because trigger rows are still unique and the input right
#      half is empty.
# d) Placement legality:
#    Sprites still occupy disjoint row bands, so shifting them flush cannot make
#    components overlap. Right-side rays are row-local, so different emitters do
#    not collide as long as trigger rows are unique; the generator preserves the
#    original no-row-sharing property of enclosed holes.
# e) Visual observability:
#    The generator fixes right-side width to 4 cells so the periodic pattern is
#    always visible. It also ensures at least one 1-wide and one 2-wide emitter
#    in each generated sample, making both trajectory branches observable within
#    a small set of examples.
# f) Visual cleanliness:
#    Only the right half changes, and each emitted row contains a simple stripe.
#    No extra markers are added, so the separator, moved sprites, and trajectory
#    stay easy to distinguish.
# g) Color inference:
#    The variant adds only one new output color, blue(1). Its use is structured:
#    blue appears only on every second ray cell of wide-hole emitters, while
#    narrow-hole emitters keep the alternating gap pattern. This keeps the
#    geometry-to-color mapping recoverable.
# h) New solver assumptions and generator guarantees:
#    The solver assumes triggering holes have widths 1 or 2. The generator
#    enforces exactly those widths for emitters and keeps distractors as
#    non-enclosed shapes, so every generated puzzle satisfies that assumption.


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


def _hole_width(pts) -> int:
    cols = [c for _, c in pts]
    return int(max(cols) - min(cols) + 1)


def _ray_cycle(hole_w: int, bg: int, red: int = 2, blue: int = 1) -> list[int]:
    assert hole_w in (1, 2), f"expected triggering hole width 1 or 2, got {hole_w}"
    if hole_w == 1:
        return [red, bg]
    return [red, blue]


def _paint_periodic_ray(out: np.ndarray, row: int, start_col: int, cycle: list[int]) -> None:
    for step, c in enumerate(range(start_col, out.shape[1])):
        out[row, c] = int(cycle[step % len(cycle)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites against a grey separator, then project periodic rays from sealed holes.

    Input structure:
    - A single full-height grey (5) column splits the grid into a left sprite area and an empty right area.
    - Black shapes lie on the left over one background color, and the right half starts entirely as background.

    Rule:
    - Shift every black connected component rightward until its right edge sits against the separator.
    - Detect enclosed background holes created inside the shifted sprites; only holes touching the inner face next to the separator can emit.
    - For each emitted row, paint the right side with a repeating cycle determined by that hole's width: 1-wide holes make red/background dashes, 2-wide holes make red/blue alternation.

    Key insight: The separator-adjacent hole still decides which rows fire, but the hole's visible width now also encodes how the ray repeats across the right half.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2
    blue = 1

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

    # Step 3: holes touching sep-2 emit periodic rays whose cycle depends on hole width.
    trigger_col = sep - 2
    trigger_specs = []
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        rows = sorted(set(r for r, _ in pts))
        hole_w = _hole_width(pts)
        trigger_specs.append((rows, hole_w))

    trigger_rows = [r for rows, _ in trigger_specs for r in rows]
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"

    for rows, hole_w in trigger_specs:
        cycle = _ray_cycle(hole_w, bg=bg, red=red, blue=blue)
        for r in rows:
            assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
            _paint_periodic_ray(out, r, sep + 1, cycle)

    return out


# --------------- generator (att4) ---------------

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
RIGHT_W_MIN, RIGHT_W_MAX = 4, 4

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
        i = random.choice(choices)
        heights[i] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _assign_emitter_hole_widths(flags):
    hole_widths = [0] * len(flags)
    emitter_ids = [i for i, is_emitter in enumerate(flags) if is_emitter]
    random.shuffle(emitter_ids)
    base = [1, 2]
    for j, idx in enumerate(emitter_ids):
        hole_widths[idx] = base[j % len(base)]
    return hole_widths


def _emitter_patch(sh, max_sw, hole_w):
    """Build an emitter sprite patch with a 1- or 2-wide triggering hole."""
    assert hole_w in (1, 2)
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
    """Decide sprite count and which sprites are emitters. Returns flags list or None."""
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


def _trigger_rows_from_patch(patch):
    return [r for r in range(patch.shape[0]) if patch[r, -1] == BLACK and patch[r, -2] == BG]


def _render_output(h, w, sep, bands, flags, hole_widths):
    """Stamp sprites flush against the separator and paint periodic rays for emitters."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for i, ((r0, sh), is_emitter) in enumerate(zip(bands, flags)):
        if is_emitter:
            hole_w = hole_widths[i]
            patch = _emitter_patch(sh, max_sw, hole_w=hole_w)
        else:
            hole_w = 0
            patch = _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            cycle = _ray_cycle(hole_w, bg=BG, red=RED, blue=BLUE)
            for lr in _trigger_rows_from_patch(patch):
                _paint_periodic_ray(out, r0 + lr, sep + 1, cycle)
        sprites.append((patch, r0, c0, hole_w))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each sprite leftward by a random amount to produce the input grid."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0, _hole_w) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    hole_widths = _assign_emitter_hole_widths(flags)
    out, sprites = _render_output(h, w, sep, bands, flags, hole_widths)
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
    raise RuntimeError("failed to generate a valid 6e453dd6 input")

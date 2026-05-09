# Combo variant of 6e453dd6 (index 10)
# combo_spec: D8/0+D10/3
# Name: 先射线后平移且保留原位

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# spec_variant:
#   concretization:
#     - The board keeps the original 6e453dd6 layout: a wide left pane, a full-height grey
#       separator column, and a narrow right pane that starts as plain background.
#     - Every black sprite still ends up flush against the separator, but this is now a copy
#       operation rather than an erase-and-move operation: the source sprite remains visible.
#     - Red rays are decided before that copy. Only holes that already touch column `sep-2`
#       in the input can emit.
#   behavior_trace:
#     - Example: two identical hollow shells occupy different row bands. Sprite A is already
#       flush in the input; sprite B is shifted two columns left.
#     - Original task:
#       1. A and B both slide flush.
#       2. Both shells now have holes touching `sep-2`.
#       3. Both row sets emit red rays, and the old positions disappear.
#     - This combo:
#       1. Inspect the input as-is. A already has a hole on `sep-2`, so A emits now.
#          B does not, so B emits nothing.
#       2. Copy both sprites to the flush position while keeping their source pixels.
#       3. Output shows source A/B plus flush copies of A/B, but only A's rows are red.
#     - Visible difference:
#       the left pane shows duplication instead of relocation, and the right pane distinguishes
#       "already aligned emitters" from "emitters that would align only after the later copy".
#   complexity_calibration:
#     - Keep the original color set, sprite families, row-band layout, separator placement,
#       emitter fraction, and narrow right pane.
#     - To avoid degeneracy, generation forces at least one emitter to be flush in the input
#       and at least one different emitter to be shifted left. That makes the D10 order change
#       visible in the same sample.
#     - It also requires at least one nontrivial left shift, usually >=2 when possible, so the
#       D8 source-retention change is visually obvious instead of collapsing into near-overlap.
#   implicit_assumption_audit:
#     - Component extraction must happen on the input left pane before adding flush copies;
#       otherwise source+copy unions could merge shapes and alter per-component shifts.
#     - Hole detection must also happen on the input left pane. Running it on the displayed
#       union would create false emitters from cavities formed only after source retention.
#     - `trigger_col = sep - 2` remains correct, but now it refers to the original input
#       geometry rather than the copied geometry.
#     - The right pane must still be pure background before ray painting; the generator keeps
#       it empty so the solver's assertions remain meaningful.
#   placement_legality:
#     - Sprites occupy disjoint vertical bands, so retaining the source cannot cause different
#       sprites to overlap even after flush copies are added.
#     - Each input shift delta lies in `[0, c0]`, where `c0` is the flush left column of the
#       sprite. Therefore the input stays in bounds and the copied version also stays left of
#       the separator.
#     - Because row bands are disjoint, different emitting holes cannot share rows; this keeps
#       `trigger_rows` unique and preserves the solver assertion.
#   visual_observability:
#     - A solver can infer the rule from 2-4 examples because each sample contains both:
#       one emitter that is already flush and therefore fires, and another emitter with the
#       same hole logic that only becomes flush later and therefore does not fire.
#     - The duplication clue is local and visible: black pixels remain at the source while
#       a second copy appears beside the separator.
#     - No hidden ordering, color-id arithmetic, or non-visual metadata is used.
#   cleanliness:
#     - Only two left-pane black layers ever appear: source sprites and their flush copies.
#     - Rays stay confined to the right pane, separated by the grey column, so the output
#       remains easy to parse even when a sprite overlaps its own copy by one column.
#   color_inference:
#     - The role mapping stays simple: black = sprite, grey = separator, red = ray,
#       magenta = background. The fusion changes timing and duplication, not the palette.
#   solver_generator_contract:
#     - `main()` assumes a unique full separator column, an empty right pane in the input,
#       and left-side sprites whose rows live in disjoint bands. `generate_input()` enforces
#       exactly those conditions.


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


def _find_trigger_rows(left: np.ndarray, *, h: int, sep: int, bg: int) -> list[int]:
    """Find rows of enclosed bg holes that already touch the separator-side column."""
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_rows = []
    trigger_col = sep - 2
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted({r for r, _ in pts}))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return trigger_rows


def _overlay_flush_copies(left: np.ndarray, *, black: int, sep: int) -> np.ndarray:
    """Return the visible left pane: original sprites plus flush copies."""
    display_left = left.copy()
    target_right = sep - 1
    for comp in _components4(left == black):
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            display_left[r, nc] = black
    return display_left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire rays from holes that are already aligned, then add a flush copy of every black sprite.

    Input structure:
    - A single full-height grey separator column divides a left sprite pane from a narrow right pane.
    - Black connected sprites appear only on the left over one background color, and the right pane starts empty.

    Rule:
    - First inspect the left pane exactly as given and locate enclosed background holes.
    - Any such hole that already touches the column immediately left of the separator emits a red ray across the right pane on each of its rows.
    - After fixing those rays, copy every black component rightward until its outer edge is flush with the separator, while keeping the original component in place.

    Key insight: The ray decision uses the original geometry, but the displayed left pane shows duplication rather than movement; later flush copies do not get a second chance to emit.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = g[:, :sep].copy()

    trigger_rows = _find_trigger_rows(left, h=h, sep=sep, bg=bg)
    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    out[:, :sep] = _overlay_flush_copies(left, black=black, sep=sep)
    return out


# --------------- generator ---------------

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

# ── Probabilities ──
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
DISTRACTOR_FLUSH_PROB = 0.35

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
    """Rows whose hole touches the inner face of the patch's right wall."""
    sh, sw = patch.shape
    assert sw >= 2
    rows = [r for r in range(sh) if patch[r, sw - 1] == BLACK and patch[r, sw - 2] == BG]
    assert rows, "emitter patch must expose at least one right-adjacent hole row"
    return rows


def _render_flush_state(h, w, sep, bands, flags):
    """Stamp all sprites flush against the separator and store their metadata."""
    flush_state = np.full((h, w), BG, dtype=int)
    flush_state[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(flush_state, patch, r0, c0, transparent=BG)
        sprites.append({
            "patch": patch,
            "r0": r0,
            "c0": c0,
            "is_emitter": bool(is_emitter),
            "trigger_rows": _emitter_rows(patch) if is_emitter else [],
        })
    return flush_state, sprites


def _choose_input_deltas(sprites):
    """Choose shifts so some emitters fire before copying and others only copy later."""
    emitter_ids = [i for i, sprite in enumerate(sprites) if sprite["is_emitter"]]
    assert len(emitter_ids) >= 2, "combo generator needs both a flush and a shifted emitter"

    forced_shift_candidates = [i for i in emitter_ids if sprites[i]["c0"] >= 2]
    forced_shift_i = random.choice(forced_shift_candidates or emitter_ids)
    forced_flush_i = random.choice([i for i in emitter_ids if i != forced_shift_i])

    deltas = []
    for i, sprite in enumerate(sprites):
        c0 = sprite["c0"]
        assert c0 >= 1, "flush sprite must leave at least one column to unslide"
        if i == forced_flush_i:
            delta = 0
        elif i == forced_shift_i:
            delta = random.randint(2, c0) if c0 >= 2 else 1
        elif sprite["is_emitter"]:
            delta = 0 if random.random() < 0.5 else random.randint(1, c0)
        else:
            delta = 0 if random.random() < DISTRACTOR_FLUSH_PROB else random.randint(1, c0)
        deltas.append(delta)

    assert any(sprites[i]["is_emitter"] and deltas[i] == 0 for i in range(len(sprites)))
    assert any(sprites[i]["is_emitter"] and deltas[i] > 0 for i in range(len(sprites)))
    assert max(deltas) >= 1
    return deltas


def _build_input(h, w, sep, sprites, deltas):
    """Apply the chosen left shifts to obtain the input grid."""
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    for sprite, delta in zip(sprites, deltas):
        paste(inp, sprite["patch"], sprite["r0"], sprite["c0"] - delta, transparent=BG)
    return inp


def _expected_output(inp, sep, sprites, deltas):
    """Build the combo output independently of `main()` for generator self-checking."""
    out = inp.copy()
    for sprite in sprites:
        paste(out, sprite["patch"], sprite["r0"], sprite["c0"], transparent=BG)
    for sprite, delta in zip(sprites, deltas):
        if not sprite["is_emitter"] or delta != 0:
            continue
        for lr in sprite["trigger_rows"]:
            out[sprite["r0"] + lr, sep + 1 :] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build input, derive expected output, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    _flush_state, sprites = _render_flush_state(h, w, sep, bands, flags)
    deltas = _choose_input_deltas(sprites)
    inp = _build_input(h, w, sep, sprites, deltas)
    expected = _expected_output(inp, sep, sprites, deltas)

    if max(deltas) < 2:
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

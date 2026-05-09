# Variant of 6e453dd6
# Dimension: D8 — 精灵平移后的残影处理 (transform)
# Change: erase——精灵平移后原位置被擦除为背景色，仅在新位置（紧贴分隔线）显示精灵... → 保留原始位置（不擦除）

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D8 keep_source variant
# spec_variant:
#   core_change:
#     old: slide each black sprite to the separator and erase the source position
#     new: slide each black sprite to the separator but keep the source pixels too
#   visible_effect:
#     output left pane shows the original sprite and its flush copy at the same time
#     right-side rays are still caused by the flush copy, not by accidental cavities formed
#     by the union of source+copy
#   concrete_trace:
#     example:
#       input: a 3x4 hollow sprite sits two columns left of the separator
#       original_task:
#         step1: erase the old 3x4 shell, redraw it flush against the separator
#         step2: detect the hole now touching sep-2 and emit red on matching rows
#       this_variant:
#         step1: keep the old shell, also draw the flush copy
#         step2: detect holes only in the flush copy pane
#         difference: the visual output is "copy" instead of "move", while ray logic stays unchanged
#   complexity_calibration:
#     keep sprite families, dimensions, separator placement, and ray rules unchanged
#     only strengthen generator shifts slightly so at least one duplicated sprite is usually
#     clearly observable; this avoids visually degenerate samples where keep_source looks
#     almost identical to erase
#   implicit_assumption_audit:
#     - original `left[left == black] = bg` depended on erase semantics and must be removed
#     - hole search must not run on the displayed union, because source remnants can create
#       extra enclosed bg pockets that were impossible in the original task
#     - connected components must still be extracted from the input pane before any copying,
#       otherwise overlapping source+copy pixels would merge components and change per-sprite shifts
#   placement_legality:
#     sprites stay in disjoint row bands, so retaining sources cannot make different sprites overlap
#     all copied pixels remain left of the separator because each shift is capped by `sep - 1 - max_c`
#   visual_observability:
#     generation keeps 2-5 sprites and usually forces at least one nontrivial left shift
#     so the solver examples show unmistakable duplication rather than a single flush silhouette
#   cleanliness:
#     only two black layers appear in the left pane: source sprites and their flush copies
#     rays remain isolated on the right side beyond the grey separator


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
    """Return (display_left, flush_only_left) for the keep_source variant."""
    comps = _components4(left == black)
    display_left = left.copy()
    flush_only_left = left.copy()
    flush_only_left[flush_only_left == black] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            display_left[r, nc] = black
            flush_only_left[r, nc] = black

    return display_left, flush_only_left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy black shapes rightward to a grey separator, then shoot red rays from the copied enclosures.

    Input structure:
    - A full-height grey (5) separator column splits the board into left and right panes.
    - Black shapes occupy only the left pane on a uniform background color, while the right pane starts empty.

    Rule:
    - For each black connected component, make a right-shifted copy whose right edge becomes flush with the separator.
    - Keep the original component in place; the output left pane shows the source sprite and its flush copy together.
    - Detect enclosed background holes only inside the flush copies, and for each hole row that touches the inner wall next to the separator, paint a red ray across the right pane.

    Key insight: The transformation is duplication rather than relocation; the copied sprite controls the rays, while the leftover source pixels are just a visible trace.
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
    display_left, flush_left = _build_flush_and_display_left(left, bg=bg, black=black, sep=sep)
    out[:, :sep] = display_left

    holes = []
    for pts in _components4(flush_left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

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


# --------------- generator (D8 keep_source) ---------------

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
FLUSH_PROB = 0.25

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


def _emitter_patch(sh, max_sw):
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


def _choose_emitter_flags(h):
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


def _render_flush_output(h, w, sep, bands, flags):
    """Build the flush-copy state used both for ray logic and for output overlay."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lr in range(sh):
                if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
                    out[r0 + lr, sep + 1:] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each flush sprite leftward to produce the input and record deltas."""
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


def _compose_keep_source_output(inp, flush_out, sprites, sep):
    out = inp.copy()
    for patch, r0, c0 in sprites:
        paste(out, patch, r0, c0, transparent=BG)
    out[:, sep + 1:] = flush_out[:, sep + 1:]
    return out


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    flush_out, sprites = _render_flush_output(h, w, sep, bands, flags)
    inp, deltas = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    if max(deltas) <= 0:
        return None

    expected = _compose_keep_source_output(inp, flush_out, sprites, sep)

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
    raise RuntimeError("failed to generate a valid 6e453dd6 D8 input")

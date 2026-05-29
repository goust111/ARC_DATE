# Combo variant of 6e453dd6 (index 9)
# combo_spec: D8/2+D17/0
# Name: 多色精灵彩色残影
#
# Design analysis for the combo variant
# - spec_variant:
#   - preserved:
#     - the grid still uses one full-height grey separator and one shared background color
#     - sprites still live only on the left, slide horizontally until flush with `sep-1`,
#       and only enclosed holes touching `sep-2` emit rays to the right edge
#     - generator still works output-first with row-band-separated sprites and >=60% emitters
#   - changed:
#     - sprites are no longer all black: each sprite uses its own foreground color
#     - the emitted ray color now equals the moved sprite color
#     - the vacated footprint is no longer erased to background; it becomes a fixed
#       partner "afterimage" color determined by the sprite color
#
# a) Concrete behavior and trace
#    Example: a blue hollow shell at cols 1..4 slides right by 2 so its flush position is 3..6.
#    Original task:
#      1. erase the old black shell to background
#      2. redraw the shell in black at cols 3..6
#      3. if the enclosed hole reaches `sep-2`, paint red rays on the matching rows
#    Combo task:
#      1. recolor the old blue shell to its partner afterimage color orange
#      2. redraw the shell in blue at cols 3..6
#      3. detect holes using only the moved blue shell, then paint blue rays on the right
#    Difference:
#      - motion is visible because the old footprint stays as a colored trail
#      - ray color is no longer fixed; it inherits the emitting shell's own color
#      - hole detection must ignore the trail, otherwise trail+sprite unions could form fake enclosures
#
# b) Complexity calibration
#    The variant uses a small fixed palette of four sprite colors, each with one fixed
#    partner trail color:
#      blue->orange, red->teal, green->maroon, yellow->black
#    This keeps the rule learnable from a few examples. The mapping is not arbitrary per
#    instance, and every sprite remains monochrome, so the solver only needs component-wise
#    color tracking rather than intra-object color segmentation.
#
# c) Old implicit assumptions that must change
#    - `left == black` is no longer a valid sprite mask; all non-bg, non-separator pixels
#      must be treated as sprite material.
#    - `left[left == black] = bg` no longer holds; vacated cells must recolor to the trail
#      color associated with that component's sprite color.
#    - `left[r, nc] = black` would destroy color identity; the moved stamp must preserve each
#      component's own color.
#    - Hole detection on the final left pane is no longer safe, because trail colors can help
#      close regions with the moved sprite. The solver therefore computes a separate moved-only
#      left pane for geometry reasoning, and only afterward builds the visible output with trails.
#    - Fixed red rays are invalid; the solver must infer the enclosing moved sprite color and use it.
#
# d) Placement legality
#    The generator keeps sprites in disjoint vertical bands with at least one full background
#    row between them. That guarantees:
#    - unslid inputs do not overlap
#    - flush outputs do not overlap
#    - ray rows from different sprites stay disjoint
#    - trail patches only interact with the corresponding moved sprite in the same band
#    Sprite count is capped at four so each sprite can have a unique foreground color from the
#    fixed paired palette without reusing colors in the same grid.
#
# e) Visual observability
#    A human comparing 2-4 examples can see three linked regularities:
#    - differently colored sprites all slide to the separator
#    - each color leaves its own repeatable partner-colored trail behind
#    - the rightward ray always matches the color of the moved sprite, not the trail
#    The rule does not depend on hidden numeric ordering of colors; color only acts as a visible label.
#
# f) Visual cleanliness
#    Trails remain on the left pane, rays remain on the right pane, and the separator stays grey.
#    The trail palette is chosen to differ from the active sprite palette, so the moved sprite and
#    its old footprint remain distinguishable even when they overlap partially.
#
# g) Color-rule inferability
#    The color mapping is intentionally small and fixed across all samples. Because every sample has
#    multiple differently colored sprites and at least one moved emitter, the observer repeatedly sees:
#      sprite color -> same-colored ray
#      sprite color -> partner-colored trail
#    That repeated structural pairing is enough to infer the combo without needing many examples.

import random

import numpy as np

from arc_utils import components_mask, paste


BG = 6
SEP_COLOR = 5
SPRITE_TO_TRAIL = {
    1: 7,
    2: 8,
    3: 9,
    4: 0,
}
SPRITE_COLORS = sorted(SPRITE_TO_TRAIL)

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 2, 4
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

MAX_RETRIES = 300


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = SEP_COLOR) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = SEP_COLOR) -> int:
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


def _sprite_mask(left: np.ndarray, bg: int, sep_color: int) -> np.ndarray:
    return (left != bg) & (left != sep_color)


def _component_color(left: np.ndarray, comp) -> int:
    colors = {int(left[r, c]) for r, c in comp}
    assert len(colors) == 1, f"sprite component must be monochrome, got {sorted(colors)}"
    color = next(iter(colors))
    assert color in SPRITE_TO_TRAIL, f"unsupported sprite color {color}"
    return color


def _hole_wall_color(left: np.ndarray, hole, bg: int) -> int:
    h, w = left.shape
    neighbors = set()
    for r, c in hole:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                val = int(left[nr, nc])
                if val != bg:
                    neighbors.add(val)
    assert len(neighbors) == 1, f"triggering hole must belong to exactly one moved sprite color, got {sorted(neighbors)}"
    return next(iter(neighbors))


def _move_components(left: np.ndarray, bg: int, sep: int):
    sprite_mask = _sprite_mask(left, bg=bg, sep_color=SEP_COLOR)
    comps = _components4(sprite_mask)
    assert comps, "expected at least one sprite on the left side"

    moved_left = np.full_like(left, bg)
    colored_comps = []
    target_right = sep - 1
    for comp in comps:
        color = _component_color(left, comp)
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        shifted = []
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            moved_left[r, nc] = color
            shifted.append((int(r), int(nc)))
        colored_comps.append((comp, shifted, color, delta))
    return moved_left, colored_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored sprites rightward to a grey separator, leave partner-colored afterimages, then fire same-colored rays.

    Input structure:
    - A single full-height grey (5) separator column splits the board into a left sprite pane and a narrow right pane.
    - The left pane contains several monochrome sprites on one background color, and different sprites use different colors from a fixed palette.

    Rule:
    - Shift every left-side sprite horizontally until its right edge is flush with the separator while preserving that sprite's own color.
    - Recolor the sprite's old footprint to a fixed partner afterimage color tied to the sprite color, then draw the moved sprite on top.
    - Using only the moved sprite layout, find enclosed background holes that touch the column just left of the separator; each such hole emits a horizontal ray across the right pane in the same color as its enclosing moved sprite.

    Key insight: Two color correspondences must be tracked per component at once: old location becomes the sprite's partner trail color, while the moved shell keeps its original color and that same original color is what the triggering hole projects to the right.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    left_in = g[:, :sep].copy()
    moved_left, colored_comps = _move_components(left_in, bg=bg, sep=sep)

    visible_left = left_in.copy()
    for comp, shifted, color, _delta in colored_comps:
        trail = SPRITE_TO_TRAIL[color]
        for r, c in comp:
            visible_left[r, c] = trail
        for r, c in shifted:
            visible_left[r, c] = color

    out = g.copy()
    out[:, :sep] = visible_left

    trigger_col = sep - 2
    row_to_color = {}
    for hole in _components4(moved_left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in hole):
            continue
        if not any(c == trigger_col for _, c in hole):
            continue
        color = _hole_wall_color(moved_left, hole, bg=bg)
        for r in sorted({int(r) for r, _ in hole}):
            if r in row_to_color:
                assert row_to_color[r] == color, f"conflicting ray colors for row {r}: {row_to_color[r]} vs {color}"
            row_to_color[r] = color

    for r, color in sorted(row_to_color.items()):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = color

    return out


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


def _emitter_patch(sh, max_sw, color):
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w, color)
    else:
        patch[0, :] = color
        patch[-1, :] = color
        patch[1:-1, left_wall] = color
        patch[1:-1, right_wall] = color
    return patch


def _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w, color):
    place_top = random.random() < 0.5
    mini_h = random.randint(3, min(4, sh - 2))
    main_h = sh - mini_h + 1
    if place_top:
        patch[0, :mini_w] = color
        for r in range(mini_h):
            patch[r, 0] = color
            patch[r, mini_w - 1] = color
        patch[mini_h - 1, :] = color
        patch[-1, :] = color
        for r in range(mini_h, sh):
            patch[r, left_wall] = color
            patch[r, right_wall] = color
    else:
        patch[0, :] = color
        for r in range(main_h - 1):
            patch[r, left_wall] = color
            patch[r, right_wall] = color
        patch[main_h - 1, :] = color
        patch[-1, :mini_w] = color
        for r in range(main_h, sh):
            patch[r, 0] = color
            patch[r, mini_w - 1] = color


def _distractor_patch(sh, max_sw, color):
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), color, dtype=int)

    min_sw = 3 if kind.startswith("c_") or sh == 2 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    if kind == "c_left":
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, -1] = color
    elif kind == "c_right":
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
    else:
        patch[:, -1] = color
        if random.random() < 0.5:
            patch[0, :] = color
        else:
            patch[-1, :] = color
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
        if _min_rows_needed(_min_emitters_for(n), n) <= h and n <= len(SPRITE_COLORS)
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


def _choose_sprite_colors(n):
    return random.sample(SPRITE_COLORS, n)


def _emitter_rows(patch, color):
    rows = []
    for lr in range(patch.shape[0]):
        if patch[lr, -1] == color and patch[lr, -2] == BG:
            rows.append(lr)
    return rows


def _trail_patch(patch, color):
    trail = SPRITE_TO_TRAIL[color]
    return np.where(patch == color, trail, BG)


def _render_flush_sprites(h, sep, bands, flags, colors):
    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter, color in zip(bands, flags, colors):
        patch = _emitter_patch(sh, max_sw, color) if is_emitter else _distractor_patch(sh, max_sw, color)
        sw = patch.shape[1]
        c0 = sep - sw
        sprites.append((patch, r0, c0, color))
    return sprites


def _unslide_sprites(h, w, sep, sprites, flags):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    movable = [i for i, (_patch, _r0, c0, _color) in enumerate(sprites) if c0 >= 1]
    emitter_movable = [i for i in movable if flags[i]]
    if not emitter_movable:
        return None, None

    flush_i = -1
    if random.random() < FLUSH_PROB:
        flush_i = random.choice(movable)

    deltas = []
    for i, (patch, r0, c0, _color) in enumerate(sprites):
        if c0 < 1:
            return None, None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
        deltas.append(delta)

    if not any(flags[i] and deltas[i] > 0 for i in range(n)):
        return None, None
    return inp, deltas


def _render_expected_output(h, w, sep, sprites, flags, deltas):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    for (patch, r0, c0, color), delta in zip(sprites, deltas):
        paste(out, _trail_patch(patch, color), r0, c0 - delta, transparent=BG)

    for (patch, r0, c0, color), is_emitter in zip(sprites, flags):
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lr in _emitter_rows(patch, color):
                assert np.all(out[r0 + lr, sep + 1 :] == BG)
                out[r0 + lr, sep + 1 :] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    colors = _choose_sprite_colors(len(flags))
    sprites = _render_flush_sprites(h, sep, bands, flags, colors)
    inp, deltas = _unslide_sprites(h, w, sep, sprites, flags)
    if inp is None:
        return None

    expected = _render_expected_output(h, w, sep, sprites, flags, deltas)
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

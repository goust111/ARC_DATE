# Combo variant of 6e453dd6 (index 34)
# combo_spec: D3/1+D26/1
# Name: 分隔线颜色决定射线颜色

import random
import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# concreteization_and_behavior_trace:
#   - The baseline geometry stays the same: black sprites live left of one full-height separator,
#     every sprite slides right until flush with the separator, and only enclosed holes whose
#     rightmost cells land at sep-2 emit horizontal rays into the right pane.
#   - D3/1 contributes the sprite-dependent part of the color rule. We use triggering-hole width
#     as the sprite attribute because it is visible in the input and preserved by sliding:
#       * width-1 trigger hole = attribute class A
#       * width-2 trigger hole = attribute class B
#   - D26/1 contributes the separator-dependent part. The separator itself is no longer fixed grey:
#       * grey separator(5) selects palette {width1->red(2), width2->blue(1)}
#       * teal separator(8) selects palette {width1->yellow(4), width2->orange(7)}
#   - Example trace:
#       * original task: both a 1-wide emitter and a 2-wide emitter would still paint red.
#       * fused task with grey separator: the same pair paints red and blue.
#       * fused task with teal separator: the same pair paints yellow and orange.
#     The separator chooses the palette; hole width chooses which entry within that palette.
# complexity_calibration:
#   - Only two separator colors and two hole-width classes are used, so the joint lookup table has
#     four outcomes total. This is enough to make the fusion visually obvious without creating a
#     high-cardinality arbitrary recoloring problem.
#   - Every generated sample contains both hole-width classes, so one sample already shows the
#     sprite-attribute modulation. Repeated calls alternate separator colors, so a short train set
#     exposes both palettes rather than risking all examples using the same separator color.
# implicit_assumption_audit:
#   - The baseline solver assumed `sep_color == 5`; that breaks under D26/1, so separator detection
#     now finds the unique full-height uniform non-background column and returns both its column and
#     its actual color.
#   - The baseline solver assumed every trigger row shared one fixed ray color; that breaks under
#     D3/1, so it now computes `row -> color` from the triggering hole width plus separator color.
#   - Hole detection still relies on enclosed background components inside the left pane after
#     sliding. This remains valid because the generator still uses closed shells for emitters.
#   - Row-color conflicts would indicate overlapping trigger bands. The original row-band separation
#     remains, so `row_to_color` should never receive incompatible colors; we keep an assertion.
# placement_legality:
#   - Sprites are still assigned to disjoint vertical bands separated by at least one background row.
#   - Emitter patches with width-2 holes require sprite width >= 4; the original `sep >= 5` bound
#     keeps that feasible.
#   - Unsliding shifts whole patches left without changing connectivity or hole topology, so the
#     trigger attribute observed after sliding is already encoded in the input sprite.
# observability_and_cleanliness:
#   - The left pane remains black-on-background, so the only new cue is separator color plus visible
#     hole width. The right pane remains clean full-row rays with at most two colors per sample.
#   - Output never stacks multiple ray colors in the same row because row bands are disjoint.
#   - The fused rule is inferable from 2-4 examples: within one example the two hole widths map to
#     two ray colors; across examples the separator color swaps the whole palette.
# generator_guardrails:
#   - Each sample contains both width-1 and width-2 emitters.
#   - Generated calls alternate separator colors across samples to avoid degenerating into a single
#     palette dataset.
#   - Output-first generation is retained and every candidate is verified by rerunning `main()`.


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _background_color(g: np.ndarray) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        border.append(int(g[0, c]))
        border.append(int(g[h - 1, c]))
    for r in range(1, h - 1):
        border.append(int(g[r, 0]))
        border.append(int(g[r, w - 1]))
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must dominate the border, got {winners.tolist()}"
    return int(winners[0])


def _find_separator(g: np.ndarray, bg: int) -> tuple[int, int]:
    _, w = g.shape
    cols = []
    for c in range(w):
        vals = np.unique(g[:, c])
        if len(vals) == 1 and int(vals[0]) != bg:
            cols.append((c, int(vals[0])))
    assert len(cols) == 1, f"expected one full-height non-background separator, got {cols}"
    return cols[0]


def _hole_width(pts) -> int:
    cols = [int(c) for _, c in pts]
    return max(cols) - min(cols) + 1


def _ray_color_from_sources(sep_color: int, hole_width: int) -> int:
    palette = SEP_COLOR_TO_RAYS[sep_color]
    if hole_width <= 1:
        return palette[1]
    return palette[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites into the separator and fire palette-coded rays from sealed holes.

    Input structure:
    - One full-height separator column splits the board into a left sprite pane and an empty right pane.
    - The separator is not always the same color; its visible color is one of the clues for ray painting.
    - Black sprites occupy only the left side on top of one shared background color.

    Rule:
    - Move every black connected component horizontally until its right edge becomes flush with the separator.
    - In that shifted layout, locate enclosed background holes whose rightmost cells sit one step left of the separator wall.
    - Such holes emit full-row rays across the right pane; the separator color chooses a two-color palette, and the triggering hole's width picks which color from that palette is used.

    Key insight: the geometry that decides which cavities are emitters is unchanged, but ray color now comes from two visible sources at once: the separator selects the palette and the hole's width selects the palette entry.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    bg = _background_color(g)
    sep, sep_color = _find_separator(g, bg)

    out = g.copy()
    left = out[:, :sep].copy()
    comps = _components4(left == BLACK)

    left[left == BLACK] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = int(c) + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[int(r), nc] = BLACK

    out[:, :sep] = left

    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_col = sep - 2
    row_to_color = {}
    for pts in holes:
        if not any(int(c) == trigger_col for _, c in pts):
            continue
        color = _ray_color_from_sources(sep_color, _hole_width(pts))
        for r in sorted({int(r) for r, _ in pts}):
            if r in row_to_color:
                assert row_to_color[r] == color, f"conflicting ray colors on row {r}"
            else:
                row_to_color[r] = color

    for r, color in sorted(row_to_color.items()):
        assert np.all(out[r, sep + 1:] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1:] = color

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

SEP_COLOR_TO_RAYS = {
    GREY: {1: RED, 2: BLUE},
    TEAL: {1: YELLOW, 2: ORANGE},
}
SEP_COLOR_SEQUENCE = [GREY, TEAL]
_GEN_CALL_COUNT = 0

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
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


def _emitter_patch(sh, max_sw, hole_w):
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter(patch, sh, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK
    return patch


def _build_split_emitter(patch, sh, left_wall, right_wall, mini_w):
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


def _choose_sprite_roles(h):
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

    assert n_emitters >= 2, "combo variant needs both triggering-hole widths"
    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)

    emitter_widths = [1, 2] + [random.choice([1, 2]) for _ in range(n_emitters - 2)]
    random.shuffle(emitter_widths)

    hole_widths = []
    idx = 0
    for is_emitter in flags:
        if is_emitter:
            hole_widths.append(emitter_widths[idx])
            idx += 1
        else:
            hole_widths.append(None)
    return flags, hole_widths


def _trigger_rows_from_patch(patch):
    rows = []
    for lr in range(patch.shape[0]):
        if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
            rows.append(lr)
    return rows


def _render_output(h, w, sep, sep_color, bands, flags, hole_widths):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = sep_color

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter, hole_w in zip(bands, flags, hole_widths):
        patch = _emitter_patch(sh, max_sw, hole_w) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            ray_color = SEP_COLOR_TO_RAYS[sep_color][hole_w]
            for lr in _trigger_rows_from_patch(patch):
                out[r0 + lr, sep + 1:] = ray_color
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sep_color, sprites):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = sep_color

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt(sep_color: int) -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    flags, hole_widths = _choose_sprite_roles(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, sep_color, bands, flags, hole_widths)
    inp = _unslide_sprites(h, w, sep, sep_color, sprites)
    if inp is None:
        return None

    right = out[:, sep + 1:]
    colors = set(int(v) for v in np.unique(right) if int(v) != BG)
    expected = set(SEP_COLOR_TO_RAYS[sep_color].values())
    if colors != expected:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def _next_separator_color() -> int:
    global _GEN_CALL_COUNT
    sep_color = SEP_COLOR_SEQUENCE[_GEN_CALL_COUNT % len(SEP_COLOR_SEQUENCE)]
    _GEN_CALL_COUNT += 1
    return sep_color


def generate_input() -> np.ndarray:
    sep_color = _next_separator_color()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(sep_color)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

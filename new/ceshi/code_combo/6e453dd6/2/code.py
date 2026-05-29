# Combo variant of 6e453dd6 (index 2)
# combo_spec: D3/0+D17/0
# Name: 多色精灵发射周期色射线

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis
# a) Concrete behavior and trace:
#    The original task has three stages: move each left sprite flush to the grey
#    separator, detect enclosed background holes that end up touching `sep-2`,
#    then paint a solid red ray on the right for every triggered row.
#    This combo changes the last stage in two coupled ways:
#    - sprites are no longer uniformly black; each component keeps its own color
#    - emitted rays are no longer solid; they repeat a dash cycle whose painted
#      cells use that same sprite color
#    Small trace after sliding:
#    - A blue emitter with a 1-wide hole touching `sep-2` fires rows like
#      `blue, bg, blue, bg`
#    - A green emitter with a 2-wide hole fires rows like
#      `green, green, bg, green`
#    The difference from the original is therefore not only "which color to
#    paint", but also "how long the repeating colored run is". Hole width and
#    sprite identity are fused into one ray rule.
# b) Complexity calibration:
#    Only two hole widths are used for emitters: 1 and 2. This keeps the periodic
#    branch count small and visible. The generator cycles these widths across
#    emitters, so examples usually contain both 1-cell dashes and 2-cell dashes.
#    Sprite colors come from a fixed seven-color palette with unique colors per
#    sprite inside one grid, which is enough to make color tracking obvious
#    without creating arbitrary mappings.
# c) Old implicit assumptions audited:
#    - `left == black` no longer defines sprites; the solver must treat every
#      non-background left pixel as sprite material.
#    - clearing and restamping can no longer use a single foreground color; each
#      component's own monochrome color must be preserved during the slide.
#    - `out[r, sep + 1:] = red` is invalid twice over: the ray color is no longer
#      fixed, and the ray is no longer solid. Row painting must be done cell by
#      cell using a cycle derived from the triggering hole.
#    - hole detection on the shifted left pane still works, but the solver now
#      also relies on two new facts: the triggering hole is enclosed by exactly
#      one sprite color, and its width is either 1 or 2. The generator enforces
#      both.
# d) Placement legality:
#    Sprites still occupy disjoint row bands separated by at least one full
#    background row. That preserves non-overlap both before and after sliding.
#    Unique sprite colors prevent visual ambiguity between different components.
#    Trigger rows from different emitters remain disjoint because each sprite
#    lives in its own row band and each triggering hole stays inside that band.
# e) Visual observability:
#    The right pane width is fixed at 4 so a periodic ray is always visible.
#    Each sample contains at least two emitters with different colors, and when
#    there are at least two emitters the generator assigns widths 1 and 2 among
#    them. A human can therefore compare outputs and read both rules directly:
#    color follows the sprite, while dash length follows the hole width.
# f) Visual cleanliness:
#    Only the right half receives the ray pattern; there are no extra markers.
#    Because row bands do not overlap, colored dashed rays do not overwrite one
#    another. Background gaps remain background-colored, which keeps the dashes
#    crisp and makes them visibly different from the original solid beams.
# g) Color-change inferability:
#    The color mapping is local and explicit: enclosing sprite color = ray color.
#    No sorting by numeric color id or hidden palette order is used. Repetition
#    across all triggered rows of one emitter provides internal evidence for the
#    mapping, and multiple emitters with different colors provide cross-example
#    support.
# h) New solver assumptions and generator guarantees:
#    - Every sprite is a single-color 4-connected component.
#    - Every triggering hole has width 1 or 2 after sliding flush.
#    - Every triggering hole is surrounded by exactly one sprite color.
#    - The right half starts as pure background, so periodic painting never needs
#      conflict resolution.


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


def _sprite_mask(left: np.ndarray, bg: int, sep_color: int) -> np.ndarray:
    return (left != bg) & (left != sep_color)


def _component_color(left: np.ndarray, comp) -> int:
    colors = {int(left[r, c]) for r, c in comp}
    assert len(colors) == 1, f"sprite component must be monochrome, got {sorted(colors)}"
    return next(iter(colors))


def _hole_width(pts) -> int:
    cols = [c for _, c in pts]
    return int(max(cols) - min(cols) + 1)


def _hole_wall_color(left: np.ndarray, hole, bg: int) -> int:
    h, w = left.shape
    neighbors = set()
    for r, c in hole:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                val = int(left[nr, nc])
                if val != bg:
                    neighbors.add(val)
    assert len(neighbors) == 1, f"triggering hole must belong to exactly one sprite color, got {sorted(neighbors)}"
    return next(iter(neighbors))


def _ray_cycle(sprite_color: int, hole_w: int, bg: int) -> list[int]:
    assert hole_w in (1, 2), f"expected triggering hole width 1 or 2, got {hole_w}"
    return [int(sprite_color)] * hole_w + [int(bg)]


def _paint_periodic_ray(out: np.ndarray, row: int, start_col: int, cycle: list[int]) -> None:
    for step, c in enumerate(range(start_col, out.shape[1])):
        out[row, c] = int(cycle[step % len(cycle)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored sprites to a grey separator, then emit same-colored dashed rays from enclosed holes.

    Input structure:
    - One full-height grey (5) column splits the grid into a left sprite pane and an empty right pane.
    - The left pane contains several monochrome sprites on one shared background, and different sprites may have different colors.

    Rule:
    - Shift each left-side connected component rightward until its right edge is flush with the separator, preserving its own color.
    - On the shifted left pane, find enclosed background holes; only holes that touch the column immediately left of the separator can emit.
    - For every row covered by such a hole, paint the right pane with a repeating dash cycle whose painted cells use the enclosing sprite color and whose period is determined by the hole width.

    Key insight: The same separator-adjacent hole still selects which rows fire, but now it also contributes the ray rhythm through its width, while the surrounding sprite contributes the ray color.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    left = out[:, :sep].copy()
    sprite_mask = _sprite_mask(left, bg=bg, sep_color=sep_color)
    comps = _components4(sprite_mask)
    assert comps, "expected at least one sprite on the left side"

    colored_comps = [(comp, _component_color(left, comp)) for comp in comps]
    left[sprite_mask] = bg

    target_right = sep - 1
    for comp, color in colored_comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = color

    out[:, :sep] = left

    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_col = sep - 2
    trigger_specs = []
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        rows = sorted(set(int(r) for r, _ in pts))
        color = _hole_wall_color(left, pts, bg=bg)
        hole_w = _hole_width(pts)
        trigger_specs.append((rows, color, hole_w))

    trigger_rows = [r for rows, _, _ in trigger_specs for r in rows]
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"

    for rows, color, hole_w in trigger_specs:
        cycle = _ray_cycle(color, hole_w=hole_w, bg=bg)
        for r in rows:
            assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
            _paint_periodic_ray(out, r, sep + 1, cycle)

    return out


# --------------- generator ---------------

BG = 6
SEP_COLOR = 5
SPRITE_COLORS = [1, 2, 3, 4, 7, 8, 9]

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 4, 4

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


def _assign_emitter_hole_widths(flags):
    hole_widths = [0] * len(flags)
    emitter_ids = [i for i, is_emitter in enumerate(flags) if is_emitter]
    random.shuffle(emitter_ids)
    base = [1, 2]
    for j, idx in enumerate(emitter_ids):
        hole_widths[idx] = base[j % len(base)]
    return hole_widths


def _emitter_patch(sh, max_sw, hole_w, color):
    assert hole_w in (1, 2)
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


def _trigger_rows_from_patch(patch, color):
    return [r for r in range(patch.shape[0]) if patch[r, -1] == color and patch[r, -2] == BG]


def _render_output(h, w, sep, bands, flags, colors, hole_widths):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for i, ((r0, sh), is_emitter, color) in enumerate(zip(bands, flags, colors)):
        if is_emitter:
            hole_w = hole_widths[i]
            patch = _emitter_patch(sh, max_sw, hole_w=hole_w, color=color)
        else:
            hole_w = 0
            patch = _distractor_patch(sh, max_sw, color=color)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            cycle = _ray_cycle(color, hole_w=hole_w, bg=BG)
            for lr in _trigger_rows_from_patch(patch, color):
                assert np.all(out[r0 + lr, sep + 1 :] == BG)
                _paint_periodic_ray(out, r0 + lr, sep + 1, cycle)
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
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


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    colors = _choose_sprite_colors(len(flags))
    hole_widths = _assign_emitter_hole_widths(flags)
    out, sprites = _render_output(h, w, sep, bands, flags, colors, hole_widths)
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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

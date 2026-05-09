# Combo variant of 6e453dd6 (index 3)
# combo_spec: D3/1+D19/2
# Name: 洞内颜色编码射线颜色

import random

import numpy as np

from arc_utils import components_mask, paste


"""
Design analysis for combo D3/1 + D19/2
======================================

combo_rule_summary:
  preserved_backbone:
    - There is still one full-height grey separator column.
    - Left-side black sprites still slide right until their outer wall is flush with the separator.
    - Only enclosed cavities that touch the column just inside that wall emit horizontal rays.
    - The right half of the input remains empty background; only the output contains rays.

  fused_change:
    - Trigger cavities are no longer plain background-only holes.
    - Each trigger cavity contains exactly one visible marker color mixed with some background cells.
    - The marker color itself is copied to the emitted ray.
    - Different sprites can therefore emit different colors in the same grid.

behavior_trace:
  concrete_example:
    - Imagine two closed shells that both become flush with the separator after sliding.
    - In the original task, both shells would emit red because every trigger row was hard-coded red.
    - In the fused task, shell A may contain green inside its trigger cavity and shell B may contain
      yellow inside its trigger cavity. After sliding, both cavities still touch the trigger column,
      but A paints green rays and B paints yellow rays.
  changed_execution_points:
    - Translation stage:
      Original code moved only black pixels. That would leave colored cavity markers behind.
      The fused solver must move each sprite's whole bounding-box patch so the marker travels with
      the black shell.
    - Hole detection stage:
      Original code searched for enclosed components of `bg`. That fails once a cavity mixes bg with
      a marker color. The fused solver must instead search enclosed components of `left != BLACK`.
    - Painting stage:
      Original code always wrote red. The fused solver must read the unique non-background color
      stored inside each triggering cavity and use that per row.

complexity_calibration:
  palette_control:
    - Marker colors are limited to four ARC-safe colors: red, green, yellow, teal.
    - Each cavity contains exactly one marker color, so there is no per-hole ambiguity.
  anti-shortcut design:
    - The generator forces at least two emitters with different marker colors in every sample.
    - Those two emitters are also forced to share the same trigger-hole width, so width cannot explain
      the ray-color difference inside that sample.
    - Additional emitters may vary in width independently, which prevents a hidden width->color mapping.
  visual_support:
    - Every marked cavity keeps at least one background cell uncolored, so the viewer still sees a hole
      with an internal color cue rather than a filled solid block.

implicit_assumption_audit:
  original_line_level_assumptions_broken:
    - `left[left == black] = bg` assumed only black pixels belong to the moving object.
      Broken: marker pixels are part of the sprite's carried patch.
      Fix: copy each black component's full bounding-box patch into a fresh left pane.
    - `_components4(left == bg)` assumed holes are pure background.
      Broken: trigger cavities are mixed background + marker.
      Fix: detect enclosed components of `left != BLACK`.
    - `out[r, sep + 1:] = red` assumed one global ray color.
      Broken: color is cavity-specific.
      Fix: build a row->color map from the marker color inside each triggering cavity.
  new_assumptions_added:
    - Every emitting cavity contains exactly one non-background color.
    - Different sprites occupy disjoint row bands, so row-color conflicts should not occur.
    - Marker pixels appear only inside enclosed trigger cavities, never elsewhere on the left pane.
  generator_guarantees_for_new_assumptions:
    - `_mark_trigger_hole()` recolors exactly one trigger cavity and leaves one bg cell inside it.
    - Row-band construction keeps sprites vertically disjoint.
    - Distractors are black-only and never introduce stray marker pixels.

placement_legality:
  spatial_constraints:
    - Grid size, separator range, right-side width, sprite count, and sprite bbox limits stay within
      the original task spec.
    - Output-first construction still places flush sprites first, then derives the input by unsliding.
    - Bounding-box translation is safe because sprites live in disjoint row bands and fit strictly on
      the left of the separator.
  cavity_legality:
    - Trigger cavities must have area >= 2 so one cell can remain background while another shows the marker.
    - Width is restricted to 1 or 2 exactly as in the original emitter family, which keeps the geometry tidy.

visual_observability_and_cleanliness:
  observability:
    - The decisive cue is local: the same cavity that causes emission visibly contains the output color.
    - Because at least two same-width emitters use different marker colors, a human can rule out width-based
      explanations from a single example.
    - The right half stays uncluttered: each active row is a single solid ray color copied from one cavity.
  cleanliness:
    - Markers appear only inside black shells, not scattered in open background.
    - Row-band gaps prevent overlap between sprites and prevent different rays from competing on one row.

post-implementation validation plan:
  - Run 50 generation attempts and require >= 40 successes.
  - Measure fusion-specific evidence by checking that generated samples contain at least two distinct emitted
    colors and that at least one pair of emitters shares a hole width while using different marker colors.
"""


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEP_COLOR = 5
BG = 6
TEAL = 8
MARKER_COLORS = (RED, GREEN, YELLOW, TEAL)

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
FLUSH_PROB = 0.35

# ── Retry budget ──
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


def _bbox_from_component(comp: np.ndarray) -> tuple[int, int, int, int]:
    rows = comp[:, 0]
    cols = comp[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _enclosed_nonblack_regions(left: np.ndarray) -> list[np.ndarray]:
    h, w = left.shape
    regions = []
    for pts in _components4(left != BLACK):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        regions.append(pts)
    return regions


def _trigger_region_infos(left: np.ndarray, bg: int) -> list[tuple[np.ndarray, int]]:
    trigger_col = left.shape[1] - 2
    infos = []
    for pts in _enclosed_nonblack_regions(left):
        if not any(c == trigger_col for _, c in pts):
            continue
        colors = sorted({int(left[r, c]) for r, c in pts if int(left[r, c]) != bg})
        assert len(colors) == 1, f"trigger cavity must contain exactly one marker color, got {colors}"
        infos.append((pts, colors[0]))
    return infos


def _trigger_holes(left: np.ndarray) -> list[np.ndarray]:
    trigger_col = left.shape[1] - 2
    return [pts for pts in _enclosed_nonblack_regions(left) if any(c == trigger_col for _, c in pts)]


def _trigger_rows_to_colors(left: np.ndarray, bg: int) -> dict[int, int]:
    row_to_color: dict[int, int] = {}
    for pts, ray_color in _trigger_region_infos(left, bg):
        for r in sorted({int(r) for r, _ in pts}):
            prev = row_to_color.get(r)
            assert prev is None or prev == ray_color, f"conflicting ray colors on row {r}: {prev} vs {ray_color}"
            row_to_color[r] = ray_color
    return row_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide left-side sprites to the separator, then emit rays whose colors are read from marked cavities.

    Input structure:
    - A single grey (5) separator column splits the board into a left sprite pane and an empty right field.
    - Black sprites occupy only the left side, and some enclosed cavities inside those sprites contain one visible marker color mixed with background.

    Rule:
    - Move each left-side black component rightward until its outer wall is flush with the separator, carrying any colored cavity contents with it.
    - In the shifted arrangement, find enclosed non-black cavities; only cavities touching the column just left of the separator-side wall are emitters.
    - For each emitting cavity, read its unique non-background interior color and paint that color across the entire right side on every row covered by the cavity.

    Key insight: the cavity still determines where emission happens, but its internal color now also acts as a legend for what color to project, so the solver must preserve and inspect hole contents instead of treating holes as plain background.
    """
    g = np.asarray(grid, dtype=int)

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    left_src = g[:, :sep].copy()
    left_dst = np.full_like(left_src, bg)

    target_right = sep - 1
    for comp in _components4(left_src == BLACK):
        r0, c0, r1, c1 = _bbox_from_component(comp)
        delta = target_right - c1
        assert delta >= 0, "left-side components must stay left of the separator"
        patch = left_src[r0 : r1 + 1, c0 : c1 + 1].copy()
        paste(left_dst, patch, r0, c0 + delta)

    out = g.copy()
    out[:, :sep] = left_dst

    for r, ray_color in _trigger_rows_to_colors(left_dst, bg).items():
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = ray_color

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


def _sprite_bands(h, roles):
    n = len(roles)
    min_heights = []
    for role in roles:
        if role is None:
            min_heights.append(2)
        else:
            _, hole_w = role
            min_heights.append(4 if hole_w == 1 else 3)
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


def _mark_trigger_hole(patch: np.ndarray, ray_color: int) -> bool:
    holes = _trigger_holes(patch)
    if len(holes) != 1:
        return False

    hole = sorted((int(r), int(c)) for r, c in holes[0])
    if len(hole) < 2:
        return False

    keep_bg = hole[0]
    for r, c in hole:
        patch[r, c] = ray_color
    patch[keep_bg] = BG
    return True


def _emitter_patch(sh, max_sw, ray_color, hole_w):
    for _ in range(60):
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

        if _mark_trigger_hole(patch, ray_color):
            return patch
    raise RuntimeError("failed to build an emitter patch with a marked trigger hole")


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


def _choose_emitter_specs(n_emitters):
    assert n_emitters >= 2, "combo variant needs at least two emitters for visible color contrast"

    pair_width = random.choice([1, 2])
    base_colors = random.sample(list(MARKER_COLORS), 2)
    specs = [(base_colors[0], pair_width), (base_colors[1], pair_width)]

    remaining = n_emitters - 2
    if remaining > 0:
        distinct = 2 if remaining == 1 else random.choice([2, 3])
        palette = random.sample(list(MARKER_COLORS), distinct)
        while len(specs) < n_emitters:
            specs.append((random.choice(palette), random.choice([1, 2])))

    random.shuffle(specs)
    return specs


def _choose_sprite_roles(h):
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(_min_emitters_for(n), n) <= h]
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

    specs = _choose_emitter_specs(n_emitters)
    roles = specs + [None] * (n - n_emitters)
    random.shuffle(roles)
    return roles


def _render_output(h, w, sep, bands, roles):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), role in zip(bands, roles):
        if role is None:
            patch = _distractor_patch(sh, max_sw)
        else:
            ray_color, hole_w = role
            patch = _emitter_patch(sh, max_sw, ray_color, hole_w)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0))

    for r, ray_color in _trigger_rows_to_colors(out[:, :sep], BG).items():
        out[r, sep + 1 :] = ray_color
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


def _ray_colors_in_output(out: np.ndarray, sep: int) -> set[int]:
    return {int(v) for v in np.unique(out[:, sep + 1 :]) if int(v) != BG}


def _same_width_diff_color_exists(left: np.ndarray, bg: int) -> bool:
    groups: dict[int, set[int]] = {}
    for pts, color in _trigger_region_infos(left, bg):
        cols = [int(c) for _, c in pts]
        width = max(cols) - min(cols) + 1
        groups.setdefault(width, set()).add(color)
    return any(len(colors) >= 2 for colors in groups.values())


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    roles = _choose_sprite_roles(h)
    bands = _sprite_bands(h, roles)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, bands, roles)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    ray_colors = _ray_colors_in_output(out, sep)
    if len(ray_colors) < 2:
        return None
    if not _same_width_diff_color_exists(out[:, :sep], BG):
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

# Variant of 6e453dd6
# Dimension: D19 — 符号编码/图例系统 (symbol_instructions)
# Change: 无显式图例——规则完全从输入输出示例对中推断... → 精灵内部标记决定射线属性

import numpy as np

from arc_utils import components_mask, paste
import random


"""
Design analysis for the D19 variant
==================================

variant_spec:
  preserved_structure:
    - The grid still has a single full-height grey separator column.
    - Black sprites still live only on the left side and slide right until flush with the separator.
    - Only enclosed cavities that reach the inner face of the separator-side wall can emit rays.
    - The right side of the input stays empty background; output paints horizontal rays on the right.

  concrete_change:
    - A triggering cavity is no longer pure background.
    - Inside each triggering cavity, some but not all cavity pixels are recolored to one marker color.
    - That marker color becomes the emitted ray color for every row covered by that cavity.
    - Distractors remain black-only shapes or open pseudo-cavities, so the new information is localized
      exactly where the old trigger already lived.

  behavior_trace:
    - Original:
      A rectangular black shell with a background hole touching the inner separator wall slides right,
      then the solver finds a background-only enclosed region and paints red on the matching rows.
    - Variant:
      The same shell slides right, but its enclosed cavity contains both background and a visible color
      such as green. The solver must treat the whole mixed-color cavity as one enclosed region, detect
      that it touches the trigger column, read the unique non-background color inside it, and paint the
      ray green instead of always red.
    - Difference:
      The hole now encodes both "emit" and "which color to emit". Any solver that only looks for pure
      background holes or blindly paints red will fail.

  complexity_calibration:
    - Marker colors are restricted to a small palette: red, green, yellow, teal.
    - Each cavity uses exactly one marker color, avoiding ambiguous per-hole color mixtures.
    - When there are multiple emitters, the generator usually includes at least two distinct ray colors
      so the color-parameter rule is visible from few examples.
    - The shell geometry stays close to the original task, so D19 is the main novelty rather than a new
      shape grammar.

  implicit_assumption_review:
    - Original step 1 moved only black pixels. That relied on the old invariant that holes were pure
      background and therefore did not need to move. In the variant, marker pixels must move with their
      enclosing sprite, so the solver now shifts each black component's bounding-box patch instead.
    - Original step 2 searched connected components of `left == bg`. That breaks once cavity pixels may
      be non-background. The solver now searches enclosed components of `left != black`, so mixed-color
      cavities remain whole.
    - Original step 3 painted every triggered row red unconditionally. That only worked because ray
      color was constant. The solver now extracts the cavity's unique non-background color and uses it.
    - The new solver assumes every non-background, non-black marker lies inside exactly one sprite bbox.
      The generator guarantees this by placing markers only inside enclosed trigger cavities.

  placement_legality:
    - Sprites keep disjoint row bands, so bbox-based shifting cannot overwrite another sprite.
    - Marker pixels are stamped only inside enclosed cavities, never on the border, separator, or right
      half, so background inference and trigger geometry remain well-posed.
    - Trigger cavities are generated with area at least 2 cells so they can visibly contain both marker
      color and background.

  observability_and_cleanliness:
    - The marker sits inside the same cavity that already governed emission in the base task, making the
      causal link visually localized and easy to compare across examples.
    - Open distractors stay uncolored, so colored pixels do not appear in visually noisy places.
    - Ray colors directly repeat the cavity marker colors, giving a clean one-hop correspondence.
"""


# ── Colors ──
BLACK = 0
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


def _trigger_rows_to_colors(left: np.ndarray, bg: int) -> dict[int, int]:
    trigger_col = left.shape[1] - 2
    row_to_color: dict[int, int] = {}
    for pts in _enclosed_nonblack_regions(left):
        if not any(c == trigger_col for _, c in pts):
            continue
        colors = sorted({int(left[r, c]) for r, c in pts if int(left[r, c]) != bg})
        assert len(colors) == 1, f"trigger cavity must contain exactly one marker color, got {colors}"
        ray_color = colors[0]
        for r in sorted({int(r) for r, _ in pts}):
            prev = row_to_color.get(r)
            assert prev is None or prev == ray_color, f"conflicting ray colors on row {r}: {prev} vs {ray_color}"
            row_to_color[r] = ray_color
    return row_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide left-side sprites to the separator, then emit color-matched rays from marked cavities.

    Input structure:
    - A single grey (5) vertical separator splits the grid into a left sprite area and an empty right field.
    - Black sprites occupy the left side, and some enclosed cavities inside those sprites contain a visible marker color.

    Rule:
    - Move each left-side sprite rightward until its black outline is flush with the separator.
    - After sliding, find enclosed non-black cavities inside the black shells; only cavities that touch the column just left of the wall can emit.
    - For each emitting cavity, read its unique non-background marker color and paint that color across the entire right side on every row covered by the cavity.

    Key insight: The same cavity still acts as the nozzle, but now its interior color is also a legend that specifies the ray color, so the solver must preserve and read the cavity contents rather than treating holes as plain background.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape

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
    min_heights = [3 if role is not None else 2 for role in roles]
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
    trigger_col = patch.shape[1] - 2
    holes = [pts for pts in _enclosed_nonblack_regions(patch) if any(c == trigger_col for _, c in pts)]
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


def _emitter_patch(sh, max_sw, ray_color):
    for _ in range(60):
        hole_w_choices = [1, 2] if max_sw >= 4 else [1]
        if sh == 3 and 2 in hole_w_choices:
            hole_w_choices = [2]
        hole_w = random.choice(hole_w_choices)
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
    raise RuntimeError("failed to build an emitter patch with a mixed-color trigger hole")


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


def _choose_emitter_colors(n_emitters):
    if n_emitters == 1:
        return [random.choice(MARKER_COLORS)]

    if n_emitters == 2:
        distinct = 2
    elif n_emitters == 3:
        distinct = 2
    else:
        distinct = random.choice([2, 3])

    palette = random.sample(list(MARKER_COLORS), distinct)
    colors = []
    while len(colors) < n_emitters:
        colors.extend(palette)
    colors = colors[:n_emitters]
    random.shuffle(colors)
    return colors


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

    roles = _choose_emitter_colors(n_emitters) + [None] * (n - n_emitters)
    random.shuffle(roles)
    return roles


def _render_output(h, w, sep, bands, roles):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), role in zip(bands, roles):
        patch = _emitter_patch(sh, max_sw, role) if role is not None else _distractor_patch(sh, max_sw)
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D19 input")

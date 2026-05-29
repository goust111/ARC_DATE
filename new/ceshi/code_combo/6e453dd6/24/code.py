# Combo variant of 6e453dd6 (index 24)
# combo_spec: D3/1+D11/3
# Name: 干扰精灵发射短射线+属性编码颜色

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   baseline_that_stays:
#     - one full-height grey separator splits left sprites from an initially empty right side
#     - all black connected components slide right until they become flush with the separator
#   D3/1 contribution:
#     - ray color is no longer globally fixed
#     - use a visible nozzle attribute: the width of the background cavity touching the component's
#       inner face after sliding
#     - mapping kept low-cardinality for learnability:
#         width_1 -> red(2)
#         width_2_or_more -> blue(1)
#   D11/3 contribution:
#     - non-emitter distractors also fire, but their cavity is open rather than enclosed
#     - enclosed cavities produce long rays to the right border
#     - open cavities produce short rays of exactly 2 cells
#   fused_generator_choice:
#     - ray length still comes from enclosure: enclosed cavities are long, open cavities are short
#     - ray color comes from a second visible attribute: nozzle width
#         width_1 -> red, width_2 -> blue
#     - both emitter and distractor families can realize either width, and every sample includes an extra
#       blue-long emitter so color is not confounded with sprite type
#
# behavior_trace:
#   small_example:
#     - sprite A is a closed shell; after sliding, it has a 1-cell cavity at local col bbox_w-2
#     - sprite B is an open-top shell; after sliding, its mouth also reaches bbox_w-2
#     - original task: only A would fire, always red, across the full right pane
#     - combo task:
#         - A fires because its cavity is enclosed; width 1 makes the ray red, and enclosure makes it long
#         - B also fires because open nozzles are now active distractors; openness makes it short, while
#           width independently selects red vs blue
#   changed_stages:
#     - hole discovery becomes per-component nozzle discovery
#     - painting now depends on two orthogonal tests:
#         1. cavity width -> color
#         2. cavity enclosure -> long vs short length
#
# complexity_calibration:
#   - only two output colors are used, and they are tied to an obvious geometric cue
#   - every generated sample contains both classes, so the learner sees red-long and blue-short evidence
#     inside a single grid instead of needing many examples
#   - right-side width is forced to be at least 3 so a 2-cell short ray is visibly different from a long ray
#
# implicit_assumption_audit:
#   baseline_lines_broken:
#     - a single global `red` constant is no longer valid
#     - scanning all left-side background components globally would merge distractor mouths with exterior
#       background and lose the distinction between open and closed nozzles
#   solver_fix:
#     - slide first, then re-segment black components, then inspect the background components inside each
#       component bbox separately
#     - for each candidate cavity touching local col bbox_w-2:
#         - `touches_border == False` means enclosed -> long ray
#         - `touches_border == True` means open -> short ray
#         - cavity width decides color
#   new_assumptions:
#     - each sprite occupies its own row band, so different rays never compete for the same row
#     - each generated sprite has at most one nozzle touching bbox_w-2, avoiding ambiguous per-row demands
#     - each generated sprite has nozzle width 1 or 2, both visually legible in a narrow left pane
#
# placement_legality_and_visual_cleanliness:
#   - output-first generation is retained because the flush state is the constrained one
#   - row bands remain disjoint with at least one full background row between sprites
#   - sprite widths stay within the original left-pane budget; width-2 nozzle variants are legal because
#     sep >= 5 leaves room for a 4-column sprite
#   - short rays only occupy the first two cells to the right of the separator, leaving a clean gap between
#     blue-short and red-long behavior
#
# observability_and_non_degeneracy:
#   - the rule uses only visible structure: cavity openness and cavity width
#   - no hidden ordering, color id arithmetic, or component enumeration is involved
#   - every sample contains at least one red-long emitter, one blue-short distractor, and one blue-long
#     emitter so the D11/3 signal stays visible while the D3/1 color cue remains independently learnable


BG = 6
BLACK = 0
BLUE = 1
RED = 2
SEP_COLOR = 5

SHORT_RAY_LEN = 2

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 3, 4

N_SPRITES_MIN, N_SPRITES_MAX = 3, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

EXTRA_DISTRACTOR_PROB = 0.35
FLUSH_PROB = 0.35
MAX_RETRIES = 400


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


def _bbox_of_points(pts):
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _slide_left_components(left: np.ndarray, sep: int, bg: int) -> np.ndarray:
    slid = left.copy()
    comps = _components4(slid == BLACK)
    slid[slid == BLACK] = bg
    target_right = sep - 1
    for comp in comps:
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = int(c) + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            slid[int(r), nc] = BLACK
    return slid


def _ray_color_from_width(width: int) -> int:
    return RED if width <= 1 else BLUE


def _component_nozzles(left: np.ndarray, comp, bg: int):
    r0, c0, r1, c1 = _bbox_of_points(comp)
    patch = left[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = patch.shape
    if pw < 2:
        return []

    target_col = pw - 2
    nozzles = []
    for pts in _components4(patch == bg):
        pts = [(int(r), int(c)) for r, c in pts]
        if not any(c == target_col for _, c in pts):
            continue
        rows = sorted({r0 + r for r, _ in pts})
        cols = [c for _, c in pts]
        width = max(cols) - min(cols) + 1
        touches_border = any(r == 0 or r == ph - 1 or c == 0 or c == pw - 1 for r, c in pts)
        nozzles.append(
            {
                "rows": rows,
                "width": int(width),
                "enclosed": not touches_border,
            }
        )
    return nozzles


def _paint_ray(out: np.ndarray, row: int, sep: int, bg: int, color: int, enclosed: bool) -> None:
    end = out.shape[1] if enclosed else min(out.shape[1], sep + 1 + SHORT_RAY_LEN)
    assert end > sep + 1, "ray must paint at least one right-side cell"
    assert np.all(out[row, sep + 1 : end] == bg), f"ray path on row {row} must be empty before painting"
    out[row, sep + 1 : end] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the separator, then fire color-coded rays whose length depends on cavity closure.

    Input structure:
    - A single full grey separator column divides the grid into a left sprite field and a blank right field.
    - Several black connected components live on the left over one shared background color.

    Rule:
    - Shift each black component horizontally until its right edge becomes flush with the separator.
    - Inspect each shifted component locally: any background cavity touching the inner column just left of the
      component's right wall becomes a nozzle.
    - A 1-cell nozzle paints red and a wider nozzle paints blue; enclosed nozzles shoot all the way to the
      right edge, while open nozzles shoot only a short 2-cell ray.

    Key insight:
    - The moved sprites must be analyzed one by one, because the important distinction is not just "hole or not"
      but whether the separator-facing cavity is sealed or open; width controls the color and enclosure controls
      the ray length.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()
    left = _slide_left_components(out[:, :sep].copy(), sep=sep, bg=bg)
    out[:, :sep] = left

    comps = _components4(left == BLACK)
    row_to_spec = {}
    for comp in comps:
        nozzles = _component_nozzles(left, comp, bg=bg)
        assert len(nozzles) <= 1, f"expected at most one separator-facing nozzle per component, got {nozzles}"
        for nozzle in nozzles:
            color = _ray_color_from_width(nozzle["width"])
            for row in nozzle["rows"]:
                spec = (color, bool(nozzle["enclosed"]))
                if row in row_to_spec:
                    assert row_to_spec[row] == spec, f"conflicting ray specs on row {row}: {row_to_spec[row]} vs {spec}"
                else:
                    row_to_spec[row] = spec

    for row, (color, enclosed) in sorted(row_to_spec.items()):
        _paint_ray(out, row=row, sep=sep, bg=bg, color=color, enclosed=enclosed)

    return out


def _scatter(total: int, slots: int):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_rows_needed(n_total: int) -> int:
    return n_total * 3 + (n_total + 1) * MIN_BAND_GAP


def _sprite_bands(h: int, n: int):
    min_heights = [3] * n
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    if sum(min_heights) + sum(base_gaps) > h:
        return None

    heights = min_heights[:]
    extra_cap = sum(SPRITE_H_MAX - x for x in heights)
    extra = random.randint(0, min(extra_cap, h - sum(heights) - sum(base_gaps)))
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


def _choose_sprite_roles(h: int):
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(n) <= h]
    n = random.choice(valid_counts)

    max_distractors = max(1, int(np.floor(0.4 * n)))
    if max_distractors > 1 and random.random() < EXTRA_DISTRACTOR_PROB:
        n_distractors = random.randint(1, max_distractors)
    else:
        n_distractors = 1

    n_emitters = n - n_distractors
    assert n_emitters * 5 >= 3 * n, "emitters must remain at least 60% of sprites"
    roles = ["emitter"] * n_emitters + ["distractor"] * n_distractors
    random.shuffle(roles)
    return roles


def _choose_sprite_specs(h: int):
    roles = _choose_sprite_roles(h)
    specs = [{"role": role, "nozzle_width": 1} for role in roles]

    emitter_ids = [i for i, spec in enumerate(specs) if spec["role"] == "emitter"]
    distractor_ids = [i for i, spec in enumerate(specs) if spec["role"] == "distractor"]
    assert emitter_ids and distractor_ids, "generator needs both emitters and distractors"

    canonical_emitter = random.choice(emitter_ids)
    canonical_distractor = random.choice(distractor_ids)
    specs[canonical_emitter]["nozzle_width"] = 1
    specs[canonical_distractor]["nozzle_width"] = 2

    reusable_emitters = [idx for idx in emitter_ids if idx != canonical_emitter]
    chosen = random.choice(reusable_emitters)
    specs[chosen]["nozzle_width"] = 2

    for idx, spec in enumerate(specs):
        if idx in (canonical_emitter, canonical_distractor, chosen):
            continue
        spec["nozzle_width"] = random.choice([1, 2])
    return specs


def _emitter_patch(sh: int, max_sw: int, nozzle_width: int) -> np.ndarray:
    sw = random.randint(nozzle_width + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, sw - nozzle_width - 2] = BLACK
    patch[1:-1, sw - 1] = BLACK
    return patch


def _distractor_patch(sh: int, max_sw: int, nozzle_width: int) -> np.ndarray:
    sw = random.randint(nozzle_width + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, sw - nozzle_width - 2] = BLACK
    patch[:, -1] = BLACK
    patch[0, sw - nozzle_width - 1 : sw - 1] = BG
    return patch


def _local_nozzle_specs_from_patch(patch: np.ndarray):
    specs = []
    for nozzle in _component_nozzles(patch, [(r, c) for r, c in np.argwhere(patch == BLACK)], bg=BG):
        specs.append((sorted({int(r) for r in nozzle["rows"]}), int(nozzle["width"]), bool(nozzle["enclosed"])))
    return specs


def _render_output(h: int, w: int, sep: int, bands, specs):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    assert max_sw >= 4, "separator must leave room for width-2 nozzle variants"

    for (r0, sh), spec in zip(bands, specs):
        role = spec["role"]
        nozzle_width = spec["nozzle_width"]
        patch = (
            _emitter_patch(sh, max_sw, nozzle_width=nozzle_width)
            if role == "emitter"
            else _distractor_patch(sh, max_sw, nozzle_width=nozzle_width)
        )
        c0 = sep - patch.shape[1]
        paste(out, patch, r0, c0, transparent=BG)

        for local_rows, width, enclosed in _local_nozzle_specs_from_patch(patch):
            color = _ray_color_from_width(width)
            for lr in local_rows:
                _paint_ray(out, row=r0 + lr, sep=sep, bg=BG, color=color, enclosed=enclosed)

        sprites.append((patch, r0, c0, spec))
    return out, sprites


def _unslide_sprites(h: int, w: int, sep: int, sprites):
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(len(sprites)) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0, _spec) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _ray_signature(grid: np.ndarray, sep: int):
    right = grid[:, sep + 1 :]
    red_rows = 0
    blue_rows = 0
    for row in right:
        non_bg = np.flatnonzero(row != BG)
        if len(non_bg) == 0:
            continue
        color = int(row[non_bg[0]])
        run_len = int(non_bg[-1] - non_bg[0] + 1)
        assert np.all(row[non_bg[0] : non_bg[-1] + 1] == color), "rays must be contiguous"
        if color == RED:
            red_rows += 1
        elif color == BLUE:
            blue_rows += 1
        else:
            raise AssertionError(f"unexpected ray color {color}")
        assert run_len in (SHORT_RAY_LEN, right.shape[1]), f"unexpected ray length {run_len}"
    return red_rows, blue_rows


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    specs = _choose_sprite_specs(h)
    bands = _sprite_bands(h, len(specs))
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, bands, specs)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    red_rows, blue_rows = _ray_signature(out, sep)
    if red_rows == 0 or blue_rows == 0:
        return None

    right = out[:, sep + 1 :]
    long_colors = set()
    short_colors = set()
    for row in right:
        non_bg = np.flatnonzero(row != BG)
        if len(non_bg) == 0:
            continue
        color = int(row[non_bg[0]])
        run_len = int(non_bg[-1] - non_bg[0] + 1)
        if run_len == right.shape[1]:
            long_colors.add(color)
        elif run_len == SHORT_RAY_LEN:
            short_colors.add(color)
    if RED not in long_colors or BLUE not in short_colors:
        return None
    if BLUE not in long_colors and RED not in short_colors:
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

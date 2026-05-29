# Variant of 6e453dd6
# Dimension: D5 — 封闭洞触发条件 (src)
# Change: enclosure_detection——洞必须是完全封闭的背景色区域（4连通分量不触及左侧子网格边界），且洞的最右列位... → 每个触发洞不再按自身占据的每一行发射红线，而是只从洞的中心行发射一条红线

# Design analysis
# source_task: none
#
# a) concretization + trace
#   - The input still has one full-height grey separator and several black sprites on its left.
#   - All black sprites still slide right until their right edge reaches the separator.
#   - A triggering hole is still an enclosed background component whose rightmost column lands on sep-2.
#   - The D5 change is only the mapping from one triggering hole to right-side rays:
#     original: if hole rows are {4,5,6}, rows 4/5/6 all turn red on the right;
#     variant: the same hole turns only its center row red, here row 5.
#   - Small example after sliding:
#       hole rows = {10,11,12,13,14}, touching sep-2
#       original output: rows 10..14 on the right half become red
#       variant output: only row 12 becomes red
#     The separator position, sprite translation, and enclosed-hole test remain unchanged.
#
# b) complexity calibration
#   - If emitters often had 1-cell-tall holes, the variant would visually collapse to the original.
#   - To keep the D5 change observable, the generator makes emitter shells tall enough that their
#     triggering holes usually have height 3 or 5. This preserves the original object vocabulary
#     while making the single-center-ray rule visible in 2-3 examples.
#   - Non-emitters still include C/L/solid distractors so the task is not reduced to "every sprite emits".
#
# c) implicit-assumption audit for main()
#   - Sliding logic assumes left sprites are separate 4-connected components. This still holds.
#   - Hole detection assumes enclosed bg components do not touch the left pane boundary. This still holds.
#   - The original code extended trigger_rows with every row in a triggering hole and asserted uniqueness.
#     That depended on the old "one ray per hole row" mapping. Under the new rule we must compute one
#     center row per triggering hole instead. Duplicates are harmless, so the solver now deduplicates
#     center rows with a set rather than asserting a row-by-row list shape.
#   - The right half is still assumed to be background before emission; the generator guarantees this.
#
# d) placement legality
#   - Row bands remain disjoint with at least one blank row between sprites, so sliding cannot create overlaps.
#   - Every emitter shell fits strictly inside the left pane and reaches sep-1 only after the output-first
#     construction, matching the solver's translation model.
#   - Emitter heights are restricted to odd values 5 or 7, yielding odd hole heights 3 or 5 and a unique
#     visual center row for the emitted ray.
#
# e) visual observability
#   - The important contrast is now "tall cavity but one red line", not "tall cavity and a full red band".
#   - Multiple emitter heights are kept, so the ray is not tied to a fixed absolute row offset.
#   - Samples still repeat the same principle several times per grid, which gives enough support to infer it.
#
# f) visual cleanliness
#   - Right-side markings remain single full-width red lines, so there is no clutter from stacked colors.
#   - Distinct row bands keep emitters and distractors easy to parse.
#
# g) color inference
#   - No new color mapping is introduced. Black sprites, grey separator, magenta background, and red rays
#     keep the original role structure, so the variant is inferable from geometry alone.
#
# h) variant metric used after implementation
#   - For each generated sample, compare:
#       original_style_ray_rows = total number of triggering-hole rows
#       variant_ray_rows = number of triggering holes
#     A non-degenerate sample should usually satisfy original_style_ray_rows > variant_ray_rows.
#
# i) new solver assumptions checked against generator
#   - Each triggering hole has a well-defined center row. Guaranteed by odd emitter heights 5 or 7.
#   - Different triggering holes do not share a center row. Guaranteed by disjoint sprite row bands.
#   - The right half is blank before painting. Guaranteed by output-first construction and unsliding.

import random

import numpy as np

from arc_utils import components_mask, paste


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


def _center_row_from_points(pts) -> int:
    rows = sorted({int(r) for r, _ in pts})
    assert rows, "hole must contain at least one row"
    return rows[(len(rows) - 1) // 2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to the grey separator, then fire one red ray from each triggering hole.

    Input structure:
    - A vertical grey (5) separator column splits the board into a left sprite pane and a right empty pane.
    - Black shapes occupy only the left pane on top of a uniform background color.

    Rule:
    - Shift every black 4-connected component right until its rightmost pixel is adjacent to the separator.
    - In the shifted left pane, find enclosed background regions whose rightmost cells touch the inner face near the separator.
    - Each such enclosed hole emits exactly one red ray from its center row across the whole right pane.

    Key insight: Sliding still creates the relevant enclosed cavities, but the variant compresses each valid cavity into a single central output line instead of lighting every row of the cavity.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
    left_mask = left == black
    comps = _components4(left_mask)

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

    # Step 3: if a hole touches column sep-2, emit one ray from the hole's center row.
    trigger_col = sep - 2
    trigger_rows = set()
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.add(_center_row_from_points(pts))

    for r in sorted(trigger_rows):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1
EMITTER_HEIGHTS = (5, 7)

EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

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
    min_emitter_h = min(EMITTER_HEIGHTS)
    return n_emitters * min_emitter_h + (n_total - n_emitters) * 2 + (n_total + 1) * MIN_BAND_GAP


def _sprite_bands(h, is_emitter):
    """Partition grid rows into disjoint bands and keep emitter heights odd for a unique center row."""
    n = len(is_emitter)
    heights = [min(EMITTER_HEIGHTS) if e else 2 for e in is_emitter]
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    height_cap = h - sum(base_gaps)
    if sum(heights) > height_cap:
        return None

    spare = height_cap - sum(heights)

    bumpable_emitters = [i for i, e in enumerate(is_emitter) if e and heights[i] + 2 <= max(EMITTER_HEIGHTS)]
    random.shuffle(bumpable_emitters)
    for i in bumpable_emitters:
        if spare >= 2 and random.random() < 0.5:
            heights[i] += 2
            spare -= 2

    while spare > 0:
        choices = [i for i, e in enumerate(is_emitter) if (not e) and heights[i] < SPRITE_H_MAX]
        if not choices:
            break
        i = random.choice(choices)
        heights[i] += 1
        spare -= 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _emitter_patch(sh, max_sw):
    """Build a closed emitter shell whose hole height is odd, so its center row is unambiguous."""
    assert sh in EMITTER_HEIGHTS, f"unexpected emitter height {sh}"
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, left_wall] = BLACK
    patch[1:-1, right_wall] = BLACK
    return patch


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
    """Decide sprite count and which sprites are emitters. Returns flags list."""
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(_min_emitters_for(n), n) <= h]
    assert valid_counts, f"no sprite counts fit height {h}"
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


def _trigger_rows_in_patch(patch):
    """Rows occupied by the triggering hole next to the right wall."""
    rows = [r for r in range(patch.shape[0]) if patch[r, -1] == BLACK and patch[r, -2] == BG]
    assert rows, "emitter patch must expose a hole beside the right wall"
    return rows


def _center_row_from_rows(rows):
    rows = sorted(int(r) for r in rows)
    return rows[(len(rows) - 1) // 2]


def _render_output(h, w, sep, bands, flags):
    """Stamp sprites flush against the separator and paint one center-row ray for each emitter."""
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
            trigger_rows = _trigger_rows_in_patch(patch)
            center_lr = _center_row_from_rows(trigger_rows)
            out[r0 + center_lr, sep + 1 :] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each sprite leftward by a random amount to produce the input grid."""
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D5 variant input")

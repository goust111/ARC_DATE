# Variant of 6e453dd6
# Dimension: D2 — 射线发射路径 (path)
# Change: straight_ray（直线射线）——从封闭洞所在行向右水平发射红色射线至网格右边缘... → 对角线射线

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and step trace
# - The left pane keeps the original semantics: black 4-connected sprites slide right until
#   their right edge is flush with the grey separator.
# - The trigger is still the same geometric fact as in the seed task: an enclosed background
#   hole whose rightmost hole cells lie on column sep-2 after the slide.
# - The changed stage is only the emission stage. Instead of painting the whole trigger row
#   red, each trigger row now acts like a nozzle pixel and launches two 45-degree rays into
#   the right pane: one up-right and one down-right. The first painted cells are at
#   (row-1, sep+1) and (row+1, sep+1); the separator itself remains grey.
# - Example: if a slid emitter has mouth rows {8, 9} and sep=10, then the original task would
#   paint rows 8 and 9 horizontally across cols 11..W-1. The variant paints:
#     row 8 -> (7,11), (6,12), ... and (9,11), (10,12), ...
#     row 9 -> (8,11), (7,12), ... and (10,11), (11,12), ...
#   The output becomes a fan/chevron pattern instead of two flat stripes.
#
# b) Complexity calibration
# - The rule stays single-mechanism: slide first, detect holes second, emit third.
# - To keep the new path visibly distinct, the right pane cannot stay too narrow. The seed
#   width 2..4 often yields only one or two diagonal pixels, which visually collapses toward
#   “just some red on the right”. This variant therefore expands right_w to 4..7 while keeping
#   the left-side object logic unchanged.
# - Rays are always red and never reflect. This imports only the 45-degree stepping idea from
#   source task 142ca369, not its reflection/color-transfer machinery.
#
# c) Implicit-assumption audit of the original main()
# - Separator detection and background inference are still valid: the separator is still one
#   full grey column and the border is still dominated by background.
# - Component sliding still works because sprites remain disjoint by row bands in the generator.
# - Hole detection still uses “background component not touching the left-pane boundary”; the
#   new variant does not change enclosure semantics.
# - The old row-wise assertion `out[r, sep+1:] == bg` is no longer valid because diagonal rays
#   from different mouth rows can overlap on the same right-half cells. The new solver instead
#   asserts once that the input right pane starts as pure background, then allows later writes
#   onto cells that are already red.
#
# d) Placement legality
# - Sprites are still generated output-first, flush with the separator, then unslid leftward.
# - Vertical row bands remain pairwise disjoint, so unsliding cannot create overlaps.
# - The widened right pane gives diagonal rays room without changing the left-side fitting logic.
# - Ray painting is clipped by grid bounds; no out-of-bounds writes occur.
#
# e) Visual observability
# - Every triggering mouth row emits both slopes whenever the grid permits, so a single emitter
#   already shows a recognizable V/fan motif.
# - The generator rejects trivial cases where the diagonal pattern would be too short to notice.
#
# f) Visual cleanliness
# - All rays keep one color and occupy only the right pane, so the left-side motion cue and the
#   right-side emission cue stay visually separated by the grey divider.
# - Overlap between rays is acceptable because the overlap color is unchanged (red), avoiding
#   ambiguous color mixing.
#
# g) Color inferability
# - No new color mapping is introduced. Black still marks sprites, grey still marks the divider,
#   magenta is still background, and red still marks emitted energy.
#
# New solver assumptions introduced by the variant
# - The input right pane must be background-only before emission; the generator guarantees this.
# - Trigger rows are read from hole pixels exactly at sep-2; the generator ensures emitter holes
#   really expose those mouth pixels after sliding flush.
# - Overlapping diagonal rays are harmless because all rays use the same red color.


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


def _paint_diagonal_branch(out: np.ndarray, start_r: int, start_c: int, dr: int, bg: int, red: int) -> int:
    """Paint one 45-degree branch, stopping at the grid boundary."""
    h, w = out.shape
    rr = int(start_r)
    cc = int(start_c)
    length = 0
    while 0 <= rr < h and 0 <= cc < w:
        assert out[rr, cc] in (bg, red), f"unexpected right-pane color {out[rr, cc]} at {(rr, cc)}"
        out[rr, cc] = red
        rr += int(dr)
        cc += 1
        length += 1
    return length


def _emit_diagonal_fan(out: np.ndarray, mouth_rows: list[int], sep: int, bg: int, red: int) -> list[int]:
    """Emit up-right and down-right branches for each trigger row."""
    lengths = []
    for r in mouth_rows:
        lengths.append(_paint_diagonal_branch(out, r - 1, sep + 1, -1, bg, red))
        lengths.append(_paint_diagonal_branch(out, r + 1, sep + 1, +1, bg, red))
    return lengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes against the grey divider, then fan red diagonal rays out from enclosed mouths.

    Input structure:
    - One full-height grey (5) separator column splits the grid into a left sprite pane and a right empty pane.
    - Black shapes occupy only the left pane on top of a single background color.

    Rule:
    - Move every black connected component rightward until its right edge is flush with the separator.
    - In that slid arrangement, detect enclosed background holes whose rightmost hole pixels touch the inner face of the component at column sep-2.
    - For each such mouth row, cast two red rays into the right pane along 45-degree diagonals: one up-right and one down-right, stopping at the grid border.

    Key insight: The slide still reveals which holes are active emitters, but the emitted signal is now encoded by diagonal fan geometry rather than flat horizontal stripes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # The right pane is empty in the input; later overlaps happen only between red rays.
    assert np.all(out[:, sep + 1 :] == bg), "input right pane must start as background only"

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

    # Step 3: mouth pixels are hole pixels at column sep-2. Each mouth row fires two diagonals.
    trigger_col = sep - 2
    mouth_rows = []
    for pts in holes:
        rows = sorted({int(r) for r, c in pts if int(c) == trigger_col})
        if rows:
            mouth_rows.extend(rows)
    assert len(mouth_rows) == len(set(mouth_rows)), f"trigger rows must be unique, got {mouth_rows}"
    _emit_diagonal_fan(out, mouth_rows, sep, bg, red)

    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 19
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 4, 7

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


def _emitter_mouth_rows(patch: np.ndarray) -> list[int]:
    """Rows whose rightmost hole cells lie against the inner face of the right wall."""
    sh, sw = patch.shape
    if sw < 2:
        return []
    return [r for r in range(sh) if patch[r, sw - 1] == BLACK and patch[r, sw - 2] == BG]


def _render_output(h, w, sep, bands, flags):
    """Stamp sprites flush against the separator and paint diagonal fan rays for emitters."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    branch_lengths = []
    n_mouth_rows = 0
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            rows = _emitter_mouth_rows(patch)
            n_mouth_rows += len(rows)
            branch_lengths.extend(_emit_diagonal_fan(out, [r0 + lr for lr in rows], sep, BG, RED))
        sprites.append((patch, r0, c0))

    red_cells = int(np.sum(out[:, sep + 1 :] == RED))
    stats = {
        "mouth_rows": int(n_mouth_rows),
        "branch_lengths": list(branch_lengths),
        "red_cells": red_cells,
        "max_branch": max(branch_lengths, default=0),
        "long_branches": sum(1 for x in branch_lengths if x >= 3),
    }
    return out, sprites, stats


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


def _visible_enough(stats, w, sep):
    """Reject outputs where the diagonal-fan change is barely visible."""
    right_w = w - sep - 1
    return (
        stats["mouth_rows"] >= 1
        and stats["max_branch"] >= 3
        and stats["long_branches"] >= 2
        and stats["red_cells"] >= right_w + 2
    )


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites, stats = _render_output(h, w, sep, bands, flags)
    if not _visible_enough(stats, w, sep):
        return None

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
    raise RuntimeError("failed to generate a valid 6e453dd6 D2-variant input")

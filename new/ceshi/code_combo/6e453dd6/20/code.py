# Combo variant of 6e453dd6 (index 20)
# combo_spec: D9/0+D14/1
# Name: 仅输出右侧区域+反转宽度比

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for combo D9/0 + D14/1
# ---------------------------------------
# concrete_behavior:
#   - The logical trigger remains the original one: black sprites slide right until
#     they are flush with the grey separator, and only enclosed background pockets
#     pressed against the separator-side interior column emit red rays.
#   - The fused visible change has two parts that reinforce each other:
#     1. D14 inversion makes the left pane tiny and the right pane wide, so the
#        sprites are miniature shells while the rays become long and salient.
#     2. D9 cropping removes the left pane and the separator from the output, so the
#        solver must infer that the returned image is just the right ray pane.
#   - Example trace: take left_w=4, right_w=8. A 3-column emitter sitting at input
#     cols 0..2 slides to cols 1..3. Its 1-cell cavity lands at col 2 (= sep-2), so
#     the corresponding rows paint red across the 8-column right pane. The final
#     output is only that 8-column pane, not the original 13-column grid.
# complexity_calibration:
#   - The spec allows left_w in 2..4, but 2 columns cannot contain an enclosed hole
#     and 3 columns leave a 3-wide emitter with no preslide slack. Fixing generation
#     at left_w=4 is therefore the smallest legal regime that still shows both
#     required effects: actual sliding on the left and a clear hole-trigger rule.
#   - Emitters stay width 3 with optional divider rows. This keeps the rule legible
#     across a few examples: tiny shells repeat on the left, while long cropped rays
#     repeat on the right.
# implicit_assumptions_audit:
#   - _find_separator_col() still assumes exactly one full-height separator column.
#     The generator preserves that.
#   - main() now assumes a narrow-left / wide-right layout. The generator guarantees
#     left_w=4 and right_w in 5..11, which satisfies the requested inversion.
#   - Sliding by target_right - max_c still works because every sprite is confined to
#     its own row band, so horizontal flushing cannot merge components.
#   - Hole detection still treats bg components touching the left subgrid boundary as
#     non-holes. With sep=4, trigger_col=sep-2=2 is always a valid interior column.
#   - Because D9 crops the output, main() no longer needs to preserve the shifted
#     left pane visually; it only uses that intermediate state to decide which right
#     rows become red.
# placement_legality:
#   - Row bands are disjoint and separated by at least one full background row, so
#     sprites do not overlap before or after sliding.
#   - Flush output placement uses cols 1..3 for 3-wide emitters and cols 1..3 or
#     2..3 for distractors, all strictly inside the 4-column left pane.
#   - Input instances are created by shifting each sprite left by at most its slack;
#     at least one sprite moves in every accepted sample, so the translation rule is
#     visible rather than vacuous.
# visual_observability:
#   - Human-visible clues are structural, not symbolic: cramped left pane, long right
#     pane, and outputs that are narrower than inputs because only the right pane is
#     retained.
#   - The crop is easy to infer from 2-3 examples because every output has the same
#     height as the input but a strictly smaller width equal to the input's right pane.
# visual_cleanliness:
#   - Input stays cleanly partitioned: sprites only on the left, separator in the
#     middle, right pane pure background.
#   - Output contains only background and red rays, so there is no layered clutter
#     from also displaying the shifted sprites.
# inferability_of_change:
#   - No invisible ordering or color-id convention is used. The only determinants are
#     separator-relative geometry, enclosure, and the consistent crop to the right pane.
# new_assumption_review:
#   - main() assumes all trigger-causing structure is recoverable from the slid left
#     pane alone. The generator constructs all emitters and distractors wholly on the
#     left, so this is satisfied.
#   - main() assumes the right pane is empty before ray painting. The generator keeps
#     every input right pane uniformly background.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide miniature black sprites into the separator, then return only the wide ray pane.

    Input structure:
    - A full-height grey (5) separator sits near the left edge, leaving a narrow sprite pane on the left and a much wider empty pane on the right.
    - Small black components occupy only the left pane on a uniform background; the right pane starts as background everywhere.

    Rule:
    - Shift every black connected component rightward until its right edge is flush with the separator.
    - In that shifted left pane, find enclosed background pockets and keep only the pockets touching the interior trigger column immediately left of the separator-side wall.
    - Paint red rows across the entire right pane for those triggering pocket rows, then crop away the separator and all left-side content so the output is only the right pane.

    Key insight: The fused variant hides the sprite motion from the final image; you must mentally reconstruct the flush left pane to know which rows fire, then keep only the resulting long right-side rays.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    right_w = w - sep - 1
    assert 3 <= sep <= 4 and 5 <= right_w <= 11 and sep < right_w, (
        f"expected narrow-left / wide-right panes, got left_w={sep}, right_w={right_w}"
    )
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)
    assert np.all(g[:, sep + 1 :] == bg), "right pane must be background-only in the input"

    left = g[:, :sep].copy()
    comps = _components4(left == black)
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

    holes = []
    for pts in _components4(left == bg):
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

    out = np.full((h, right_w), bg, dtype=int)
    for r in trigger_rows:
        out[r, :] = red
    return out


# --------------- generator (combo variant) ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
LEFT_W = 4
RIGHT_W_MIN, RIGHT_W_MAX = 5, 11

# ── Sprite constraints ──
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
EMITTER_W = 3
SPRITE_W_MAX = 3
MIN_BAND_GAP = 1

# ── Probabilities ──
DIVIDER_PROB = 0.45
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.2

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


def _divider_rows(sh):
    """Choose non-adjacent internal divider rows for a tiny emitter shell."""
    if sh < 5:
        return []
    rows = []
    candidates = list(range(2, sh - 2))
    random.shuffle(candidates)
    for r in candidates:
        if random.random() >= DIVIDER_PROB:
            continue
        if any(abs(r - prev) <= 1 for prev in rows):
            continue
        rows.append(r)
    return sorted(rows)


def _emitter_patch(sh):
    """Build a 3-column closed shell whose cavity lands on sep-2 when flush."""
    patch = np.full((sh, EMITTER_W), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    for r in _divider_rows(sh):
        patch[r, :] = BLACK
    return patch


def _distractor_patch(sh, max_sw):
    """Build a non-emitter distractor that fits the cramped left pane."""
    choices = ["solid", "l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.choice([2, max_sw])
        return np.full((sh, sw), BLACK, dtype=int)

    if kind in {"c_left", "c_right"}:
        patch = np.full((sh, 3), BG, dtype=int)
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        if kind == "c_left":
            patch[:, -1] = BLACK
        else:
            patch[:, 0] = BLACK
        return patch

    sw = random.choice([2, max_sw])
    patch = np.full((sh, sw), BG, dtype=int)
    patch[:, -1] = BLACK
    if random.random() < 0.5:
        patch[0, :] = BLACK
    else:
        patch[-1, :] = BLACK
    return patch


def _choose_grid_dims():
    """Pick grid height and the inverted left/right pane widths."""
    h = random.randint(H_MIN, H_MAX)
    sep = LEFT_W
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    w = sep + 1 + right_w
    assert W_MIN <= w <= W_MAX
    return h, w, sep


def _choose_emitter_flags(h):
    """Decide sprite count and which sprites are emitters."""
    valid_counts = [
        n
        for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
        if _min_rows_needed(_min_emitters_for(n), n) <= h
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


def _render_target_and_sprites(h, w, sep, bands, flags):
    """Build the cropped target pane and the flush sprite placements."""
    target = np.full((h, w - sep - 1), BG, dtype=int)
    sprites = []
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh) if is_emitter else _distractor_patch(sh, SPRITE_W_MAX)
        sw = patch.shape[1]
        c0 = sep - sw
        if is_emitter:
            for lr in range(sh):
                if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
                    target[r0 + lr, :] = RED
        sprites.append((patch, r0, c0))
    return target, sprites


def _unslide_sprites(h, w, sep, sprites):
    """Shift each sprite leftward to produce the input grid."""
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    moved = 0
    for i, (patch, r0, c0) in enumerate(sprites):
        assert c0 >= 1, "variant sprites should keep at least one preslide slack cell"
        delta = 0 if i == flush_i else random.randint(1, c0)
        moved += int(delta > 0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    if moved == 0:
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build target, derive input, verify with oracle."""
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    target, sprites = _render_target_and_sprites(h, w, sep, bands, flags)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), target):
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

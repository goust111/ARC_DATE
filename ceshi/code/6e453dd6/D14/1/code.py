# Variant of 6e453dd6
# Dimension: D14 — 左右分区宽度比 (meta)
# Change: 左侧宽(5-11列)，右侧窄(2-4列)——不对称分区，精灵空间大但射线短... → 左侧窄右侧宽（反转）

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D14 inversion
# ------------------------------------
# concrete_behavior:
#   - The logical rule stays the same: black sprites slide right until flush with the
#     grey separator, then enclosed background pockets that touch the separator-side
#     interior column fire red rays across the entire right pane.
#   - The visible change is the layout ratio. The separator is moved close to the
#     left edge so the sprite pane is tiny and the ray pane is long. Generated
#     instances use a 4-column left pane and a 5-11-column right pane. This keeps
#     the inversion obvious while preserving enough room for nontrivial sliding.
#   - Example trace:
#       original regime: left_w=8, right_w=3, a 4-wide shell can shift several cells,
#       but each emitted ray is short.
#       variant regime: left_w=4, right_w=9, the same idea becomes a 3-wide mini-shell
#       that shifts by exactly one cell; the emitted red segment is much longer and
#       dominates the output visually.
# complexity_calibration:
#   - A left pane of only 2 columns cannot host an enclosed hole, and a 3-column pane
#     leaves no preslide slack for a 3-wide emitter. Fixing generated instances at
#     left_w=4 is the smallest setting that still shows both the sliding action and
#     the hole-trigger rule. This is an allowed adaptation because it serves the D14
#     inversion directly rather than introducing a different mechanic.
#   - Emitters are constrained to width 3 with optional internal divider rows. That
#     keeps the rule readable from a few examples: tiny shells on the left, long rays
#     on the right, repeated across several bands.
# implicit_assumptions_audit:
#   - main(): finding the full-height separator still works unchanged; only its
#     position range changes.
#   - main(): translating each component by target_right-max_c remains valid because
#     sprites are still wholly to the left of the separator and row bands prevent
#     inter-component overlap after the slide.
#   - main(): hole detection assumes a trigger column sep-2 that is strictly inside
#     the left pane. The generator guarantees sep=4, so sep-2=2 is always valid.
#   - main(): red rays are written only onto background cells on the right. The
#     generator still keeps the right pane pure background in the input.
# placement_legality:
#   - Row bands are disjoint with at least one background row between them, so sprites
#     never overlap before or after sliding.
#   - Output sprites are placed flush at columns 1..3 (for 3-wide emitters) or 2..3 /
#     1..3 (for distractors), all inside the 4-column left pane.
#   - Input sprites are derived by shifting each sprite left by at most its available
#     slack; at least one sprite is forced to move so the slide is visible.
# visual_observability:
#   - The inversion is easy to see: the left side is a cramped strip with miniature
#     black components, while the right side has long red rays spanning many columns.
#   - No invisible ordering or color-id trick is used; only geometric relations to
#     the separator matter.
# visual_cleanliness:
#   - The separator cleanly splits the panes, sprites live only on the left, and rays
#     live only on the right. The two effects do not overlap visually.


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
    Slide mini black sprites into a narrow left slot, then project long red rays rightward.

    Input structure:
    - A full-height grey (5) separator stands close to the left edge, leaving only a tiny left pane and a much wider right pane.
    - Small black sprites occupy the left pane on a uniform background, while the right pane starts as background only.

    Rule:
    - Translate every black connected component rightward until its right edge is flush with the separator.
    - In the shifted left pane, detect enclosed background pockets and keep only the ones touching the interior column immediately left of the separator-side wall.
    - For each row occupied by those triggering pockets, paint a red ray across the entire right pane.

    Key insight: The D14 variant compresses the sprites into miniature shells, but the trigger is unchanged: only sealed pockets pressed against the separator side fire, and the now-wide right pane makes those rays the dominant output feature.
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

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
    comps = _components4(left == black)

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

    # Step 2: find enclosed background regions (holes) inside the shifted left pane.
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    # Step 3: holes that reach sep-2 fire into the long right pane on their occupied rows.
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


# --------------- generator (D14 variant) ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
LEFT_W = 4
SEP_MIN, SEP_MAX = LEFT_W, LEFT_W
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
    """Choose non-adjacent internal divider rows for a 3-wide emitter shell."""
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
    """Build a tiny closed shell whose 1-cell cavity touches the trigger column when flush."""
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
    """Pick grid height, width, and separator column for the inverted ratio."""
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


def _render_output(h, w, sep, bands, flags):
    """Stamp sprites flush against the separator and paint red rays for emitters."""
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh) if is_emitter else _distractor_patch(sh, SPRITE_W_MAX)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lr in range(sh):
                if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
                    out[r0 + lr, sep + 1 :] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


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
    raise RuntimeError("failed to generate a valid 6e453dd6 D14 input")

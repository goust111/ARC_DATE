# Combo variant of 6e453dd6 (index 6)
# combo_spec: D1/0+D8/0
# Name: 反向平移保留原位

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the fused D1/0 + D8/0 variant
# --------------------------------------------------
# a) Concrete behavior and trace
# - Baseline 6e453dd6:
#   sprites start somewhere on the left, slide right until flush with the separator,
#   then only the holes touching the separator-facing inner wall emit red rays rightward.
# - This combo changes two parts together:
#   1. reverse the translation direction inside the left pane,
#   2. keep the source sprite instead of erasing it.
# - The resulting behavior is:
#   input already shows separator-flush sprites on the left;
#   output keeps those source sprites, adds a left-shifted copy whose left edge reaches
#   column 0, and still emits rightward rays from holes in the original separator-flush
#   sprite, not from the moved copy.
# - Small trace for one 3x4 emitter with a 1-cell hole near its right wall:
#   input left pane:  [..####][sep]
#                     [..#..#][sep]
#                     [..####][sep]
#   baseline on that flush input would leave the sprite in place and only add rays.
#   combo output becomes:
#                     [####..####][sep]
#                     [#..#..#..#][sep]
#                     [####..####][sep]
#   plus red rows to the right. The visible difference is duplication toward the left edge.
#
# b) Complexity calibration
# - The sprite family stays the same: 2-5 vertically separated black components, with
#   >=60% emitters and the same emitter/distractor vocabulary as the original task.
# - The copy target is a single discrete destination (left edge), which keeps the rule
#   human-readable. At the same time, the source remains flush against the separator, so
#   the duplication is visually strong and cannot be mistaken for the original task.
# - To avoid ambiguous heavy overlap between source and copy, generation restricts sprite
#   widths so the two silhouettes are separated by at least one background column.
#
# c) Implicit assumption audit
# - The original solver assumed "black lives on the left before motion" and "erase source
#   before redrawing". The first still holds; the second must be removed.
# - The original hole test ran on the post-slide display pane. That is invalid here:
#   source remnants plus copied pixels could create accidental enclosed background pockets.
#   The fused solver must therefore detect holes only on the original source pane.
# - Component extraction must happen on the input source pane before adding copies. If it
#   were delayed until after drawing the copy, wide shapes could merge with their copies
#   and destroy per-component motion.
# - Ray direction stays rightward. That means the D1 reversal applies only to sprite
#   translation, not to separator side or beam destination.
#
# d) Placement legality
# - The generator builds input directly in the constrained source state: every sprite is
#   already flush with the separator.
# - Row bands are disjoint and separated by at least one background row, so neither the
#   source sprites nor their left-edge copies can merge across different objects.
# - Width is capped by floor((sep-1)/2), guaranteeing each copied sprite fits at column 0
#   and leaves a visible gap before the separator-flush source copy.
#
# e) Visual observability
# - Across examples, the user sees three repeated cues:
#   source sprites all hug the separator in the input,
#   output adds a second black silhouette against the left border,
#   red rays still align with the separator-side holes of the original sprite.
# - This is easy to distinguish from both the original task (no duplicate remains) and the
#   full mirror task (sprites would live on the right and rays would go left).
#
# f) Visual cleanliness
# - The left pane contains only two black layers for each object: the source and its copy.
# - Because generator forces a background gap between them, the duplication reads as two
#   separate silhouettes rather than a messy thickened blob.
# - The right pane stays purely reserved for red rays.
#
# g) New solver assumptions and generator guarantees
# - `main()` assumes the source sprites are the black components already present on the left.
# - It assumes right-side rows are background before ray emission.
# - It assumes trigger rows are unique across sprites because row bands do not overlap.
# - `generate_input()` enforces all three and also ensures the combo is non-degenerate:
#   every sample has at least one added left-edge copy and all source pixels remain visible.


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


def _build_left_edge_copy_display(
    left: np.ndarray,
    *,
    black: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (display_left, source_left) for the keep-source leftward-copy rule."""
    comps = _components4(left == black)
    display_left = left.copy()
    source_left = left.copy()

    for comp in comps:
        min_c = min(c for _, c in comp)
        delta = min_c
        assert delta >= 0, "left-side components must stay inside the left pane"
        for r, c in comp:
            nc = c - delta
            assert 0 <= nc < left.shape[1], f"shifted pixel {(r, c)} left the pane"
            display_left[r, nc] = black

    return display_left, source_left


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep separator-flush black shapes in place, add left-shifted copies, and fire rightward red rays from the original holes.

    Input structure:
    - A full-height grey (5) separator column divides the grid into a wide left sprite pane and a narrow right ray pane.
    - Black shapes already sit on the left pane with their right edges flush against the separator, while the right pane is background.

    Rule:
    - For each black connected component, keep the original sprite and add a copy shifted left until its left edge reaches column 0.
    - Detect enclosed background holes only inside the original separator-flush sprites.
    - For every row belonging to a source hole that touches the inner column next to the separator, paint a red ray across the entire right pane.

    Key insight: The transformation is a reverse-direction copy, not a move: the separator-side source sprite remains the ray-emitting nozzle, and the new information is the extra silhouette pasted against the left border.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = g[:, :sep].copy()
    display_left, source_left = _build_left_edge_copy_display(left, black=black)
    out[:, :sep] = display_left

    holes = []
    for pts in _components4(source_left == bg):
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
    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    return out


# --------------- generator (combo D1/0 + D8/0) ---------------

# Colors
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# Grid dimensions
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 7, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

# Sprite constraints
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# Probabilities
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25

# Retry budget
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


def _emitter_patch(sh, max_sw):
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


def _choose_emitter_flags(h):
    valid_counts = [
        n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
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


def _render_source_input(h, w, sep, bands, flags):
    """Build the input as separator-flush source sprites."""
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, (sep - 1) // 2)
    assert max_sw >= 3, "left pane must support visible leftward copies and emitters"

    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(inp, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0, is_emitter))
    return inp, sprites


def _compose_expected_output(inp, sprites, sep):
    """Compose the combo output from the known source sprites without using `main()`."""
    out = inp.copy()
    for patch, r0, _c0, _is_emitter in sprites:
        paste(out, patch, r0, 0, transparent=BG)

    for patch, r0, _c0, is_emitter in sprites:
        if not is_emitter:
            continue
        for lr in range(patch.shape[0]):
            if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
                out[r0 + lr, sep + 1 :] = RED
    return out


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    inp, sprites = _render_source_input(h, w, sep, bands, flags)
    expected = _compose_expected_output(inp, sprites, sep)

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

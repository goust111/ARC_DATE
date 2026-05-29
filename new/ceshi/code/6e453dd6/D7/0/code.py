# Variant of 6e453dd6
# Dimension: D7 — 射线终止条件 (dst)
# Change: 射线延伸到网格右边缘（grid.right_edge）——填充从sep+1到w-1的所有列... → 射线延伸到目标像素处停止

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis
# ----------------
# spec:
#   variant_core:
#     - Left-half behavior stays the same: slide every black component right until it is flush with the grey separator.
#     - Trigger detection stays the same: enclosed bg holes that touch the inner face of the separator determine which rows emit.
#     - The changed part is the destination of each emitted ray. The right side now contains isolated colored target pixels
#       already present in the input, and each red ray stops immediately before the first target pixel on its row.
#   concrete_behavior_and_trace:
#     - Small example:
#       before slide, an emitter sprite sits 2 columns away from the separator; after slide, one hole row becomes
#       "... bg hole at sep-2 | black wall at sep-1 | grey sep | bg bg teal bg".
#       Original task: output row on the right became "red red red red".
#       Variant: if the first target on that row is teal at sep+3, output becomes "red red teal bg";
#       a farther orange target on the same row does nothing because the nearer teal one already stops the ray.
#   complexity_calibration:
#     - The right pane is widened from 2-4 columns to 4-6 columns so the stop position is visually observable.
#     - Every triggering row gets at least one interior target (never on the outer border), so the D7 change is active
#       instead of degenerating back to "fill to the edge".
#     - Some rows also get a second, farther target, and some non-trigger rows get decoy targets. This keeps the rule from
#       collapsing to "paint until the only colored pixel in the pane" while remaining easy to read.
#   implicit_assumptions_review:
#     - Sliding code only depends on black components being fully left of the separator; unchanged and still valid.
#     - Hole finding only depends on bg components inside the slid left pane; unchanged and still valid.
#     - The original `assert np.all(out[r, sep + 1:] == bg)` and `out[r, sep + 1:] = red` rely on the old invariant that the
#       right side is empty. They are invalid here because targets occupy some right-side cells and must not be overwritten.
#     - `_background_color()` infers bg from border majority, so the generator keeps target pixels off the top, bottom, and
#       outer-right border. That preserves the old inference method.
#   placement_legality:
#     - Sprite row bands remain disjoint, so trigger rows from different sprites never collide.
#     - Targets are single right-pane pixels only; they do not overlap the separator or each other.
#     - Each trigger row is guaranteed to have a first target at columns sep+2..w-2, ensuring at least one red cell and a
#       visible stop before the edge.
#   visual_observability:
#     - In input/output pairs, the only new non-empty content on the right is the colored target pixels.
#       Output rays always terminate exactly at the nearest such pixel on triggering rows.
#     - Because non-trigger rows can still contain targets, the learner must combine both halves:
#       left-side enclosed-hole rows decide whether a ray exists; right-side nearest target decides where it stops.
#   cleanliness_and_color_logic:
#     - Target colors come from a small palette distinct from black/red/grey/bg and carry no per-color behavior.
#       All target colors block identically, so the rule is positional rather than based on hidden color ordering.
#   new_solver_assumptions_checked_against_generator:
#     - For every trigger row there is at least one target pixel to the right of the separator.
#     - Cells between `sep+1` and the first target are bg in the input, so filling them red is safe.
#     - Right-side colored pixels in the input are intended blockers; the generator never places any other right-side motifs.


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


def _first_target_col(g: np.ndarray, row: int, sep: int, bg: int) -> int | None:
    for c in range(sep + 1, g.shape[1]):
        if int(g[row, c]) != int(bg):
            return int(c)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to a grey separator, then emit red rays that stop at colored targets.

    Input structure:
    - A full-height grey (5) separator column splits the board into a left sprite area and a right target area.
    - Black shapes occupy only the left side; the right side contains background plus isolated colored blocker pixels.

    Rule:
    - Shift every black 4-connected component right until its right edge is flush with the separator.
    - In the slid left pane, locate enclosed background holes; rows belonging to holes that touch the separator's inner face become emitter rows.
    - On each emitter row, paint red from the first column right of the separator up to, but not including, the nearest colored target pixel.

    Key insight: The left half still decides which rows fire, but the right half now decides how far each ray can travel because the first target pixel blocks it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
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

    # Step 3: rows from holes touching the inner face of the separator emit.
    trigger_col = sep - 2
    trigger_rows = []
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted(set(r for r, _ in pts)))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"

    # Step 4: each ray stops at the nearest pre-existing target pixel on its row.
    for r in trigger_rows:
        stop_col = _first_target_col(g, r, sep, bg)
        assert stop_col is not None, f"trigger row {r} must contain a right-side target"
        assert stop_col >= sep + 2, f"target on row {r} must leave room for a visible ray"
        assert np.all(out[r, sep + 1 : stop_col] == bg), (
            f"cells before the first target on row {r} must be background"
        )
        out[r, sep + 1 : stop_col] = red

    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5
TARGET_COLORS = [1, 3, 4, 7, 8, 9]

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 4, 6

# ── Sprite constraints ──
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# ── Probabilities ──
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35
SECOND_TARGET_PROB = 0.45

# ── Retry budget ──
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


def _trigger_rows_for_patch(patch, r0):
    rows = []
    sh, sw = patch.shape
    if sw < 2:
        return rows
    for lr in range(sh):
        if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
            rows.append(r0 + lr)
    return rows


def _choose_target_col(sep, w):
    candidates = list(range(sep + 2, w - 1))
    assert candidates, "right pane must allow interior blocker columns"
    return random.choice(candidates)


def _stamp_targets(out, sep, trigger_rows):
    h, w = out.shape
    target_rows = {}
    for r in sorted(trigger_rows):
        cols = { _choose_target_col(sep, w) }
        farther = [c for c in range(min(cols) + 1, w - 1)]
        if farther and random.random() < SECOND_TARGET_PROB:
            cols.add(random.choice(farther))
        for c in sorted(cols):
            out[r, c] = random.choice(TARGET_COLORS)
        target_rows[r] = sorted(cols)

    eligible_decoys = [r for r in range(1, h - 1) if r not in target_rows]
    random.shuffle(eligible_decoys)
    max_decoys = min(len(eligible_decoys), max(1, len(trigger_rows) // 2))
    n_decoys = random.randint(0, max_decoys) if max_decoys > 0 else 0
    for r in eligible_decoys[:n_decoys]:
        cols = { _choose_target_col(sep, w) }
        if w - sep >= 6 and random.random() < 0.25:
            farther = [c for c in range(min(cols) + 1, w - 1)]
            if farther:
                cols.add(random.choice(farther))
        for c in cols:
            out[r, c] = random.choice(TARGET_COLORS)

    return target_rows


def _render_output(h, w, sep, bands, flags):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    trigger_rows = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            trigger_rows.extend(_trigger_rows_for_patch(patch, r0))
        sprites.append((patch, r0, c0))

    assert trigger_rows, "at least one emitter row is required"
    assert len(trigger_rows) == len(set(trigger_rows)), "sprite bands should make trigger rows unique"

    target_rows = _stamp_targets(out, sep, trigger_rows)
    for r in trigger_rows:
        stop_col = target_rows[r][0]
        out[r, sep + 1 : stop_col] = RED
    return out, sprites, target_rows


def _unslide_sprites(h, w, sep, sprites, right_template):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    right = right_template[:, sep + 1 :].copy()
    right[right == RED] = BG
    inp[:, sep + 1 :] = right

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites, _ = _render_output(h, w, sep, bands, flags)
    inp = _unslide_sprites(h, w, sep, sprites, out)
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D7 input")

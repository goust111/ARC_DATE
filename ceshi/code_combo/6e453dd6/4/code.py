# Combo variant of 6e453dd6 (index 4)
# combo_spec: D6/1+D19/0
# Name: 分隔线缺口与标记双重过滤

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) concrete behavior trace:
#    - Original task: after sliding each black sprite flush to the separator, any enclosed bg hole that
#      reaches column sep-2 emits a red ray on every row it spans.
#    - D6 adds a broken separator: only candidate rows whose separator cell is a bg gap may pass.
#    - D19/0 adds explicit row marks: some rows are visibly annotated, and only annotated candidate rows
#      may emit.
#    - Directly stacking both changes on one cell is impossible because a separator cell cannot be both a
#      bg gap and a colored marker at once. The fused design therefore uses a two-cell gate:
#        * separator column `sep`: grey barrier or bg gap
#        * first right-side column `sep+1`: optional colored marker pixel
#      A row fires iff it is simultaneously:
#        1. a hole row after sliding,
#        2. a separator gap row,
#        3. a marked row in the adjacent marker lane.
#    - Small trace:
#        hole rows = {7,8,9}
#        gap rows  = {7,8}
#        marks     = {7,9}
#        output rays appear only on row 7. Row 8 fails the mark test; row 9 fails the gap test.
#
# b) complexity calibration:
#    - The generator forces at least one candidate row in each of three categories:
#        * allowed = hole + gap + mark
#        * gap-only candidate = hole + gap without mark
#        * mark-only candidate = hole + mark without gap
#      This prevents collapse to the base task or to the D6-only single-filter variant.
#    - It also adds a few non-candidate decoy gaps and marks when possible, so humans can see that neither
#      signal is sufficient without an enclosed trigger hole.
#    - Marker color is fixed to one accent color to keep the rule "presence of mark" rather than "decode a
#      color vocabulary", which would add unnecessary freedom.
#
# c) implicit assumption review of main():
#    - `_find_separator_col`: the old full-height-grey assumption is invalid; the new valid assumption is
#      that all grey separator pixels still live in one unique column.
#    - background inference still works because marker pixels are placed only at `sep+1`, never on the grid
#      border, and the separator column itself contains only grey or bg.
#    - sliding still moves only black pixels, because marks are not part of sprites and holes remain pure bg.
#    - hole detection may keep using `left == bg`; unlike the color-in-hole D19 variant, this combo does not
#      recolor cavities.
#    - ray painting can no longer assume the right side is entirely bg on emitting rows; the marker cell at
#      `sep+1` may already be occupied and is intentionally overwritten by the red ray.
#
# d) placement legality:
#    - Sprites stay in disjoint row bands, so horizontal sliding and unsliding never create overlaps.
#    - The separator column is reserved for grey/bg only; the marker lane is reserved for bg/marker only.
#    - The right side width remains 2-4, so the dedicated marker lane at `sep+1` always exists.
#
# e) visual observability:
#    - Input examples visibly show three row-wise cues in the middle-right corridor: hole-aligned rows,
#      separator gaps, and colored markers.
#    - Because the generator includes allowed, gap-only, and mark-only candidate rows, the training pairs
#      demonstrate that all three conditions matter.
#
# f) cleanliness:
#    - Marks occupy a single dedicated lane next to the separator instead of being mixed into black sprites.
#    - Rays overwrite only their own marked rows, while blocked marks remain visible, keeping the scene easy
#      to parse.
#
# g) inferability:
#    - No hidden use of numeric color ids is involved. The only new symbolic signal is a repeated visible
#      marker color in a consistent lane adjacent to the separator.


# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5
MARKER_COLOR = 8

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
    coords = np.argwhere(g == sep_color)
    assert len(coords) > 0, "expected grey pixels for the separator"
    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, f"expected grey pixels in one separator column, got {cols.tolist()}"
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


def _separator_gap_rows(g: np.ndarray, sep_col: int, bg: int, sep_color: int = SEP_COLOR) -> set[int]:
    sep_vals = g[:, sep_col]
    assert np.all((sep_vals == sep_color) | (sep_vals == bg)), "separator column may only contain grey and bg"
    gaps = {int(r) for r in np.flatnonzero(sep_vals == bg)}
    assert 0 < len(gaps) < g.shape[0], "separator must contain both solid cells and gaps"
    return gaps


def _marked_rows(g: np.ndarray, sep_col: int, bg: int, marker_color: int = MARKER_COLOR) -> set[int]:
    assert sep_col + 1 < g.shape[1], "marker lane requires at least one right-side column"
    marker_vals = g[:, sep_col + 1]
    assert np.all((marker_vals == bg) | (marker_vals == marker_color)), "marker lane may only contain bg or marker"
    return {int(r) for r in np.flatnonzero(marker_vals == marker_color)}


def _candidate_trigger_rows(left: np.ndarray, bg: int) -> list[int]:
    h, sep = left.shape
    trigger_col = sep - 2
    rows = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if not any(c == trigger_col for _, c in pts):
            continue
        rows.extend(sorted({int(r) for r, _ in pts}))
    assert len(rows) == len(set(rows)), f"trigger rows must be unique, got {rows}"
    return rows


def _right_row_ready_for_ray(out: np.ndarray, r: int, sep: int, bg: int, marker_color: int) -> bool:
    marker_ok = int(out[r, sep + 1]) in (bg, marker_color)
    tail_ok = np.all(out[r, sep + 2 :] == bg)
    return bool(marker_ok and tail_ok)


def _combo_row_sets_from_input(grid: np.ndarray) -> dict[str, set[int]]:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g)
    bg = _background_color(g, sep)

    left = g[:, :sep].copy()
    moved = left.copy()
    moved[moved == BLACK] = bg
    target_right = sep - 1
    for comp in _components4(left == BLACK):
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            moved[r, c + delta] = BLACK

    candidates = set(_candidate_trigger_rows(moved, bg))
    gaps = _separator_gap_rows(g, sep, bg)
    marks = _marked_rows(g, sep, bg)
    return {
        "candidates": candidates,
        "gap_rows": gaps,
        "marked_rows": marks,
        "allowed": candidates & gaps & marks,
        "gap_only": (candidates & gaps) - marks,
        "mark_only": (candidates & marks) - gaps,
        "blocked": candidates - gaps - marks,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites against a broken separator, then fire only through rows that pass both gates.

    Input structure:
    - A single separator column still splits the grid, but some separator cells are bg gaps instead of grey.
    - Black sprites occupy the left pane, and the first column on the right may contain colored row markers.

    Rule:
    - Move every left-side black component rightward until it is flush with the separator's inner face.
    - After sliding, find enclosed bg holes that reach the trigger column just inside the sprite's right wall.
    - A candidate hole row emits a red ray only when that same row is both a separator gap and a marked row in the adjacent marker lane.

    Key insight: The hole still nominates candidate emitter rows, but two visible row filters sit beside it: the separator must open on that row and the dedicated marker lane must also flag that row.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)
    gap_rows = _separator_gap_rows(g, sep, bg, sep_color=SEP_COLOR)
    marked_rows = _marked_rows(g, sep, bg, marker_color=MARKER_COLOR)

    out = g.copy()

    left = out[:, :sep].copy()
    comps = _components4(left == BLACK)
    left[left == BLACK] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = BLACK

    out[:, :sep] = left

    candidate_rows = _candidate_trigger_rows(left, bg)
    trigger_rows = [r for r in candidate_rows if r in gap_rows and r in marked_rows]
    for r in trigger_rows:
        assert _right_row_ready_for_ray(out, r, sep, bg, MARKER_COLOR), f"right half of row {r} is not a clean marker lane"
        out[r, sep + 1 :] = RED

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


def _candidate_rows_from_patch(patch: np.ndarray) -> list[int]:
    h, w = patch.shape
    if h < 3 or w < 2:
        return []
    return [int(r) for r in range(h) if patch[r, w - 1] == BLACK and patch[r, w - 2] == BG]


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


def _choose_gate_rows(h: int, candidate_rows: list[int]) -> tuple[set[int], set[int], set[int]]:
    candidates = sorted(set(int(r) for r in candidate_rows))
    assert len(candidates) >= 3, "combo needs allowed, gap-only, and mark-only candidate rows"

    seeds = random.sample(candidates, 3)
    allowed_rows = {seeds[0]}
    gap_rows = {seeds[0], seeds[1]}
    marked_rows = {seeds[0], seeds[2]}

    for r in candidates:
        if r in seeds:
            continue
        state = random.choice(["allowed", "gap_only", "mark_only", "blocked"])
        if state == "allowed":
            allowed_rows.add(r)
            gap_rows.add(r)
            marked_rows.add(r)
        elif state == "gap_only":
            gap_rows.add(r)
        elif state == "mark_only":
            marked_rows.add(r)

    other_rows = [r for r in range(h) if r not in candidates]
    if other_rows:
        extra_gap_max = min(3, len(other_rows))
        extra_gap_count = random.randint(1, extra_gap_max)
        gap_rows.update(random.sample(other_rows, extra_gap_count))

        remaining_for_mark = [r for r in other_rows if r not in marked_rows]
        if remaining_for_mark:
            extra_mark_max = min(3, len(remaining_for_mark))
            extra_mark_count = random.randint(1, extra_mark_max)
            marked_rows.update(random.sample(remaining_for_mark, extra_mark_count))

    assert allowed_rows == (set(candidates) & gap_rows & marked_rows)
    assert (set(candidates) & gap_rows) - marked_rows
    assert (set(candidates) & marked_rows) - gap_rows
    assert 0 < len(gap_rows) < h
    assert 0 < len(marked_rows) < h
    return gap_rows, marked_rows, allowed_rows


def _render_output(h, w, sep, bands, flags):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    candidate_rows = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            candidate_rows.extend(r0 + lr for lr in _candidate_rows_from_patch(patch))
        sprites.append((patch, r0, c0))

    candidate_rows = sorted(set(candidate_rows))
    if len(candidate_rows) < 3:
        return None

    gap_rows, marked_rows, allowed_rows = _choose_gate_rows(h, candidate_rows)
    for r in gap_rows:
        out[r, sep] = BG
    for r in marked_rows:
        out[r, sep + 1] = MARKER_COLOR
    for r in allowed_rows:
        out[r, sep + 1 :] = RED

    return out, sprites, gap_rows, marked_rows, allowed_rows


def _unslide_sprites(h, w, sep, sprites, gap_rows, marked_rows):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    for r in gap_rows:
        inp[r, sep] = BG
    for r in marked_rows:
        inp[r, sep + 1] = MARKER_COLOR

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

    rendered = _render_output(h, w, sep, bands, flags)
    if rendered is None:
        return None
    out, sprites, gap_rows, marked_rows, allowed_rows = rendered

    inp = _unslide_sprites(h, w, sep, sprites, gap_rows, marked_rows)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
        combo = _combo_row_sets_from_input(inp)
        if not combo["allowed"] or not combo["gap_only"] or not combo["mark_only"]:
            return None
        if combo["allowed"] != allowed_rows:
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

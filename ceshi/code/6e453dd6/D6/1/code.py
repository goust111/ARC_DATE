# Variant of 6e453dd6
# Dimension: D6 — 分隔线结构 (obstacle)
# Change: separator_line——单列全高灰色(5)垂直分隔线，将网格分为左右两区；左侧宽(5-11列)，右侧窄(2-4列)... → 分隔线带有缺口

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   preserved:
#     grid: 16-25 rows x 10-16 cols, bg=6
#     colors: black sprites=0, separator uses grey=5, rays use red=2
#     layout: one separator column still splits the board into a wide left pane and narrow right pane
#     action: black components slide right until flush with the separator's inner face
#     trigger source: enclosed bg holes inside the slid black sprites still define candidate emitter rows
#   changed_core:
#     separator:
#       old: the separator column was solid grey on every row
#       new: the separator column mixes grey and bg, so it contains visible gaps
#     emission filter:
#       old: every candidate hole row emitted
#       new: only candidate rows whose separator cell is a bg gap emit to the right
#   concrete_trace:
#     old:
#       after sliding, a hole touching column sep-2 marked rows r={7,8,9}; all three rows turned red on the right
#     new:
#       the same sprite still marks candidate rows r={7,8,9}, but if the separator has gaps only at rows {7,9},
#       then row 8 is blocked by a grey barrier and only rows 7 and 9 become red
#     visible_difference:
#       the separator itself now encodes which candidate rows are allowed through, so the output no longer
#       depends on holes alone; it depends on the overlap between holes and separator gaps
#   complexity_calibration:
#     gaps are chosen as a nontrivial subset of candidate rows, not all-or-none
#     each sample also adds 1+ decoy gaps on non-candidate rows when possible
#     this avoids collapsing back to the original task and makes the filtering rule visible from few examples
#   implicit_assumptions_review:
#     _find_separator_col:
#       old assumption: one full-height grey column exists
#       new assumption: all grey pixels still belong to one unique column, but some rows in that column are bg
#     hole scan:
#       old assumption: candidate rows are exactly rows of enclosed bg components touching sep-2
#       this remains valid because the variant changes only the obstacle, not sprite interiors
#     ray painting:
#       old assumption: every candidate row fires
#       new assumption: a row fires iff the separator cell on that row is bg
#     generator support:
#       it must guarantee at least one allowed candidate row and at least one blocked candidate row, otherwise
#       the variant can visually degenerate back to the original behavior
#   placement_legality:
#     sprites still occupy disjoint row bands, so sliding them horizontally cannot create overlaps
#     the separator column is reserved; gaps are bg cells inside that column, never black or red
#     the right side starts as all bg in the input, so painting red rays cannot overwrite other objects
#   observability_and_cleanliness:
#     the only new signal is the punctured separator column, placed exactly between the holes and the rays
#     this keeps the board readable: left pane = black sprites, center = grey/bg obstacle, right pane = red outputs


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = 5) -> int:
    coords = np.argwhere(g == sep_color)
    assert len(coords) > 0, "expected grey pixels for the broken separator"
    cols = np.unique(coords[:, 1])
    assert len(cols) == 1, f"expected grey pixels in one separator column, got {cols.tolist()}"
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


def _separator_gap_rows(g: np.ndarray, sep_col: int, bg: int, sep_color: int = 5) -> set[int]:
    sep_vals = g[:, sep_col]
    assert np.all((sep_vals == sep_color) | (sep_vals == bg)), "separator column may only contain grey and bg gaps"
    gaps = {int(r) for r in np.flatnonzero(sep_vals == bg)}
    assert 0 < len(gaps) < g.shape[0], "broken separator must contain both grey cells and bg gaps"
    return gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to a punctured grey separator, then fire only through aligned gaps.

    Input structure:
    - One vertical separator column still splits the grid, but it is broken: some cells are grey (5) and some are bg gaps.
    - Black shapes sit on the left side against a uniform background, while the right side starts empty.

    Rule:
    - Translate each black connected component rightward until its right edge is flush with the separator's left side.
    - After sliding, find enclosed bg regions (holes) whose cells reach the usual trigger column just inside the right wall.
    - For each row covered by such a hole, emit a red ray on the right only if that same row is a gap in the separator column.

    Key insight: The hole rows still nominate candidate emitters, but the punctured separator acts as a row-wise gate, so only hole rows aligned with visible openings can cross into the right pane.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)
    gap_rows = _separator_gap_rows(g, sep, bg, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
    left_mask = left == black
    comps = _components4(left_mask)

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

    # Step 3: candidate hole rows emit only when the separator row is an actual gap.
    trigger_col = sep - 2
    trigger_rows = []
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        for r in sorted(set(int(r) for r, _ in pts)):
            if r in gap_rows:
                trigger_rows.append(r)
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

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


def _candidate_rows_from_patch(patch: np.ndarray) -> list[int]:
    h, w = patch.shape
    if w < 2:
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


def _choose_gap_rows(h: int, candidate_rows: list[int]) -> set[int]:
    candidates = sorted(set(int(r) for r in candidate_rows))
    assert len(candidates) >= 2, "variant needs at least one open and one blocked candidate row"

    open_count = random.randint(1, len(candidates) - 1)
    open_rows = set(random.sample(candidates, open_count))

    other_rows = [r for r in range(h) if r not in candidates]
    extra_min = 1 if other_rows else 0
    extra_max = min(3, len(other_rows))
    extra_count = random.randint(extra_min, extra_max) if extra_max > 0 else 0
    extra_rows = set(random.sample(other_rows, extra_count)) if extra_count else set()

    gap_rows = open_rows | extra_rows
    blocked_rows = set(candidates) - open_rows
    assert gap_rows, "separator must have at least one gap"
    assert blocked_rows, "separator must still block at least one candidate row"
    assert len(gap_rows) < h, "separator cannot disappear completely"
    return gap_rows


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
    if len(candidate_rows) < 2:
        return None

    gap_rows = _choose_gap_rows(h, candidate_rows)
    for r in gap_rows:
        out[r, sep] = BG
    for r in candidate_rows:
        if r in gap_rows:
            out[r, sep + 1 :] = RED
    return out, sprites, gap_rows


def _unslide_sprites(h, w, sep, sprites, gap_rows):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    for r in gap_rows:
        inp[r, sep] = BG

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
    out, sprites, gap_rows = rendered

    inp = _unslide_sprites(h, w, sep, sprites, gap_rows)
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
    raise RuntimeError("failed to generate a valid 6e453dd6 input")

# Combo variant of 6e453dd6 (index 33)
# combo_spec: D6/1+D23/1
# Name: 分隔线缺口+精灵与分隔线间隙

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# source_task: original 6e453dd6, fused with D6/1 + D23/1.
#
# a) Concrete behavior and step trace
# - Original task: after solving, every black component slides until its right edge reaches
#   `sep-1`. An enclosed bg hole then triggers when it reaches `sep-2`, and every triggering
#   row paints a red ray across the right pane.
# - D23/1 changes the stopping geometry itself: the solved position is no longer flush with the
#   separator. Every component must stop at `sep-2`, leaving the entire column `sep-1` blank.
#   The hole column that matters therefore shifts left from `sep-2` to `sep-3`.
# - D6/1 keeps the broken separator from the single-variant reference: candidate rows only emit
#   if the separator row is a bg gap instead of a grey barrier.
# - Small trace:
#   original solved shell: right wall at `sep-1`, hole rows touch `sep-2`, so rows {7,8,9} fire.
#   combo solved shell: right wall stops at `sep-2`, column `sep-1` stays blank, hole rows touch
#   `sep-3`; if the separator has gaps only at rows {7,9}, then row 8 is still a candidate from
#   the hole but is blocked by grey and only rows 7 and 9 paint red.
# - The visible difference from the seed is therefore twofold and coupled: the left pane ends one
#   column earlier, and even those shifted hole rows still need separator openings to pass.
#
# b) Complexity calibration
# - The one-column lane at `sep-1` is forced in every solved output so the spatial offset is
#   obvious to a human; it is not an occasional corner case.
# - The generator requires at least two candidate rows in total, then chooses a nontrivial subset
#   of them as open separator gaps. This prevents collapse to either:
#   1. D23/1 only (all candidate rows open), or
#   2. a trivial no-ray sample (all candidate rows blocked).
# - Extra gap rows are also added away from the candidate set when possible, so the punctured
#   separator is not merely a stencil copied from the final red rows.
#
# c) Implicit-assumption review of the original main()
# - `_find_separator_col()` originally assumed a full grey column. Under D6/1 this is false, so
#   the solver must instead assume all grey separator pixels lie in one unique column and that
#   bg cells in that same column are intentional gaps.
# - The original motion code assumed `target_right = sep-1`. Under D23/1 this is false; the new
#   invariant is `target_right = sep-2`, leaving one empty lane before the separator.
# - The original hole trigger assumed `trigger_col = sep-2`. With the new lane, the correct
#   trigger becomes `sep-3`. Keeping the old trigger would incorrectly miss every emitter.
# - The original emission code assumed every candidate row fires. Under D6/1 it must intersect
#   candidate rows with the separator-gap rows.
# - The hole-detection boundary rule still works because holes are judged inside the whole left
#   pane. Any bg component touching row 0, row h-1, col 0, or col `sep-1` is outside space, not a
#   closed hole. The generator preserves that topology.
#
# d) Placement legality
# - Output-first generation remains the cleanest approach: first build the solved left pane with
#   the mandatory lane at `sep-1`, add separator gaps and red rays, then shift sprites left to
#   derive the input.
# - Sprites stay in disjoint row bands separated by at least one full bg row, so horizontal motion
#   cannot merge them and their ray rows cannot clash.
# - Every solved sprite patch satisfies `rightmost_col = sep-2`, so `sep-1` is guaranteed to stay
#   bg and no black pixel ever touches the separator.
#
# e) Visual observability
# - A human can infer the fusion from 2-4 examples because both cues are visible on the board:
#   1. solved sprites stop one column before the divider rather than hugging it;
#   2. only some hole rows continue through the separator, exactly where the divider is punctured.
# - No hidden ordering by component index or color id is used. All decisions come from visible
#   spatial relations: one-cell lane, enclosed holes, and separator openings.
#
# f) Visual cleanliness
# - The board remains cleanly partitioned: left = black sprites plus a magenta lane, center =
#   punctured grey separator, right = red rays.
# - The extra lane is only one column wide and always uniform bg, so it adds a strong clue without
#   cluttering the grid.
#
# g) Degeneration checks and new solver assumptions
# - To prove the fusion is active, post-test metrics should show:
#   1. `blocked_candidate_rows > 0` in each accepted sample, so D6/1 matters;
#   2. `diff_vs_d23_only > 0`, where D23-only means "same one-cell lane but ignore separator gaps";
#   3. `diff_vs_d6_only > 0`, where D6-only means "use the broken separator but slide flush to
#      `sep-1` instead of leaving the lane".
# - New solver assumptions that the generator guarantees:
#   1. the separator column contains both grey cells and bg gaps;
#   2. the right pane starts as pure bg;
#   3. candidate hole rows are unique because sprite bands are row-disjoint;
#   4. all input components satisfy `max_col <= sep-2`, so sliding rightward to `sep-2` is legal.


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


def _slide_left_pane(left: np.ndarray, sep: int, bg: int, target_right: int, black: int = 0) -> np.ndarray:
    moved = left.copy()
    comps = _components4(moved == black)
    moved[moved == black] = bg
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must not start beyond the allowed stopping column"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            moved[r, nc] = black
    return moved


def _trigger_rows_from_left(left: np.ndarray, bg: int, trigger_col: int) -> list[int]:
    h, sep = left.shape
    if trigger_col < 0 or trigger_col >= sep:
        return []

    trigger_rows = []
    for pts in _components4(left == bg):
        pts = [(int(r), int(c)) for r, c in pts]
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted({r for r, _ in pts}))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return sorted(trigger_rows)


def _solve_variant(grid: np.ndarray, *, lane_gap: int, gate_with_separator: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)
    gap_rows = _separator_gap_rows(g, sep, bg, sep_color=sep_color)

    out = g.copy()
    target_right = sep - 1 - lane_gap
    left = _slide_left_pane(g[:, :sep].copy(), sep, bg, target_right, black=black)
    out[:, :sep] = left

    trigger_col = target_right - 1
    candidate_rows = _trigger_rows_from_left(left, bg, trigger_col)
    active_rows = [r for r in candidate_rows if (r in gap_rows or not gate_with_separator)]

    for r in active_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red
    return out


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)
    gap_rows = _separator_gap_rows(g, sep, bg, sep_color=SEP_COLOR)

    fused_left = _slide_left_pane(g[:, :sep].copy(), sep, bg, sep - 2, black=BLACK)
    moved_components = 0
    for comp in _components4(g[:, :sep] == BLACK):
        if max(int(c) for _, c in comp) < sep - 2:
            moved_components += 1
    candidate_rows = _trigger_rows_from_left(fused_left, bg, sep - 3)
    open_rows = [r for r in candidate_rows if r in gap_rows]
    blocked_rows = [r for r in candidate_rows if r not in gap_rows]
    decoy_gap_rows = [r for r in sorted(gap_rows) if r not in set(candidate_rows)]

    fused_out = _solve_variant(g, lane_gap=1, gate_with_separator=True)
    d23_only_out = _solve_variant(g, lane_gap=1, gate_with_separator=False)
    d6_only_out = _solve_variant(g, lane_gap=0, gate_with_separator=True)
    return {
        "candidate_rows": candidate_rows,
        "open_candidate_rows": open_rows,
        "blocked_candidate_rows": blocked_rows,
        "decoy_gap_rows": decoy_gap_rows,
        "lane_clear": bool(np.all(fused_left[:, sep - 1] == bg)),
        "moved_components": moved_components,
        "diff_vs_d23_only": int(np.count_nonzero(fused_out != d23_only_out)),
        "diff_vs_d6_only": int(np.count_nonzero(fused_out != d6_only_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes so they stop one column before a broken grey separator, then fire only through matching openings.

    Input structure:
    - One vertical separator column still divides the board, but it is punctured: grey cells and bg gaps share the same column.
    - Black sprites live in the left pane, the right pane starts empty, and the solved position must leave a one-cell bg lane between each sprite's right edge and the separator.

    Rule:
    - Translate every black connected component rightward until its right edge lands at column `sep-2`, not `sep-1`, so column `sep-1` stays blank.
    - In that shifted left pane, find enclosed bg holes whose cells reach column `sep-3`; those rows are the candidate emitters under the new lane geometry.
    - Paint red rays across the right pane only for candidate rows whose separator cell is a bg gap rather than a grey wall.

    Key insight: The lane and the broken separator must be read together: the lane shifts the effective nozzle one column left, and the punctured divider then acts as a row-wise gate that decides which of those shifted hole rows can cross.
    """
    return _solve_variant(np.asarray(grid, dtype=int), lane_gap=1, gate_with_separator=True)


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

SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
ALIGNED_GAP_PROB = 0.35

MAX_RETRIES = 600


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

    slack = h - sum(heights) - sum(base_gaps)
    gaps = [a + b for a, b in zip(base_gaps, _scatter(slack, n + 1))]
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
    assert extra_rows or not other_rows, "generator should expose at least one decoy gap row when possible"
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
        c0 = sep - sw - 1
        if c0 < 0:
            return None
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

    assert np.all(out[:, sep - 1] == BG), "combo requires a full bg lane before the separator"
    return out, sprites, gap_rows


def _unslide_sprites(h, w, sep, sprites, gap_rows):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    for r in gap_rows:
        inp[r, sep] = BG

    free_to_stay = [i for i, (_patch, _r0, c0) in enumerate(sprites) if c0 > 0]
    aligned_i = random.choice(free_to_stay) if free_to_stay and random.random() < ALIGNED_GAP_PROB else -1

    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 == 0 or i == aligned_i:
            delta = 0
        else:
            delta = random.randint(1, c0)
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
    try:
        if not np.array_equal(main(inp), out):
            return None
        metrics = _analyze_input(inp)
    except Exception:
        return None

    if not metrics["lane_clear"]:
        return None
    if metrics["moved_components"] <= 0:
        return None
    if not metrics["blocked_candidate_rows"]:
        return None
    if not metrics["decoy_gap_rows"]:
        return None
    if metrics["diff_vs_d23_only"] <= 0:
        return None
    if metrics["diff_vs_d6_only"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

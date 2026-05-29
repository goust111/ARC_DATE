# Combo variant of 6e453dd6 (index 23)
# combo_spec: D6/1+D7/2
# Name: 分隔线缺口+射线互相阻挡

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and step trace
# - The left half keeps the seed task's geometry: black 4-connected sprites slide right until
#   their right edge is flush with the separator's inner face, then enclosed bg holes touching
#   column `sep-2` nominate candidate emitter rows.
# - D6/1 keeps only the rows whose separator cell is a bg gap. That is the first filter.
# - D7/2 is rendered as endpoint blocking inside each consecutive run of open separator gaps:
#   after a row fires, it claims the farthest reachable endpoint for that run, and the next lower
#   open row must stop one cell earlier. A blocked row breaks the run, so a later open row starts
#   over at full length. This keeps the "later rays are limited by earlier ones" dependency
#   visible while preserving the seed task's horizontal-ray family.
# - Small trace with `right_w = 4` and one hole spanning candidate rows {7,8,9}:
#   old task: all three rows fill four red cells.
#   D6-only: if the separator has gaps on rows {7,8}, then rows 7 and 8 both fill four cells.
#   fused task: rows {7,8} are still the only rows allowed through the separator, but they are
#   processed top-to-bottom, so row 7 gets length 4 and row 8 is shortened to length 3. If row 9
#   is blocked by grey and row 10 is open again, the staircase stops at row 8 and row 10 resets
#   to length 4.
#
# b) Complexity calibration
# - The right pane is narrowed only slightly less than the seed task: width is kept in 3..4.
#   That is enough room to display a full row and at least one visibly shortened row.
# - Each accepted sample must contain:
#   1. at least one consecutive run of open candidate rows, so D7/2 actually shortens something;
#   2. at least one blocked candidate row, so D6/1 actually filters something;
#   3. at least one non-candidate gap row, so the broken separator is not just a copy of the
#      eventual ray rows.
#
# c) Implicit-assumption review of the original main()
# - `_find_separator_col()` originally assumed one full-height grey column. That breaks under D6/1;
#   the new version instead requires all grey separator pixels to stay in one column, while some
#   rows of that column may be bg gaps.
# - Hole detection remains valid because the left-side topology is unchanged.
# - The original `out[r, sep+1:] = red` assumed:
#   1. every candidate row always fires;
#   2. all firing rows have the same maximal length;
#   3. row order does not matter.
#   All three assumptions fail in the combo. The solver must first intersect candidate rows with
#   separator gaps, then paint in top-down order while carrying the current run's remaining
#   rightmost endpoint.
#
# d) Placement legality
# - Sprites still live in disjoint row bands, so horizontal sliding cannot create overlaps.
# - The separator column is reserved for only grey and bg. No black or red is ever placed there.
# - The generator stays output-first: construct the flush left pane and gap pattern, paint the
#   staircase rays, then unslide sprites to derive the input. Inputs therefore always satisfy the
#   solver's assumptions.
#
# e) Visual observability
# - A human can see three coupled clues directly from a few examples:
#   1. only gap rows can cross the separator;
#   2. some gap-aligned candidate rows still produce shorter rays than the row above;
#   3. the shortening propagates only through consecutive open rows and resets after a blocked row.
# - No hidden numeric ordering is used: "top to bottom" is spatially visible, and the collision
#   state is read only from visible consecutive open-row runs.
#
# f) Visual cleanliness
# - The board remains cleanly partitioned: left = black sprites, middle = punctured separator,
#   right = red rays. No extra marker colors or overlapping object families are introduced.
# - Because sprite bands are vertically separated, different emitters do not create ambiguous
#   cross-interference; the only collisions happen within visible consecutive row runs.
#
# g) Post-test degeneration check
# - After implementation we should measure, over generated samples:
#   `blocked_candidate_rows > 0`, `shortened_rows > 0`, `decoy_gap_rows > 0`, and, when there are
#   enough candidate rows, `reset_rows > 0`. If any of these collapses to zero frequently, the
#   generator has regressed to a simpler task.
#
# h) New solver assumptions that the generator must guarantee
# - The separator contains at least one grey cell and at least one bg gap.
# - There is at least one candidate row with a gap, at least one consecutive pair among those open
#   candidate rows, and usually one later open row after a blocked candidate so the reset is visible.
# - The selected consecutive open run length never exceeds the right-pane width, so the collision
#   effect stays visible instead of immediately disappearing.


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


def _trigger_rows_from_left(left: np.ndarray, sep: int, bg: int) -> list[int]:
    h, _ = left.shape
    trigger_col = sep - 2
    trigger_rows = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted({int(r) for r, _ in pts}))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return sorted(trigger_rows)


def _emit_collision_rays(
    out: np.ndarray, active_rows: list[int], sep: int, bg: int, red: int = 2
) -> dict[int, int]:
    ray_lengths: dict[int, int] = {}
    run_end = out.shape[1] - 1
    prev_row = None
    for r in sorted(active_rows):
        if prev_row is None or r != prev_row + 1:
            run_end = out.shape[1] - 1
        end = run_end
        if end >= sep + 1:
            assert np.all(out[r, sep + 1 : end + 1] == bg), f"ray path on row {r} must be background"
            out[r, sep + 1 : end + 1] = red
            ray_lengths[int(r)] = int(end - sep)
            run_end = end - 1
        else:
            ray_lengths[int(r)] = 0
        prev_row = int(r)
    return ray_lengths


def _analyze_active_rows(candidate_rows: list[int], gap_rows: set[int]) -> dict:
    candidate_set = set(candidate_rows)
    active_rows = sorted(r for r in candidate_rows if r in gap_rows)
    active_set = set(active_rows)
    blocked_candidate_rows = sorted(r for r in candidate_rows if r not in gap_rows)
    active_segments = _consecutive_segments(active_rows)
    shortened_rows = [r for r in active_rows if (r - 1) in active_set]
    reset_rows = [r for r in active_rows if (r - 1) not in active_set and any(b < r for b in blocked_candidate_rows)]
    decoy_gap_rows = sorted(r for r in gap_rows if r not in candidate_set)
    return {
        "active_rows": active_rows,
        "blocked_candidate_rows": blocked_candidate_rows,
        "active_segments": active_segments,
        "shortened_rows": shortened_rows,
        "reset_rows": reset_rows,
        "decoy_gap_rows": decoy_gap_rows,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes to a broken grey separator, then let allowed rays shorten each other from top to bottom.

    Input structure:
    - One vertical separator column still divides the board, but it is punctured: grey cells and bg gaps share that column.
    - Black sprites occupy only the left pane, while the right pane starts empty.

    Rule:
    - Shift each black connected component rightward until its right edge is flush with the separator's left side.
    - In the slid left pane, find enclosed bg holes that touch the usual trigger column next to the separator; their rows are candidate emitters.
    - Keep only the candidate rows whose separator cell is a gap, then process each consecutive open run from top to bottom. The first row of a run paints to the right edge; each next row in the same run must stop one cell earlier than the row above, and any blocked row resets the run.

    Key insight: The holes still choose candidate rows, the separator gaps decide which of those rows may cross, and consecutive open rows then compete for the same rightmost endpoint, so later rays in a run become a visible staircase while reopened rows after a block return to full length.
    """
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)
    gap_rows = _separator_gap_rows(g, sep, bg, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
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

    out[:, :sep] = left

    # Step 2: find hole rows that would emit in the original task.
    candidate_rows = _trigger_rows_from_left(left, sep, bg)

    # Step 3: only gap-aligned rows may pass through the separator.
    active_rows = [r for r in candidate_rows if r in gap_rows]

    # Step 4: emit rays top-to-bottom; a row shortens if the row above already emitted.
    _emit_collision_rays(out, active_rows, sep, bg, red=red)
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
RIGHT_W_MIN, RIGHT_W_MAX = 3, 4

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


def _consecutive_segments(rows: list[int]) -> list[list[int]]:
    if not rows:
        return []
    segs = [[int(rows[0])]]
    for r in rows[1:]:
        if int(r) == segs[-1][-1] + 1:
            segs[-1].append(int(r))
        else:
            segs.append([int(r)])
    return segs


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


def _choose_gap_rows(h: int, candidate_rows: list[int], right_w: int) -> set[int] | None:
    candidates = sorted(set(int(r) for r in candidate_rows))
    if len(candidates) < 3:
        return None

    segments = [seg for seg in _consecutive_segments(candidates) if len(seg) >= 2]
    random.shuffle(segments)
    for seg in segments:
        max_open_len = min(len(seg), right_w, len(candidates) - 1)
        if max_open_len < 2:
            continue
        open_lens = list(range(2, max_open_len + 1))
        random.shuffle(open_lens)
        for open_len in open_lens:
            starts = list(range(len(seg) - open_len + 1))
            random.shuffle(starts)
            for start in starts:
                run_rows = set(seg[start : start + open_len])
                blocked_pool = [r for r in candidates if r not in run_rows]
                if not blocked_pool:
                    continue

                extra_choices = [
                    r
                    for r in blocked_pool
                    if all(abs(r - rr) > 1 for rr in run_rows) and any(b < r for b in blocked_pool if b != r)
                ]
                fallback_choices = [r for r in blocked_pool if all(abs(r - rr) > 1 for rr in run_rows)]

                open_rows = set(run_rows)
                if extra_choices:
                    open_rows.add(int(random.choice(extra_choices)))
                elif fallback_choices and len(candidates) >= 4:
                    open_rows.add(int(random.choice(fallback_choices)))

                row_stats = _analyze_active_rows(candidates, open_rows)
                if not row_stats["blocked_candidate_rows"]:
                    continue
                if not row_stats["shortened_rows"]:
                    continue
                if len(candidates) >= 4 and not row_stats["reset_rows"]:
                    continue

                decoy_pool = [r for r in range(h) if r not in candidates]
                if not decoy_pool:
                    continue
                decoy_count = random.randint(1, min(3, len(decoy_pool)))
                decoys = set(random.sample(decoy_pool, decoy_count))

                gap_rows = set(open_rows) | decoys
                if 0 < len(gap_rows) < h and _analyze_active_rows(candidates, gap_rows)["decoy_gap_rows"]:
                    return gap_rows
    return None


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
    gap_rows = _choose_gap_rows(h, candidate_rows, right_w=w - sep - 1)
    if gap_rows is None:
        return None

    for r in gap_rows:
        out[r, sep] = BG

    row_stats = _analyze_active_rows(candidate_rows, gap_rows)
    ray_lengths = _emit_collision_rays(out, row_stats["active_rows"], sep, BG, red=RED)

    full_len = w - sep - 1
    shortened_rows = [r for r, ln in ray_lengths.items() if 0 < ln < full_len]
    reset_rows = [r for r, ln in ray_lengths.items() if ln == full_len and r in row_stats["reset_rows"]]
    if not row_stats["blocked_candidate_rows"] or not shortened_rows or not row_stats["decoy_gap_rows"]:
        return None

    metrics = {
        "candidate_rows": candidate_rows,
        "gap_rows": sorted(gap_rows),
        "active_rows": row_stats["active_rows"],
        "blocked_candidate_rows": row_stats["blocked_candidate_rows"],
        "shortened_rows": shortened_rows,
        "reset_rows": reset_rows,
        "decoy_gap_rows": row_stats["decoy_gap_rows"],
        "active_segments": row_stats["active_segments"],
        "ray_lengths": ray_lengths,
    }
    return out, sprites, gap_rows, metrics


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
    out, sprites, gap_rows, metrics = rendered

    inp = _unslide_sprites(h, w, sep, sprites, gap_rows)
    if inp is None:
        return None

    try:
        got = main(inp)
        if not np.array_equal(got, out):
            return None
    except Exception:
        return None

    full_len = w - sep - 1
    if not metrics["blocked_candidate_rows"]:
        return None
    if not any(0 < ln < full_len for ln in metrics["ray_lengths"].values()):
        return None
    if not metrics["decoy_gap_rows"]:
        return None
    if len(metrics["candidate_rows"]) >= 4 and not metrics["reset_rows"]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

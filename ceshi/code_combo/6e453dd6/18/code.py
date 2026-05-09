# Combo variant of 6e453dd6 (index 18)
# combo_spec: D3/2+D8/1
# Name: 限长射线+平移轨迹

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        sys.path.append(str(_arc_utils.parent))
        break

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# a_concretization_and_trace:
#   - Preserved base geometry:
#     * a single full-height grey separator splits the grid
#     * black sprites originate on the left and slide horizontally until flush with sep-1
#     * only enclosed bg holes that touch sep-2 after sliding are emitters
#   - D3/2 choice concretized here:
#     * the bounded ray length is the slid sprite width, not hole area
#     * reason: width is directly visible in the final black sprite, repeats across several rows,
#       and matches the provided single-variant reference for this task
#   - D8/1 concretized here:
#     * the trail is the union of every intermediate sprite position before the final flush pose
#     * because motion is purely horizontal, this produces a continuous blue left-to-right trace
#   - Trace example:
#     * a hollow 3x4 sprite starts at cols 1..4 and slides right by 3 into cols 4..7
#     * original task output: old pose disappears; rows of the trigger hole paint red to the border
#     * combo output:
#       1. cols 1..6 visited by the moving black pixels become blue wherever the final sprite does not end
#       2. the final sprite stays black at cols 4..7
#       3. trigger rows emit only 4 red cells on the right because the slid sprite width is 4
#     * so both left and right halves visibly change relative to the base task
# b_complexity_calibration:
#   - Width is kept in a small visible range (3..5 typical), so several emitters can share counts.
#   - Right width is widened to 5..7, making "short of the border" common rather than accidental.
#   - The trail color is a single dedicated color and the motion remains one-directional, so the
#     added state is visible without creating many case splits.
# c_hidden_assumptions_from_original_main:
#   - `left[left == black] = bg` is no longer valid: clearing the old pose would erase the trail.
#   - Hole detection on the displayed left pane would also become wrong, because blue trail pixels
#     can split bg regions. The solver therefore constructs a separate moved-black-only canvas for
#     hole detection while drawing the visible trail on the displayed canvas.
#   - `out[r, sep + 1 :] = red` assumed all rays reach the edge. Under D3/2 this must become a
#     per-emitter bounded segment using the enclosing moved component width.
#   - Matching a hole to its enclosing component now matters. The solver stores each slid component's
#     bbox, pixel set, and width, then asserts that every triggering hole has exactly one enclosure.
# d_placement_legality:
#   - Generator keeps sprites in disjoint row bands with at least one bg row between bands.
#   - That guarantees no overlap in the input, no overlap after all sprites slide flush, and no
#     cross-sprite ambiguity when associating holes with enclosing components.
#   - Emitter width is capped at `right_w - 1`, guaranteeing every bounded ray stops before the edge.
#   - Inputs are rejected unless at least one sprite moves by >=2 columns, so the continuous trail is
#     not visually reducible to a simple recolored source footprint.
# e_visual_observability:
#   - A human sees two coupled signals:
#     * left pane: blue marks the exact horizontal travel corridor of each moving sprite
#     * right pane: red length mirrors the visible width of the flush black emitter
#   - Because emitters and distractors share the same motion/trail behavior but only true holes fire,
#     the puzzle still requires understanding enclosure rather than reading a fixed row pattern.
# f_visual_cleanliness:
#   - Blue is confined to the left pane, red to the right pane, grey stays the separator, black stays
#     the final sprite color. The regions are therefore readable instead of being overlaid in one area.
#   - The trail is continuous but still sparse enough because sprites occupy disjoint row bands.
# g_change_inferability:
#   - Only one extra color is introduced: blue(1), always meaning "visited during horizontal slide".
#   - The red-length rule is structural, not color-id-based: it follows sprite width visible in the
#     same output image, so the bounded-ray rule can be inferred from a few examples.
# h_generator_metrics_against_degeneration:
#   - The generator computes fusion metrics and rejects samples unless:
#     * there is visible blue trail
#     * some blue trail also appears on cells that were bg in the input, proving the path extends
#       beyond a simple recolored source footprint
#     * at least one sprite has a multi-step path (delta >= 2)
#     * at least one red row is strictly shorter than the right pane
#     * the combo output differs from the plain original output on the same input
# i_new_solver_assumptions_checked_against_generator:
#   - Solver assumption: each triggering hole belongs to one moved component only.
#     Generator guarantee: one connected sprite per band and disjoint vertical bands.
#   - Solver assumption: right-pane target cells are background before painting rays.
#     Generator guarantee: input right pane is pure bg and no other process writes there.
#   - Solver assumption: bounded rays fit before the border.
#     Generator guarantee: emitter width <= right_w - 1.


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


def _component_info_after_slide(comp, delta: int):
    pts = {(int(r), int(c + delta)) for r, c in comp}
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return {
        "pts": pts,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
        "width": max(cols) - min(cols) + 1,
    }


def _component_width_for_hole(hole, comp_infos) -> int:
    hole_rows = [int(r) for r, _ in hole]
    hole_cols = [int(c) for _, c in hole]
    hr0, hc0, hr1, hc1 = min(hole_rows), min(hole_cols), max(hole_rows), max(hole_cols)

    matches = []
    for info in comp_infos:
        r0, c0, r1, c1 = info["bbox"]
        if not (r0 <= hr0 <= hr1 <= r1 and c0 <= hc0 <= hc1 <= c1):
            continue
        if any(
            (int(r + dr), int(c + dc)) in info["pts"]
            for r, c in hole
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            matches.append(info)
    assert len(matches) == 1, f"expected unique enclosing component for hole, got {len(matches)}"
    return int(matches[0]["width"])


def _paint_component_trail(canvas: np.ndarray, comp, delta: int, trail_color: int) -> None:
    for step in range(delta):
        for r, c in comp:
            canvas[int(r), int(c + step)] = int(trail_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the grey divider, keep a blue motion trail, then fire width-limited red rays.

    Input structure:
    - A single full-height grey (5) separator column splits the board into a left sprite pane and an empty right pane.
    - The left pane contains black connected sprites on one background color; some sprites contain enclosed background holes.

    Rule:
    - Translate every black connected component horizontally until its rightmost black pixel becomes flush with the separator.
    - Leave a blue trail on every left-pane cell visited during that slide, then draw the final flush sprite in black on top.
    - In the moved black layout, any enclosed background hole that touches the separator-side inner column emits red on its rows, but only for as many cells as that moved sprite is wide.

    Key insight: The same separator-adjacent hole rows still decide which rows fire, but the output now also exposes the travel corridor on the left, while the right-side ray length is encoded by the width of the flush black emitter rather than extending to the border.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    blue = 1
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    left_in = g[:, :sep]
    left_out = left_in.copy()
    left_out[left_out == black] = bg
    moved_only = np.full(left_in.shape, bg, dtype=int)

    comps = _components4(left_in == black)
    target_right = sep - 1
    comp_infos = []
    for comp in comps:
        max_c = max(int(c) for _, c in comp)
        min_c = min(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"

        _paint_component_trail(left_out, comp, delta, blue)

        info = _component_info_after_slide(comp, delta)
        assert info["width"] == max_c - min_c + 1
        for r, c in info["pts"]:
            assert 0 <= c < sep, f"shifted pixel {(r, c)} left the pane"
            left_out[r, c] = black
            moved_only[r, c] = black
        comp_infos.append(info)

    out[:, :sep] = left_out

    holes = []
    for pts in _components4(moved_only == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_col = sep - 2
    for pts in holes:
        if not any(int(c) == trigger_col for _, c in pts):
            continue
        ray_len = _component_width_for_hole(pts, comp_infos)
        ray_end = min(w, sep + 1 + ray_len)
        trigger_rows = sorted({int(r) for r, _ in pts})
        for r in trigger_rows:
            assert np.all(out[r, sep + 1 : ray_end] == bg), f"target ray segment on row {r} must be empty"
            out[r, sep + 1 : ray_end] = red

    return out


BG = 6
BLACK = 0
BLUE = 1
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 11, 16
SEP_MIN, SEP_MAX = 5, 10
RIGHT_W_MIN, RIGHT_W_MAX = 5, 7

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.2

MAX_RETRIES = 400
MIN_TRAIL_PIXELS = 10
MIN_DIFF_FROM_PLAIN = 10


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
    assert sh >= 3 and max_sw >= 3
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
    return h, w, sep, right_w


def _choose_emitter_flags(h):
    valid_counts = [
        n
        for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
        if _min_rows_needed(_min_emitters_for(n), n) <= h and _min_emitters_for(n) < n
    ]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = min_em
    for k in range(min_em + 1, n):
        if _min_rows_needed(k, n) <= h:
            max_em = k
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _trigger_rows_from_patch(patch):
    rows = []
    for r in range(patch.shape[0]):
        if patch.shape[1] >= 2 and patch[r, -1] == BLACK and patch[r, -2] == BG:
            rows.append(r)
    return rows


def _trail_patch(patch):
    return np.where(patch == BLACK, BLUE, BG)


def _paint_patch_trail(out: np.ndarray, patch: np.ndarray, r0: int, start_c: int, delta: int) -> None:
    trail = _trail_patch(patch)
    for step in range(delta):
        paste(out, trail, r0, start_c + step, transparent=BG)


def _render_flush_sprites(h, sep, right_w, bands, flags):
    sprites = []
    shared_max_sw = min(SPRITE_W_MAX, sep - 1, right_w - 1)
    assert shared_max_sw >= 3, "need enough width for emitters and visible bounded rays"
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, shared_max_sw) if is_emitter else _distractor_patch(sh, shared_max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        sprites.append({
            "patch": patch,
            "r0": r0,
            "c0": c0,
            "is_emitter": bool(is_emitter),
            "width": sw,
        })
    return sprites


def _unslide_sprites(h, w, sep, sprites):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    deltas = []
    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, sprite in enumerate(sprites):
        patch, r0, c0 = sprite["patch"], sprite["r0"], sprite["c0"]
        if c0 < 1:
            return None, None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
        deltas.append(delta)
    return inp, deltas


def _render_expected_output(h, w, sep, sprites, deltas):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    for sprite, delta in zip(sprites, deltas):
        if delta > 0:
            _paint_patch_trail(out, sprite["patch"], sprite["r0"], sprite["c0"] - delta, delta)

    for sprite in sprites:
        patch, r0, c0 = sprite["patch"], sprite["r0"], sprite["c0"]
        paste(out, patch, r0, c0, transparent=BG)
        if sprite["is_emitter"]:
            ray_end = sep + 1 + sprite["width"]
            for lr in _trigger_rows_from_patch(patch):
                out[r0 + lr, sep + 1 : ray_end] = RED
    return out


def _render_plain_output(h, w, sep, sprites):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR
    for sprite in sprites:
        patch, r0, c0 = sprite["patch"], sprite["r0"], sprite["c0"]
        paste(out, patch, r0, c0, transparent=BG)
        if sprite["is_emitter"]:
            for lr in _trigger_rows_from_patch(patch):
                out[r0 + lr, sep + 1 :] = RED
    return out


def _fusion_metrics(inp: np.ndarray, expected: np.ndarray, plain: np.ndarray, sep: int, deltas) -> dict[str, int]:
    right = expected[:, sep + 1 :]
    trail_pixels = int(np.count_nonzero(expected[:, :sep] == BLUE))
    extended_trail_pixels = int(np.count_nonzero((expected[:, :sep] == BLUE) & (inp[:, :sep] != BLACK)))
    moved_sprites = sum(delta > 0 for delta in deltas)
    long_moves = sum(delta >= 2 for delta in deltas)
    short_ray_rows = int(sum(np.any(row == RED) and np.any(row != RED) for row in right))
    diff_vs_plain = int(np.count_nonzero(expected != plain))
    return {
        "trail_pixels": trail_pixels,
        "extended_trail_pixels": extended_trail_pixels,
        "moved_sprites": int(moved_sprites),
        "long_moves": int(long_moves),
        "short_ray_rows": short_ray_rows,
        "diff_vs_plain": diff_vs_plain,
    }


def _gen_attempt() -> np.ndarray | None:
    h, w, sep, right_w = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    sprites = _render_flush_sprites(h, sep, right_w, bands, flags)
    inp, deltas = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    expected = _render_expected_output(h, w, sep, sprites, deltas)
    plain = _render_plain_output(h, w, sep, sprites)
    metrics = _fusion_metrics(inp, expected, plain, sep, deltas)

    if metrics["trail_pixels"] < MIN_TRAIL_PIXELS:
        return None
    if metrics["extended_trail_pixels"] <= 0:
        return None
    if metrics["moved_sprites"] <= 0 or metrics["long_moves"] <= 0:
        return None
    if metrics["short_ray_rows"] <= 0:
        return None
    if metrics["diff_vs_plain"] < MIN_DIFF_FROM_PLAIN:
        return None

    try:
        if BLUE in set(map(int, inp.flatten())):
            return None
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

# Combo variant of 6e453dd6 (index 14)
# combo_spec: D11/0+D13/0
# Name: 单精灵全发射

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# This fusion combines:
# - D11/0: remove distractors so every sprite is a true emitter.
# - D13/0: reduce the count to exactly one sprite.
#
# concreteization_and_trace:
# - Original task: 2-5 black components on the left, only a subset are emitters. After
#   sliding all components flush to the grey separator, only enclosed holes that reach
#   the inner face near the separator produce red rays on the right.
# - Fused task: there is exactly one black component, and it is always an emitter. A
#   typical input therefore shows one shell-like object somewhere left of the separator.
#   Example trace: if the component's right edge is at col 2 and the separator is at
#   col 7, main slides the whole component right by 4 so its right edge lands at col 6.
#   Its enclosed magenta cavity then has cells on col 5 (= sep-2), so the rows of that
#   cavity paint red across cols 8..w-1.
# - Difference from the original: there is no multi-object discrimination and no
#   emitter-vs-distractor ambiguity. The only remaining reasoning is the core mechanism:
#   horizontal translation first, then hole detection, then ray emission.
#
# complexity_calibration:
# - The fusion intentionally simplifies the task, but it still leaves visible structure:
#   the sprite size, height, width, vertical position, left shift amount, and whether it
#   is a plain shell or a split shell all vary. The right-side red pattern also varies by
#   the cavity height. This keeps samples from collapsing into one repeated picture.
# - To keep the "slide" step observable, the generator always unslides the sprite by at
#   least one column. That avoids accidental examples where input already equals output on
#   the left half.
#
# implicit_assumption_review_for_main:
# - _find_separator_col assumes exactly one full-height separator column. The generator
#   still creates exactly one such column.
# - _background_color assumes the border majority is the background. The generator keeps
#   the outer border magenta except for the separator column, so the inference remains
#   valid.
# - Step 1 in main shifts each black component independently. With one component this is
#   trivially safe; no component-merging or row-band separation logic is needed anymore.
# - Step 2 classifies bg components not touching the left-pane boundary as holes. The
#   generator therefore places the sprite away from the top and bottom borders and uses
#   only closed-shell emitter shapes, so the triggering cavity is always a true enclosed
#   hole.
# - Step 3 looks for hole cells on col sep-2 and expects the right side to be blank
#   before painting. The emitter patch is constructed to guarantee at least one such row,
#   and the input right side is always pure background.
#
# placement_legality:
# - Output-first generation is still the easiest safe construction. First place one valid
#   emitter flush against the separator in the output state, then shift it left to create
#   the input.
# - The sprite width is capped by sep-1, so the flush placement fits inside the left pane.
#   The unslide delta is chosen in [1, c0], where c0 is the flush left column, so the
#   shifted input also stays fully inside the left pane.
# - The sprite start row r0 satisfies 1 <= r0 and r0 + sh <= h - 1, leaving a full
#   background margin above and below the component. That prevents the intended hole from
#   touching the subgrid boundary.
#
# visual_observability_and_cleanliness:
# - The change is easy to read from examples: one lone black sprite slides to the grey
#   wall, and every red row on the right lines up with the internal opening of that same
#   sprite. There are no extra objects competing for attention.
# - The right pane contains only solid horizontal red runs on a magenta field, so the
#   output stays visually clean and unlayered.
#
# color_inference_and_non_degeneracy:
# - Colors keep their original roles: black=sprite, grey=separator, magenta=background,
#   red=emitted ray. No hidden use of numeric color ids is introduced.
# - This generator does not degenerate back to the original distribution: every sample has
#   exactly one sprite, whereas the original spec required 2-5 sprites. A simple metric
#   such as (# black components, # trigger rows) should therefore show a constant first
#   coordinate of 1 and a strictly positive second coordinate across generated samples.
#
# new_post-edit_main_assumptions:
# - main itself is almost unchanged, so it does not add new geometric assumptions beyond
#   the original solver. The only effective new assumption is dataset-level: there is one
#   component and it emits. generate_input enforces this by construction and by oracle
#   verification against main.


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
    Slide the lone black emitter against the grey divider, then project red rays from its sealed inner pocket.

    Input structure:
    - One full-height grey (5) separator column splits the board into a left sprite pane and a right empty pane.
    - Exactly one black connected component sits somewhere on the left over a magenta background, and the right pane starts fully magenta.

    Rule:
    - Translate the black component horizontally right until its rightmost black cells become flush with the column just left of the separator.
    - In that shifted left pane, find enclosed magenta regions; any enclosed region that reaches the inner column next to the separator contributes all of its rows as ray rows.
    - Paint those rows red across the entire right pane, while preserving the shifted black sprite and the grey separator.

    Key insight: With only one sprite and no distractors, the puzzle reduces to the essential causal chain: move the shell to the wall, inspect which rows belong to the sealed cavity touching that wall, and copy those rows as rightward red beams.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    left = out[:, :sep].copy()
    comps = _components4(left == black)
    assert len(comps) == 1, f"single-emitter fusion expects exactly one left-side component, got {len(comps)}"

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

    assert trigger_rows, "single-emitter fusion expects at least one triggering hole row"
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    return out


BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

SPRITE_H_MIN, SPRITE_H_MAX = 3, 8
SPRITE_W_MAX = 6
SPLIT_PROB = 0.45
MAX_RETRIES = 300
_NEXT_STYLE = 0


def _trigger_rows_from_patch(patch: np.ndarray) -> list[int]:
    sh, sw = patch.shape
    if sw < 2:
        return []
    return [r for r in range(sh) if patch[r, sw - 1] == BLACK and patch[r, sw - 2] == BG]


def _build_split_emitter(patch: np.ndarray, sh: int, sw: int, left_wall: int, right_wall: int, mini_w: int) -> None:
    """Fill patch with a connected split-shell emitter that still has one triggering cavity."""
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


def _emitter_patch(sh: int, max_sw: int, prefer_split: bool = False) -> np.ndarray:
    """Build the single sprite: always one connected closed-shell emitter."""
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    min_sw = hole_w + 2
    if prefer_split:
        min_sw = max(min_sw, 4)
    if min_sw > max_sw:
        return _emitter_patch(sh, max_sw, prefer_split=False)

    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and (prefer_split or random.random() < SPLIT_PROB):
        _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK

    assert _components4(patch == BLACK)
    assert len(_components4(patch == BLACK)) == 1, "emitter patch must stay connected"
    assert _trigger_rows_from_patch(patch), "emitter patch must contain at least one triggering hole row"
    return patch


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_sprite_band(h: int, require_split: bool = False) -> tuple[int, int] | None:
    max_sh = min(SPRITE_H_MAX, h - 2)
    min_sh = max(SPRITE_H_MIN, 5) if require_split else SPRITE_H_MIN
    if max_sh < min_sh:
        return None
    sh = random.randint(min_sh, max_sh)
    r0 = random.randint(1, h - sh - 1)
    return r0, sh


def _render_output(h: int, w: int, sep: int, prefer_split: bool = False):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    band = _choose_sprite_band(h, require_split=prefer_split)
    if band is None:
        return None
    r0, sh = band
    max_sw = min(SPRITE_W_MAX, sep - 1)
    patch = _emitter_patch(sh, max_sw, prefer_split=prefer_split)
    sw = patch.shape[1]
    c0 = sep - sw
    paste(out, patch, r0, c0, transparent=BG)

    trigger_rows = _trigger_rows_from_patch(patch)
    for lr in trigger_rows:
        out[r0 + lr, sep + 1 :] = RED

    return out, (patch, r0, c0), trigger_rows


def _unslide_sprite(h: int, w: int, sep: int, sprite):
    patch, r0, c0 = sprite
    if c0 < 1:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    delta = random.randint(1, c0)
    paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt(prefer_split: bool = False) -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    rendered = _render_output(h, w, sep, prefer_split=prefer_split)
    if rendered is None:
        return None
    out, sprite, trigger_rows = rendered
    inp = _unslide_sprite(h, w, sep, sprite)
    if inp is None:
        return None

    try:
        if len(_components4(inp[:, :sep] == BLACK)) != 1:
            return None
        if not trigger_rows:
            return None
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    global _NEXT_STYLE
    prefer_split = bool(_NEXT_STYLE % 2)
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(prefer_split=prefer_split)
        if grid is not None:
            _NEXT_STYLE += 1
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

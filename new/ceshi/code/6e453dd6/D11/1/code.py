# Variant of 6e453dd6
# Dimension: D11 — 发射精灵与干扰精灵的比例 (meta)
# Change: >=60%的精灵为射线发射体（emitter），其余为干扰项（distractor）。干扰精灵形状类似但不满足封闭洞条件... → 仅一个精灵为发射体（多数为干扰项）

import random

import numpy as np

from arc_utils import components_mask, paste

# Design analysis for the D11 inversion variant
# ---------------------------------------------
# concrete_behavior:
#   - The geometric rule is unchanged: all black components slide right until they touch the
#     grey separator, then enclosed background holes that touch the inner face of the shell
#     launch red rays to the right pane.
#   - The variant changes which sprites satisfy that condition. Instead of most sprites being
#     emitters, each generated grid has exactly one true emitter and at least two distractors.
#   - Example trace after sliding:
#       original-style flags: [emitter, emitter, distractor, emitter]
#       variant flags:        [distractor, emitter, distractor, distractor]
#     The solver still scans all holes, but only the single true shell contributes rows whose
#     hole pixels reach column sep-2. The difference is statistical and visual, not procedural.
# complexity_calibration:
#   - The sprite count is restricted to 3-5 so one emitter is visibly a minority.
#   - Distractors are biased toward near-miss shapes, especially "false-hole" shells that do
#     contain an enclosed pocket but keep a double right wall, so the pocket stops at sep-3.
#   - This preserves the original search problem while making the filtering step harder.
# implicit_assumptions_review:
#   - `main()` moves each 4-connected black component independently. The generator therefore
#     keeps sprites in disjoint row bands so they never merge before or after sliding.
#   - Hole detection treats any background component touching the left pane border as non-hole.
#     Open C/L distractors remain connected to exterior background, while false-hole distractors
#     are fully closed but deliberately blocked away from sep-2.
#   - Ray rows are asserted unique. With exactly one emitter and non-overlapping row bands,
#     no two triggering sprites can compete for the same output rows.
# placement_legality:
#   - Every patch is built to fit strictly left of the separator once flush, and unsliding only
#     translates patches left within that pane.
#   - The right side of every input stays pure background so `main()` can safely paint rays.
# visual_observability_and_tidiness:
#   - The inversion is visible because most grids contain several black shapes yet only one of
#     them produces red rows in the output.
#   - The distractors remain shape-similar to emitters, but the output stays clean because row
#     bands keep sprites separated and rays only appear in the right pane.


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


def _trigger_rows_from_left(left: np.ndarray, bg: int) -> list[int]:
    """Return rows of enclosed bg holes that touch the column just inside the separator."""
    h, sep = left.shape
    trigger_col = sep - 2
    rows = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if any(c == trigger_col for _, c in pts):
            rows.extend(sorted({r for r, _ in pts}))
    assert len(rows) == len(set(rows)), f"trigger rows must be unique, got {rows}"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the black sprites into the grey divider, then let the true closed nozzle fire red rays.

    Input structure:
    - A full-height grey (5) separator column splits the grid into a left sprite pane and a right empty pane.
    - Several black sprites sit on the left over a uniform background; in this variant only one of them is a real emitter and the others are near-miss distractors.

    Rule:
    - Translate every black connected component rightward until its right edge becomes flush with the separator.
    - In the slid arrangement, locate enclosed background pockets whose cells touch the inner face of the separator-side wall.
    - For each row occupied by such a qualifying pocket, paint a red ray across the right side of that row.

    Key insight: The answer is not based on counting sprites but on isolating the single shell whose enclosed pocket reaches the separator-facing side after the slide; lookalikes either stay open to the outside or keep their pocket one column too far left.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

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

    for r in _trigger_rows_from_left(left, bg):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 3, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

SPLIT_PROB = 0.4
FLUSH_PROB = 0.35
FALSE_HOLE_WEIGHT = 3
MAX_RETRIES = 300


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_rows_needed(n_total: int) -> int:
    """One emitter needs height 3, each distractor needs height 2, plus gaps and margins."""
    return 3 + (n_total - 1) * 2 + (n_total - 1) + 2


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


def _trigger_rows_in_patch(patch: np.ndarray) -> list[int]:
    """Evaluate a patch in the same flush-to-separator geometry used by the solver."""
    ph, pw = patch.shape
    left = np.full((ph + 2, pw + 1), BG, dtype=int)
    left[1:-1, 1:] = patch
    return [r - 1 for r in _trigger_rows_from_left(left, BG)]


def _assert_single_component(patch: np.ndarray) -> None:
    comps = _components4(patch == BLACK)
    assert len(comps) == 1, "sprite patch must be a single 4-connected component"


def _emitter_patch(sh, max_sw):
    """Closed shell whose pocket reaches the separator-facing inner wall when flush."""
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

    _assert_single_component(patch)
    assert _trigger_rows_in_patch(patch), "emitter must expose at least one trigger row"
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


def _false_hole_distractor_patch(sh, max_sw):
    """Closed shell with a pocket that stops one column short of the triggering position."""
    sw = random.randint(5, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, 0] = BLACK
    patch[1:-1, -1] = BLACK
    patch[1:-1, -2] = BLACK
    return patch


def _distractor_patch(sh, max_sw):
    """Near-miss sprite that never exposes a hole at the separator-facing inner wall."""
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh >= 3 and max_sw >= 5:
        choices.extend(["false_hole"] * FALSE_HOLE_WEIGHT)
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        patch = np.full((sh, sw), BLACK, dtype=int)
    elif kind == "false_hole":
        patch = _false_hole_distractor_patch(sh, max_sw)
    else:
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

    _assert_single_component(patch)
    assert not _trigger_rows_in_patch(patch), "distractor must not mimic a true emitter row"
    return patch


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_emitter_flags(h):
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(n) <= h]
    n = random.choice(valid_counts)
    flags = [False] * n
    flags[random.randrange(n)] = True
    return flags


def _render_output(h, w, sep, bands, flags):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    emitter_count = 0
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            emitter_count += 1
            for lr in _trigger_rows_in_patch(patch):
                out[r0 + lr, sep + 1 :] = RED
        sprites.append((patch, r0, c0, is_emitter))

    assert emitter_count == 1, f"expected exactly one emitter, got {emitter_count}"
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0, _is_emitter) in enumerate(sprites):
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
    raise RuntimeError("failed to generate a valid 6e453dd6 D11 input")

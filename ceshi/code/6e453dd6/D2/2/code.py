# Variant of 6e453dd6
# Dimension: D2 — 射线发射路径 (path)
# Change: straight_ray（直线射线）——从封闭洞所在行向右水平发射红色射线至网格右边缘... → 射线从洞口向四个方向发射

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the D2 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     original: every triggering hole row paints a horizontal red strip on the right side
#     variant: every triggering hole row still projects through the separator-facing hole,
#       but the visible emission point is moved one extra cell into the right chamber so
#       that the ray can branch in four directions. The horizontal branch still fills the
#       whole right-side row; the new vertical branch fills one fixed right-side column.
#       The left branch therefore runs from that launch cell back toward the separator and
#       stops there, which is why the horizontal branch still covers cols sep+1..w-1.
#   concretization:
#     - The trigger condition is unchanged: after sliding, only enclosed bg holes that
#       touch col sep-2 are emitters.
#     - For any triggering row r, the launch point is (r, min(sep+2, w-1)).
#     - From that point, the union of four rays is:
#         right: row r to the outer right edge
#         left: row r back to col sep+1, stopping before the grey separator
#         up: the launch column upward to row 0
#         down: the same launch column downward to row h-1
#       Visually this is a cross inside the right chamber. Because the horizontal left+right
#       union spans the full right-side row, the only visible delta from the original task
#       is the added full-height vertical red bar.
#   behavior_trace:
#     original_example:
#       - Suppose a slid emitter has a hole touching sep-2 on rows 5..7.
#       - Original output paints rows 5,6,7 red across cols sep+1..w-1.
#     variant_example:
#       - The same rows 5,6,7 still paint those horizontal strips.
#       - In addition, all three crosses share the same launch column sep+2 (or w-1 when
#         the right chamber is only two cells wide), so the union adds a full red column
#         on that right-side launch column.
#       - This makes the new path rule observable on every emitting example while leaving
#         the emitter-detection logic intact.
#   complexity_calibration:
#     - Only one extra structural degree of freedom is introduced: the common launch column.
#       This keeps the mapping easy to infer from 2-4 examples.
#     - The right chamber keeps width 2-4, so the cross never becomes visually noisy.
#     - The generator still uses 2-5 disjoint sprite bands, preserving repeated evidence.
#   assumption_audit_main:
#     - Finding the separator column and inferring the background still rely on the input,
#       where the right side is pure background. These assumptions remain valid.
#     - Sliding black components is unchanged and still assumes left-side components do not
#       overlap after being moved flush to the separator. The generator already guarantees
#       disjoint row bands, so horizontal sliding cannot cause collisions.
#     - Hole detection still assumes emitters are closed bg regions not touching the left
#       subgrid boundary. This remains the emitter criterion.
#     - The original assertion that each triggered row was untouched before painting is no
#       longer sufficient, because vertical rays can overlap horizontal rays. The variant
#       therefore only asserts that target cells are bg or already red.
#     - The new launch column assumes right_w >= 2 so that sep+2 is inside the right pane
#       in the usual case. The generator keeps the original right-side width range 2-4, so
#       this is guaranteed.
#   placement_legality:
#     - Sprites still occupy disjoint row bands with at least one bg row between them.
#     - The right side starts as all bg in the input and is only painted during output
#       construction, so no red/black overlap can occur there.
#     - Using a single shared launch column prevents competing vertical rays from creating
#       ambiguous patterns; overlaps collapse cleanly to red.
#   observability_and_cleanliness:
#     - The added vertical bar is a large, stable visual cue and is easy to compare against
#       the original row-only projection behavior.
#     - Colors stay minimal: black sprites, grey separator, magenta background, red rays.
#       No hidden color ordering or numeric-color logic is used.


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


def _trigger_rows_from_left(left: np.ndarray, h: int, sep: int, bg: int) -> list[int]:
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
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return trigger_rows


def _launch_col(sep: int, w: int) -> int:
    launch = min(sep + 2, w - 1)
    assert launch > sep, "right pane must contain at least one launch cell"
    return int(launch)


def _paint_cross_rays(out: np.ndarray, sep: int, trigger_rows: list[int], bg: int, red: int) -> None:
    if not trigger_rows:
        return
    h, w = out.shape
    launch = _launch_col(sep, w)

    col_vals = out[:, launch]
    assert np.all((col_vals == bg) | (col_vals == red)), "launch column must be free before emission"
    out[:, launch] = red

    for r in trigger_rows:
        seg = out[r, sep + 1 :]
        assert np.all((seg == bg) | (seg == red)), f"right half of row {r} must be free before ray emission"
        out[r, sep + 1 :] = red


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to a grey separator, then emit four-direction crosses from enclosed holes.

    Input structure:
    - A full-height grey (5) separator column divides the grid into a left sprite area and a narrow right chamber.
    - Black shapes occupy the left side on a uniform background color, while the right chamber starts empty.

    Rule:
    - Shift every left-side black connected component horizontally until its right edge becomes flush with the separator.
    - In that slid arrangement, detect enclosed background holes whose cells reach the inner trigger column one step left of the black wall next to the separator.
    - For every row touched by such a hole, emit a red cross into the right chamber: the horizontal arm spans the whole right-side row, and a shared vertical arm rises and falls through the launch column.

    Key insight: The enclosing hole test is unchanged from the original task; the variant only changes the ray path after activation, turning each row-wise outlet into a cross-shaped projection in the right chamber.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

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

    trigger_rows = _trigger_rows_from_left(left, h, sep, bg)
    _paint_cross_rays(out, sep, trigger_rows, bg, red)

    return out


# --------------- generator (D2 variant) ---------------

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
        i = random.choice(choices)
        heights[i] += 1

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
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_rows_needed(_min_emitters_for(n), n) <= h]
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
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    trigger_rows = set()
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            for lr in range(sh):
                if patch[lr, -1] == BLACK and patch[lr, -2] == BG:
                    trigger_rows.add(r0 + lr)
        sprites.append((patch, r0, c0))

    _paint_cross_rays(out, sep, sorted(trigger_rows), BG, RED)
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

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
    raise RuntimeError("failed to generate a valid 6e453dd6 D2 input")

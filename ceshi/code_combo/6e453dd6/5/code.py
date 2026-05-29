# Combo variant of 6e453dd6 (index 5)
# combo_spec: D7/0+D25/0
# Name: 右侧障碍物阻挡射线

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis
# ----------------
# spec:
#   variant_core:
#     - The left pane keeps the original mechanics: slide each black component right until it becomes flush with the
#       grey separator, then detect enclosed background holes whose cells reach the inner face at `sep-2`.
#     - The fused change is entirely on the right pane: it is no longer blank in the input. Instead it already contains
#       colored obstacle pixels, and each emitted red ray stops immediately before the first obstacle on its row.
#     - This combines two dependencies in one inference chain: the left pane decides whether a row fires at all, while
#       the right pane decides the ray's destination by nearest-collision logic.
#   concrete_behavior_and_trace:
#     - Example row after sliding an emitter:
#       left side becomes `... hole at sep-2 | black wall at sep-1 | grey sep`.
#       Original task: the output paints every cell on the right through the edge.
#       Fused task: if the input right row is `bg bg orange bg cyan`, the ray paints only the two bg cells before the
#       first obstacle and leaves `orange bg cyan` unchanged. A non-trigger row with the same obstacle pattern emits
#       nothing, so the learner must use both halves together.
#   complexity_calibration:
#     - The right pane is widened to 4-6 columns so stopping before an obstacle is visible instead of collapsing into
#       a one-cell difference.
#     - Obstacles are not a one-off special marker: the generator chooses 2-3 reusable interior obstacle lanes and
#       places decoys on non-trigger rows too. This prevents the task from degenerating into "find the single colored
#       pixel and stop there" while keeping the right-side vocabulary small and repeated.
#     - Trigger rows always have an interior first obstacle at `sep+2..w-2`, so the blocking behavior is active in
#       every generated sample and differs from the original full-width ray.
#   implicit_assumptions_review:
#     - Sliding black components still assumes all black pixels lie strictly left of the separator; the generator keeps
#       that invariant.
#     - Hole detection still assumes enclosed holes are bg components not touching the left-pane border; unchanged.
#     - The original assertion that the whole right row is bg before painting is no longer valid because obstacles are
#       pre-existing right-side content. The solver must only require bg between `sep+1` and the first obstacle.
#     - `_background_color()` infers the bg color from the border, so obstacles are kept off the top border, bottom
#       border, and outer-right border. This preserves the original bg inference logic.
#   placement_legality:
#     - Sprite bands remain row-disjoint, so trigger rows from different sprites cannot overlap.
#     - Obstacles only occupy right-pane cells in interior columns and interior rows. They never touch the separator,
#       overwrite left sprites, or collide with each other in a way that matters beyond "first obstacle on the row".
#     - Each trigger row receives at least one obstacle and at least one bg cell before it, so a visible red segment is
#       always produced and the solver's collision assertion is satisfied.
#   visual_observability:
#     - Across examples, the right pane visibly contains colored blocker pixels before any output is produced, so the
#       new behavior is human-visible rather than hidden in solver-only logic.
#     - Because some non-trigger rows also contain blockers, the rule is not "any obstacle causes red"; only rows
#       selected by left-side holes fire, and those rays terminate at the nearest blocker on the same row.
#   cleanliness_and_color_logic:
#     - Obstacle colors come from a small palette distinct from bg/black/red/grey and carry no per-color semantics.
#       Reusing a few lane colors makes the obstacle field look intentional instead of noisy.
#   new_solver_assumptions_checked_against_generator:
#     - Every trigger row has at least one obstacle pixel in the input right pane.
#     - Cells strictly before the first obstacle on a trigger row are bg, so writing red there is safe.
#     - Any non-bg pixel on the right is intended to be an obstacle; the generator does not place unrelated motifs there.


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


def _first_obstacle_col(g: np.ndarray, row: int, sep: int, bg: int) -> int | None:
    for c in range(sep + 1, g.shape[1]):
        if int(g[row, c]) != int(bg):
            return int(c)
    return None


def _trigger_rows_from_left(left: np.ndarray, sep: int, bg: int) -> list[int]:
    h, _ = left.shape
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_rows = []
    trigger_col = sep - 2
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted(set(r for r, _ in pts)))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return trigger_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the separator, then fire red rays that are blocked by right-side obstacles.

    Input structure:
    - A single full-height grey (5) separator splits the grid into a left sprite pane and a right obstacle pane.
    - Black components live only on the left; the right pane already contains background plus colored obstacle pixels.

    Rule:
    - Shift every black 4-connected component right until its right edge is flush with the separator.
    - In the shifted left pane, find enclosed background holes; rows from holes that reach the inner face at `sep-2`
      become emitter rows.
    - On each emitter row, paint red from `sep+1` rightward until just before the first obstacle pixel already present
      on that row; the obstacle itself remains unchanged and blocks further painting.

    Key insight: The left half decides which rows are allowed to shoot, but the right half is no longer empty space;
    its pre-existing obstacles determine each ray's stopping point via nearest-collision logic.
    """
    g = np.asarray(grid, dtype=int)
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

    for r in _trigger_rows_from_left(left, sep, bg):
        stop_col = _first_obstacle_col(g, r, sep, bg)
        assert stop_col is not None, f"trigger row {r} must contain a right-side obstacle"
        assert stop_col >= sep + 2, f"obstacle on row {r} must leave room for a visible ray"
        assert np.all(out[r, sep + 1 : stop_col] == bg), (
            f"cells before the first obstacle on row {r} must be background"
        )
        out[r, sep + 1 : stop_col] = red

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5
OBSTACLE_COLORS = [1, 3, 4, 7, 8, 9]

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 4, 6

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35
SECOND_OBSTACLE_PROB = 0.45
DECOY_ROW_PROB = 0.45

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


def _obstacle_lanes(sep, w):
    cols = list(range(sep + 2, w - 1))
    assert len(cols) >= 2, "right pane must allow at least two interior obstacle columns"
    n_lanes = 2 if len(cols) == 2 else random.choice([2, 3])
    lanes = sorted(random.sample(cols, n_lanes))
    # Keep obstacle lanes visually distinct so the blocker field reads as deliberate structure, not noise.
    lane_colors = {c: color for c, color in zip(lanes, random.sample(OBSTACLE_COLORS, n_lanes))}
    return lanes, lane_colors


def _stamp_obstacles(board, sep, trigger_rows):
    h, w = board.shape
    lanes, lane_colors = _obstacle_lanes(sep, w)
    obstacle_rows = {}

    for r in sorted(trigger_rows):
        stop_idx = random.randrange(len(lanes))
        cols = {lanes[stop_idx]}
        farther = [c for c in lanes if c > lanes[stop_idx]]
        if farther and random.random() < SECOND_OBSTACLE_PROB:
            cols.add(random.choice(farther))
        for c in sorted(cols):
            board[r, c] = lane_colors[c]
        obstacle_rows[r] = sorted(cols)

    candidate_rows = [r for r in range(1, h - 1) if r not in obstacle_rows]
    random.shuffle(candidate_rows)
    for r in candidate_rows:
        if random.random() >= DECOY_ROW_PROB:
            continue
        cols = {random.choice(lanes)}
        farther = [c for c in lanes if c > min(cols)]
        if farther and random.random() < 0.25:
            cols.add(random.choice(farther))
        for c in cols:
            board[r, c] = lane_colors[c]

    return obstacle_rows


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

    obstacle_rows = _stamp_obstacles(out, sep, trigger_rows)
    for r in trigger_rows:
        stop_col = obstacle_rows[r][0]
        out[r, sep + 1 : stop_col] = RED
    return out, sprites, obstacle_rows


def _unslide_sprites(h, w, sep, sprites, right_template):
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    right = right_template[:, sep + 1 :].copy()
    right[right == RED] = BG
    inp[:, sep + 1 :] = right

    flush_i = random.randrange(len(sprites)) if random.random() < FLUSH_PROB else -1
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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

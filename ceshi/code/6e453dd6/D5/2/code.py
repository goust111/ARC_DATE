# Variant of 6e453dd6
# Dimension: D5 — 封闭洞触发条件 (src)
# Change: enclosure_detection——洞必须是完全封闭的背景色区域（4连通分量不触及左侧子网格边界），且洞的最右列位... → 洞的面积决定是否触发

# Design analysis for the D5 variant
# a) Concrete behavior and trace:
#    - The left pane still contains black sprites on a magenta background, split from the right pane by a full grey column.
#    - Each black component still slides right until it touches the separator.
#    - After sliding, we still look for enclosed background holes whose rightmost cells lie on the inner-face column `sep-2`.
#    - The changed step is the trigger filter: only holes with area >= 2 fire red rays. A 3x3 ring creates a single-cell hole
#      at `sep-2`; under the original rule it would emit, but in this variant it stays silent. A 4x3 shell (2x1 hole) or 3x4
#      shell (1x2 hole) still emits because its enclosed hole area is at least 2.
# b) Complexity calibration:
#    - Every generated grid includes both a single-cell enclosed hole that reaches `sep-2` and a larger enclosed hole that also
#      reaches `sep-2`. This makes the D5 change observable within one example instead of relying on cross-example luck.
#    - Larger holes vary between tall-thin and short-wide cases, so the pattern is not reducible to "height >= 2" or
#      "width >= 2"; the stable cue is enclosed area.
# c) Hidden assumptions from the original solver:
#    - Separator detection and border-based background inference stay valid because the generator still keeps a unique full grey
#      column and a border dominated by the background color.
#    - Component sliding assumes sprites do not collide after being shifted flush. The generator preserves disjoint row bands with
#      full background gaps, so horizontal motion cannot merge sprites.
#    - Hole detection treats any bg component touching the left pane boundary as non-enclosed. Therefore generated hole-bearing
#      sprites must stay at least one column away from the left boundary after sliding; widths are capped at `sep-1` so flush
#      placement always leaves column 0 available as outside background.
#    - The original code implicitly triggered every enclosed hole at `sep-2`. That assumption is the exact D5 point being removed,
#      so the solver now classifies trigger-side holes by area before painting rays.
# d) Placement legality:
#    - Patches are built with widths <= `sep-1`, so their flush position stays strictly inside the left pane and each sprite can be
#      shifted left by at least one cell when producing the input.
#    - Vertical placement uses disjoint bands with >=1 full background row between sprites and between sprites and the outer border.
#      This prevents overlap and preserves hole enclosure.
# e) Visual observability:
#    - The key contrast is visible: two closed hollow shapes can sit equally close to the separator, but only the one whose hole
#      spans at least two pixels produces a red ray.
#    - Because both cases are present in every sample, the rule does not depend on rare branch coverage.
# f) Visual cleanliness:
#    - Rays are confined to the right pane, sprites are isolated by empty rows, and each hole-bearing object occupies its own band.
#      There is no multi-layer overlap that would make the source of a ray ambiguous.
# g) Color reasoning:
#    - Colors keep their original roles: black=sprite, magenta=background, grey=separator, red=ray. No hidden color ordering or
#      color-dependent branching is introduced; only geometric area matters.
# h) New solver assumptions to audit:
#    - `main()` now assumes at least one triggered hole may coexist with one or more ignored single-cell holes. The generator
#      explicitly constructs both kinds and verifies the produced output with `main()`.
#    - Trigger rows must remain unique. The row-band layout guarantees that different hole components live on disjoint row ranges.

import random

import numpy as np

from arc_utils import components_mask, paste


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


def _enclosed_holes(left: np.ndarray, bg: int) -> list[np.ndarray]:
    h, sep = left.shape
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _classify_trigger_side_holes(
    left: np.ndarray,
    bg: int,
    min_area: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    trigger_col = left.shape[1] - 2
    large, small = [], []
    for pts in _enclosed_holes(left, bg):
        if not any(c == trigger_col for _, c in pts):
            continue
        if len(pts) >= min_area:
            large.append(pts)
        else:
            small.append(pts)
    return large, small


def _trigger_rows_from_left(left: np.ndarray, bg: int, min_area: int) -> list[int]:
    large_holes, _ = _classify_trigger_side_holes(left, bg, min_area=min_area)
    rows = []
    for pts in large_holes:
        rows.extend(sorted({int(r) for r, _ in pts}))
    assert len(rows) == len(set(rows)), f"trigger rows must be unique, got {rows}"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black shapes rightward to the grey separator, then emit rays only from sufficiently large enclosed holes.

    Input structure:
    - A single full-height grey (5) separator column splits the grid into a left sprite pane and a right empty pane.
    - Black shapes occupy the left pane on a uniform background color, while the right pane starts as pure background.

    Rule:
    - Shift every black connected component right until its right edge becomes flush with the separator.
    - In the shifted left pane, find enclosed background regions whose cells reach the column just inside the separator (`sep-2`).
    - Ignore 1-cell holes; only enclosed trigger-side holes with area at least 2 paint red rays across the corresponding rows on the right pane.

    Key insight: proximity to the separator is no longer enough by itself — two closed holes can touch the same inner face, but only the larger cavity behaves like an active nozzle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2
    trigger_hole_min_area = 2

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

    for r in _trigger_rows_from_left(left, bg, min_area=trigger_hole_min_area):
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

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

FLUSH_PROB = 0.35
MAX_RETRIES = 400
TRIGGER_HOLE_MIN_AREA = 2

K_BIG = "big_emitter"
K_TINY = "tiny_hole"
K_DISTRACTOR = "distractor"


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _rect_shell(hole_h: int, hole_w: int) -> np.ndarray:
    sh = hole_h + 2
    sw = hole_w + 2
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    return patch


def _big_emitter_patch(max_sw: int, max_sh: int) -> np.ndarray:
    options = []
    for hole_h in range(1, max_sh - 1):
        for hole_w in range(1, max_sw - 1):
            if hole_h * hole_w >= TRIGGER_HOLE_MIN_AREA:
                options.append((hole_h, hole_w))
    assert options, "expected at least one large-hole option"
    hole_h, hole_w = random.choice(options)
    return _rect_shell(hole_h, hole_w)


def _tiny_hole_patch() -> np.ndarray:
    return _rect_shell(1, 1)


def _distractor_patch(max_sw: int, max_sh: int) -> np.ndarray:
    choices = ["l_shape", "solid"]
    if max_sw >= 3 and max_sh >= 3:
        choices.extend(["c_left", "c_right"])
    kind = random.choice(choices)

    if kind == "solid":
        sh = random.randint(2, min(3, max_sh))
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), BLACK, dtype=int)

    if kind == "l_shape":
        sh = random.randint(2, max_sh)
        sw = random.randint(2, max_sw)
        patch = np.full((sh, sw), BG, dtype=int)
        patch[:, -1] = BLACK
        if random.random() < 0.5:
            patch[0, :] = BLACK
        else:
            patch[-1, :] = BLACK
        return patch

    sh = random.randint(3, max_sh)
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    if kind == "c_left":
        patch[:, -1] = BLACK
    else:
        patch[:, 0] = BLACK
    return patch


def _choose_grid_dims() -> tuple[int, int, int]:
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_kinds() -> list[str]:
    n = random.randint(N_SPRITES_MIN, N_SPRITES_MAX)
    kinds = [K_BIG, K_TINY]
    pool = [K_BIG, K_TINY, K_DISTRACTOR]
    weights = [0.45, 0.3, 0.25]
    for _ in range(n - 2):
        kinds.append(random.choices(pool, weights=weights, k=1)[0])
    random.shuffle(kinds)
    return kinds


def _build_patches(kinds: list[str], sep: int) -> list[np.ndarray] | None:
    max_sw = min(SPRITE_W_MAX, sep - 1)
    max_sh = SPRITE_H_MAX
    if max_sw < 3:
        return None

    patches = []
    for kind in kinds:
        if kind == K_BIG:
            patch = _big_emitter_patch(max_sw, max_sh)
        elif kind == K_TINY:
            patch = _tiny_hole_patch()
        else:
            patch = _distractor_patch(max_sw, max_sh)
        if patch.shape[1] > max_sw or patch.shape[0] > max_sh:
            return None
        patches.append(patch)
    return patches


def _place_patches(h: int, patches: list[np.ndarray]) -> list[tuple[np.ndarray, int]] | None:
    total_h = sum(int(p.shape[0]) for p in patches)
    min_needed = total_h + MIN_BAND_GAP * (len(patches) + 1)
    if min_needed > h:
        return None

    gaps = [MIN_BAND_GAP + x for x in _scatter(h - min_needed, len(patches) + 1)]
    placed = []
    r = gaps[0]
    for i, patch in enumerate(patches):
        placed.append((patch, r))
        r += patch.shape[0] + gaps[i + 1]
    return placed


def _render_output(h: int, w: int, sep: int, placed: list[tuple[np.ndarray, int]]):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    for patch, r0 in placed:
        c0 = sep - patch.shape[1]
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0))

    left = out[:, :sep]
    for r in _trigger_rows_from_left(left, BG, min_area=TRIGGER_HOLE_MIN_AREA):
        assert np.all(out[r, sep + 1 :] == BG), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = RED
    return out, sprites


def _unslide_sprites(h: int, w: int, sep: int, sprites) -> np.ndarray | None:
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(len(sprites)) if random.random() < FLUSH_PROB else -1
    shifted = 0
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        shifted += int(delta > 0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    if shifted == 0:
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    kinds = _choose_kinds()
    patches = _build_patches(kinds, sep)
    if patches is None:
        return None

    placed = _place_patches(h, patches)
    if placed is None:
        return None

    out, sprites = _render_output(h, w, sep, placed)
    left = out[:, :sep]
    large_holes, small_holes = _classify_trigger_side_holes(left, BG, min_area=TRIGGER_HOLE_MIN_AREA)
    if not large_holes or not small_holes:
        return None

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
    raise RuntimeError("failed to generate a valid 6e453dd6 D5-area-threshold input")

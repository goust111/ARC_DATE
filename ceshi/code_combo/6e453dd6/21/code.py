# Combo variant of 6e453dd6 (index 21)
# combo_spec: D5/2+D22/1
# Name: 面积阈值触发+射线高度扩展

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The shared first phase is unchanged: the grid has one full grey separator, black
#      components live on the left, and every black component slides right until its
#      right edge reaches `sep-1`.
#    - After sliding, the original task treated every enclosed background hole that
#      touches `sep-2` as an emitter and painted only the hole's own rows on the right.
#    - This combo changes both the trigger and the painted geometry. Example:
#      a 3x3 ring has a 1x1 hole touching `sep-2`; originally it emitted one thin row,
#      but here it emits nothing because area=1 < 2. A 4x3 ring has a 2x1 hole; here it
#      does emit because area=2, and its hole width is 1 so the red block extends one row
#      above and below the hole rows. A 3x4 ring has a 1x2 hole; it also emits, but with
#      a stronger expansion because width=2, so the red block is visibly taller.
#    - The fusion is therefore sequential rather than independent: enclosure decides the
#      candidate holes, area>=2 decides which candidates survive, and only the surviving
#      holes contribute width-dependent ray rectangles.
# b) Complexity calibration:
#    - The generator always includes three contrasting cases in one sample:
#      one ignored 1x1 trigger-side hole, one narrow-but-valid area>=2 hole with width 1,
#      and one wider valid hole with width >=2. This makes both sub-rules observable from
#      a single input/output pair instead of relying on dataset-level luck.
#    - Hole widths are kept to a small visible set {1,2,3}, and valid areas come from a
#      small family of rectangular holes. That avoids combinatorial noise while still
#      making "area threshold" and "width controls thickness" separately inferable.
# c) Hidden assumptions audit:
#    - Separator detection and border-majority background inference still hold because the
#      generator keeps exactly one full grey column and a border dominated by the bg color.
#    - Sliding assumes left-side components never merge after translation. The generator
#      uses disjoint vertical bands with full background rows between sprites, so horizontal
#      motion cannot create collisions.
#    - Hole detection assumes any bg component touching the left-pane boundary is exterior.
#      Therefore hole-bearing patches are always stamped flush with width <= `sep-1`, leaving
#      exterior bg around them and keeping enclosed holes away from pane borders.
#    - The original row-wise ray assertion no longer holds: two expanded blocks may overlap
#      vertically, so the solver must write red rectangles directly without asserting that a
#      row is untouched beforehand.
#    - The new width-based expansion assumes hole width is readable from the enclosed hole's
#      occupied columns. The generator only uses rectangular enclosed holes, so that width is
#      stable and unambiguous.
# d) Placement legality:
#    - All patches have width <= min(6, sep-1) and height <= 8, so flush placement stays
#      strictly inside the left pane and every sprite can be unslid left by at least one cell.
#    - Row bands provide non-overlap on the left. The generator additionally rejects samples
#      whose expanded red intervals would clip against the top/bottom border or merge the key
#      wide and narrow emitters into a single indistinct block.
# e) Visual observability:
#    - A solver can compare three nearby shell-like objects: the tiny 1x1 cavity reaches the
#      same separator-facing column as the others but produces no red; a narrow valid cavity
#      produces a modest block; a wider valid cavity produces a clearly taller block.
#    - Because the ignored and emitted cases are structurally similar and share colors, the
#      informative signal is geometric and visible to a human, not tied to hidden ids.
# f) Visual cleanliness:
#    - The right pane remains dedicated to red projections only. Key expanded blocks are kept
#      vertically separated, so the output does not devolve into one giant merged red slab.
#    - Optional distractors are simple open or solid black shapes; they do not add competing
#      colors or ambiguous overlays.
# g) Color reasoning:
#    - Colors retain the original semantics: black=sprite, magenta=background, grey=separator,
#      red=projection. The combo introduces no extra color mapping; only geometric conditions
#      on holes change the right-side red footprint.
# h) Generator-side verification of the fusion:
#    - Beyond `main(inp) == out`, generation checks for at least one ignored trigger-side hole,
#      at least one triggered width-1 hole, at least one triggered width>=2 hole, and a red
#      area strictly larger than the thin-ray baseline. If those checks fail, the sample is
#      rejected as visually weak or degenerate.
# i) New solver assumptions to audit:
#    - Every hole that should be considered lies in the slid left pane and is 4-connected bg.
#      The generator only creates rectangular enclosed holes, so `_trigger_side_hole_specs()`
#      can safely summarize each hole by row span, width, and area.
#    - The right side starts as background-only in the input. The generator preserves that,
#      ensuring the solver's red painting is the only source of non-bg cells on the right.

import random

import numpy as np

from arc_utils import components_mask, paste


BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

N_SPRITES_MIN, N_SPRITES_MAX = 3, 4
FLUSH_PROB = 0.35
MAX_RETRIES = 500

TRIGGER_HOLE_MIN_AREA = 2
MAX_HOLE_WIDTH = 3

K_TALL = "tall_emitter"
K_WIDE = "wide_emitter"
K_TINY = "tiny_hole"
K_DISTRACTOR = "distractor"


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = SEP_COLOR) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
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


def _slide_left_components(left: np.ndarray, bg: int) -> np.ndarray:
    """Translate every black component so its right edge becomes flush with the separator."""
    sep = left.shape[1]
    slid = left.copy()
    comps = _components4(slid == BLACK)
    slid[slid == BLACK] = bg
    target_right = sep - 1
    for comp in comps:
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = int(c) + delta
            assert 0 <= nc < sep, f"shifted pixel {(int(r), int(c))} left the pane"
            slid[int(r), nc] = BLACK
    return slid


def _enclosed_holes(left: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    """Return enclosed bg components inside the left pane."""
    h, left_w = left.shape
    holes = []
    for pts in _components4(left == bg):
        if any(int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == left_w - 1 for r, c in pts):
            continue
        holes.append([(int(r), int(c)) for r, c in pts])
    return holes


def _trigger_side_hole_specs(left: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    """
    Return [(row_min, row_max, hole_width, hole_area), ...] for enclosed holes touching `sep-2`.
    """
    trigger_col = left.shape[1] - 2
    specs = []
    for pts in _enclosed_holes(left, bg):
        if not any(c == trigger_col for _, c in pts):
            continue
        rows = [r for r, _ in pts]
        cols = {c for _, c in pts}
        hole_w = len(cols)
        hole_area = len(pts)
        assert hole_w >= 1, "trigger-side hole must occupy at least one column"
        specs.append((min(rows), max(rows), hole_w, hole_area))
    return specs


def _paint_expanded_rays(out: np.ndarray, sep: int, specs: list[tuple[int, int, int, int]], red: int) -> None:
    """Paint right-side rectangles for area-qualified holes, expanding by hole width."""
    h, _ = out.shape
    for row_min, row_max, hole_w, hole_area in specs:
        if hole_area < TRIGGER_HOLE_MIN_AREA:
            continue
        top = max(0, row_min - hole_w)
        bottom = min(h - 1, row_max + hole_w)
        out[top : bottom + 1, sep + 1 :] = red


def _expanded_interval(spec: tuple[int, int, int, int], h: int) -> tuple[int, int]:
    row_min, row_max, hole_w, _ = spec
    return max(0, row_min - hole_w), min(h - 1, row_max + hole_w)


def _thin_ray_area(specs: list[tuple[int, int, int, int]], right_w: int) -> int:
    return sum((row_max - row_min + 1) * right_w for row_min, row_max, _, area in specs if area >= TRIGGER_HOLE_MIN_AREA)


def _expanded_red_area(specs: list[tuple[int, int, int, int]], h: int, right_w: int) -> int:
    mask = np.zeros(h, dtype=bool)
    for spec in specs:
        _, _, _, area = spec
        if area < TRIGGER_HOLE_MIN_AREA:
            continue
        top, bottom = _expanded_interval(spec, h)
        mask[top : bottom + 1] = True
    return int(mask.sum()) * right_w


def _combo_metrics_from_slid_left(left: np.ndarray, right_w: int, bg: int) -> dict[str, int]:
    specs = _trigger_side_hole_specs(left, bg)
    triggered = [spec for spec in specs if spec[3] >= TRIGGER_HOLE_MIN_AREA]
    ignored = [spec for spec in specs if spec[3] < TRIGGER_HOLE_MIN_AREA]
    width1 = sum(spec[2] == 1 for spec in triggered)
    width2p = sum(spec[2] >= 2 for spec in triggered)
    return {
        "trigger_side_holes": len(specs),
        "triggered_holes": len(triggered),
        "ignored_holes": len(ignored),
        "width1_triggers": width1,
        "width2p_triggers": width2p,
        "thin_red_area": _thin_ray_area(specs, right_w),
        "expanded_red_area": _expanded_red_area(specs, left.shape[0], right_w),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the black sprites to the separator, then let only large enough enclosure nozzles emit widened red blocks.

    Input structure:
    - A single full-height grey (5) separator column splits the board into a left sprite pane and a right empty pane.
    - Black connected components occupy the left pane on one background color, and the right pane starts as uniform background.

    Rule:
    - Shift every black connected component horizontally until its rightmost black cell is flush with the separator.
    - In the shifted left pane, find enclosed background holes whose cells touch the inner-face column `sep-2`.
    - Ignore trigger-side holes of area 1; each trigger-side hole with area at least 2 paints a red rectangle across the whole right pane, expanding upward and downward by an amount equal to that hole's width.

    Key insight: two separator-facing holes can look equally well-positioned, but only sufficiently large enclosed cavities activate, and among the activated ones the hole width controls how tall the emitted red block becomes.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()
    left = _slide_left_components(out[:, :sep], bg)
    out[:, :sep] = left

    specs = _trigger_side_hole_specs(left, bg)
    _paint_expanded_rays(out, sep=sep, specs=specs, red=RED)
    return out


def _scatter(total: int, slots: int) -> list[int]:
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


def _tall_emitter_patch(max_sh: int) -> np.ndarray:
    hole_h = random.choice([h for h in (2, 3) if h + 2 <= max_sh])
    return _rect_shell(hole_h, 1)


def _wide_emitter_patch(max_sw: int, max_sh: int) -> np.ndarray:
    valid = []
    max_hole_w = min(MAX_HOLE_WIDTH, max_sw - 2)
    for hole_h in (1, 2):
        for hole_w in range(2, max_hole_w + 1):
            if hole_h + 2 <= max_sh and hole_h * hole_w >= TRIGGER_HOLE_MIN_AREA:
                valid.append((hole_h, hole_w))
    assert valid, "expected at least one wide emitter shape"
    hole_h, hole_w = random.choice(valid)
    return _rect_shell(hole_h, hole_w)


def _tiny_hole_patch() -> np.ndarray:
    return _rect_shell(1, 1)


def _distractor_patch(max_sw: int, max_sh: int) -> np.ndarray:
    choices = ["solid", "l_shape"]
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
    kinds = [K_TALL, K_WIDE, K_TINY]
    for _ in range(n - len(kinds)):
        kinds.append(random.choices([K_DISTRACTOR, K_TALL, K_WIDE], weights=[0.45, 0.25, 0.30], k=1)[0])
    random.shuffle(kinds)
    return kinds


def _build_patches(kinds: list[str], sep: int) -> list[np.ndarray] | None:
    max_sw = min(SPRITE_W_MAX, sep - 1)
    max_sh = SPRITE_H_MAX
    if max_sw < 4:
        return None

    patches = []
    for kind in kinds:
        if kind == K_TALL:
            patch = _tall_emitter_patch(max_sh)
        elif kind == K_WIDE:
            patch = _wide_emitter_patch(max_sw, max_sh)
        elif kind == K_TINY:
            patch = _tiny_hole_patch()
        else:
            patch = _distractor_patch(max_sw, max_sh)
        if patch.shape[0] > max_sh or patch.shape[1] > max_sw:
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


def _render_output(h: int, w: int, sep: int, placed: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]], list[tuple[int, int, int, int]]]:
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    for patch, r0 in placed:
        c0 = sep - patch.shape[1]
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0))

    left = out[:, :sep]
    specs = _trigger_side_hole_specs(left, BG)
    _paint_expanded_rays(out, sep=sep, specs=specs, red=RED)
    return out, sprites, specs


def _intervals_separated(specs: list[tuple[int, int, int, int]], h: int) -> bool:
    triggered = [spec for spec in specs if spec[3] >= TRIGGER_HOLE_MIN_AREA]
    if not triggered:
        return False
    intervals = []
    for spec in triggered:
        top, bottom = _expanded_interval(spec, h)
        row_min, row_max, hole_w, _ = spec
        if top == 0 or bottom == h - 1:
            return False
        intervals.append((top, bottom, hole_w, row_min, row_max))

    width1 = [it for it in intervals if it[2] == 1]
    width2p = [it for it in intervals if it[2] >= 2]
    if not width1 or not width2p:
        return False

    for t1, b1, _, _, _ in width1:
        for t2, b2, _, _, _ in width2p:
            if b1 < t2 or b2 < t1:
                return True
    return False


def _unslide_sprites(h: int, w: int, sep: int, sprites: list[tuple[np.ndarray, int, int]]) -> np.ndarray | None:
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

    out, sprites, specs = _render_output(h, w, sep, placed)
    metrics = _combo_metrics_from_slid_left(out[:, :sep], w - sep - 1, BG)
    if metrics["ignored_holes"] < 1:
        return None
    if metrics["width1_triggers"] < 1 or metrics["width2p_triggers"] < 1:
        return None
    if metrics["expanded_red_area"] <= metrics["thin_red_area"]:
        return None
    if not _intervals_separated(specs, h):
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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo D5/2+D22/1 input")

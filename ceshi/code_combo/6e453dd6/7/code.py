# Combo variant of 6e453dd6 (index 7)
# combo_spec: D5/2+D18/2
# Name: 仅最大封闭洞且面积达标触发
#
# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The original task slides every black sprite flush to the grey separator, then fires from every enclosed hole that reaches
#      the inner face near the separator. In the combo, hole processing becomes sprite-local and comparative.
#    - Example after sliding: one sprite contains two closed chambers with areas 3 and 2, and another sprite contains one 1-cell
#      hole. Originally all three hole rows would emit. Under D5 alone, the 3-cell and 2-cell holes would emit, while the 1-cell
#      hole would not. Under this combo, only the unique largest hole of the first sprite emits, and the single-pixel-hole sprite
#      stays silent because its maximum enclosed area is still below 2.
#    - The operational order is therefore: slide components -> collect each sprite's enclosed holes -> compare hole areas within
#      the sprite -> keep only the maximum-area hole(s) -> apply the >=2 threshold -> paint red rows.
# b) Complexity calibration:
#    - Every generated grid includes at least one multi-hole sprite whose non-maximum hole also has area >=2, so the sample cannot
#      be explained by the D5 threshold-only rule.
#    - Every generated grid also includes at least one sprite whose largest hole has area 1, so the sample cannot be explained by
#      a "largest hole always fires" rule either.
#    - Hole sizes come from a small repeated family (areas 1, 2, 3, 4) to keep the concept learnable from a few examples.
# c) Hidden assumptions from the original solver:
#    - The original global hole scan assumed the trigger condition depended only on hole position. That is no longer valid, so the
#      solver now re-groups holes by owning sprite via each component's tight bbox before comparing areas.
#    - Separator detection, background inference, and rightward sliding remain sound because the generator still preserves a single
#      full-height grey column, border-dominant background color, and row-disjoint sprites.
#    - Hole finding by bbox-boundary exclusion still works because any true cavity is fully enclosed by black, while all outside
#      background strips inside the bbox still touch the bbox boundary.
# d) Placement legality:
#    - Output-first construction stamps every sprite flush against the separator inside its own vertical band, so sprites never
#      overlap and horizontal unsliding cannot merge them.
#    - Patch widths are capped at `sep-1`, guaranteeing that every sprite remains strictly inside the left pane and can be shifted
#      left by at least one column when producing the input.
# e) Visual observability:
#    - The decisive evidence is visible inside a single sprite: several closed holes exist, but rays appear only on the rows of the
#      largest cavity.
#    - A separate tiny-hole sprite shows that "largest" is not enough by itself; the maximum hole must also meet the area >=2 rule.
# f) Visual cleanliness:
#    - Sprites live in disjoint row bands with full-background gaps, and rays occupy only the right pane, so there is no ambiguity
#      about which left-side object caused which red rows.
#    - Multi-hole sprites use vertically stacked chambers, making the selected vs ignored holes easy to compare by eye.
# g) Change inferability:
#    - No color remapping is introduced. Black remains the sprite, magenta the background, grey the separator, and red the emitted
#      ray; the only new reasoning burden is geometric comparison of enclosed areas.
# h) New solver assumptions to audit:
#    - `main()` assumes sprite-local hole comparison is meaningful, so the generator guarantees that each multi-hole sprite has a
#      unique largest hole and that equal-area ties are not needed to explain the samples.
#    - The generator also guarantees at least one emitting sprite, at least one ignored non-maximum large hole, and at least one
#      sprite suppressed solely by the >=2 threshold, so the fusion behavior is present rather than accidental.

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

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

FLUSH_PROB = 0.35
MAX_RETRIES = 500
TRIGGER_HOLE_MIN_AREA = 2

K_MULTI = "multi_hole"
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


def _bbox_of_points(pts) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _holes_in_patch(patch: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = patch.shape
    holes = []
    for pts in _components4(patch == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        holes.append([(int(r), int(c)) for r, c in pts])
    return holes


def _component_holes(left: np.ndarray, bg: int) -> list[list[list[tuple[int, int]]]]:
    holes_by_component = []
    for comp in _components4(left == BLACK):
        r0, c0, r1, c1 = _bbox_of_points(comp)
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        global_holes = []
        for hole in _holes_in_patch(patch, bg):
            global_holes.append([(r0 + r, c0 + c) for r, c in hole])
        if global_holes:
            holes_by_component.append(global_holes)
    return holes_by_component


def _combo_trigger_rows_from_left(left: np.ndarray, bg: int, min_area: int) -> list[int]:
    rows = set()
    for holes in _component_holes(left, bg):
        max_area = max(len(hole) for hole in holes)
        if max_area < min_area:
            continue
        for hole in holes:
            if len(hole) == max_area:
                rows.update(int(r) for r, _ in hole)
    return sorted(rows)


def _threshold_only_rows_from_left(left: np.ndarray, bg: int, min_area: int) -> list[int]:
    rows = set()
    for holes in _component_holes(left, bg):
        for hole in holes:
            if len(hole) >= min_area:
                rows.update(int(r) for r, _ in hole)
    return sorted(rows)


def _largest_only_rows_from_left(left: np.ndarray, bg: int) -> list[int]:
    rows = set()
    for holes in _component_holes(left, bg):
        max_area = max(len(hole) for hole in holes)
        for hole in holes:
            if len(hole) == max_area:
                rows.update(int(r) for r, _ in hole)
    return sorted(rows)


def _left_behavior_summary(left: np.ndarray, bg: int, min_area: int) -> dict:
    combo_rows = set()
    d5_rows = set()
    d18_rows = set()
    ignored_large_holes = 0
    tiny_max_sprites = 0
    multi_hole_sprites = 0
    emitting_sprites = 0

    for holes in _component_holes(left, bg):
        if len(holes) >= 2:
            multi_hole_sprites += 1
        areas = [len(hole) for hole in holes]
        max_area = max(areas)
        if max_area < min_area:
            tiny_max_sprites += 1
        else:
            emitting_sprites += 1
        for hole in holes:
            area = len(hole)
            if min_area <= area < max_area:
                ignored_large_holes += 1
            if area >= min_area:
                d5_rows.update(int(r) for r, _ in hole)
            if area == max_area:
                d18_rows.update(int(r) for r, _ in hole)
                if max_area >= min_area:
                    combo_rows.update(int(r) for r, _ in hole)

    return {
        "combo_rows": sorted(combo_rows),
        "d5_rows": sorted(d5_rows),
        "d18_rows": sorted(d18_rows),
        "ignored_large_holes": int(ignored_large_holes),
        "tiny_max_sprites": int(tiny_max_sprites),
        "multi_hole_sprites": int(multi_hole_sprites),
        "emitting_sprites": int(emitting_sprites),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites against the grey divider, then let only each sprite's largest qualifying cavity fire.

    Input structure:
    - A single full-height grey separator column splits the board into a left sprite pane and a right empty pane.
    - Black connected components sit in the left pane on one uniform background color; the right pane begins as pure background.

    Rule:
    - Translate every black connected component rightward until its right edge is flush with the separator.
    - For each shifted sprite separately, find all enclosed background holes inside its bbox and compare their areas.
    - Only the maximum-area hole of that sprite may trigger, and it triggers only if that maximum area is at least 2; the rows of
      that winning hole are then painted red across the full right pane.

    Key insight: the active "nozzle" is no longer chosen by mere existence or position alone; you must first identify all closed
    cavities inside the same sprite, keep only the largest one, and then discard it too if its area is only a single pixel.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()
    left = out[:, :sep].copy()
    comps = _components4(left == BLACK)

    left[left == BLACK] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = int(c) + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[int(r), nc] = BLACK

    out[:, :sep] = left

    for r in _combo_trigger_rows_from_left(left, bg, min_area=TRIGGER_HOLE_MIN_AREA):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = RED

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


def _multi_hole_patch(max_sw: int, max_sh: int) -> np.ndarray:
    presets = [
        [(1, 2), (1, 1)],
        [(2, 1), (1, 1)],
        [(2, 2), (1, 2)],
        [(1, 3), (1, 2)],
        [(2, 2), (1, 2), (1, 1)],
        [(1, 3), (1, 2), (1, 1)],
    ]
    choices = []
    for chambers in presets:
        sw = 2 + max(w for _, w in chambers)
        sh = sum(h for h, _ in chambers) + len(chambers) + 1
        if sw <= max_sw and sh <= max_sh:
            areas = [h * w for h, w in chambers]
            max_area = max(areas)
            if max_area >= TRIGGER_HOLE_MIN_AREA and sum(a >= TRIGGER_HOLE_MIN_AREA and a < max_area for a in areas) >= 1:
                choices.append(chambers)
    assert choices, "expected at least one legal multi-hole patch"

    chambers = [tuple(x) for x in random.choice(choices)]
    if random.random() < 0.5:
        chambers.reverse()

    sw = 2 + max(w for _, w in chambers)
    sh = sum(h for h, _ in chambers) + len(chambers) + 1
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK

    r = 1
    for hole_h, hole_w in chambers:
        left_wall = sw - hole_w - 2
        right_wall = sw - 1
        for rr in range(r, r + hole_h):
            patch[rr, left_wall] = BLACK
            patch[rr, right_wall] = BLACK
        r += hole_h
        patch[r, :] = BLACK
        r += 1

    return patch


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
    kinds = [K_MULTI, K_TINY]
    pool = [K_MULTI, K_TINY, K_DISTRACTOR]
    weights = [0.35, 0.2, 0.45]
    for _ in range(n - 2):
        kinds.append(random.choices(pool, weights=weights, k=1)[0])
    random.shuffle(kinds)
    return kinds


def _build_patches(kinds: list[str], sep: int) -> list[np.ndarray] | None:
    max_sw = min(SPRITE_W_MAX, sep - 1)
    max_sh = SPRITE_H_MAX
    if max_sw < 4:
        return None

    patches = []
    for kind in kinds:
        if kind == K_MULTI:
            patch = _multi_hole_patch(max_sw, max_sh)
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


def _render_output(h: int, w: int, sep: int, placed: list[tuple[np.ndarray, int]]):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    for patch, r0 in placed:
        c0 = sep - patch.shape[1]
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0))

    left = out[:, :sep]
    for r in _combo_trigger_rows_from_left(left, BG, min_area=TRIGGER_HOLE_MIN_AREA):
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
    summary = _left_behavior_summary(left, BG, min_area=TRIGGER_HOLE_MIN_AREA)
    if summary["emitting_sprites"] < 1:
        return None
    if summary["ignored_large_holes"] < 1:
        return None
    if summary["tiny_max_sprites"] < 1:
        return None
    if summary["combo_rows"] == summary["d5_rows"]:
        return None
    if summary["combo_rows"] == summary["d18_rows"]:
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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")

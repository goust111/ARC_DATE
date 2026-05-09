# Combo variant of 6e453dd6 (index 31)
# combo_spec: D5/2+D16/2
# Name: 面积阈值+洞内颜色干扰触发
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Start from the original task: black sprites on the left slide right until their right edge becomes flush with the grey separator.
#      In the base task, any enclosed trigger-side background hole at `sep-2` emits a red ray.
#    - D16/2 adds filled-hole distractors: some closed shells contain a non-background color inside the cavity. Those colored cavities move
#      together with the shell, but they are inert and must never emit.
#    - D5/2 then further filters the remaining background-colored cavities by area: after the color check, only background-filled trigger-side
#      cavities with area >= 2 emit. A 1x1 background hole is therefore silent even though it is enclosed and reaches `sep-2`.
#    - Small example after sliding:
#        * a 4x3 rectangular shell with a 2x1 magenta cavity touching `sep-2` emits;
#        * a same-sized shell with a blue 2x1 cavity does not emit because the cavity color is not the background;
#        * a 3x3 shell with a single magenta cavity at `sep-2` does not emit because its area is 1.
#      Relative to the original task, the second and third objects would have emitted, so the fusion changes two distinct stages.
# b) Complexity calibration:
#    - Every generated puzzle contains all three decisive cases at once: one large bg cavity, one large colored cavity, and one 1-cell bg cavity.
#      This prevents the generator from collapsing back to the original rule and makes both new filters visually testable from a few examples.
#    - The large bg cavity and the large colored cavity are generated with matching dimensions in at least one pair, so the visible contrast is
#      driven by fill color rather than by shape size. The small bg cavity separately isolates the area threshold.
# c) Hidden assumptions from the original solver:
#    - The original code only translated black pixels because the source task had no colored pixels inside sprites. That assumption breaks here:
#      a colored cavity fill must move with its enclosing shell. The new solver therefore shifts black components and any non-background interior
#      payload inside their bounding boxes by the same horizontal delta.
#    - The original hole detector only inspected background components. That is no longer enough because enclosed colored cavities now exist and
#      must be rejected explicitly. The new detector enumerates enclosed components for every non-black color, then checks trigger-column contact,
#      then cavity color, then area.
#    - Separator uniqueness, border-based background inference, and non-overlapping row bands still hold. The generator keeps a single full-height
#      grey separator and leaves the outer border dominated by the magenta background.
# d) Placement legality:
#    - All patches have width <= `sep-1`, so flush placement stays strictly in the left pane and every sprite can still be unslid leftward.
#    - Sprites occupy disjoint row bands with at least one full background row between them, so horizontal sliding cannot merge components and
#      trigger rows remain unique.
#    - Colored cavity fills are only placed strictly inside enclosing shells, never on the shell boundary, so they can be assigned unambiguously
#      to the surrounding black component during sliding.
# e) Visual observability:
#    - A human can compare two equally large closed shells near the separator and notice that only the one whose hole looks like background emits.
#    - The single-cell background hole gives an independent counterexample showing that "background-colored hole" is still not sufficient: size
#      matters after color has been checked.
# f) Visual cleanliness:
#    - Rays only appear in the right pane, sprites live in separate row bands, and cavity fill colors are restricted to a small palette. This keeps
#      the causal source of each emitted or suppressed row legible.
# g) Change inferability:
#    - No hidden numeric ordering is used. The decisive cues are geometric and visual: enclosure, whether the cavity fill matches the surrounding
#      background, and whether the cavity spans one cell or multiple cells.
# h) New solver assumptions to audit:
#    - The solver assumes any non-background colored pixel on the left lies inside the bbox interior of exactly one black shell. The generator
#      guarantees this by only introducing colored pixels as enclosed cavity fills.
#    - The solver assumes all trigger-side cavities belonging to different sprites live on disjoint rows. The row-band placement enforces that.

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

N_SPRITES_MIN, N_SPRITES_MAX = 3, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

FLUSH_PROB = 0.35
MAX_RETRIES = 500
TRIGGER_HOLE_MIN_AREA = 2
NOISE_COLORS = [1, 3, 8]

K_BIG_BG = "big_bg"
K_BIG_NOISE = "big_noise"
K_SMALL_BG = "small_bg"
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


def _bbox_from_points(pts) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_color_components(left: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, sep = left.shape
    enclosed = []
    for color in sorted(int(x) for x in np.unique(left) if int(x) != BLACK):
        for pts in _components4(left == color):
            if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
                continue
            enclosed.append((color, [(int(r), int(c)) for r, c in pts]))
    return enclosed


def _classify_trigger_side_cavities(
    left: np.ndarray,
    bg: int,
    min_area: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, list[tuple[int, int]]]], list[list[tuple[int, int]]]]:
    trigger_col = left.shape[1] - 2
    active = []
    wrong_color = []
    small_bg = []
    for color, pts in _enclosed_color_components(left):
        if not any(c == trigger_col for _, c in pts):
            continue
        if color != bg:
            wrong_color.append((color, pts))
            continue
        if len(pts) >= min_area:
            active.append(pts)
        else:
            small_bg.append(pts)
    return active, wrong_color, small_bg


def _trigger_rows_from_left(left: np.ndarray, bg: int, min_area: int) -> list[int]:
    active, _, _ = _classify_trigger_side_cavities(left, bg, min_area=min_area)
    rows = []
    for pts in active:
        rows.extend(sorted({int(r) for r, _ in pts}))
    assert len(rows) == len(set(rows)), f"trigger rows must be unique, got {rows}"
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the left-side sprites to the separator, then fire rays only from background-colored trigger cavities that are large enough.

    Input structure:
    - A single full-height grey (5) separator splits the board into a left sprite pane and a right empty pane.
    - Black shells live on the left over one shared background color, and some shells may contain enclosed interior fills whose color is either the background itself or a small non-background distractor color.

    Rule:
    - Move each black connected component rightward until its right edge is flush with the separator; any colored fill enclosed inside that shell moves by the same horizontal amount.
    - In the shifted left pane, inspect enclosed same-color cavity components whose cells reach the inner-face column `sep-2`.
    - First reject any such cavity whose fill color is not the background; among the remaining background-colored cavities, emit red rays only for those with area at least 2.

    Key insight: being a closed hole next to the separator is no longer sufficient; a cavity must both look like genuine background inside the shell and occupy more than one cell before it behaves like a nozzle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()
    left = out[:, :sep].copy()
    comps = _components4(left == BLACK)

    moved_payloads = []
    target_right = sep - 1
    for comp in comps:
        pts = [(int(r), int(c)) for r, c in comp]
        r0, c0, r1, c1 = _bbox_from_points(pts)
        payload = []
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    val = int(left[r, c])
                    if val not in (bg, BLACK):
                        payload.append((r, c, val))
        max_c = max(c for _, c in pts)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        moved_payloads.append((pts, payload, delta))
        for r, c in pts:
            left[r, c] = bg
        for r, c, _ in payload:
            left[r, c] = bg

    for pts, payload, delta in moved_payloads:
        for r, c in pts:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = BLACK
        for r, c, val in payload:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted payload pixel {(r, c)} left the pane"
            left[r, nc] = val

    out[:, :sep] = left

    for r in _trigger_rows_from_left(left, bg, min_area=TRIGGER_HOLE_MIN_AREA):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = RED

    return out


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _rect_shell(hole_h: int, hole_w: int, fill_color: int) -> np.ndarray:
    sh = hole_h + 2
    sw = hole_w + 2
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    patch[1:-1, 1:-1] = fill_color
    return patch


def _large_hole_options(max_sw: int, max_sh: int) -> list[tuple[int, int]]:
    options = []
    for hole_h in range(1, max_sh - 1):
        for hole_w in range(1, max_sw - 1):
            if hole_h * hole_w >= TRIGGER_HOLE_MIN_AREA:
                options.append((hole_h, hole_w))
    return options


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
    kinds = [K_BIG_BG, K_BIG_NOISE, K_SMALL_BG]
    for _ in range(n - len(kinds)):
        # Keep exactly one example of each decisive branch and use any extra sprite
        # only as a non-triggering distractor so the fusion rule stays legible.
        kinds.append(K_DISTRACTOR)
    random.shuffle(kinds)
    return kinds


def _build_patches(kinds: list[str], sep: int) -> list[np.ndarray] | None:
    max_sw = min(SPRITE_W_MAX, sep - 1)
    max_sh = SPRITE_H_MAX
    if max_sw < 3:
        return None

    large_options = _large_hole_options(max_sw, max_sh)
    if not large_options:
        return None

    paired_dims = random.choice(large_options)
    paired_noise = random.choice(NOISE_COLORS)
    used_bg_pair = False
    used_noise_pair = False
    patches = []

    for kind in kinds:
        if kind == K_BIG_BG:
            if not used_bg_pair:
                hole_h, hole_w = paired_dims
                used_bg_pair = True
            else:
                hole_h, hole_w = random.choice(large_options)
            patch = _rect_shell(hole_h, hole_w, BG)
        elif kind == K_BIG_NOISE:
            if not used_noise_pair:
                hole_h, hole_w = paired_dims
                fill = paired_noise
                used_noise_pair = True
            else:
                hole_h, hole_w = random.choice(large_options)
                fill = random.choice(NOISE_COLORS)
            patch = _rect_shell(hole_h, hole_w, fill)
        elif kind == K_SMALL_BG:
            patch = _rect_shell(1, 1, BG)
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
    active, wrong_color, small_bg = _classify_trigger_side_cavities(left, BG, min_area=TRIGGER_HOLE_MIN_AREA)
    large_wrong_color = [pts for _, pts in wrong_color if len(pts) >= TRIGGER_HOLE_MIN_AREA]
    if len(active) != 1 or len(large_wrong_color) != 1 or len(small_bg) != 1:
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

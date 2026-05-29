# Combo variant of 6e453dd6 (index 29)
# combo_spec: D1/1+D20/2
# Name: 垂直平移+对称性决定射线方向
#
# Design analysis
# ===============
# a) Concretization + behavior trace
#    - The source task slides left-pane black sprites horizontally into a vertical separator and
#      uses enclosed holes as the trigger for rightward red rays.
#    - D1/1 rotates the mechanics: the separator becomes a full-width grey row, sprites live
#      above it, and every black component drops straight downward until it is flush with the
#      row just above the separator.
#    - D20/2 replaces the topological trigger with a geometric one: after dropping, a sprite
#      emits based only on its symmetry inside its own bbox.
#      * left-right symmetric -> emit a horizontal red segment directly below the sprite
#      * up-down symmetric -> emit a vertical red segment below the sprite's center column(s)
#      * both symmetries -> emit both
#      * no symmetry -> emit nothing
#    - Small trace:
#      * Original ring-like emitter: moved right, hole touched the separator-side wall, rows of
#        the hole produced red on the far side.
#      * This combo: a dropped "T" shape is left-right symmetric but not up-down symmetric, so
#        it paints a horizontal red bar on the first row below the separator over its projected
#        width. A dropped "C" shape is up-down symmetric but not left-right symmetric, so it
#        paints a vertical red bar in its projected center column(s). A plain asymmetric "L"
#        moves down the same way but stays silent.
#      * Difference: the decisive feature is visible geometry of the whole sprite, not whether
#        a background pocket is enclosed.
# b) Complexity calibration
#    - The vertical-motion variant needs sprites in disjoint column bands so they can all drop
#      independently without collisions. The width budget is therefore kept broad (16-25 cols),
#      while the bottom pane remains compact (3-5 rows) so the emitted markers stay legible.
#    - The generator always includes at least one horizontal emitter and at least one vertical
#      emitter, and often mixes in a non-emitter. This gives repeated support for the rule in a
#      few training examples without exploding the case count.
# c) Implicit-assumption audit of the original main()
#    - A unique full separator column is no longer valid; the solver must find a unique full
#      separator row instead.
#    - The active pane is `top = out[:sep, :]`, not the area left of a separator.
#    - The movement delta is computed from the component's maximum row, not maximum column.
#    - Hole detection and hole-touch tests disappear entirely; replacing them with symmetry tests
#      means the moved component must be re-extracted from its bbox after dropping.
#    - Red painting can no longer assert "blank before paint" cell-by-cell in the original way,
#      because horizontal and vertical emissions may intersect. The generator instead guarantees
#      that the lower pane starts as background, and overlapping rays are both red anyway.
# d) Placement legality
#    - Sprites are assigned disjoint column bands with at least one full background column between
#      neighboring bands and also at the outer margins.
#    - Every rendered output sprite satisfies `r0 >= 1`, so unsliding upward by at least one row
#      is possible for almost every sprite and never leaves the grid.
#    - The lower pane is kept free of black pixels; all red marks live there, so there is no
#      overwrite ambiguity with the moved sprites.
# e) Visual observability
#    - A human can spot the fused rule from a few examples because three cues line up:
#      * sprites clearly move downward toward a horizontal separator,
#      * symmetric sprites receive red marks directly below their own projection,
#      * asymmetric sprites do not produce red at all.
#    - Horizontal vs vertical red orientation is visually tied to left-right vs up-down symmetry,
#      which is inferable from the black geometry itself rather than hidden metadata.
# f) Visual cleanliness
#    - All red output stays in the lower pane, so the top pane remains a clean before/after view
#      of the dropped sprites.
#    - Horizontal emitters produce disjoint segments because column bands do not overlap; vertical
#      emitters use center columns inside those same bands, so crossings are sparse and readable.
# g) New main() assumptions and generator guarantees
#    - Assumption: all black components lie strictly above the separator. Guaranteed by generator.
#    - Assumption: the separator row is unique and full width. Guaranteed by generator.
#    - Assumption: the lower pane has at least one row. Guaranteed by `BOTTOM_H_MIN >= 3`.
#    - Assumption: each moved component remains isolated after dropping. Guaranteed by disjoint
#      column bands.

import random

import numpy as np

from arc_utils import components_mask, paste


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_row(g: np.ndarray, sep_color: int = 5) -> int:
    h, _ = g.shape
    rows = [r for r in range(h) if np.all(g[r, :] == sep_color)]
    assert len(rows) == 1, f"expected a unique full separator row, got {rows}"
    return int(rows[0])


def _background_color(g: np.ndarray, sep_row: int, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if r != sep_row and g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if r != sep_row and g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    assert border, "expected non-separator border pixels to infer the background"
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must be unique on the border, got {winners.tolist()}"
    return int(winners[0])


def _bbox_of_points(pts: np.ndarray) -> tuple[int, int, int, int]:
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_patch(mask: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_of_points(pts)
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    return patch, (r0, c0, r1, c1)


def _symmetry_flags(patch: np.ndarray) -> tuple[bool, bool]:
    lr = bool(np.array_equal(patch, patch[:, ::-1]))
    ud = bool(np.array_equal(patch, patch[::-1, :]))
    return lr, ud


def _center_columns(c0: int, c1: int) -> list[int]:
    width = c1 - c0 + 1
    if width % 2 == 1:
        return [c0 + width // 2]
    mid = c0 + width // 2
    return [mid - 1, mid]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop black sprites toward a grey horizontal separator, then emit red markers according to symmetry.

    Input structure:
    - A full-width grey (5) separator row splits the grid into an upper sprite pane and a lower empty pane.
    - Black connected components start above the separator on a uniform background color.

    Rule:
    - Move every black component straight downward until its bottom edge is flush with the row just above the separator.
    - Inspect each dropped component inside its bbox: left-right symmetry produces a horizontal red segment below the sprite, up-down symmetry produces a vertical red segment below the sprite's center column(s), and a component with both symmetries produces both.
    - Components with neither symmetry emit nothing.

    Key insight: The old trigger came from enclosed background topology; here the trigger is purely geometric, so the solver must read each moved sprite's symmetry and project the corresponding ray direction into the lower pane.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_row(g, sep_color=sep_color)
    bg = _background_color(g, sep_row=sep, sep_color=sep_color)
    assert sep + 1 < h, "expected at least one row below the separator"

    out = g.copy()

    top = out[:sep, :].copy()
    comps = _components4(top == black)

    top[top == black] = bg
    target_bottom = sep - 1
    for comp in comps:
        max_r = max(r for r, _ in comp)
        delta = target_bottom - max_r
        assert delta >= 0, "upper-pane components must stay above the separator"
        for r, c in comp:
            nr = r + delta
            assert 0 <= nr < sep, f"shifted pixel {(r, c)} left the pane"
            top[nr, c] = black

    out[:sep, :] = top

    moved_mask = top == black
    moved_comps = _components4(moved_mask)
    for comp in moved_comps:
        patch, (_r0, c0, _r1, c1) = _component_patch(moved_mask, comp)
        lr_sym, ud_sym = _symmetry_flags(patch)
        if lr_sym:
            out[sep + 1, c0 : c1 + 1] = red
        if ud_sym:
            for c in _center_columns(c0, c1):
                out[sep + 1 :, c] = red

    return out


# --------------- generator ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 10, 16
W_MIN, W_MAX = 16, 25
SEP_MIN, SEP_MAX = 5, 11
BOTTOM_H_MIN, BOTTOM_H_MAX = 3, 5

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 6
SPRITE_W_MAX = 7
MIN_BAND_GAP = 1

FLUSH_PROB = 0.35
MAX_RETRIES = 400

TYPE_MIN_WIDTH = {
    "lr": 3,
    "ud": 3,
    "both": 2,
    "none": 2,
}

TYPE_MIN_HEIGHT = {
    "lr": 2,
    "ud": 3,
    "both": 2,
    "none": 2,
}


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_cols_needed(types: list[str]) -> int:
    return sum(TYPE_MIN_WIDTH[t] for t in types) + MIN_BAND_GAP * (len(types) + 1)


def _min_cols_for_count(n: int) -> int:
    base = ["lr", "ud"] + ["both"] * max(0, n - 2)
    return _min_cols_needed(base)


def _choose_grid_dims() -> tuple[int, int, int]:
    h = random.randint(H_MIN, H_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if BOTTOM_H_MIN <= h - s - 1 <= BOTTOM_H_MAX]
    sep = random.choice(valid_seps)
    w = random.randint(W_MIN, W_MAX)
    return h, w, sep


def _random_sprite_types(n: int) -> list[str]:
    types = ["lr", "ud"]
    if random.random() < 0.35:
        types[random.randrange(2)] = "both"
    while len(types) < n:
        types.append(random.choices(["lr", "ud", "both", "none"], weights=[3, 3, 2, 3])[0])
    if n >= 3 and "none" not in types and random.random() < 0.7:
        types[random.randrange(2, n)] = "none"
    random.shuffle(types)
    return types


def _choose_sprite_types(w: int) -> list[str] | None:
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1) if _min_cols_for_count(n) <= w]
    if not valid_counts:
        return None
    for _ in range(200):
        n = random.choice(valid_counts)
        types = _random_sprite_types(n)
        if _min_cols_needed(types) <= w:
            return types
    return None


def _sprite_bands(w: int, types: list[str]) -> list[tuple[int, int]] | None:
    n = len(types)
    widths = [TYPE_MIN_WIDTH[t] for t in types]
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    width_cap = w - sum(base_gaps)
    if sum(widths) > width_cap:
        return None

    extra_cap = sum(SPRITE_W_MAX - x for x in widths)
    extra = random.randint(0, min(extra_cap, width_cap - sum(widths)))
    for _ in range(extra):
        choices = [i for i, sw in enumerate(widths) if sw < SPRITE_W_MAX]
        i = random.choice(choices)
        widths[i] += 1

    free = w - sum(widths) - sum(base_gaps)
    gaps = [a + b for a, b in zip(base_gaps, _scatter(free, n + 1))]
    bands = []
    c = gaps[0]
    for i, sw in enumerate(widths):
        bands.append((c, sw))
        c += sw + gaps[i + 1]
    return bands


def _classify_patch(patch: np.ndarray) -> str:
    lr, ud = _symmetry_flags(patch == BLACK)
    if lr and ud:
        return "both"
    if lr:
        return "lr"
    if ud:
        return "ud"
    return "none"


def _lr_patch(sh: int, sw: int) -> np.ndarray:
    patch = np.full((sh, sw), BG, dtype=int)
    centers = _center_columns(0, sw - 1)
    patch[0, :] = BLACK
    for r in range(1, sh):
        for c in centers:
            patch[r, c] = BLACK
    foot_w = min(sw, 3 if sw % 2 == 1 else 4)
    foot_c0 = (sw - foot_w) // 2
    patch[sh - 1, foot_c0 : foot_c0 + foot_w] = BLACK
    return patch


def _ud_patch(sh: int, sw: int) -> np.ndarray:
    patch = np.full((sh, sw), BG, dtype=int)
    patch[:, 0] = BLACK
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    if sh >= 5 and sw >= 4 and random.random() < 0.5:
        patch[1, 1 : min(sw, 3)] = BLACK
        patch[-2, 1 : min(sw, 3)] = BLACK
    return patch


def _both_patch(sh: int, sw: int) -> np.ndarray:
    return np.full((sh, sw), BLACK, dtype=int)


def _none_patch(sh: int, sw: int) -> np.ndarray:
    patch = np.full((sh, sw), BG, dtype=int)
    patch[:, 0] = BLACK
    patch[-1, :] = BLACK
    if sw > 1 and sh >= 3:
        patch[0, 1] = BLACK
    return patch


def _build_patch(kind: str, max_sh: int, sw: int) -> np.ndarray | None:
    min_h = TYPE_MIN_HEIGHT[kind]
    if max_sh < min_h:
        return None
    for _ in range(50):
        sh = random.randint(min_h, max_sh)
        if kind == "lr":
            patch = _lr_patch(sh, sw)
        elif kind == "ud":
            patch = _ud_patch(sh, sw)
        elif kind == "both":
            patch = _both_patch(sh, sw)
        else:
            patch = _none_patch(sh, sw)
        if _classify_patch(patch) == kind:
            return patch
    return None


def _paint_emission(out: np.ndarray, sep: int, c0: int, patch: np.ndarray) -> None:
    mask = patch == BLACK
    lr_sym, ud_sym = _symmetry_flags(mask)
    if lr_sym:
        out[sep + 1, c0 : c0 + patch.shape[1]] = RED
    if ud_sym:
        for c in _center_columns(c0, c0 + patch.shape[1] - 1):
            out[sep + 1 :, c] = RED


def _render_output(h: int, w: int, sep: int, bands: list[tuple[int, int]], types: list[str]):
    out = np.full((h, w), BG, dtype=int)
    out[sep, :] = SEP_COLOR

    sprites = []
    max_sh = min(SPRITE_H_MAX, sep - 1)
    for (c0, sw), kind in zip(bands, types):
        patch = _build_patch(kind, max_sh, sw)
        if patch is None:
            return None
        sh = patch.shape[0]
        r0 = sep - sh
        if r0 < 1:
            return None
        paste(out, patch, r0, c0, transparent=BG)
        _paint_emission(out, sep, c0, patch)
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h: int, w: int, sep: int, sprites: list[tuple[np.ndarray, int, int]]) -> np.ndarray:
    inp = np.full((h, w), BG, dtype=int)
    inp[sep, :] = SEP_COLOR

    flush_i = random.randrange(len(sprites)) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        delta = 0 if i == flush_i else random.randint(1, r0)
        paste(inp, patch, r0 - delta, c0, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    types = _choose_sprite_types(w)
    if types is None:
        return None
    bands = _sprite_bands(w, types)
    if bands is None:
        return None

    rendered = _render_output(h, w, sep, bands, types)
    if rendered is None:
        return None
    out, sprites = rendered
    inp = _unslide_sprites(h, w, sep, sprites)

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

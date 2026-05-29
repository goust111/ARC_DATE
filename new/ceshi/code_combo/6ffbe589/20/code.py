# Combo variant of 6ffbe589 (index 20)
# combo_spec: D1/2+D3/0+D7/2
# Name: 尺寸排序+镜像翻转+交替方向
#
# Design analysis for the combo variant
# =====================================
# a) concretization_and_trace:
#    - The board is still one largest square multi-color component with 3 concentric color layers:
#      inner sprite, middle inset layer, outer frame layer.
#    - Outside the board there are exactly 3 monochrome cue glyphs, but their colors are disjoint from
#      the board colors, so color matching is impossible.
#    - Cue area is now used only for pairing by size rank, while cue silhouette encodes a base reflection:
#        * full `1 x L` bar -> `h`  -> left-right mirror (`np.fliplr`)
#        * full `L x 1` bar -> `v`  -> top-bottom mirror (`np.flipud`)
#        * `L x L` main diagonal -> `d` -> main-diagonal reflection (`mask.T`)
#      with `L` varying so cue areas are distinct and sortable.
#    - The D7 fusion is applied after pairing: sorted layer indices `0,1,2` (smallest->largest area)
#      use transforms `[base, companion(base), base]`, where companions are `h<->v` and `d->a`
#      (`a` = anti-diagonal reflection). Because the generated board always satisfies
#      inner_area < middle_area < outer_area, this is exactly inner direct / middle opposite / outer direct.
#    - Concrete trace:
#      layer areas after sorting = [6, 15, 28];
#      cue areas after sorting   = [2, 3, 4];
#      cue shapes after sorting  = [`h`, `d`, `v`].
#      Original task would try same-color cue lookup and rotate by cue area mod 4.
#      The fused rule instead does:
#        rank-0 layer (inner, area 6)    <- smallest cue area 2, shape `h` -> apply `h`
#        rank-1 layer (middle, area 15)  <- middle cue area 3, shape `d` -> apply companion `a`
#        rank-2 layer (outer, area 28)   <- largest cue area 4, shape `v` -> apply `v`
#      The differences are simultaneous: matching is global by size, transform type is reflection rather
#      than rotation, and the middle layer reverses the cue-indicated direction inside the same family.
#
# b) complexity_calibration:
#    - Exactly 3 cues are always present, one per layer, which avoids under-constrained subset matching.
#    - Cue colors are disjoint from board colors, so size ordering is visibly necessary.
#    - Cue areas are forced distinct, and board layer areas are forced distinct, so sorted pairing is unique.
#    - Only 3 directly encoded cue families are used (`h`, `v`, `d`); the anti-diagonal reflection appears
#      only as the middle-layer companion of `d`, keeping the family small enough to infer from examples.
#    - The middle layer is forced to break 180-degree symmetry, because the alternating-direction effect
#      would otherwise collapse to the no-alternation reflection baseline.
#
# c) implicit_assumption_audit_for_main:
#    - `crop_components_near_largest(...)` still assumes the board is the unique largest foreground
#      component. The generator preserves that.
#    - The original same-color cue lookup is invalid now; `main()` must instead sort all outside cue
#      components by area and pair them with board layers sorted by area.
#    - Sorted pairing assumes cue count == layer count and both area lists are unique. The generator
#      always produces exactly 3 cues with distinct areas and 3 board layers with distinct areas.
#    - Reflection decoding assumes every cue is one of the supported templates. The generator only emits
#      horizontal bars, vertical bars, and main diagonals.
#    - Alternation now depends on sorted layer rank, so the generator must guarantee that layer area order
#      matches visible depth order (inner < middle < outer). The concentric construction enforces this.
#    - The anti-diagonal companion reflection assumes a square crop. The board crop is always `N x N`.
#
# d) placement_legality:
#    - The board is placed wholly inside the `20 x 20` canvas and remains the unique largest component.
#    - Cue glyphs are placed in the background with Chebyshev distance >= 4 from the board bbox and from
#      each other, so the crop helper excludes them and they never overlap.
#    - Cue glyphs have maximum footprint `4 x 4`, so they fit comfortably without touching the board crop.
#
# e) visual_observability:
#    - A human can see cue colors do not match the board, so color identity is not a viable rule.
#    - Cue length visibly orders the 3 cues, while cue silhouette visibly separates horizontal / vertical /
#      diagonal families.
#    - The alternating-direction effect is exposed by forcing the middle layer to differ from its
#      no-alternation counterpart; on generated samples, the fused solver output differs from the simpler
#      "size-sort + direct reflection" baseline.
#
# f) visual_cleanliness:
#    - The scene remains the same clean ARC layout: one coherent board plus a few detached cue marks.
#    - There is no stacking of multiple encodings onto one color channel inside the board; all control
#      information lives in detached cues.
#
# g) color_change_inference:
#    - Board colors are preserved; there is no recoloring rule.
#    - Cue colors are intentionally non-informative and disjoint from board colors, preventing hidden
#      color-based matching and keeping the task about geometry and sorting.
#
# 9) new_main_assumptions_and_generator_guarantees:
#    - Assumption: each board color corresponds to one unique layer area. Guaranteed by the outer/middle/
#      inner size construction.
#    - Assumption: cue areas are all distinct. Guaranteed by using 3 different cue lengths.
#    - Assumption: the middle layer is not 180-symmetric, so the alternating companion rule is visible.
#      Guaranteed by explicit generator checks.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper (only used by this solver): crop the cluster near the largest non-bg component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox is in the original grid coordinates (inclusive).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    mr0, mc0, mr1, mc1 = map(int, largest[0]["bbox"])

    h, w = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(h - 1), int(mr1 + margin))
    ec1 = min(int(w - 1), int(mc1 + margin))

    keep_mask = np.zeros((h, w), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    assert keep_coords.size > 0, "expected the expanded board crop to retain the largest component"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


BASE_TRANSFORMS = ("h", "v", "d")
ALL_TRANSFORMS = ("h", "v", "d", "a")
COMPANION_TRANSFORM = {"h": "v", "v": "h", "d": "a"}


def apply_transform(mask: np.ndarray, transform: str) -> np.ndarray:
    """Apply one reflection to a square layer mask."""
    if transform == "h":
        return np.fliplr(mask)
    if transform == "v":
        return np.flipud(mask)
    if transform == "d":
        assert mask.shape[0] == mask.shape[1], "main-diagonal reflection expects a square crop"
        return mask.T
    if transform == "a":
        assert mask.shape[0] == mask.shape[1], "anti-diagonal reflection expects a square crop"
        return np.flipud(np.fliplr(mask)).T
    raise AssertionError(f"unknown transform: {transform}")


def _effective_transform(layer_idx: int, base_transform: str) -> str:
    """Even sorted layer ranks use the base cue direction; odd rank uses its companion direction."""
    if int(layer_idx) % 2 == 0:
        return str(base_transform)
    return COMPANION_TRANSFORM[str(base_transform)]


def reflect_color_layer(grid: np.ndarray, color: int, transform: str, *, bg: int | None = None) -> np.ndarray:
    """Mirror one color layer inside the fixed board crop."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    col = int(color)
    mask = grid == col
    tmask = apply_transform(mask, str(transform))

    out = grid.copy()
    out[out == col] = bg
    out[tmask] = col
    return out


def _sorted_layer_records(crop: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Return board layers sorted by total color area (smallest to largest)."""
    records = [{"color": int(col), "area": int(np.sum(crop == int(col)))} for col in colors_present(crop, exclude={int(bg)})]
    assert len(records) == 3, "expected exactly three board colors"
    areas = [int(rec["area"]) for rec in records]
    assert len(set(areas)) == len(areas), "expected distinct board-layer areas for size-order pairing"
    return sorted(records, key=lambda rec: (int(rec["area"]), int(rec["color"])))


def _decode_cue_transform(grid: np.ndarray, comp: dict) -> str:
    """Decode one outside cue component into its base reflection family."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    cue_color = int(comp["color"])
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == cue_color
    h, w = patch.shape

    if h == 1 and w >= 2 and bool(np.all(patch)):
        return "h"
    if w == 1 and h >= 2 and bool(np.all(patch)):
        return "v"
    if h == w and h >= 2 and np.array_equal(patch, np.eye(h, dtype=bool)):
        return "d"
    raise AssertionError(f"unrecognized cue shape for color {cue_color}")


def _sorted_outside_cues(outside_grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    """Return outside cue components sorted by area (smallest to largest)."""
    cues = find_components(outside_grid, bg=int(bg), connectivity=8, monochromatic=True, return_format="dict")
    assert len(cues) == 3, "expected exactly three outside cue sprites"
    areas = [int(comp["area"]) for comp in cues]
    assert len(set(areas)) == len(areas), "expected distinct cue areas for size-order pairing"
    return sorted(cues, key=lambda comp: (int(comp["area"]), int(comp["color"])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board and mirror its layers by size-ranked cue glyphs with an alternating direction rule.

    Input structure:
    - The dominant color is the background.
    - One largest square foreground cluster is a 3-layer concentric board.
    - Three detached monochrome cue glyphs sit outside the board; their colors do not match the board.

    Rule:
    - Crop tightly around the main board cluster.
    - Sort the board's 3 color layers by pixel count from smallest to largest.
    - Sort the 3 outside cue glyphs by pixel count from smallest to largest and pair the two lists by rank.
    - Each cue silhouette gives a base reflection family: horizontal bar -> left-right mirror, vertical bar
      -> top-bottom mirror, main diagonal -> main-diagonal reflection.
    - Apply that base reflection to the smallest and largest matched layers, but apply the companion
      direction to the middle matched layer (`h<->v`, `d->anti-diagonal`).

    Key insight:
    - Solving requires all three signals at once: cue size decides which layer is targeted, cue shape decides
      which mirror family is active, and the matched layer rank decides whether the cue direction is used
      directly or flipped to its companion.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the main board and recover its layers in smallest->largest area order ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    layers = _sorted_layer_records(crop, int(bg))
    out = crop.copy()

    # --- STEP 2) Pair sorted layers with sorted outside cues, decode the cue family, then alternate direction ---
    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    cues = _sorted_outside_cues(outside_grid, int(bg))

    for layer_idx, (layer, cue) in enumerate(zip(layers, cues)):
        base_transform = _decode_cue_transform(outside_grid, cue)
        transform = _effective_transform(layer_idx, base_transform)
        out = reflect_color_layer(out, int(layer["color"]), transform, bg=int(bg))

    return out


# --------------- generator (combo D1/2 + D3/0 + D7/2) ---------------

# ---- Configuration constants ----
CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 3

OUTER_DENSITY_LO = 0.17
OUTER_DENSITY_HI = 0.32
MID_DENSITY_LO = 0.30
MID_DENSITY_HI = 0.50
MID_OUTER_CAP = 0.32

LAYER_BAND_WIDTH = 2
MIDDLE_INSET = 2
ASYM_MARKER = (1, 2)
INNER_SPAN_MARGIN = 8

CUE_LENGTHS = (2, 3, 4)
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4

GEN_RETRIES = 500


def _mirror_partner(r: int, c: int, n: int, transform: str) -> tuple[int, int]:
    """Return the partner cell paired with (r, c) by the requested reflection."""
    if transform == "h":
        return r, n - 1 - c
    if transform == "v":
        return n - 1 - r, c
    if transform == "d":
        return c, r
    if transform == "a":
        return n - 1 - c, n - 1 - r
    raise AssertionError(f"unknown transform: {transform}")


def _is_visible_under(mask: np.ndarray, transform: str) -> bool:
    """True iff applying the requested reflection changes the mask."""
    return not np.array_equal(mask, apply_transform(mask, str(transform)))


def _is_rot180_asym(mask: np.ndarray) -> bool:
    """True iff the mask is not invariant under 180-degree rotation."""
    return not np.array_equal(mask, np.rot90(mask, 2))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    """Checkerboard pattern restricted to the perimeter of an n*n grid."""
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + phase) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    """All (r, c) within `width` rows/cols of any edge in an n*n grid."""
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < width
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    """Randomly set cells in mask until it reaches target count. Returns success."""
    random.shuffle(candidates)
    for r, c in candidates:
        if int(mask.sum()) >= int(target):
            return True
        mask[r, c] = True
    return int(mask.sum()) >= int(target)


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    *,
    transform: str,
    extra_lo_terms: list[int] = (),
    require_rot180_asym: bool = False,
) -> np.ndarray | None:
    """Build a perimeter-heavy layer, forcing visibility under one requested reflection."""
    mask = _checker_perimeter(n, random.randrange(2))
    partner = _mirror_partner(*ASYM_MARKER, n, str(transform))
    assert partner != ASYM_MARKER, "asymmetry marker must not lie on the chosen reflection axis"

    mask[ASYM_MARKER] = True
    mask[partner] = False
    blocked = {partner}

    lo = max(int(np.ceil(density_lo * n * n)), int(mask.sum()), *extra_lo_terms)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    candidates = [
        (r, c)
        for r, c in _band_cells(n, LAYER_BAND_WIDTH)
        if (r, c) not in blocked and not mask[r, c]
    ]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_visible_under(mask, str(transform)):
        return None
    if require_rot180_asym and not _is_rot180_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int, transform: str) -> np.ndarray | None:
    """Build the middle concentric layer, inset by MIDDLE_INSET from each edge."""
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(
        m,
        MID_DENSITY_LO,
        min(MID_DENSITY_HI, hi_cap / (m * m)),
        transform=str(transform),
        require_rot180_asym=True,
    )
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_visible_under(mask, str(transform)):
        return None
    if not _is_rot180_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int, transform: str) -> np.ndarray | None:
    """Build the outer frame layer spanning the full n*n grid."""
    return _make_asym_layer(
        n,
        OUTER_DENSITY_LO,
        OUTER_DENSITY_HI,
        transform=str(transform),
        extra_lo_terms=[int(min_area)],
    )


def _valid_inner_spans(n: int) -> list[int]:
    """Return valid inner-sprite span sizes for a board of side n."""
    max_span = n - INNER_SPAN_MARGIN
    return [s for s in range(2, max_span + 1) if (s % 2) == (n % 2)]


def _make_inner_local(span: int) -> np.ndarray:
    """Build a small centered local sprite before embedding it into the board."""
    if span == 2:
        local = np.ones((2, 2), dtype=bool)
        local[random.randrange(2), random.randrange(2)] = False
        return local
    if span == 3:
        variants = [
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ]
        local = random.choice(variants).copy()
        return np.rot90(local, random.randrange(4))

    local = np.zeros((span, span), dtype=bool)
    if span == 4:
        pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
    else:
        pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
    for r, c in pts:
        local[r, c] = True
    if random.randrange(2):
        local = np.fliplr(local)
    return np.rot90(local, random.randrange(4))


def _make_inner_mask(n: int, transform: str) -> np.ndarray | None:
    """Build the small central sprite layer and ensure cue-driven flips stay visible."""
    spans = _valid_inner_spans(n)
    if not spans:
        return None

    for _ in range(40):
        span = random.choice(spans)
        local = _make_inner_local(span)
        mask = np.zeros((n, n), dtype=bool)
        top = (n - span) // 2
        mask[top : top + span, top : top + span] = local
        if _is_visible_under(mask, str(transform)):
            return mask
    return None


def _cue_patch(color: int, cue_length: int, transform: str) -> np.ndarray:
    """Return the monochrome cue glyph for one base reflection family and one sortable length."""
    cue_length = int(cue_length)
    if transform == "h":
        patch = np.full((1, cue_length), int(color), dtype=int)
    elif transform == "v":
        patch = np.full((cue_length, 1), int(color), dtype=int)
    elif transform == "d":
        patch = np.zeros((cue_length, cue_length), dtype=int)
        np.fill_diagonal(patch, int(color))
    else:
        raise AssertionError(f"unexpected cue transform: {transform}")
    return patch


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev (L-inf) gap distance between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Place a cue patch on the canvas, respecting distance constraints. Returns success."""
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            candidates.append((r, c, bb))
    if not candidates:
        return False
    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int) -> tuple[int, int]:
    """Pick a random board top-left position, flush against one edge of the canvas."""
    spare = CANVAS_SIZE - n
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


def _solve_without_alternation(grid: np.ndarray) -> np.ndarray:
    """Reference baseline: size-sorted pairing plus direct cue reflection, without D7 alternation."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    layers = _sorted_layer_records(crop, int(bg))
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    cues = _sorted_outside_cues(outside_grid, int(bg))

    for layer, cue in zip(layers, cues):
        base_transform = _decode_cue_transform(outside_grid, cue)
        out = reflect_color_layer(out, int(layer["color"]), base_transform, bg=int(bg))
    return out


def _solve_original_like(grid: np.ndarray) -> np.ndarray:
    """Reference baseline close to the original puzzle: color-match cues and rotate by area mod 4."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    for col in colors_present(crop, exclude={int(bg)}):
        outside_comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        if not outside_comps:
            continue
        cue_size = int(outside_comps[0]["area"])
        rot_k = (-cue_size) % 4
        if rot_k:
            mask = crop == int(col)
            rmask = np.rot90(mask, rot_k)
            updated = out.copy()
            updated[updated == int(col)] = int(bg)
            updated[rmask] = int(col)
            out = updated
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    palette = random.sample(range(1, 10), 2 * NUM_COLORS)
    board_colors = [int(c) for c in palette[:NUM_COLORS]]
    cue_colors = [int(c) for c in palette[NUM_COLORS:]]
    c_outer, c_middle, c_inner = board_colors

    base_transforms_sorted = [random.choice(BASE_TRANSFORMS) for _ in range(NUM_COLORS)]
    actual_transforms_sorted = [_effective_transform(i, base) for i, base in enumerate(base_transforms_sorted)]
    cue_lengths_sorted = list(CUE_LENGTHS)

    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)

    outer_actual = actual_transforms_sorted[2]
    middle_actual = actual_transforms_sorted[1]
    inner_actual = actual_transforms_sorted[0]

    mid_mask = _make_middle_mask(n, middle_actual)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, inner_actual)
    if inner_mask is None:
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD, outer_actual)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = int(c_outer)
    out_board[mid_mask] = int(c_middle)
    out_board[inner_mask] = int(c_inner)

    layer_records = [{"color": int(col), "area": int(np.sum(out_board == int(col)))} for col in board_colors]
    sorted_layers = sorted(layer_records, key=lambda rec: (int(rec["area"]), int(rec["color"])))
    sorted_areas = [int(rec["area"]) for rec in sorted_layers]
    if len(set(sorted_areas)) != NUM_COLORS:
        return None

    # The area order is part of the solver logic: inner < middle < outer.
    if [int(rec["color"]) for rec in sorted_layers] != [int(c_inner), int(c_middle), int(c_outer)]:
        return None

    color_to_actual = {}
    color_to_base = {}
    for idx, rec in enumerate(sorted_layers):
        color_to_actual[int(rec["color"])] = str(actual_transforms_sorted[idx])
        color_to_base[int(rec["color"])] = str(base_transforms_sorted[idx])

    in_board = np.zeros((n, n), dtype=int)
    for col in board_colors:
        inv = apply_transform(out_board == int(col), color_to_actual[int(col)])
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    if np.array_equal(in_board, out_board):
        return None

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = []
    for idx, rec in enumerate(sorted_layers):
        cue_items.append(
            (
                int(cue_colors[idx]),
                int(cue_lengths_sorted[idx]),
                str(base_transforms_sorted[idx]),
            )
        )
    random.shuffle(cue_items)
    for cue_color, cue_length, base_transform in cue_items:
        patch = _cue_patch(int(cue_color), int(cue_length), str(base_transform))
        if not _place_cue(canvas, patch, board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    for bb in cue_bboxes:
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            return None

    combo_out = main(canvas)
    if not np.array_equal(combo_out, out_board):
        return None

    no_alt_out = _solve_without_alternation(canvas)
    if np.array_equal(no_alt_out, out_board):
        return None

    original_like_out = _solve_original_like(canvas)
    if np.array_equal(original_like_out, out_board):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

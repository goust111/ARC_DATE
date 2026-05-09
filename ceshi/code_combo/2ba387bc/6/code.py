# Combo variant of 2ba387bc (index 6)
# combo_spec: D1/3+D16/0
# Name: 对称性分类+框架旋转
#
# Design analysis for combo D1/3 + D16/0
# ---------------------------------------
# a) Concrete behavior trace
# - Original task:
#   each 4x4 object was either
#     frame = uniform border + all-bg 2x2 interior
#     solid = fully filled 4x4
#   The solver classified by `inner == bg`, sorted by input row, and stacked
#   frames in the left column and solids in the right column.
# - Fused variant used here:
#   every object is now a "framed tile": uniform 4x4 border with a 2x2 internal
#   marker made of colors 1 and 2. No object's interior is all background.
#   Two marker layouts are used:
#     symmetric bucket:
#       [[1, 2],
#        [1, 2]]
#       This has a horizontal reflection axis because the two rows are equal.
#     asymmetric bucket:
#       [[1, 2],
#        [1, 1]]
#       This has no horizontal/vertical reflection axis.
#   Classification is therefore based on the marker's symmetry, not on whether
#   the interior is empty.
# - Output rule:
#   tiles from the symmetric bucket go to the left column and are rotated 90 deg
#   CCW before pasting; asymmetric tiles go to the right column unchanged.
#   Example:
#     input inner [[1,2],[1,2]] -> output inner after rotation [[2,2],[1,1]]
#   so the stripe direction changes, making the conditional rotation observable.
#
# b) Complexity calibration
# - If symmetric markers were also 90-degree symmetric, the rotation would be
#   invisible and D16/0 would collapse into decoration. The chosen symmetric
#   marker has exactly one visible stripe direction, so every rotation matters.
# - If the two buckets differed only by color count or by "has background hole",
#   humans could ignore symmetry entirely. Here both buckets are dense 4x4 tiles
#   with the same two marker colors; the difference is the arrangement.
# - Each generated sample contains at least one object from each bucket, so the
#   two-column split is always supported by repeated evidence.
#
# c) Implicit assumptions audit against the original main()
# - `bg = get_bg(g)` still assumes the background is the most common color.
#   The generator keeps a large empty canvas so this remains true.
# - `find_components(..., monochromatic=False)` assumes each logical object is a
#   single non-bg connected component. That still holds because every tile is a
#   fully connected 4x4 patch, even though it uses multiple colors.
# - The original branch
#       if border is uniform and inner == bg: frame else solid
#   is invalid now because inner is never all bg. The new solver replaces that
#   assumption with an explicit inner-marker symmetry test.
# - Sorting by `(r0, c0)` is still valid because the fused variant changes only
#   the classification criterion and the post-classification transform.
# - Output width 8 and 4-row stacking still hold because every object remains 4x4.
#
# d) Placement legality
# - The generator places only 4x4 tiles.
# - `random_free_location(..., padding=1, padding_connectivity=4)` guarantees
#   no overlap and at least one background cell of 4-neighborhood separation, so
#   neighboring tiles never merge into one component.
#
# e) Visual observability
# - A human can see that all objects share the same outer structure while their
#   internal 2x2 markers differ by symmetry.
# - The left-column outputs show the striped marker turned by 90 degrees, which
#   ties the classification rule and the transform rule together.
# - We do not use hidden features such as absolute color ids for sorting or for
#   deciding the transform; only visible shape/layout information matters.
#
# f) Visual cleanliness
# - Only one object occupies each 4x4 bbox and objects are spaced apart.
# - The marker palette is fixed (1 and 2), while the border color varies per
#   object, so objects are easy to track from input to output without clutter.
#
# g) Anti-degeneration checks
# - Under the original solver, all objects would fall into the "solid" bucket
#   because no 2x2 interior is fully background. The fused generator therefore
#   forces a behavioral disagreement with the original task in every sample.
# - We later quantify this by checking that generated samples disagree with the
#   baseline original solver and that at least one rotated symmetric tile is
#   present in every sample.
#
# New solver assumptions introduced by this variant
# - Every relevant object is exactly 4x4 with a uniform outer border color.
# - The class cue is entirely inside the 2x2 center.
# - Symmetric markers are generated to have a horizontal or vertical reflection
#   axis, and asymmetric markers are generated to have neither; the generator
#   constructs exactly those two templates, so the solver's classification test
#   is guaranteed to be decisive.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


BG = 0
MARK_A = 1
MARK_B = 2
BORDER_COLORS = (3, 4, 5, 6, 7, 8, 9)

SYMMETRIC_INNER = np.array([[MARK_A, MARK_B], [MARK_A, MARK_B]], dtype=int)
ASYMMETRIC_INNER = np.array([[MARK_A, MARK_B], [MARK_A, MARK_A]], dtype=int)


def _has_hv_reflection(inner: np.ndarray) -> bool:
    """Return True when the 2x2 marker has a horizontal or vertical mirror axis."""
    return bool(np.array_equal(inner, np.flipud(inner)) or np.array_equal(inner, np.fliplr(inner)))


def _is_valid_tile(sub: np.ndarray, bg: int) -> bool:
    """Accept only dense 4x4 tiles with a single border color and no background inside."""
    if sub.shape != (4, 4):
        return False
    if bool((sub == int(bg)).any()):
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:3, 0], sub[1:3, -1]])
    return len(set(map(int, border.tolist()))) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 framed tiles into two columns by inner-marker symmetry.

    Bucket rule:
    - symmetric marker (has horizontal/vertical reflection axis) -> left column
    - asymmetric marker -> right column

    Transform rule:
    - left-column tiles are rotated 90 degrees CCW before placement
    - right-column tiles stay unchanged

    In both buckets, objects are ordered top-to-bottom by their input row, with
    column as a tiebreaker, exactly as in the source task.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    symmetric_tiles: list[tuple[int, int, np.ndarray]] = []
    asymmetric_tiles: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        if not _is_valid_tile(sub, bg):
            continue

        inner = sub[1:3, 1:3]
        if _has_hv_reflection(inner):
            symmetric_tiles.append((r0, c0, sub))
        else:
            asymmetric_tiles.append((r0, c0, sub))

    symmetric_tiles.sort(key=lambda t: (int(t[0]), int(t[1])))
    asymmetric_tiles.sort(key=lambda t: (int(t[0]), int(t[1])))

    if not symmetric_tiles and not asymmetric_tiles:
        return np.zeros((0, 8), dtype=int)

    out_h = 4 * max(len(symmetric_tiles), len(asymmetric_tiles))
    out = np.full((out_h, 8), bg, dtype=int)

    for i, (_r0, _c0, sub) in enumerate(symmetric_tiles):
        paste(out, np.rot90(sub, k=1), i * 4, 0)
    for i, (_r0, _c0, sub) in enumerate(asymmetric_tiles):
        paste(out, sub, i * 4, 4)

    return out


def _make_tile(border_color: int, inner: np.ndarray) -> np.ndarray:
    tile = np.full((4, 4), int(border_color), dtype=int)
    tile[1:3, 1:3] = np.array(inner, dtype=int)
    return tile


def _make_symmetric_tile(border_color: int) -> np.ndarray:
    return _make_tile(border_color, SYMMETRIC_INNER)


def _make_asymmetric_tile(border_color: int) -> np.ndarray:
    return _make_tile(border_color, ASYMMETRIC_INNER)


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(18, 31))
    w = int(np.random.randint(18, 31))
    g = np.full((h, w), BG, dtype=int)

    n_symmetric = int(np.random.randint(1, 4))
    n_asymmetric = int(np.random.randint(1, 4))
    total = n_symmetric + n_asymmetric

    if total > len(BORDER_COLORS):
        return None

    border_colors = np.random.permutation(np.array(BORDER_COLORS, dtype=int))[:total].tolist()
    np.random.shuffle(border_colors)

    tiles: list[np.ndarray] = []
    for col in border_colors[:n_symmetric]:
        tiles.append(_make_symmetric_tile(int(col)))
    for col in border_colors[n_symmetric:]:
        tiles.append(_make_asymmetric_tile(int(col)))

    np.random.shuffle(tiles)

    try:
        for tile in tiles:
            r0, c0 = random_free_location(
                g,
                tile,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, tile, r0, c0, transparent=BG)
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")

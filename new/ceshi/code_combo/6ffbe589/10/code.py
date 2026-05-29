# Combo variant of 6ffbe589 (index 10)
# combo_spec: D1/0+D3/1
# Name: 形状匹配+旋转翻转组合变换
#
# Design analysis for the fused variant:
# a) Concrete behavior:
#    - Original task matched each outside sprite to a board layer by color, then rotated that
#      color mask by a size-derived quarter turn.
#    - This variant removes color pairing entirely. Each active board layer contains a visible
#      bg "bite" carved out of a deterministic concentric support region. The outside cue uses
#      the exact same bite shape, so pairing is done by shape equality between cue and board gap.
#    - The matched cue then contributes two independent parameters: `area % 4` gives the CW
#      rotation count, and diagonal symmetry (main or anti diagonal) decides whether a left-right
#      reflection is also applied before the rotation.
# b) Complexity calibration:
#    - The board still has only 3 nested layers, so the object vocabulary stays small.
#    - Cue parameters live in a tiny discrete space: rotation in {0,1,2,3}, flip in {False,True}.
#    - Exact-shape matching stays visually legible because each example uses three small, distinct
#      cue/gap shapes, avoiding ambiguous many-to-many correspondence.
# c) Implicit assumptions changed from the original:
#    - `main()` can no longer assume cue color == layer color.
#    - The solver now assumes each layer occupies a known concentric support region, and any gap
#      inside that support is a single cue-sized bite rather than arbitrary noise.
#    - The generator therefore guarantees: centered square supports, unique gap shape per active
#      layer, cue distance >= 4 from the board crop, and non-identity transforms that are visible.
# d) Placement legality:
#    - Board supports partition the NxN crop into outer band / middle band / inner square, so
#      layers never overlap and always remain in-bounds.
#    - Cue placement reuses the original far-away placement constraints to keep the board as the
#      unique largest component and to prevent accidental inclusion in the crop.
# e) Visual observability:
#    - Humans can see the same small shape appear both as a layer's bite and as a detached cue.
#    - Cue color is intentionally unrelated to board colors, so shape matching is the only stable
#      correspondence signal. Rotation/flip remain visible because the bite changes side/orientation.
# f) Visual cleanliness:
#    - Only one gap is carved from each support region, and supports are nested, non-overlapping,
#      and centered, so the board remains tidy rather than becoming a cluttered collage.
# g) Non-degeneracy target:
#    - The generator forces all cue colors to differ from board colors and includes flipped cues in
#      the sample set, so the fused behavior is observably different from the original color-only,
#      rotation-only task.

from __future__ import annotations

import random
import numpy as np
from arc_utils import colors_present, find_components, get_bg, paste


CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 3

OUTER_WIDTH = 2
MIDDLE_INSET = 2
MIDDLE_WIDTH = 2
INNER_INSET = MIDDLE_INSET + MIDDLE_WIDTH

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
GEN_RETRIES = 600


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """Crop the cluster near the unique largest non-bg component."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    mr0, mc0, mr1, mc1 = map(int, largest[0]["bbox"])

    H, W = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(H - 1), int(mr1 + margin))
    ec1 = min(int(W - 1), int(mc1 + margin))

    keep_mask = np.zeros((H, W), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    assert keep_coords.size > 0
    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=grid.dtype)
    sub = keep_mask[r0 : r1 + 1, c0 : c1 + 1]
    crop[sub] = grid[r0 : r1 + 1, c0 : c1 + 1][sub]
    return crop, (r0, c0, r1, c1), bg


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    bbox = _mask_bbox(mask)
    assert bbox is not None, "expected non-empty mask"
    r0, c0, r1, c1 = bbox
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_mask(mask)
    return tight.shape, tight.astype(np.uint8).tobytes()


def _diag_symmetric(mask: np.ndarray) -> bool:
    m = np.array(mask, dtype=bool)
    if m.shape[0] != m.shape[1]:
        return False
    return bool(np.array_equal(m, m.T) or np.array_equal(m, np.rot90(m.T, 2)))


def _transform_mask(mask: np.ndarray, rot_k: int, do_flip: bool) -> np.ndarray:
    out = np.fliplr(mask) if do_flip else mask.copy()
    return np.rot90(out, -int(rot_k) % 4)


def _inverse_transform_mask(mask: np.ndarray, rot_k: int, do_flip: bool) -> np.ndarray:
    out = np.rot90(mask, int(rot_k) % 4)
    return np.fliplr(out) if do_flip else out


def _apply_color_transform(grid: np.ndarray, color: int, rot_k: int, do_flip: bool, *, bg: int) -> np.ndarray:
    mask = grid == int(color)
    moved = _transform_mask(mask, rot_k, do_flip)
    out = grid.copy()
    out[out == int(color)] = int(bg)
    out[moved] = int(color)
    return out


def _support_masks(n: int) -> list[np.ndarray]:
    rr, cc = np.indices((n, n))
    depth = np.minimum.reduce([rr, cc, n - 1 - rr, n - 1 - cc])
    outer = depth < OUTER_WIDTH

    m = n - 2 * MIDDLE_INSET
    rr2, cc2 = np.indices((m, m))
    depth2 = np.minimum.reduce([rr2, cc2, m - 1 - rr2, m - 1 - cc2])
    middle_local = depth2 < MIDDLE_WIDTH
    middle = np.zeros((n, n), dtype=bool)
    middle[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = middle_local

    inner = np.zeros((n, n), dtype=bool)
    inner_span = n - 2 * INNER_INSET
    inner[INNER_INSET : INNER_INSET + inner_span, INNER_INSET : INNER_INSET + inner_span] = True

    assert inner_span >= 2
    assert not np.any(outer & middle)
    assert not np.any(outer & inner)
    assert not np.any(middle & inner)
    return [outer, middle, inner]


def _one_component(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, connectivity=8, monochromatic=True, colors=[1], return_format="dict")
    return len(comps) == 1


def _hole_candidates(support: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = support.shape
    ph, pw = patch.shape
    on = np.argwhere(patch)
    support_bbox = _mask_bbox(support)
    assert support_bbox is not None
    candidates: list[tuple[int, int]] = []
    for r0 in range(sh - ph + 1):
        for c0 in range(sw - pw + 1):
            pts = on + np.array([r0, c0])
            if not np.all(support[pts[:, 0], pts[:, 1]]):
                continue
            remain = support.copy()
            remain[pts[:, 0], pts[:, 1]] = False
            if not _one_component(remain):
                continue
            if _mask_bbox(remain) != support_bbox:
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _cue_patch_from_hole(hole_mask: np.ndarray) -> np.ndarray:
    return _tight_mask(hole_mask).astype(int)


def _array_from_bool(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(color)
    return out


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
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
    spare = CANVAS_SIZE - n
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


HOLE_LIBRARY: dict[tuple[int, bool], list[np.ndarray]] = {
    (0, False): [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    ],
    (0, True): [
        np.array([[1, 1], [1, 1]], dtype=bool),
    ],
    (1, False): [
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 1]], dtype=bool),
    ],
    (1, True): [
        np.array([[1]], dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    ],
    (2, False): [
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
    ],
    (2, True): [
        np.array([[1, 0], [0, 1]], dtype=bool),
        np.array([[0, 1], [1, 0]], dtype=bool),
    ],
    (3, False): [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
    ],
    (3, True): [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    ],
}


def _valid_patch(mask: np.ndarray, rot_k: int, do_flip: bool) -> bool:
    return int(mask.sum()) % 4 == int(rot_k) and _diag_symmetric(mask) == bool(do_flip)


for (_rk, _flip), _patches in HOLE_LIBRARY.items():
    for _patch in _patches:
        assert _valid_patch(_patch, _rk, _flip)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the main board and realign its color layers by shape-matching detached cue sprites.

    Input structure:
    - bg is the most frequent color.
    - The largest foreground component is a square 3-layer board made of concentric support regions.
    - Some layers contain a small bg bite, and detached monochrome cue sprites repeat those bite shapes.

    Rule:
    - Tight-crop the board.
    - For each board layer, find whether its bite shape appears among the outside cues; pairing depends on
      exact shape equality, not color equality.
    - A matched cue says how to transform that entire color layer: `cue_area % 4` gives the CW rotation
      count, and diagonal symmetry of the cue says whether to do a left-right flip before rotating.
    - Layers with no matching cue stay unchanged.

    Key insight:
    - The cue serves two roles at once: its silhouette identifies which board gap it belongs to, and its
      own geometric features encode the layer's rotation and flip parameters.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    assert crop.shape[0] == crop.shape[1], "expected a square board crop"
    n = int(crop.shape[0])
    supports = _support_masks(n)

    layer_colors = []
    for col in colors_present(crop, exclude={bg}):
        mask = crop == int(col)
        bbox = _mask_bbox(mask)
        assert bbox is not None
        br0, bc0, br1, bc1 = bbox
        layer_colors.append(
            {
                "color": int(col),
                "bbox_area": int((br1 - br0 + 1) * (bc1 - bc0 + 1)),
                "area": int(mask.sum()),
            }
        )
    layer_colors.sort(key=lambda d: (-d["bbox_area"], -d["area"], d["color"]))
    assert len(layer_colors) == 3, "expected three concentric board colors"

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    cue_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    cue_map: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for comp in cue_comps:
        rr0, cc0, rr1, cc1 = map(int, comp["bbox"])
        patch = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        coords = comp["coords"]
        patch[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        sig = patch.shape, patch.astype(np.uint8).tobytes()
        cue_map.setdefault(sig, []).append(
            {
                "rot_k": int(patch.sum()) % 4,
                "do_flip": _diag_symmetric(patch),
            }
        )

    plan: list[tuple[int, int, bool]] = []
    for idx, info in enumerate(layer_colors):
        col = int(info["color"])
        support = supports[idx]
        actual = crop == col
        assert np.all(actual <= support), "layer pixels escaped their support region"
        hole = support & (~actual)
        if not hole.any():
            plan.append((col, 0, False))
            continue
        sig = _shape_signature(hole)
        assert sig in cue_map and len(cue_map[sig]) == 1, "expected a unique cue for each visible bite"
        cue = cue_map[sig][0]
        plan.append((col, int(cue["rot_k"]), bool(cue["do_flip"])))

    out = crop.copy()
    for col, rot_k, do_flip in plan:
        if rot_k == 0 and not do_flip:
            continue
        out = _apply_color_transform(out, col, rot_k, do_flip, bg=bg)
    return out


def _choose_transforms() -> list[tuple[int, bool]]:
    options = [(r, f) for r in range(4) for f in (False, True) if not (r == 0 and not f)]
    while True:
        picked = [random.choice(options) for _ in range(NUM_COLORS)]
        if any(r != 0 for r, _f in picked) and any(f for _r, f in picked) and any(not f for _r, f in picked):
            return picked


def _build_layer_with_hole(
    support: np.ndarray,
    patch: np.ndarray,
    rot_k: int,
    do_flip: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    candidates = _hole_candidates(support, patch)
    random.shuffle(candidates)
    support_bbox = _mask_bbox(support)
    assert support_bbox is not None
    on = np.argwhere(patch)
    for r0, c0 in candidates:
        output_layer = support.copy()
        pts = on + np.array([r0, c0])
        output_layer[pts[:, 0], pts[:, 1]] = False
        input_layer = _inverse_transform_mask(output_layer, rot_k, do_flip)
        if _mask_bbox(input_layer) != support_bbox:
            continue
        if not _one_component(input_layer):
            continue
        if np.array_equal(input_layer, output_layer):
            continue
        rot_only = _transform_mask(input_layer, rot_k, False)
        full = _transform_mask(input_layer, rot_k, do_flip)
        if do_flip and np.array_equal(rot_only, full):
            continue
        return output_layer, input_layer
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)
    supports = _support_masks(n)
    board_colors = random.sample(range(1, 10), NUM_COLORS)
    cue_colors = random.sample([c for c in range(1, 10) if c not in board_colors], NUM_COLORS)
    transforms = _choose_transforms()

    out_layers: list[np.ndarray] = []
    in_layers: list[np.ndarray] = []
    cue_patches: list[np.ndarray] = []
    seen_sigs: set[tuple[tuple[int, int], bytes]] = set()

    for idx in range(NUM_COLORS):
        support = supports[idx]
        rot_k, do_flip = transforms[idx]
        patch_options = HOLE_LIBRARY[(rot_k, do_flip)].copy()
        random.shuffle(patch_options)
        built = None
        for patch in patch_options:
            res = _build_layer_with_hole(support, patch, rot_k, do_flip)
            if res is None:
                continue
            out_layer, input_layer = res
            input_hole = support & (~input_layer)
            sig = _shape_signature(input_hole)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            built = (out_layer, input_layer, _cue_patch_from_hole(input_hole))
            break
        if built is None:
            return None
        out_layers.append(built[0])
        in_layers.append(built[1])
        cue_patches.append(built[2])

    out_board = np.zeros((n, n), dtype=int)
    in_board = np.zeros((n, n), dtype=int)
    for col, mask in zip(board_colors, out_layers):
        out_board[mask] = int(col)
    for col, mask in zip(board_colors, in_layers):
        in_board[mask] = int(col)

    if not all(_mask_bbox(in_board == col) == _mask_bbox(supports[idx]) for idx, col in enumerate(board_colors)):
        return None

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = list(zip(cue_colors, cue_patches))
    random.shuffle(cue_items)
    for cue_color, cue_mask in cue_items:
        patch = _array_from_bool(cue_mask.astype(bool), cue_color)
        if not _place_cue(canvas, patch, board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    if any(_bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST for bb in cue_bboxes):
        return None

    got = main(canvas)
    if not np.array_equal(got, out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 6ffbe589 input")

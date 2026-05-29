# Combo variant of f560132c (index 5)
# combo_spec: D2/2+D16/0
# Name: 对角线标记对+对角线分割

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The original input showed four filled polyominoes, one of which carried the palette. Here that
#    representation is replaced by two sparse layers that must be fused:
#      - a connected 4x4 legend block whose side labels tell which marker color means top/right/
#        bottom/left in the final diagonal split;
#      - four non-connected marker pairs, each only two same-colored pixels on opposite bbox corners.
#    A small mental trace is:
#      - Old rule: isolate a filled piece -> try D4 placements -> color it by rectangular quadrant.
#      - New rule: isolate a color-coded pixel pair -> infer only its bounding box -> look up its role
#        from the legend side label -> recover the staircase triangle that belongs in that box under a
#        diagonal partition -> assemble the four triangles into one square.
#      - Difference: the sparse markers do not reveal area or connectivity; the solver must combine
#        color-role information and bbox geometry before any assembly is possible.
# b) Complexity calibration:
#    The task still has only four roles and a fixed 2x2 palette, so the hypothesis space remains
#    compact. To avoid a visually weak variant, the generator uses actual diagonal sector masks rather
#    than plain rectangles, and it rejects configurations where the old axis-aligned quadrant coloring
#    would accidentally match the new diagonal rule.
# c) Implicit assumptions audited from the original solver:
#      - "each piece is a connected component" is no longer true; marker pairs are single-pixel
#        components, so component-based piece extraction must be replaced.
#      - "one component is the legend piece" is also false; the legend is now a separate connected
#        block, not one of the four source sprites.
#      - "piece shape is directly visible" is false; only the bbox is visible, so the solver must
#        search over generator-legal diagonal partitions and match them by observed crop sizes.
#      - "centroid quantization picks palette cells" is false; palette cells now correspond to the
#        diagonal side sectors top/right/bottom/left.
# d) Placement legality:
#    Each pair reserves its whole inferred bbox during scattering even though only two marker pixels
#    are painted. This prevents overlapping inferred rectangles, keeps objects in-bounds, and makes
#    the intended boxes readable to a human.
# e) Visual observability:
#    A human sees two clear structural cues across examples:
#      - the legend block visibly binds the four marker colors to the four sides of the 2x2 palette;
#      - each color then appears exactly twice elsewhere, marking a rectangle whose hidden content is
#        one of the diagonal staircase triangles. The output shows the four side sectors meeting along
#        diagonal boundaries, which is visibly different from the original cross-quadrant output.
# f) Visual cleanliness:
#    No cell carries multiple meanings. The legend is one compact block; each source sprite is just
#    one color repeated twice; reserved bbox separation avoids accidental interleaving of different
#    inferred rectangles.
# g) Color inference:
#    Only eight non-background colors are used: four marker colors and four palette colors. The color
#    change is fully signposted by the legend block, so the mapping is structural rather than based on
#    raw color ids.


ROLE_TOP = 0
ROLE_RIGHT = 1
ROLE_BOTTOM = 2
ROLE_LEFT = 3
ROLE_ORDER = (ROLE_TOP, ROLE_RIGHT, ROLE_BOTTOM, ROLE_LEFT)
ROLE_TO_LEGEND_CELL = {
    ROLE_TOP: (0, 0),
    ROLE_RIGHT: (0, 1),
    ROLE_BOTTOM: (1, 1),
    ROLE_LEFT: (1, 0),
}
ROLE_TO_DIAG = {
    ROLE_TOP: "TLBR",
    ROLE_RIGHT: "TRBL",
    ROLE_BOTTOM: "TLBR",
    ROLE_LEFT: "TRBL",
}


SIDE_CHOICES = [10, 12]
K = 2
NUM_PIECES = 4
NUM_COLORS = 8
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
LEGEND_PATCH_SIZE = 4
INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 34
RESERVED_DENSITY_LO = 0.28
RESERVED_DENSITY_HI = 0.42
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 400


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _build_sector_masks(side: int, row_shift: int, col_shift: int) -> list[np.ndarray]:
    masks = [np.zeros((side, side), dtype=bool) for _ in range(NUM_PIECES)]
    for r in range(side):
        for c in range(side):
            vr = 2 * r + 1 - side - row_shift
            vc = 2 * c + 1 - side - col_shift
            if abs(vr) > abs(vc):
                idx = ROLE_TOP if vr < 0 else ROLE_BOTTOM
            else:
                idx = ROLE_LEFT if vc < 0 else ROLE_RIGHT
            masks[idx][r, c] = True
    return masks


def _old_cross_cell(mask: np.ndarray, side: int) -> tuple[int, int]:
    pts = np.argwhere(mask)
    cr = float(pts[:, 0].mean())
    cc = float(pts[:, 1].mean())
    rr = max(0, min(1, int(cr * K / side)))
    cc2 = max(0, min(1, int(cc * K / side)))
    return rr, cc2


def _expected_from_masks(masks: list[np.ndarray], legend_map: np.ndarray, use_diagonal: bool) -> np.ndarray:
    side = masks[0].shape[0]
    out = np.zeros((side, side), dtype=int)
    for role, mask in enumerate(masks):
        if use_diagonal:
            rr, cc = ROLE_TO_LEGEND_CELL[role]
        else:
            rr, cc = _old_cross_cell(mask, side)
        out[mask] = int(legend_map[rr, cc])
    return out


def _candidate_sector_configs(side: int) -> list[tuple[int, int]]:
    min_area, max_area = _area_bounds(side)
    candidates = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for row_shift in range(-(side // 3), side // 3 + 1):
        for col_shift in range(-(side // 3), side // 3 + 1):
            if (row_shift - col_shift) % 2 == 0:
                continue
            masks = _build_sector_masks(side, row_shift, col_shift)
            areas = [int(m.sum()) for m in masks]
            if any(a < min_area or a > max_area for a in areas):
                continue
            sig = tuple(tuple(map(int, _crop_mask(m).shape)) for m in masks)
            test_map = np.array([[1, 2], [3, 4]], dtype=int)
            if np.array_equal(_expected_from_masks(masks, test_map, True), _expected_from_masks(masks, test_map, False)):
                continue
            assert sig not in signatures, "sector signature must uniquely identify the diagonal split"
            signatures.add(sig)
            candidates.append((row_shift, col_shift))
    assert candidates, f"no valid diagonal sector configs for side={side}"
    return candidates


def _all_candidates() -> list[tuple[int, int, int]]:
    out = []
    for side in SIDE_CHOICES:
        for row_shift, col_shift in _candidate_sector_configs(side):
            out.append((side, row_shift, col_shift))
    return out


CANDIDATE_CONFIGS = _all_candidates()


def _legend_patch(marker_colors: dict[int, int], legend_map: np.ndarray) -> np.ndarray:
    patch = np.zeros((LEGEND_PATCH_SIZE, LEGEND_PATCH_SIZE), dtype=int)
    patch[0, 1:3] = marker_colors[ROLE_TOP]
    patch[1:3, 3] = marker_colors[ROLE_RIGHT]
    patch[3, 1:3] = marker_colors[ROLE_BOTTOM]
    patch[1:3, 0] = marker_colors[ROLE_LEFT]
    patch[1:3, 1:3] = legend_map
    return patch


def _extract_legend_info(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    comps = components_mask(g != bg, connectivity=4)
    legend_candidates = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape == (LEGEND_PATCH_SIZE, LEGEND_PATCH_SIZE):
            legend_candidates.append((pts, sub))
    assert len(legend_candidates) == 1, "there must be exactly one 4x4 legend block"

    pts, sub = legend_candidates[0]
    assert int(sub[0, 0]) == bg and int(sub[0, 3]) == bg and int(sub[3, 0]) == bg and int(sub[3, 3]) == bg, \
        "legend corners must be background"

    top = int(sub[0, 1])
    assert int(sub[0, 2]) == top and top != bg, "top legend label must be a repeated non-bg color"
    right = int(sub[1, 3])
    assert int(sub[2, 3]) == right and right != bg, "right legend label must be a repeated non-bg color"
    bottom = int(sub[3, 1])
    assert int(sub[3, 2]) == bottom and bottom != bg, "bottom legend label must be a repeated non-bg color"
    left = int(sub[1, 0])
    assert int(sub[2, 0]) == left and left != bg, "left legend label must be a repeated non-bg color"

    legend_map = np.asarray(sub[1:3, 1:3], dtype=int).copy()
    label_colors = [top, right, bottom, left]
    palette_colors = list(map(int, legend_map.ravel()))
    assert len(set(label_colors)) == 4, "legend labels must use four distinct marker colors"
    assert len(set(palette_colors)) == 4, "legend palette must use four distinct colors"
    assert set(label_colors).isdisjoint(palette_colors), "marker colors and palette colors must be disjoint"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, c in pts:
        legend_mask[int(r), int(c)] = True

    marker_of_role = {
        ROLE_TOP: top,
        ROLE_RIGHT: right,
        ROLE_BOTTOM: bottom,
        ROLE_LEFT: left,
    }
    return marker_of_role, legend_map, legend_mask


def _pair_signature_for_role(coords: np.ndarray, role: int) -> tuple[int, int]:
    assert len(coords) == 2, "each marker sprite must contain exactly two pixels"
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h >= 2 and w >= 2, "marker pair bbox must be at least 2x2"
    cells = {tuple(map(int, rc)) for rc in coords}
    if ROLE_TO_DIAG[role] == "TLBR":
        expected = {(r0, c0), (r1, c1)}
    else:
        expected = {(r0, c1), (r1, c0)}
    assert cells == expected, "marker pixels must occupy the expected diagonal corners of their bbox"
    return h, w


def _match_partition_from_signature(signature: tuple[tuple[int, int], ...]) -> tuple[int, int, int, list[np.ndarray]]:
    matches = []
    for side, row_shift, col_shift in CANDIDATE_CONFIGS:
        masks = _build_sector_masks(side, row_shift, col_shift)
        cand_sig = tuple(tuple(map(int, _crop_mask(m).shape)) for m in masks)
        if cand_sig == signature:
            matches.append((side, row_shift, col_shift, masks))
    assert len(matches) == 1, "marker bbox sizes must determine a unique diagonal partition"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a diagonal-partition square from sparse marker pairs and recolor it from a side-labeled legend.

    Input structure:
    - bg is the most common color.
    - One connected 4×4 legend block contains a 2×2 output palette in its center and four side labels
      whose colors identify the top/right/bottom/left triangle roles.
    - Outside that legend, each source sprite is only a same-colored diagonal marker pair: two pixels at
      opposite corners of a hidden rectangle, not a filled connected piece.

    Rule:
    - Read the legend side labels to map marker colors to the four diagonal sectors of the final square.
    - For each role, use its two marker pixels to recover the sprite bbox, then find the unique generator-
      legal diagonal partition whose cropped sector boxes match those four observed bboxes.
    - Rebuild the four staircase triangles from that partition and output the full square recolored by the
      legend palette in clockwise side order: top <- TL, right <- TR, bottom <- BR, left <- BL.

    Key insight:
    The sparse marker pairs do not show the pieces themselves. Their colors tell which side-sector each
    hidden triangle represents, while the pair geometry tells only the bounding box. The legend and the
    diagonal split must be fused to recover the actual piece shapes.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    marker_of_role, legend_map, legend_mask = _extract_legend_info(g, bg)

    signature = []
    for role in ROLE_ORDER:
        color = marker_of_role[role]
        coords = np.argwhere((g == color) & (~legend_mask))
        signature.append(_pair_signature_for_role(coords, role))

    _, _, _, masks = _match_partition_from_signature(tuple(signature))
    return _expected_from_masks(masks, legend_map, True)


def _pair_patch(shape: tuple[int, int], role: int, color: int) -> np.ndarray:
    h, w = map(int, shape)
    patch = np.zeros((h, w), dtype=int)
    if ROLE_TO_DIAG[role] == "TLBR":
        patch[0, 0] = color
        patch[h - 1, w - 1] = color
    else:
        patch[0, w - 1] = color
        patch[h - 1, 0] = color
    return patch


def _scatter_objects(pair_shapes: dict[int, tuple[int, int]], marker_colors: dict[int, int], legend_map: np.ndarray) -> np.ndarray | None:
    reserve_total = LEGEND_PATCH_SIZE * LEGEND_PATCH_SIZE
    reserve_total += sum(int(h) * int(w) for h, w in pair_shapes.values())
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(reserve_total / RESERVED_DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(reserve_total / RESERVED_DENSITY_LO)))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    objects: list[tuple[int, np.ndarray, np.ndarray]] = []
    legend_patch = _legend_patch(marker_colors, legend_map)
    objects.append((-1, legend_patch, np.ones_like(legend_patch, dtype=int)))
    for role in random.sample(list(ROLE_ORDER), len(ROLE_ORDER)):
        shape = pair_shapes[role]
        patch = _pair_patch(shape, role, marker_colors[role])
        box = np.ones(shape, dtype=int)
        objects.append((role, patch, box))

    for _, patch, box in objects:
        try:
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + box.shape[0], c0 : c0 + box.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _gen_attempt() -> np.ndarray | None:
    side, row_shift, col_shift = random.choice(CANDIDATE_CONFIGS)
    masks = _build_sector_masks(side, row_shift, col_shift)

    pair_shapes = {role: tuple(map(int, _crop_mask(masks[role]).shape)) for role in ROLE_ORDER}
    signature = tuple(pair_shapes[role] for role in ROLE_ORDER)
    _, _, _, matched_masks = _match_partition_from_signature(signature)
    assert tuple(tuple(map(int, _crop_mask(m).shape)) for m in matched_masks) == signature

    colors = random.sample(range(1, 10), NUM_COLORS)
    marker_palette = colors[:NUM_PIECES]
    out_palette = colors[NUM_PIECES:]
    random.shuffle(marker_palette)
    random.shuffle(out_palette)
    marker_colors = {role: marker_palette[idx] for idx, role in enumerate(ROLE_ORDER)}
    legend_map = np.array(out_palette, dtype=int).reshape(K, K)

    expected = _expected_from_masks(masks, legend_map, True)
    old_like = _expected_from_masks(masks, legend_map, False)
    if np.array_equal(expected, old_like):
        return None

    grid = _scatter_objects(pair_shapes, marker_colors, legend_map)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")

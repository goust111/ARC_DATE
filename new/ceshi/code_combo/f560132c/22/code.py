# Combo variant of f560132c (index 22)
# combo_spec: D3/1+D4/2+D7/3
# Name: 分层叠加+颜色混合+图例指示顺序

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# fused_rule_summary:
#   - Keep the D3 stacking reinterpretation: the four scattered sprites are not tiled into
#     a disjoint rectangle; instead they are anchored to the four corners of one common square
#     and may overlap near quadrant boundaries.
#   - Add the D4 color-mix reinterpretation in a concrete, visible way:
#       each sprite contains an enclosed hole in the input;
#       in the output, the hole is filled and uses a second legend-derived color,
#       while the original sprite pixels use the first color.
#   - Use the D7 ordering reinterpretation to control both painting order and the fill-color cycle:
#       legend 2x2 cells are read row-major (TL, TR, BL, BR);
#       layer i is painted in that order,
#       its body uses legend color i,
#       and its filled-hole region uses legend color i+1 mod 4.
#
# concrete_pixel_realization:
#   - Output side S stays in {8, 10, 12}; S is even, so the common square splits into four
#     S/2-by-S/2 home quadrants.
#   - TL legend layer:
#       a solid TL home block,
#       plus a thin top band across the full square and a thin left band down the full square,
#       so the legend bbox itself reveals the whole output side.
#       It also contains one small enclosed hole offset inside the TL block, away from the palette slot.
#   - TR / BL / BR layers:
#       each has a solid home block in its own corner,
#       plus one horizontal inward arm and one vertical inward arm, creating overlap near the center.
#       Each also has one enclosed hole near its outer corner, so the secondary color stays visible
#       even after the center-region overlaps are resolved by stack order.
#   - Because the four home blocks already cover the whole square, the final output is still a
#     tight crop with no background. The arms only create extra overlap on top of that base cover.
#
# behavior_trace_with_small_example:
#   - Suppose S = 8. Then each home quadrant is 4x4.
#   - Original task:
#       1. find four scattered sprites,
#       2. search an exact-cover tiling,
#       3. recolor each whole piece by one legend cell color.
#   - This combo task:
#       1. find four scattered sprites and the unique multicolor legend,
#       2. read the legend 2x2 map in row-major order,
#       3. anchor the legend at TL and classify the other sprites by which corner-half of their bbox
#          contains the dominant 4x4 block,
#       4. for each anchored layer, fill its enclosed hole,
#          paint the filled hole with the next legend color in reading order,
#          paint the original pixels with the current legend color,
#       5. paint TL, then TR, then BL, then BR, so later layers overwrite earlier layers.
#   - Difference from the original:
#       area is no longer conserved,
#       write collisions are intentional,
#       and a layer now contributes two structured colors rather than one flat recolor.
#
# complexity_calibration:
#   - The family of shapes uses only a few discrete parameters:
#       side in {8,10,12}, band width in {1,2}, arm extent in small integer ranges,
#       hole size 1 for S=8 and 2 for S>=10.
#   - This keeps the rule learnable while still making the combo visually different from both the
#     original exact-cover task and a simpler pure-stacking variant.
#   - Generator rejection criteria force:
#       visible overlap near the center,
#       visible filled-hole pixels in the final output,
#       and a measurable difference between row-major paint order and a reversed order.
#
# implicit_assumption_audit_for_main:
#   - old assumption: total piece area determines the output rectangle
#     new status: invalid, because stacking allows overlap and hole-filling adds painted cells
#     replacement: the legend bbox is the output square.
#   - old assumption: exact-cover search over arbitrary placements is necessary
#     new status: invalid
#     replacement: all layers are corner-anchored and classified by their dominant corner block.
#   - old assumption: each piece has one output color
#     new status: invalid
#     replacement: each layer has a body color and a fill color derived from adjacent legend cells.
#   - old assumption: recolor writes commute because pieces are disjoint
#     new status: invalid
#     replacement: row-major legend order defines the overwrite sequence.
#   - old D3-only assumption: a home corner can be detected by an all-filled half-square
#     new status: weakened by D4, because the hole intrudes into that block
#     replacement: classify by the unique largest occupancy among the four half-square corner windows.
#
# placement_legality:
#   - Input sprites are scattered independently with one-cell padding, so they never overlap in input.
#   - Output anchoring uses only four corner offsets inside one SxS square; all layer bboxes fit by
#     construction, and the hole is fully enclosed before filling.
#   - The legend palette is embedded only in a 2x2 body region strictly inside the legend bbox,
#     and the generator checks that the solver's legend-window extractor reads back that exact block.
#
# visual_observability:
#   - All cues are visible in the pixels:
#       the legend is the only multicolor sprite,
#       the layer order is the natural reading order of the 2x2 legend,
#       and the two-color behavior is attached to visible enclosed holes in the sprites.
#   - No hidden numeric property such as color id magnitude is used for sorting or matching.
#
# visual_cleanliness:
#   - Overlap is concentrated near quadrant seams and the center rather than across the whole canvas.
#   - Each piece contributes one compact secondary-color patch rather than many noisy exceptions.
#   - The final image still reads as four quadrant-rooted layers instead of an unstructured pileup.
#
# color_inference:
#   - The body color of each layer is the legend color at the layer's own corner slot.
#   - The fill color is the next legend color in row-major order, cycling after BR back to TL.
#   - This creates repeated evidence across all four layers in each sample, which makes the two-color
#     rule inferable from only a few examples.
#
# new_main_assumptions_guaranteed_by_generator:
#   - Exactly four components exist and exactly one is the multicolor legend.
#   - The legend bbox is a square whose side is the output side.
#   - Each non-legend piece has a unique dominant half-square corner score.
#   - Every piece has at least one enclosed hole, and at least three layers retain visible filled-hole
#     pixels in the final stacked rendering.
#   - Row-major order has observable impact: reversing the order changes the output on generated samples.
# ---------------------------------------------------------------------------


K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
SIDE_CHOICES = [8, 10, 12]

INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 30
DENSITY_LO = 0.14
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 500

LEGEND_BAND_RANGE = (1, 2)
ARM_EXT_RANGES = {
    8: (2, 3),
    10: (3, 4),
    12: (3, 5),
}
MIN_OVERLAP_CELLS = {
    8: 8,
    10: 12,
    12: 16,
}
MIN_CENTER_OVERLAP = 2
MIN_VISIBLE_HOLE_CELLS = {
    8: 3,
    10: 6,
    12: 8,
}
MIN_ORDER_SENSITIVE_CELLS = {
    8: 4,
    10: 8,
    12: 10,
}

CORNERS = ["TL", "TR", "BL", "BR"]
CORNER_TO_CELL = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}
CORNER_TO_INDEX = {corner: i for i, corner in enumerate(CORNERS)}


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    """
    Inside the legend bbox, pick the kxk window with the most distinct colors.
    Ties go to the top-left window.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    return _best_legend_window(sub, k)[1]


def _best_legend_window(sub: np.ndarray, k: int) -> tuple[tuple[int, int], np.ndarray]:
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= k, "legend component must fit kxk map"
    best = None
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    assert best_rc is not None, "legend component must contain a valid palette"
    rr, cc = best_rc
    return (int(rr), int(cc)), np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool).copy()


def _rotate_mask(mask: np.ndarray, corner: str) -> np.ndarray:
    if corner == "TL":
        out = mask
    elif corner == "TR":
        out = np.rot90(mask, k=3)
    elif corner == "BL":
        out = np.rot90(mask, k=1)
    elif corner == "BR":
        out = np.rot90(mask, k=2)
    else:
        raise ValueError(corner)
    return _crop_mask(np.asarray(out, dtype=bool))


def _corner_offset(corner: str, h: int, w: int, side: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, side - w
    if corner == "BL":
        return side - h, 0
    if corner == "BR":
        return side - h, side - w
    raise ValueError(corner)


def _classify_corner(mask: np.ndarray, half: int) -> str:
    h, w = mask.shape
    assert h >= half and w >= half, "piece must be at least half-size in both dimensions"
    scores = {
        "TL": int(mask[:half, :half].sum()),
        "TR": int(mask[:half, w - half : w].sum()),
        "BL": int(mask[h - half : h, :half].sum()),
        "BR": int(mask[h - half : h, w - half : w].sum()),
    }
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    assert ordered[0][1] > ordered[1][1], "piece must have a unique dominant corner block"
    return ordered[0][0]


def _fill_holes_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    bgmask = ~mask
    vis = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for rr in range(h):
        for cc in range(w):
            if rr in (0, h - 1) or cc in (0, w - 1):
                if bgmask[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    stack.append((rr, cc))
                    while stack:
                        r, c = stack.pop()
                        for r2, c2 in neighbors4(r, c):
                            if 0 <= r2 < h and 0 <= c2 < w and bgmask[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                stack.append((r2, c2))

    holes = bgmask & ~vis
    return mask | holes, holes


def _legend_sequence(legend_map: np.ndarray) -> list[int]:
    return [int(legend_map[r, c]) for r in range(K) for c in range(K)]


def _paint_layers(
    side: int,
    ordered_masks: dict[str, np.ndarray],
    legend_map: np.ndarray,
    order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order is None:
        order = CORNERS
    seq = _legend_sequence(legend_map)

    out = np.zeros((side, side), dtype=int)
    owner = np.full((side, side), -1, dtype=int)
    kind = np.zeros((side, side), dtype=int)  # 0 unset, 1 fill-hole, 2 original-body

    for corner in order:
        mask = ordered_masks[corner]
        filled, holes = _fill_holes_mask(mask)
        assert int(holes.sum()) > 0, "each piece must contain an enclosed hole"
        h, w = mask.shape
        r0, c0 = _corner_offset(corner, h, w, side)
        idx = CORNER_TO_INDEX[corner]
        body_color = seq[idx]
        fill_color = seq[(idx + 1) % len(seq)]

        patch = out[r0 : r0 + h, c0 : c0 + w]
        patch_owner = owner[r0 : r0 + h, c0 : c0 + w]
        patch_kind = kind[r0 : r0 + h, c0 : c0 + w]

        patch[filled] = fill_color
        patch_owner[filled] = idx
        patch_kind[filled] = 1

        patch[mask] = body_color
        patch_owner[mask] = idx
        patch_kind[mask] = 2

    return out, owner, kind


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack four scattered sprites into one square, using legend order for both overwriting and two-color filling.

    Input structure:
    - The background is the most common color.
    - There are exactly four separated connected sprites, and exactly one is multicolor because it contains the 2x2 legend.
    - The legend sprite has a square bbox that directly gives the output side; each other sprite has one dominant half-square corner block that indicates which output corner it belongs to.
    - Every sprite also contains an enclosed hole, visible as background inside the sprite.

    Rule:
    - Read the legend's 2x2 palette in row-major order: top-left, top-right, bottom-left, bottom-right.
    - Anchor the legend to TL and anchor the other three sprites to the corners indicated by their dominant half-square block.
    - For each anchored layer, fill its enclosed hole; color the filled-hole cells with the next legend color in reading order, color the original sprite pixels with the layer's own legend color, and paint the four layers in that same row-major order so later layers overwrite earlier ones.

    Key insight: The legend is not only a color key; its reading order defines the stack order, and each sprite contributes two structured colors because its enclosed hole becomes a second painted region after assembly.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == NUM_PIECES, "this combo variant expects exactly four components"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)
    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend must be uniquely identified by color variety"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], K)

    pieces: list[np.ndarray] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        pieces.append(sub != bg)

    legend_mask = pieces[legend_idx]
    side_h, side_w = legend_mask.shape
    assert side_h == side_w, "legend bbox must define a square output"
    side = int(side_h)
    half = side // 2
    assert side in SIDE_CHOICES and half * 2 == side, "unexpected output side"

    ordered_masks: dict[str, np.ndarray] = {"TL": legend_mask}
    for i, mask in enumerate(pieces):
        if i == legend_idx:
            continue
        corner = _classify_corner(mask, half)
        assert corner != "TL", "only the legend may occupy the TL slot"
        assert corner not in ordered_masks, "corner assignment must be unique"
        ordered_masks[corner] = mask

    assert set(ordered_masks) == set(CORNERS), "all four corner layers must be present"
    return _paint_layers(side, ordered_masks, legend_map)[0]


def _hole_size(side: int) -> int:
    return 1 if side == 8 else 2


def _build_legend_mask(side: int, band_h: int, band_w: int) -> np.ndarray:
    half = side // 2
    mask = np.zeros((side, side), dtype=bool)
    mask[:half, :half] = True
    mask[:band_h, :] = True
    mask[:, :band_w] = True
    mask[half - 2 : half - 1, 1:2] = False
    return mask


def _build_corner_piece(side: int, corner: str, ext_right: int, ext_down: int) -> np.ndarray:
    half = side // 2
    hole = _hole_size(side)
    base = np.zeros((half + ext_down, half + ext_right), dtype=bool)
    base[:half, :half] = True
    base[half - hole : half, half:] = True
    base[half:, half - hole : half] = True
    base[1 : 1 + hole, 1 : 1 + hole] = False
    return _rotate_mask(base, corner)


def _palette_spots(mask: np.ndarray) -> list[tuple[int, int]]:
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all():
                spots.append((r, c))
    return spots


def _place_palette(legend_patch: np.ndarray, legend_mask: np.ndarray, legend_map: np.ndarray) -> bool:
    spots = _palette_spots(legend_mask)
    if not spots:
        return False
    base = legend_patch.copy()
    for r, c in spots:
        trial = base.copy()
        trial[r : r + K, c : c + K] = legend_map
        best_rc, best_map = _best_legend_window(trial, K)
        if best_rc == (r, c) and np.array_equal(best_map, legend_map):
            legend_patch[:, :] = trial
            return True
    return False


def _scatter_pieces(patches: list[np.ndarray]) -> np.ndarray | None:
    total = sum(int(np.count_nonzero(p)) for p in patches)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, max(nlo, math.floor(math.sqrt(total / DENSITY_LO))))
    if nlo > nhi:
        return None

    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    order = list(range(len(patches)))
    random.shuffle(order)
    for pi in order:
        patch = patches[pi]
        try:
            r0, c0 = random_free_location(
                occupied,
                patch,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=0)
        paste(occupied, (patch != 0).astype(int), r0, c0, transparent=0)
    return grid


def _coverage_stats(side: int, ordered_masks: dict[str, np.ndarray]) -> tuple[int, int]:
    cover = np.zeros((side, side), dtype=int)
    for corner in CORNERS:
        mask = ordered_masks[corner]
        filled, _ = _fill_holes_mask(mask)
        h, w = mask.shape
        r0, c0 = _corner_offset(corner, h, w, side)
        cover[r0 : r0 + h, c0 : c0 + w] += filled.astype(int)
    half = side // 2
    center = cover[half - 1 : half + 1, half - 1 : half + 1]
    return int((cover > 1).sum()), int((center > 1).sum())


def _visible_hole_cells(kind: np.ndarray) -> int:
    return int((kind == 1).sum())


def _visible_hole_cells_by_layer(owner: np.ndarray, kind: np.ndarray) -> list[int]:
    return [int(((owner == i) & (kind == 1)).sum()) for i in range(NUM_PIECES)]


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    side = random.choice(SIDE_CHOICES)
    half = side // 2
    hole = _hole_size(side)

    band_h = random.randint(*LEGEND_BAND_RANGE)
    band_w = random.randint(*LEGEND_BAND_RANGE)
    ext_lo, ext_hi = ARM_EXT_RANGES[side]
    legend_mask = _build_legend_mask(side, band_h, band_w)

    ordered_masks: dict[str, np.ndarray] = {"TL": legend_mask}
    for corner in ("TR", "BL", "BR"):
        ok = False
        for _ in range(30):
            ext_r = random.randint(ext_lo, ext_hi)
            ext_d = random.randint(ext_lo, ext_hi)
            mask = _build_corner_piece(side, corner, ext_r, ext_d)
            if _classify_corner(mask, half) != corner:
                continue
            filled, holes = _fill_holes_mask(mask)
            if int(holes.sum()) != hole * hole:
                continue
            ordered_masks[corner] = mask
            ok = True
            break
        if not ok:
            return None

    if _classify_corner(legend_mask, half) != "TL":
        return None

    overlap_cells, center_overlap = _coverage_stats(side, ordered_masks)
    if overlap_cells < MIN_OVERLAP_CELLS[side] or center_overlap < MIN_CENTER_OVERLAP:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    legend_map = np.array(colors[NUM_PIECES:], dtype=int).reshape(K, K)

    expected, owner, kind = _paint_layers(side, ordered_masks, legend_map)
    if (expected == 0).any():
        return None

    visible_hole_cells = _visible_hole_cells(kind)
    per_layer_holes = _visible_hole_cells_by_layer(owner, kind)
    if visible_hole_cells < MIN_VISIBLE_HOLE_CELLS[side]:
        return None
    if sum(v > 0 for v in per_layer_holes) < 3:
        return None

    reverse_expected = _paint_layers(side, ordered_masks, legend_map, order=list(reversed(CORNERS)))[0]
    order_sensitive_cells = int(np.count_nonzero(expected != reverse_expected))
    if order_sensitive_cells < MIN_ORDER_SENSITIVE_CELLS[side]:
        return None

    legend_patch = np.zeros((side, side), dtype=int)
    legend_patch[legend_mask] = body_colors[0]
    if not _place_palette(legend_patch, legend_mask, legend_map):
        return None

    patches = [legend_patch]
    for idx, corner in enumerate(("TR", "BL", "BR"), start=1):
        patch = np.zeros(ordered_masks[corner].shape, dtype=int)
        patch[ordered_masks[corner]] = body_colors[idx]
        patches.append(patch)

    grid = _scatter_pieces(patches)
    if grid is None:
        return None

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    stats = {
        "side": side,
        "hole": hole,
        "overlap_cells": overlap_cells,
        "center_overlap": center_overlap,
        "visible_hole_cells": visible_hole_cells,
        "per_layer_holes": per_layer_holes,
        "order_sensitive_cells": order_sensitive_cells,
        "legend_bands": (band_h, band_w),
    }
    return grid, stats


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _stats = result
            return grid
    raise RuntimeError("failed to generate combo variant for f560132c")

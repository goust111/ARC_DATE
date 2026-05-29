# Variant of f560132c
# Dimension: D3 — 目标结构：拼装目标的形状与组织 (dst)
# Change: 4个精灵拼装为一个矩形复合体，图例精灵的轮廓定义了该矩形的大小和各象限的分割方式... → 分层叠加而非平面拼合

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# variant_spec:
#   core_change:
#     from: 4 pieces form a disjoint exact cover of one square
#     to:   4 layers are aligned onto one square and painted in stack order;
#           later layers overwrite earlier layers on overlapping cells
#   retained_structure:
#     - still 4 scattered connected sprites
#     - still one multicolor legend sprite with a 2x2 palette
#     - still one square output whose size is read from the legend sprite
#     - still recolor each sprite by the legend cell corresponding to its assembled position
#   concrete_pixel_realization:
#     - output side S stays in {8, 10, 12}; S is even, so the square splits into four S/2 x S/2 quadrants
#     - legend/TL layer:
#         a full solid top-left quadrant block, plus a thin band across the top edge and a thin band down the left edge
#         this keeps the legend bbox equal to SxS, so the legend alone reveals output size
#     - TR / BL / BR layers:
#         each has a solid S/2 x S/2 corner block at its home corner plus two thin inward arms that reach toward the center
#         the four base corner blocks still cover the full square, while the arms create visible overlap
#     - stack order is fixed and visible across examples: TL -> TR -> BL -> BR
#         therefore BR is topmost in the center if it reaches there
#
# behavior_trace:
#   original_local_flow:
#     - extract four piece masks
#     - search an exact-cover tiling with no overlaps
#     - recolor each piece once and paint it; later writes are harmless because masks are disjoint
#   new_local_flow:
#     - extract four piece masks
#     - classify each piece by the solid half-quadrant block that sits in one corner of its own bbox
#     - anchor each piece to the matching corner of the common SxS square
#     - recolor by assembled centroid quadrant, then paint in fixed layer order
#   practical difference:
#     - `out[r, c] = color` is no longer harmless unconditional filling: overlaps are intended,
#       so the draw order itself is part of the rule
#
# complexity_calibration:
#   - overlap is not optional decoration: generator rejects weak-overlap samples and requires
#     overlap in the central 2x2 neighborhood
#   - the layer family uses only a few discrete arm widths/extents, so the rule is still learnable
#   - every sample repeats the same ordering logic (reading order stack), which gives humans
#     repeated evidence instead of many unrelated subcases
#
# implicit_assumption_audit_for_main:
#   - old assumption: total piece area implies the output rectangle
#     new status: invalid, because areas now exceed S*S due to overlap
#     replacement: legend bbox is exactly SxS
#   - old assumption: exact-cover search over arbitrary translations is required
#     new status: invalid, because pieces are corner-anchored layers, not free tiles
#     replacement: corner classification + deterministic anchoring
#   - old assumption: recolor-then-fill can ignore write collisions
#     new status: invalid, because collisions are the intended behavior
#     replacement: fixed stack order with deliberate overwrite
#
# placement_legality:
#   - input sprites are still scattered with no overlap and 1-cell padding
#   - output assembly uses only four corner anchors inside an SxS square, so no layer is placed out of bounds
#   - generator validates that each non-legend piece has exactly one detectable solid S/2 x S/2 corner block,
#     which is the assumption used by the solver's corner classifier
#
# visual_observability:
#   - no sorting uses color ids or hidden metadata
#   - the new behavior is visible because central cells are covered by multiple layers and the topmost color changes
#     with the fixed stack order
#   - the legend still advertises four colors, but the output is no longer a patchwork of disjoint regions:
#     it contains overwritten crossings and capped arms
#
# visual_cleanliness:
#   - only four layers exist, each with one dominant color in the output
#   - overlap is concentrated near quadrant boundaries and the center, avoiding chaotic full-canvas occlusion
#
# color_inference:
#   - each layer keeps one output color, chosen from the legend by its assembled centroid quadrant
#   - the number of active colors remains four in the output, matching the 2x2 legend
#
# new_main_assumptions_guaranteed_by_generator:
#   - legend sprite is the unique component with the most distinct colors
#   - legend bbox is square and equal to the final output size
#   - each non-legend sprite has exactly one solid half-square corner block
#   - centroid quantization of each anchored layer still lands in its home quadrant
# ---------------------------------------------------------------------------


K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
SIDE_CHOICES = [8, 10, 12]

INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 28
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 400

LEGEND_BAND_RANGE = (1, 2)
ARM_EXT_RANGE = (1, 2)
ARM_WIDTH_RANGE = (1, 2)
MIN_OVERLAP_CELLS = 10
MIN_CENTER_OVERLAP = 2

CORNER_TO_CELL = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}
STACK_ORDER = ["TL", "TR", "BL", "BR"]


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    """
    Find the kxk palette block inside the legend bbox.
    Ties go to the top-left window, so the generator fixes the palette at the top-left-most
    valid interior location.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
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
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


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
    hits = []
    if mask[:half, :half].all():
        hits.append("TL")
    if mask[:half, w - half : w].all():
        hits.append("TR")
    if mask[h - half : h, :half].all():
        hits.append("BL")
    if mask[h - half : h, w - half : w].all():
        hits.append("BR")
    assert len(hits) == 1, "piece must expose exactly one solid half-square corner block"
    return hits[0]


def _piece_cell(mask: np.ndarray, corner: str, side: int, k: int) -> tuple[int, int]:
    h, w = mask.shape
    r0, c0 = _corner_offset(corner, h, w, side)
    pts = np.argwhere(mask)
    cr = float(pts[:, 0].mean() + r0)
    cc = float(pts[:, 1].mean() + c0)
    rr = int(cr * float(k) / float(side))
    cc2 = int(cc * float(k) / float(side))
    rr = max(0, min(k - 1, rr))
    cc2 = max(0, min(k - 1, cc2))
    return rr, cc2


def _render_stacked(side: int, ordered_masks: dict[str, np.ndarray], legend_map: np.ndarray) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    for corner in STACK_ORDER:
        mask = ordered_masks[corner]
        h, w = mask.shape
        r0, c0 = _corner_offset(corner, h, w, side)
        rr, cc = _piece_cell(mask, corner, side, K)
        color = int(legend_map[rr, cc])
        out[r0 : r0 + h, c0 : c0 + w][mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack scattered layer sprites into one square, recoloring each layer from a legend palette.

    Input structure:
    - bg is the most common color.
    - There are four separated connected sprites; exactly one is a multicolor legend sprite.
    - Each sprite contains one solid half-square corner block that indicates which output corner it anchors to.

    Rule:
    - Read the output side length from the legend sprite's square bounding box.
    - Anchor the legend at the top-left and anchor the other sprites at the corner encoded by their solid half-square block.
    - Recolor each anchored sprite by the legend color of its assembled 2x2 centroid cell, then paint the layers in fixed reading order so later layers overwrite earlier ones.

    Key insight: The pieces are no longer disjoint tiles; they are overlapping layers aligned to the same square, so solving requires both corner assignment and the intended overwrite order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain non-background components"

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps) == NUM_PIECES, "this variant expects exactly four pieces"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)
    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend must be uniquely identified by color variety"
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)

    pieces: list[np.ndarray] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        pieces.append(sub != bg)

    legend_mask = pieces[legend_idx]
    side_h, side_w = legend_mask.shape
    assert side_h == side_w, "legend bbox must directly define a square output"
    side = int(side_h)
    half = side // 2
    assert half * 2 == side and half >= 4, "output side must be an even square side"

    ordered_masks: dict[str, np.ndarray] = {"TL": legend_mask}
    for i, mask in enumerate(pieces):
        if i == legend_idx:
            continue
        corner = _classify_corner(mask, half)
        assert corner != "TL", "only the legend may occupy the TL slot"
        assert corner not in ordered_masks, "corner assignment must be unique"
        ordered_masks[corner] = mask

    assert set(ordered_masks) == set(STACK_ORDER), "all four corner layers must be present"
    return _render_stacked(side, ordered_masks, legend_map)


def _build_legend_mask(side: int, band_h: int, band_w: int) -> np.ndarray:
    half = side // 2
    mask = np.zeros((side, side), dtype=bool)
    mask[:half, :half] = True
    mask[:band_h, :] = True
    mask[:, :band_w] = True
    return mask


def _build_corner_piece(side: int, corner: str, ext_right: int, ext_down: int, arm_h: int, arm_w: int) -> np.ndarray:
    half = side // 2
    base = np.zeros((half + ext_down, half + ext_right), dtype=bool)
    base[:half, :half] = True
    base[half - arm_h : half, half:] = True
    base[half:, half - arm_w : half] = True
    return _rotate_mask(base, corner)


def _place_palette(legend_patch: np.ndarray, legend_map: np.ndarray) -> None:
    legend_patch[1 : 1 + K, 1 : 1 + K] = legend_map


def _scatter_pieces(patches: list[np.ndarray]) -> np.ndarray | None:
    total = sum(int(np.count_nonzero(p)) for p in patches)
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = max(nlo, INPUT_SIZE_HI)
    if nlo > INPUT_SIZE_HI:
        return None
    grid_size = random.randint(nlo, INPUT_SIZE_HI)
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
    for corner, mask in ordered_masks.items():
        h, w = mask.shape
        r0, c0 = _corner_offset(corner, h, w, side)
        cover[r0 : r0 + h, c0 : c0 + w] += mask.astype(int)
    half = side // 2
    center = cover[half - 1 : half + 1, half - 1 : half + 1]
    return int((cover > 1).sum()), int((center > 1).sum())


def _home_quadrant_ok(side: int, ordered_masks: dict[str, np.ndarray]) -> bool:
    for corner, cell in CORNER_TO_CELL.items():
        rr, cc = _piece_cell(ordered_masks[corner], corner, side, K)
        if (rr, cc) != cell:
            return False
    return True


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    side = random.choice(SIDE_CHOICES)
    half = side // 2

    band_h = random.randint(*LEGEND_BAND_RANGE)
    band_w = random.randint(*LEGEND_BAND_RANGE)
    legend_mask = _build_legend_mask(side, band_h, band_w)

    ordered_masks: dict[str, np.ndarray] = {"TL": legend_mask}
    seen_shapes: set[tuple] = set()
    for corner in ("TR", "BL", "BR"):
        ok = False
        for _ in range(40):
            ext_right = random.randint(*ARM_EXT_RANGE)
            ext_down = random.randint(*ARM_EXT_RANGE)
            arm_h = random.randint(1, min(ext_down, ARM_WIDTH_RANGE[1]))
            arm_w = random.randint(1, min(ext_right, ARM_WIDTH_RANGE[1]))
            mask = _build_corner_piece(side, corner, ext_right, ext_down, arm_h, arm_w)
            key = (corner, mask.shape, tuple(mask.ravel()))
            if key in seen_shapes:
                continue
            if _classify_corner(mask, half) != corner:
                continue
            ordered_masks[corner] = mask
            seen_shapes.add(key)
            ok = True
            break
        if not ok:
            return None

    overlap_cells, center_overlap = _coverage_stats(side, ordered_masks)
    if overlap_cells < MIN_OVERLAP_CELLS or center_overlap < MIN_CENTER_OVERLAP:
        return None
    if not _home_quadrant_ok(side, ordered_masks):
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    legend_map = np.array(colors[NUM_PIECES:], dtype=int).reshape(K, K)

    expected = _render_stacked(side, ordered_masks, legend_map)
    if (expected == 0).any():
        return None

    legend_patch = np.zeros((side, side), dtype=int)
    legend_patch[legend_mask] = body_colors[0]
    _place_palette(legend_patch, legend_map)

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
        "overlap_cells": overlap_cells,
        "center_overlap": center_overlap,
        "legend_bands": (band_h, band_w),
    }
    return grid, stats


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            grid, _stats = result
            return grid
    raise RuntimeError("failed to generate stacking variant for f560132c")

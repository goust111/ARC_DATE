# Combo variant of f560132c (index 6)
# combo_spec: D3/1+D7/3
# Name: 分层叠加+图例指示叠放顺序

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# combo_spec:
#   - D3/1: flat exact-cover assembly becomes overlapping corner-anchored layers
#   - D7/3: the legend's 2x2 reading order decides stack order
#
# concrete_pixel_realization:
#   - keep four scattered connected sprites and one unique multicolor legend sprite
#   - keep one even square output with side in {8, 10, 12}
#   - every sprite still advertises one home corner via a solid S/2 x S/2 block
#   - the legend sprite is no longer forced to be the top-left layer; it may occupy any one
#     of the four corners, and its 2x2 palette rotates with that orientation
#   - after reassembly, the four corner layers overlap near the center as in the stacking variant
#
# behavior_trace:
#   - original task:
#       1. find four disconnected pieces
#       2. exact-cover tile them into a square
#       3. recolor by quadrant and paint; order is irrelevant because there is no overlap
#   - D3-only stacking:
#       1. classify each piece by its solid half-square corner block
#       2. anchor all pieces to one shared square
#       3. paint them in a fixed TL -> TR -> BL -> BR order
#   - fused combo:
#       1. classify corners exactly as in the stacking family
#       2. read the legend's displayed 2x2 palette in row-major order
#       3. use the legend sprite's own orientation to map those four local palette cells onto
#          global corners; that mapped order becomes the stack order
#       4. the same mapping assigns colors to corners, so orientation affects both recoloring
#          and which layer overwrites which
#   - practical difference:
#       the D7 rule is not cosmetic: if the legend sits at TR / BL / BR, the overwrite order
#       differs from the fixed D3 order, and the generator rejects samples where the two outputs
#       accidentally coincide
#
# tiny thought experiment:
#   - suppose the legend is the BR-shaped layer and its palette reads
#       a b
#       c d
#   - local row-major is a, b, c, d
#   - because the legend is rotated 180 degrees relative to the TL canonical frame,
#     those local cells correspond globally to BR, BL, TR, TL
#   - so BR paints first with a, BL second with b, TR third with c, TL last with d
#   - this differs from a fixed TL-first stack even though the four geometric layers are unchanged
#
# complexity_calibration:
#   - only four stack permutations are allowed, one per legend orientation
#   - the same corner-block cue is reused across all pieces, so assignment stays human-readable
#   - overlap is concentrated around quadrant boundaries and the center instead of the whole square
#   - the legend orientation is repeated evidence across examples rather than a one-off hidden key
#
# implicit_assumption_audit_for_main:
#   - old assumption: the legend is anchored at TL
#     status: false in the combo
#     replacement: classify the legend corner exactly like every other layer
#   - old assumption: fixed stack order is sufficient
#     status: false in the combo
#     replacement: derive global order from the legend palette's local row-major order plus legend orientation
#   - old assumption: color lookup uses a fixed global legend orientation
#     status: false in the combo
#     replacement: color assignment and stack order share the same orientation-aware corner mapping
#   - old assumption: later writes are harmless
#     status: false in the combo because collisions are intentional and visible
#
# placement_legality:
#   - input sprites are still scattered without overlap and with padding
#   - output anchoring is deterministic and keeps every mask inside the common SxS square
#   - generator validates corner classification, overlap strength, and non-degeneracy against a
#     fixed-order baseline before accepting a sample
#
# visual_observability:
#   - no hidden sorting by numeric color id is used
#   - the only order cue is visible: where the multicolor legend sits and how its 2x2 patch is rotated
#   - because layers overlap near the center, changing the order changes visible top colors there
#
# visual_cleanliness:
#   - each layer still has one final color
#   - only four layers exist, and each keeps a dominant corner region, so the output remains legible
#   - overlap stays local instead of turning the whole square into noise
#
# new_main_assumptions_guaranteed_by_generator:
#   - exactly four components exist and exactly one is uniquely multicolor
#   - the legend bbox is square and equals the final output side
#   - every piece exposes exactly one solid half-square corner block
#   - the legend orientation is one of TR / BL / BR so the fused order differs from the fixed D3 order
# ---------------------------------------------------------------------------


K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
SIDE_CHOICES = [8, 10, 12]

INPUT_SIZE_LO = 22
INPUT_SIZE_HI = 30
DENSITY_LO = 0.12
DENSITY_HI = 0.24
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 500

LEGEND_BAND_RANGE = (1, 2)
ARM_EXT_RANGE = (1, 2)
ARM_WIDTH_RANGE = (1, 2)
MIN_OVERLAP_CELLS = 10
MIN_CENTER_OVERLAP = 2

CORNERS = ["TL", "TR", "BL", "BR"]
LEGEND_CORNER_CHOICES = ["TR", "BL", "BR"]
CORNER_TO_CELL = {
    "TL": (0, 0),
    "TR": (0, 1),
    "BL": (1, 0),
    "BR": (1, 1),
}
CELL_TO_CORNER = {v: k for k, v in CORNER_TO_CELL.items()}


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    """Find the kxk palette block inside the legend bbox, preferring top-left ties."""
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


def _rotate_patch(patch: np.ndarray, corner: str) -> np.ndarray:
    if corner == "TL":
        out = patch
    elif corner == "TR":
        out = np.rot90(patch, k=3)
    elif corner == "BL":
        out = np.rot90(patch, k=1)
    elif corner == "BR":
        out = np.rot90(patch, k=2)
    else:
        raise ValueError(corner)
    return np.asarray(out, dtype=int).copy()


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


def _local_cell_to_global_corner(legend_corner: str, rr: int, cc: int) -> str:
    if legend_corner == "TL":
        gcell = (rr, cc)
    elif legend_corner == "TR":
        gcell = (cc, K - 1 - rr)
    elif legend_corner == "BL":
        gcell = (K - 1 - cc, rr)
    elif legend_corner == "BR":
        gcell = (K - 1 - rr, K - 1 - cc)
    else:
        raise ValueError(legend_corner)
    return CELL_TO_CORNER[gcell]


def _legend_order_and_colors(legend_map: np.ndarray, legend_corner: str) -> tuple[list[str], dict[str, int]]:
    order: list[str] = []
    corner_to_color: dict[str, int] = {}
    for rr in range(K):
        for cc in range(K):
            corner = _local_cell_to_global_corner(legend_corner, rr, cc)
            order.append(corner)
            corner_to_color[corner] = int(legend_map[rr, cc])
    assert len(order) == NUM_PIECES and len(set(order)) == NUM_PIECES
    return order, corner_to_color


def _render_stacked(
    side: int,
    ordered_masks: dict[str, np.ndarray],
    legend_map: np.ndarray,
    legend_corner: str,
) -> np.ndarray:
    order, corner_to_color = _legend_order_and_colors(legend_map, legend_corner)
    out = np.zeros((side, side), dtype=int)
    for corner in order:
        mask = ordered_masks[corner]
        h, w = mask.shape
        r0, c0 = _corner_offset(corner, h, w, side)
        out[r0 : r0 + h, c0 : c0 + w][mask] = int(corner_to_color[corner])
    return out


def _render_fixed_baseline(side: int, ordered_masks: dict[str, np.ndarray], legend_map: np.ndarray) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    for corner in CORNERS:
        mask = ordered_masks[corner]
        h, w = mask.shape
        r0, c0 = _corner_offset(corner, h, w, side)
        rr, cc = CORNER_TO_CELL[corner]
        out[r0 : r0 + h, c0 : c0 + w][mask] = int(legend_map[rr, cc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-stack scattered corner layers into one square, using the legend's displayed 2x2 reading order as the depth order.

    Input structure:
    - Background is the most common color.
    - There are exactly four disconnected sprites; exactly one is a multicolor legend sprite.
    - Each sprite contains one solid half-square corner block that identifies which output corner it belongs to.

    Rule:
    - Read the output side from the legend sprite's square bounding box and extract its visible 2x2 palette block.
    - Align every sprite to the common square by its corner block.
    - Use the legend sprite's own orientation to map the palette's local row-major order onto the four global corners; that mapped order is the stack order, and the same mapping chooses each corner layer's final color.

    Key insight: The legend is not just a color key. Because the palette rotates with the legend layer, its visible reading order tells both which corner gets which color and which overlapping layer is painted later on top.
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

    ordered_masks: dict[str, np.ndarray] = {}
    for mask in pieces:
        corner = _classify_corner(mask, half)
        assert corner not in ordered_masks, "corner assignment must be unique"
        ordered_masks[corner] = mask
    assert set(ordered_masks) == set(CORNERS), "all four corner layers must be present"

    legend_corner = _classify_corner(legend_mask, half)
    return _render_stacked(side, ordered_masks, legend_map, legend_corner)


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


def _fusion_effect_ok(side: int, ordered_masks: dict[str, np.ndarray], legend_map: np.ndarray, legend_corner: str) -> bool:
    fused = _render_stacked(side, ordered_masks, legend_map, legend_corner)
    baseline = _render_fixed_baseline(side, ordered_masks, legend_map)
    return not np.array_equal(fused, baseline)


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    side = random.choice(SIDE_CHOICES)
    half = side // 2

    legend_corner = random.choice(LEGEND_CORNER_CHOICES)
    band_h = random.randint(*LEGEND_BAND_RANGE)
    band_w = random.randint(*LEGEND_BAND_RANGE)
    legend_mask = _rotate_mask(_build_legend_mask(side, band_h, band_w), legend_corner)

    ordered_masks: dict[str, np.ndarray] = {legend_corner: legend_mask}
    seen_shapes: set[tuple] = set()
    for corner in CORNERS:
        if corner == legend_corner:
            continue
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
    canonical_legend_map = np.array(colors[NUM_PIECES:], dtype=int).reshape(K, K)
    displayed_legend_map = _rotate_patch(canonical_legend_map, legend_corner)
    if not _fusion_effect_ok(side, ordered_masks, displayed_legend_map, legend_corner):
        return None

    expected = _render_stacked(side, ordered_masks, displayed_legend_map, legend_corner)
    if (expected == 0).any():
        return None

    corner_to_body = dict(zip(CORNERS, body_colors))
    canonical_legend_patch = np.zeros((side, side), dtype=int)
    canonical_legend_mask = _build_legend_mask(side, band_h, band_w)
    canonical_legend_patch[canonical_legend_mask] = corner_to_body[legend_corner]
    _place_palette(canonical_legend_patch, canonical_legend_map)
    legend_patch = _rotate_patch(canonical_legend_patch, legend_corner)

    patches = []
    for corner in CORNERS:
        if corner == legend_corner:
            patches.append(legend_patch)
            continue
        patch = np.zeros(ordered_masks[corner].shape, dtype=int)
        patch[ordered_masks[corner]] = corner_to_body[corner]
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

    order, _corner_to_color = _legend_order_and_colors(displayed_legend_map, legend_corner)
    stats = {
        "side": side,
        "legend_corner": legend_corner,
        "stack_order": tuple(order),
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
    raise RuntimeError("failed to generate combo stacking-order variant for f560132c")

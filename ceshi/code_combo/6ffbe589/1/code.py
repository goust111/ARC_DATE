# Combo variant of 6ffbe589 (index 1)
# combo_spec: D1/2+D2/2
# Name: 尺寸排序匹配+距离编码旋转
#
# Design analysis for the combo variant
# ====================================
# a) concretization_and_trace:
#    - The board stays the same kind of object: one square, largest, multicolor crop with three concentric
#      color layers whose masks are rotated independently.
#    - The fusion changes both "which cue goes with which layer" and "what numeric code the cue carries".
#      Cue color is now deliberately useless, so the solver must first sort board layers by area and sort
#      outside cues by area, then pair rank-to-rank. After pairing, the cue's own area no longer determines
#      the angle; instead, the cue centroid's Manhattan distance to the board center determines the angle.
#    - Concrete trace:
#      board-layer areas sorted small->large = [5, 18, 31]
#      cue areas sorted small->large = [1, 4, 5]
#      cue centroid distances to board center = [10, 9, 7]
#      Old task would color-match and then use cue areas, giving residues [1, 0, 1]; that is impossible here
#      because cue colors are disjoint from board colors and would produce no matches at all.
#      A D1-only reading would pair by size correctly but still use areas, producing rotations
#      [(-1) % 4, (-4) % 4, (-5) % 4] = [3, 0, 3].
#      The fused rule pairs by size, then uses distances, producing
#      [(-10) % 4, (-9) % 4, (-7) % 4] = [2, 3, 1].
#      So both modified links are necessary on the same sample.
# b) complexity_calibration:
#    - To keep the assignment human-solvable, the generator always gives exactly three cues, one for each
#      board layer, and forces all board-layer areas distinct plus all cue areas distinct.
#    - To make the distance code readable, the board side length is restricted to odd values. Then the board
#      center is a lattice point, and the small cue template family uses centroids that keep Manhattan
#      distances integral.
#    - To keep the fusion visible rather than degenerate, cue colors come from a palette disjoint from the
#      board colors, and the cue-area set always includes an area-4 sprite so that the old area-code cannot
#      accidentally agree with the new distance-code for every pair.
# c) implicit_assumption_audit_for_main:
#    - `crop_components_near_largest(...)` still assumes all cues lie outside the board's margin-3 expansion.
#      The generator keeps cue bboxes at Chebyshev gap >= 4 from the board, so the crop remains exactly the
#      board.
#    - `rotate_color_layer(...)` still assumes the crop is square. The generator still creates an NxN board.
#    - The original `find same-color cue for each layer` assumption is invalid. `main()` now assumes global
#      rank matching: board-layer areas must be unique, cue areas must be unique, and cue count must equal
#      layer count. The generator enforces all three.
#    - Distance coding adds a new assumption that centroid-to-center Manhattan distances are integral. The
#      generator guarantees this with odd board sizes and symmetric cue templates, and `main()` asserts it.
#    - Sequential layer rotation remains safe because layers are color-disjoint; each pass clears only one
#      color before repainting that rotated mask.
# d) placement_legality:
#    - The board is still the unique largest 8-connected foreground component.
#    - Cue patches are monochrome, 8-connected, placed fully in bounds, non-overlapping, and mutually
#      separated. Their bboxes stay at least 4 cells away from the board bbox in Chebyshev distance, so they
#      never leak into the crop.
#    - Cue placement additionally filters for a required Manhattan-distance residue relative to the board
#      center, so every placed cue is legal for both geometry and encoded rotation.
# e) visual_observability:
#    - Across examples, cue colors visibly fail to match any board color, so color matching is ruled out.
#    - The board layers have clearly different scales, and cue sprites also have visibly different scales,
#      making the size-order pairing available directly from pixels.
#    - Rotation is not tied to cue size anymore: a larger cue can encode a smaller turn than a smaller cue if
#      it sits on a different distance ring. That difference is visible from input-output examples.
# f) visual_cleanliness:
#    - The board stays a single compact crop. The cues stay outside that crop with clear spacing and no
#      stacking. Only three cues are used, which keeps the scene readable.
#    - Cue shapes come from a very small template family with distinct areas, so the size ranking is easy to
#      inspect by eye.
# g) change_inferability:
#    - No board recoloring happens; board colors are preserved.
#    - The only role of cue color is to mark cue components as separate objects. Because cue colors are
#      disjoint from board colors, they cannot hide an alternate pairing rule.

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
    assert keep_coords.size > 0, "expected the expanded board crop to retain the largest component"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    """
    Local helper (only used by this solver): rotate a single color's mask inside a fixed-size grid.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    if k == 0:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    rmask = np.rot90(mask, k)
    assert rmask.shape == mask.shape, "expected a square crop for layer rotation"

    out = grid.copy()
    out[out == col] = bg
    out[rmask == 1] = col
    return out


def _sorted_layer_records(crop: np.ndarray, bg: int) -> list[dict[str, int]]:
    """Return board layers sorted by total color area (smallest to largest)."""
    records = [{"color": int(col), "area": int(np.sum(crop == int(col)))} for col in colors_present(crop, exclude={int(bg)})]
    assert records, "expected the cropped board to contain foreground colors"
    areas = [int(rec["area"]) for rec in records]
    assert len(set(areas)) == len(areas), "expected distinct board-layer areas for size-order pairing"
    return sorted(records, key=lambda rec: (int(rec["area"]), int(rec["color"])))


def _sorted_outside_cues(outside_grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    """Return outside cue components sorted by area (smallest to largest)."""
    cues = find_components(outside_grid, bg=int(bg), connectivity=8, monochromatic=True, return_format="dict")
    assert cues, "expected outside cue sprites"
    areas = [int(comp["area"]) for comp in cues]
    assert len(set(areas)) == len(areas), "expected distinct cue areas for size-order pairing"
    return sorted(cues, key=lambda comp: (int(comp["area"]), int(comp["color"])))


def _board_center_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = map(int, bbox)
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _coords_centroid(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _integral_manhattan_distance(a: tuple[float, float], b: tuple[float, float]) -> int:
    dist = abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))
    dist_i = int(round(dist))
    assert abs(dist - dist_i) < 1e-6, "expected an integral centroid-to-center Manhattan distance"
    return dist_i


def _extract_crop_and_outside(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, bbox, bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    r0, c0, r1, c1 = bbox
    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    return crop, bbox, outside_grid, int(bg)


def _apply_combo_rule(crop: np.ndarray, bbox: tuple[int, int, int, int], outside_grid: np.ndarray, bg: int) -> np.ndarray:
    layers = _sorted_layer_records(crop, int(bg))
    cues = _sorted_outside_cues(outside_grid, int(bg))
    assert len(cues) == len(layers), "expected exactly one outside cue per board layer"

    board_center = _board_center_from_bbox(bbox)
    out = crop.copy()
    for layer, cue in zip(layers, cues):
        cue_center = _coords_centroid(cue["coords"])
        dist = _integral_manhattan_distance(cue_center, board_center)
        rot_k = (-dist) % 4
        if rot_k:
            out = rotate_color_layer(out, int(layer["color"]), int(rot_k), bg=int(bg))
    return out


def _apply_size_area_baseline(crop: np.ndarray, outside_grid: np.ndarray, bg: int) -> np.ndarray:
    """Ablation used only to reject generator samples that collapse to the D1-only behavior."""
    layers = _sorted_layer_records(crop, int(bg))
    cues = _sorted_outside_cues(outside_grid, int(bg))
    if len(cues) != len(layers):
        raise AssertionError("expected exactly one outside cue per board layer")
    out = crop.copy()
    for layer, cue in zip(layers, cues):
        rot_k = (-int(cue["area"])) % 4
        if rot_k:
            out = rotate_color_layer(out, int(layer["color"]), int(rot_k), bg=int(bg))
    return out


def _apply_color_distance_baseline(crop: np.ndarray, bbox: tuple[int, int, int, int], outside_grid: np.ndarray, bg: int) -> np.ndarray:
    """Ablation used only to reject generator samples that collapse to the D2-only behavior."""
    board_center = _board_center_from_bbox(bbox)
    out = crop.copy()
    for col in colors_present(crop, exclude={int(bg)}):
        outside_comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        assert len(outside_comps) <= 1, f"expected at most one same-color outside cue for color {col}"
        if not outside_comps:
            continue
        cue_center = _coords_centroid(outside_comps[0]["coords"])
        dist = _integral_manhattan_distance(cue_center, board_center)
        rot_k = (-dist) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board and re-orient its color layers by combining size-ranked matching with distance-coded turns.

    Input structure:
    - The background is the dominant color.
    - One large multicolor square board is the unique largest foreground component.
    - Three small monochrome cue sprites sit away from the board; their colors do not identify board layers.

    Rule:
    - Crop around the board cluster and treat that crop as the working square.
    - Sort board color layers by in-crop area from smallest to largest.
    - Sort outside cue sprites by area from smallest to largest, then pair the two sorted lists.
    - For each matched pair, compute the cue centroid's Manhattan distance to the board center.
    - Rotate that paired board layer by a number of quarter-turns encoded by the distance mod 4.

    Key insight: Cue size rank chooses which layer to affect, but cue position relative to the board center chooses how much to rotate it.
    """
    crop, bbox, outside_grid, bg = _extract_crop_and_outside(grid)
    return _apply_combo_rule(crop, bbox, outside_grid, int(bg))


# --------------- generator (combo D1/2 + D2/2) ---------------

# ---- Configuration constants ----
CANVAS_SIZE = 20
BOARD_NS = (11, 13)
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

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 3
OUTER_AREA_PAD = 4
GEN_RETRIES = 400

CUE_TEMPLATES_BY_AREA = {
    1: (
        np.array([[1]], dtype=int),
    ),
    3: (
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
    ),
    4: (
        np.array([[1, 1], [1, 1]], dtype=int),
    ),
    5: (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    ),
}
CUE_AREA_OPTIONS = (
    (1, 3, 4),
    (1, 4, 5),
    (3, 4, 5),
)


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    """Return the 4 positions in the 90-degree rotation orbit of (r, c) in an n*n grid."""
    pts = set()
    rr, cc = r, c
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _is_rot_asym(mask: np.ndarray) -> bool:
    """True if mask is not invariant under any 90/180/270 rotation."""
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


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
    extra_lo_terms: list[int] = (),
) -> np.ndarray | None:
    """Build an n*n boolean mask with checker-perimeter base, asymmetry marker, and random fill."""
    mask = _checker_perimeter(n, random.randrange(2))

    orbit = _rot_orbit(*ASYM_MARKER, n)
    for p in orbit - {ASYM_MARKER}:
        mask[p] = False
    mask[ASYM_MARKER] = True

    lo = max(int(np.ceil(density_lo * n * n)), int(mask.sum()), *extra_lo_terms)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    blocked = orbit - {ASYM_MARKER}
    candidates = [(r, c) for r, c in _band_cells(n, LAYER_BAND_WIDTH) if (r, c) not in blocked and not mask[r, c]]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int) -> np.ndarray | None:
    """Build the middle concentric layer, inset by MIDDLE_INSET from each edge."""
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(m, MID_DENSITY_LO, min(MID_DENSITY_HI, hi_cap / (m * m)))
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int) -> np.ndarray | None:
    """Build the outer frame layer spanning the full n*n grid."""
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[min_area])


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    """Return valid inner-sprite span sizes for a board of side n."""
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray:
    """Build the small central sprite layer."""
    mask = np.zeros((n, n), dtype=bool)
    span = random.choice(_valid_inner_spans(n, require_asym))
    top = (n - span) // 2

    if span == 1:
        local = np.ones((1, 1), dtype=bool)
    elif span == 2:
        local = np.ones((2, 2), dtype=bool)
        local[random.randrange(2), random.randrange(2)] = False
    elif span == 3:
        variants = [
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ]
        local = random.choice(variants)
        local = np.rot90(local, random.randrange(4))
    else:
        local = np.zeros((span, span), dtype=bool)
        pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
        for r, c in pts:
            local[r, c] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))
    mask[top : top + local.shape[0], top : top + local.shape[1]] = local
    return mask


def _pick_rotations() -> list[int]:
    """Pick rotation amounts while covering the zero-turn branch without weakening the puzzle."""
    patterns: list[list[int]] = []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                rot_ks = [a, b, c]
                if len(set(rot_ks)) < 2:
                    continue
                nonzero = sum(int(k != 0) for k in rot_ks)
                if nonzero < 2:
                    continue
                patterns.append(rot_ks)
    return list(random.choice(patterns))


def _cue_patch(color: int, area: int) -> np.ndarray:
    """Colorize a template with the requested area."""
    template = random.choice(CUE_TEMPLATES_BY_AREA[int(area)])
    return np.where(template != 0, int(color), 0).astype(int)


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev (L-inf) gap distance between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _patch_centroid_offset(patch: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(patch != 0)
    return _coords_centroid(coords)


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
    target_residue: int,
) -> bool:
    """Place a cue patch on the canvas while matching the requested distance residue."""
    h, w = patch.shape
    mask = patch != 0
    board_center = _board_center_from_bbox(board_bbox)
    off_r, off_c = _patch_centroid_offset(patch)

    candidates: list[tuple[int, int, int]] = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue

            cue_center = (r + off_r, c + off_c)
            dist = _integral_manhattan_distance(cue_center, board_center)
            if dist % 4 != int(target_residue):
                continue
            candidates.append((dist, r, c))

    if not candidates:
        return False

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    keep = candidates[: max(1, min(24, len(candidates)))]
    _dist, r, c = random.choice(keep)
    bb = (r, c, r + h - 1, c + w - 1)
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


def _choose_cue_areas(rot_ks_in_sorted_layer_order: list[int]) -> list[int] | None:
    """Pick distinct cue areas that visibly disagree with the obsolete area-based rotation code."""
    options = list(CUE_AREA_OPTIONS)
    random.shuffle(options)
    residues = [(-int(k)) % 4 for k in rot_ks_in_sorted_layer_order]
    for areas in options:
        mismatch_count = sum((int(area) % 4) != int(residue) for area, residue in zip(areas, residues))
        if mismatch_count >= 2:
            return [int(a) for a in areas]
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    palette = random.sample(range(1, 10), 2 * NUM_COLORS)
    colors = [int(c) for c in palette[:NUM_COLORS]]
    cue_colors = [int(c) for c in palette[NUM_COLORS:]]
    rot_ks = _pick_rotations()
    n = random.choice(BOARD_NS)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=True)
    if not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    for color, mask in zip(colors, [outer_mask, mid_mask, inner_mask]):
        out_board[mask] = int(color)

    layer_specs = []
    for col, k in zip(colors, rot_ks):
        layer_specs.append({"color": int(col), "rot_k": int(k), "area": int(np.sum(out_board == int(col)))})
    layer_areas = [int(spec["area"]) for spec in layer_specs]
    if len(set(layer_areas)) != NUM_COLORS:
        return None
    sorted_layers = sorted(layer_specs, key=lambda spec: (int(spec["area"]), int(spec["color"])))

    cue_areas = _choose_cue_areas([int(spec["rot_k"]) for spec in sorted_layers])
    if cue_areas is None:
        return None

    in_board = np.zeros((n, n), dtype=int)
    for col, k in zip(colors, rot_ks):
        inv = np.rot90(out_board == int(col), (-int(k)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_specs = []
    shuffled_cue_colors = random.sample(cue_colors, len(cue_colors))
    for idx, (layer_spec, cue_area) in enumerate(zip(sorted_layers, cue_areas)):
        cue_specs.append(
            {
                "cue_color": int(shuffled_cue_colors[idx]),
                "cue_area": int(cue_area),
                "target_residue": (-int(layer_spec["rot_k"])) % 4,
            }
        )

    cue_bboxes: list[tuple[int, int, int, int]] = []
    for spec in sorted(cue_specs, key=lambda item: (-int(item["cue_area"]), int(item["cue_color"]))):
        patch = _cue_patch(int(spec["cue_color"]), int(spec["cue_area"]))
        if not _place_cue(canvas, patch, board_bbox, cue_bboxes, int(spec["target_residue"])):
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

    crop, bbox, outside_grid, bg = _extract_crop_and_outside(canvas)
    combo_out = _apply_combo_rule(crop, bbox, outside_grid, int(bg))
    if not np.array_equal(combo_out, out_board):
        return None
    if np.array_equal(_apply_size_area_baseline(crop, outside_grid, int(bg)), out_board):
        return None
    if np.array_equal(_apply_color_distance_baseline(crop, bbox, outside_grid, int(bg)), out_board):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

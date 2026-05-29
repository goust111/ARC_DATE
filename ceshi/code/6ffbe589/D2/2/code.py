# Variant of 6ffbe589
# Dimension: D2 — 变换参数编码方式 (match_cond)
# Change: cue_size mod 4 编码旋转次数，即提示精灵的像素数量对4取模决定旋转角度... → 提示精灵到板中心的曼哈顿距离编码旋转次数

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the D2 variant
# - source_task: none. The only intentional rule change is how each same-color cue
#   encodes the layer rotation.
# - Concrete behavior:
#   The large square board is still the unique largest component and still contains
#   three concentric, rotation-sensitive color layers. Small monochrome cues still
#   live outside the board and still match board colors 1:1. The difference is that
#   cue area is now irrelevant. For a cue of color X, the solver must find the cue's
#   centroid, measure its Manhattan distance to the board center, reduce that distance
#   mod 4, and use that value as the clockwise quarter-turn count for layer X.
# - Behavior trace:
#   Suppose a layer needs solver-side `rot_k = 1` (one CCW `np.rot90` step) to reach
#   the target crop. In the original task, a 3-cell cue could encode this because
#   `(-3) % 4 == 1`. In this variant, the cue can instead be a 2x2 square (area 4,
#   which would wrongly suggest "no rotation" under the old rule) placed so that its
#   centroid is Manhattan distance 7 from the board center; then `(-7) % 4 == 1`.
#   This makes position, not size, the controlling signal.
# - Complexity calibration:
#   The board structure, number of colors, and output-first inversion strategy remain
#   unchanged so the task stays close to the original. To keep the new distance signal
#   crisp and human-readable, the generator restricts board sizes to odd N so the board
#   center is an integer grid cell, and it uses a small family of cue templates whose
#   centroids are either integer/integer or half/half. That keeps centroid Manhattan
#   distances integral and visually traceable.
# - Hidden-assumption audit for `main()`:
#   `crop_components_near_largest(...)` still assumes all cues sit outside the board's
#   margin-3 expansion; the generator preserves Chebyshev gap >= 4, so the crop remains
#   exactly the board.
#   `colors_present(crop, exclude={bg})` still assumes the crop contains only board
#   colors; preserved by the same spacing guarantee.
#   `find_components(... monochromatic=True, colors=[col])` assumes at most one outside
#   cue per board color; the generator still creates at most one cue per rotating layer.
#   The new logic additionally assumes the cue centroid / board-center Manhattan distance
#   is an integer; the generator guarantees that through odd board sizes and cue template
#   choices, and the solver asserts that this holds.
#   Rotating layers sequentially remains safe because each layer is color-disjoint and
#   `rotate_color_layer` clears only its own color before repainting that mask.
# - Placement legality:
#   Cue placement now has to satisfy the old non-overlap / spacing rules and also match
#   a target centroid distance residue. The generator therefore enumerates candidate cue
#   positions, filters by exact Manhattan distance, board gap, and inter-cue gap, then
#   samples one valid location.
# - Visual observability:
#   Rotation-relevant cues appear on different distance rings (9/10/11 under the
#   preserved board-size and spacing constraints), while cue size/shape varies
#   independently. That makes the D2 change visible: same-looking cue sizes can cause
#   different rotations, and different-looking cue sizes can cause the same rotation if
#   they sit at matching distances from the board center.
# - Visual cleanliness:
#   The board remains a single compact crop, cues stay well separated from it and from
#   one another, and there is no stacking of multiple markers in one local region.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and re-orient its color layers using outside cue positions.

    Input structure:
    - The background is the dominant color.
    - One large multicolor square board is the unique largest foreground component.
    - Small monochrome cue sprites sit away from the board; each cue color matches one board layer.

    Rule:
    - Crop around the board cluster and treat that crop as the working square.
    - Compute the board center from the cropped board bbox.
    - For each color in the board, locate its same-color cue outside the board.
    - Rotate that color layer by a number of quarter-turns determined by the cue centroid's Manhattan distance to the board center, taken mod 4.
    - If a color has no outside cue, leave that layer unchanged.

    Key insight: The cue's placement relative to the board center, not its size, encodes the layer rotation.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster (ignore distant cue sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    board_center = _board_center_from_bbox((r0, c0, r1, c1))

    # --- STEP 2) For each layer, rotate according to its same-color cue distance ---
    colors = colors_present(crop, exclude={int(bg)})
    assert colors, "expected the cropped board to contain foreground colors"
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    for col in colors:
        outside_comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        assert len(outside_comps) <= 1, f"expected at most one outside cue for color {col}"
        if not outside_comps:
            continue
        cue_center = _coords_centroid(outside_comps[0]["coords"])
        dist = _integral_manhattan_distance(cue_center, board_center)
        rot_k = (-dist) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator (D2 variant) ---------------

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

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4

GEN_RETRIES = 300

CUE_TEMPLATES = (
    np.array([[1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
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
        if mask.sum() >= target:
            return True
        mask[r, c] = True
    return mask.sum() >= target


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
        if span == 4:
            pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
        else:
            pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
        for r, c in pts:
            local[r, c] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))
    mask[top : top + local.shape[0], top : top + local.shape[1]] = local
    return mask


def _pick_rotations() -> list[int]:
    """Pick rotation amounts for 3 color layers; at least one must be non-zero."""
    rot_ks = [random.randrange(4) for _ in range(3)]
    if not any(rot_ks):
        rot_ks[random.randrange(3)] = random.randint(1, 3)
    return rot_ks


def _cue_patch(color: int, template: np.ndarray) -> np.ndarray:
    """Colorize a cue template."""
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


def _cue_distance_targets(rot_k: int) -> list[int]:
    residue = (-int(rot_k)) % 4
    assert residue in (1, 2, 3), "generator only places cues for non-zero rotations"
    return [8 + residue]


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
    target_dists: list[int],
) -> bool:
    """Place a cue patch on the canvas, matching one of the requested centroid distances."""
    h, w = patch.shape
    mask = patch != 0
    board_center = _board_center_from_bbox(board_bbox)
    off_r, off_c = _patch_centroid_offset(patch)

    for target_dist in target_dists:
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

                cue_center = (r + off_r, c + off_c)
                dist = _integral_manhattan_distance(cue_center, board_center)
                if dist != int(target_dist):
                    continue
                candidates.append((r, c, bb))
        if candidates:
            r, c, bb = random.choice(candidates)
            paste(canvas, patch, r, c, transparent=0)
            cue_bboxes.append(bb)
            return True
    return False


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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    c_frame, c_mid, c_inner = colors
    rot_ks = _pick_rotations()

    valid_ns = [
        n
        for n in range(BOARD_N_MIN, BOARD_N_MAX + 1)
        if (n % 2 == 1) and _valid_inner_spans(n, rot_ks[2] > 0)
    ]
    n = random.choice(valid_ns)

    # Build concentric layer masks
    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=rot_ks[2] > 0)
    if rot_ks[2] > 0 and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    # Compose output board and invert rotations for input board
    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = c_frame
    out_board[mid_mask] = c_mid
    out_board[inner_mask] = c_inner

    in_board = np.zeros((n, n), dtype=int)
    for col, k in zip(colors, rot_ks):
        inv = np.rot90(out_board == col, (-k) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = col

    # Place board and cue sprites on canvas
    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(col, _cue_distance_targets(k)) for col, k in zip(colors, rot_ks) if k != 0]
    random.shuffle(cue_items)
    for col, target_dists in cue_items:
        patch = _cue_patch(col, random.choice(CUE_TEMPLATES))
        if not _place_cue(canvas, patch, board_bbox, cue_bboxes, target_dists):
            return None

    # Validate: board is unique largest component and oracle check passes
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

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 input")

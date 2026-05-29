# Variant of 6ffbe589
# Dimension: D4 — 源对象：提示精灵结构 (src)
# Change: 板外的1-3个单色小精灵，每个1-5像素，8-连通色块... → 提示精灵为L形/T形等特定形状

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for the D4 variant
#
# variant_spec:
#   preserved_backbone:
#     - Keep the original 20x20 canvas, square board crop, three centered color
#       layers, and output-first generation pipeline.
#     - Keep color matching the same: an outside cue still controls the layer of
#       the same color, and missing cue still means no rotation.
#   changed_dimension:
#     - Old src cue: arbitrary monochrome 8-connected blob, rotation decoded from
#       pixel-count mod 4.
#     - New src cue: a monochrome template chosen from a small semantic lexicon:
#         L tetromino -> rotate by 1 quarter-turn
#         T tetromino -> rotate by 2 quarter-turns
#         Cross pentomino -> rotate by 3 quarter-turns
#     - The solver must recognize the silhouette, not count cells.
#   concrete_arc_realization:
#     - Board colors remain visually layered and mostly asymmetric so rotation is
#       visible once a cue exists.
#     - Cue templates are all 8-connected, monochrome, and small:
#         L: shape 3x2 with 4 cells
#         T: shape 2x3 with 4 cells
#         Cross: shape 3x3 with 5 cells
#     - Cue orientation is randomized by rotation, but template type is the only
#       thing that matters logically.
#   behavior_trace:
#     - Suppose the red inner layer should be restored by one quarter-turn.
#     - Original task: place a red blob of size 1 or 5; solver computes
#       (-area) % 4 == 1 and rotates the red mask once.
#     - New task: place a red L-shaped tetromino; solver extracts that component,
#       normalizes its local mask, matches it against the L template family, and
#       rotates the red mask once.
#     - Difference: the control channel moved from arithmetic over area to shape
#       classification over a fixed vocabulary.
#   complexity_calibration:
#     - Only three cue categories exist, matching the three non-zero rotations.
#       This keeps the hypothesis space small enough for ARC.
#     - Using both L and T as 4-cell cues prevents degeneration into the original
#       size rule, because old size decoding would treat area 4 as zero rotation.
#     - Cross remains 5 cells; that is acceptable because the task still shows
#       repeated examples where L/T behave differently from size-based decoding.
#   implicit_assumption_review_for_main:
#     - crop_components_near_largest(...) assumes cues stay outside the margin-3
#       expansion around the board; generator enforces cue-board Chebyshev gap >= 4.
#     - colors_present(crop, exclude={bg}) assumes the cropped board still contains
#       every active layer color; generator keeps the board as the unique largest
#       component and never lets cues merge into it.
#     - find_components(..., colors=[col], monochromatic=True) assumes each board
#       color has at most one outside cue; generator creates at most one cue per
#       color and never duplicates cue colors.
#     - _decode_cue_rotation(...) assumes any present cue exactly matches one of
#       the fixed template families up to rotation/flip of its tight bbox; the
#       generator only emits those templates.
#     - rotate_color_layer(...) assumes the board crop is square; generator keeps
#       the board square and the crop helper returns the board bbox exactly.
#   placement_legality:
#     - Cue patches are placed only where they do not overlap existing pixels.
#     - Cue bbox distance from board bbox is at least 4, so the crop helper does
#       not accidentally absorb cue pixels.
#     - Cue bboxes are also mutually separated, which keeps them visually isolated.
#   visual_observability:
#     - Across a few examples, the same role is encoded by repeated recognizable
#       silhouettes (L/T/cross), so humans can infer "shape type -> rotation".
#     - The board colors are unchanged by color-id tricks; only geometry changes.
#   cleanliness:
#     - The board occupies one side of the canvas and cues live in empty space,
#       avoiding clutter and overlap.
#   new_solver_assumptions_to_audit:
#     - If a cue exists, its bbox must be tight enough that template recognition
#       is not padded by background. find_components bbox cropping guarantees this.
#     - If a layer gets a non-zero cue, its mask should be rotation-visible; the
#       generator keeps the original asymmetry safeguards for rotating layers.
# ---------------------------------------------------------------------------


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


CUE_TEMPLATES: dict[int, np.ndarray] = {
    1: np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    2: np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    3: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
}


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    arr = np.array(mask, dtype=np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), tuple(int(x) for x in arr.ravel())


def _template_signatures(mask: np.ndarray) -> set[tuple[int, int, tuple[int, ...]]]:
    sigs: set[tuple[int, int, tuple[int, ...]]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            sigs.add(_mask_signature(np.rot90(base, k)))
    return sigs


CUE_SIGNATURES = {rot_k: _template_signatures(mask) for rot_k, mask in CUE_TEMPLATES.items()}


def _component_patch(comp: dict) -> np.ndarray:
    coords = np.array(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _decode_cue_rotation(comp: dict) -> int:
    sig = _mask_signature(_component_patch(comp))
    matches = [rot_k for rot_k, sigs in CUE_SIGNATURES.items() if sig in sigs]
    assert len(matches) == 1, "expected cue to match exactly one template family"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and rotate its color layers by reading outside cue shapes.

    Input structure:
    - bg is the most common color.
    - The largest foreground component is a square multi-color board built from three centered layers.
    - Some board colors have a matching monochrome cue outside the board, drawn as an L, T, or cross template.

    Rule:
    - Tight-crop the main board cluster and ignore the remote cue sprites.
    - For each color present in the crop, look for the unique outside cue of that same color.
    - Decode the cue shape as quarter-turn count: L -> 1, T -> 2, cross -> 3.
    - Rotate that color layer by the decoded amount; if there is no cue, leave the layer unchanged.

    Key insight: The rotation amount is encoded by cue silhouette category rather than by cue size.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster while excluding distant cue shapes ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) For each color layer, decode its same-color cue template and rotate ---
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
        rot_k = _decode_cue_rotation(outside_comps[0]) if outside_comps else 0
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator (att2) ---------------

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


def _cue_patch(color: int, rot_k: int) -> np.ndarray:
    """Return a same-color cue sprite whose template type encodes the rotation amount."""
    template = CUE_TEMPLATES[int(rot_k)]
    patch = np.where(template, int(color), 0)
    return np.rot90(patch, random.randrange(4))


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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    c_frame, c_mid, c_inner = colors
    rot_ks = _pick_rotations()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, rot_ks[2] > 0)]
    n = random.choice(valid_ns)

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

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(col, k) for col, k in zip(colors, rot_ks) if k != 0]
    random.shuffle(cue_items)
    for col, rot_k in cue_items:
        if not _place_cue(canvas, _cue_patch(col, rot_k), board_bbox, cue_bboxes):
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

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 input")

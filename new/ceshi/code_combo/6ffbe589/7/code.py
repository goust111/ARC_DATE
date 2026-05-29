# Combo variant of 6ffbe589 (index 7)
# combo_spec: D2/3+D4/3
# Name: 孔洞数编码+L/T形精灵

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, label_mask, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# combo_spec:
#   preserved_backbone:
#     - Keep the original 20x20 canvas, largest-component crop, square board,
#       three concentric color layers, and output-first generation pipeline.
#     - Keep same-color binding: an outside cue still controls the board layer
#       drawn in that same color, and a missing cue still means no rotation.
#   fused_change:
#     - D4/3 contributes the cue vocabulary: every outside cue must belong to a
#       small recognizable family of semantic silhouettes: L, T, or cross.
#     - D2/3 changes the control channel: the quarter-turn count is not read
#       from cue area and not from family id; it is read from the number of
#       4-connected background holes enclosed by that cue.
#     - The implemented fusion is therefore: use L/T/cross-shaped carriers whose
#       internal room count equals the desired rotation (1, 2, or 3).
#   concrete_arc_realization:
#     - Each cue is a compact monochrome mask hand-designed to look like an L,
#       T, or cross while containing exactly 1/2/3 enclosed holes.
#     - Cue orientation is randomized by rotation and optional mirror, but hole
#       count is invariant, so topology rather than pose carries the signal.
#     - Because these carriers are larger than the original 1-5 cell blobs, the
#       board side is fixed to 10 and anchored to a corner; that leaves enough
#       empty space to keep the margin-3 crop from absorbing cues.
#   behavior_trace:
#     - Example: suppose the red middle layer must be restored by 2 quarter-turns.
#     - Original task: place a red blob whose area mod 4 implies 2, then rotate
#       the red layer accordingly after cropping the board.
#     - Single D4 variant: place a red T cue and decode T -> 2 directly from
#       silhouette category.
#     - This combo: place a red cue that still looks like one of {L,T,cross},
#       but the solver first verifies it belongs to that vocabulary and then
#       counts its enclosed holes; if the cue has 2 holes, rotate the red layer
#       twice regardless of whether the outer silhouette is L, T, or cross.
#     - Difference from both parents: outer shape is semantic, but the usable
#       parameter is topological.
#   complexity_calibration:
#     - The rotation domain is still only {1,2,3} for present cues, so humans
#       only need to infer a 3-level discrete code.
#     - Family and rotation are intentionally decoupled: the same rotation can be
#       shown by different cue families, and the same family appears with
#       different rotations across samples. This blocks degeneration into either
#       "area mod 4" or "family id" decoding.
#     - The cue library is small and repeated, so examples can support human
#       induction without exploding the hypothesis space.
#   implicit_assumption_review_for_main:
#     - crop_components_near_largest(...) assumes cue bboxes stay outside the
#       board's margin-3 expansion; generator enforces cue-board Chebyshev gap >= 4.
#     - colors_present(crop, exclude={bg}) assumes the crop still contains every
#       active board color; generator keeps the board as one isolated largest
#       multicolor component.
#     - find_components(... colors=[col], monochromatic=True) assumes at most one
#       outside cue per board color; generator creates at most one cue per color.
#     - _classify_cue_patch(...) assumes each cue exactly matches one library
#       template up to rotation/flip; generator emits only those templates.
#     - _count_internal_holes(...) assumes hole count is measured on the cue's
#       tight bbox and uses 4-connected background; templates are built so each
#       intended hole is separated under that connectivity.
#     - rotate_color_layer(...) assumes the crop is square; generator keeps the
#       board square and the crop helper returns the board bbox exactly.
#   placement_legality:
#     - Cue patches are placed only on empty cells, entirely within bounds.
#     - Cue placement uses backtracking because cues are larger than in the
#       source task and may compete for the corner-side free space.
#     - The board is anchored to a corner so the remaining background forms a
#       clean L-shaped region where cues can sit without overlap.
#   visual_observability:
#     - Humans can notice two stable regularities across examples:
#         1. outside markers keep reusing L/T/cross-like silhouettes;
#         2. the decisive variation is how many enclosed rooms those silhouettes
#            contain, not how many pixels they have and not which family they are.
#     - Rotating board layers remain asymmetric whenever a cue exists, so the
#       input/output difference is visible.
#   cleanliness:
#     - The board and cues occupy disjoint regions with a guaranteed moat.
#     - Cue families are monochrome and separated from each other, so hole rooms
#       are visually legible instead of being buried in clutter.
#   new_solver_assumptions_to_audit:
#     - If a cue exists, its foreground mask must be one of the recognized
#       library members after tight cropping. The generator guarantees this.
#     - If a color gets a cue, its board layer must be rotation-visible. The
#       generator preserves the original asymmetry guard for rotating layers.
#     - The board must stay larger than every cue component. The generator checks
#       board area against the largest placed cue before accepting a sample.
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


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


CUE_FAMILY_TEMPLATES: dict[str, dict[int, np.ndarray]] = {
    "L": {
        1: _mask_from_rows(
            [
                "111100",
                "101100",
                "111100",
                "100000",
                "100000",
                "111111",
            ]
        ),
        2: _mask_from_rows(
            [
                "111100",
                "101100",
                "111100",
                "110000",
                "101111",
                "111111",
            ]
        ),
        3: _mask_from_rows(
            [
                "111100",
                "101100",
                "111100",
                "111110",
                "101010",
                "111111",
            ]
        ),
    },
    "T": {
        1: _mask_from_rows(
            [
                "111111",
                "101111",
                "111111",
                "001100",
                "001100",
                "001100",
            ]
        ),
        2: _mask_from_rows(
            [
                "111111",
                "101101",
                "111111",
                "001100",
                "001100",
                "001100",
            ]
        ),
        3: _mask_from_rows(
            [
                "111111",
                "101101",
                "111111",
                "011110",
                "010010",
                "011110",
            ]
        ),
    },
    "X": {
        1: _mask_from_rows(
            [
                "0001000",
                "0011100",
                "1111111",
                "1110111",
                "1111111",
                "0011100",
                "0001000",
            ]
        ),
        2: _mask_from_rows(
            [
                "0001000",
                "0011100",
                "1111111",
                "1011101",
                "1111111",
                "0011100",
                "0001000",
            ]
        ),
        3: _mask_from_rows(
            [
                "0001000",
                "0111110",
                "1111111",
                "1010101",
                "1111111",
                "0111110",
                "0001000",
            ]
        ),
    },
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


SIGNATURE_TO_CUE_INFO: dict[tuple[int, int, tuple[int, ...]], tuple[str, int]] = {}
for _family, _templates in CUE_FAMILY_TEMPLATES.items():
    for _holes, _mask in _templates.items():
        for _sig in _template_signatures(_mask):
            prev = SIGNATURE_TO_CUE_INFO.get(_sig)
            assert prev is None or prev == (_family, _holes), "cue template signatures must be unambiguous"
            SIGNATURE_TO_CUE_INFO[_sig] = (_family, int(_holes))


def _component_patch(comp: dict) -> np.ndarray:
    coords = np.array(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _count_internal_holes(mask: np.ndarray) -> int:
    bg_mask = ~np.array(mask, dtype=bool)
    lab, n = label_mask(bg_mask, connectivity=4)
    holes = 0
    H, W = mask.shape
    for idx in range(1, int(n) + 1):
        coords = np.argwhere(lab == idx)
        if coords.size == 0:
            continue
        if (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == H - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == W - 1)
        ):
            continue
        holes += 1
    return int(holes)


def _classify_cue_patch(patch: np.ndarray) -> tuple[str, int]:
    sig = _mask_signature(np.array(patch, dtype=bool))
    assert sig in SIGNATURE_TO_CUE_INFO, "expected cue to match an L/T/cross template family"
    family, expected_holes = SIGNATURE_TO_CUE_INFO[sig]
    actual_holes = _count_internal_holes(np.array(patch, dtype=bool))
    assert actual_holes == expected_holes, "template signature and hole topology disagreed"
    return family, int(actual_holes)


def _decode_cue_rotation(comp: dict) -> int:
    _family, holes = _classify_cue_patch(_component_patch(comp))
    assert 1 <= holes <= 3, "expected present cues to encode a non-zero rotation"
    return int(holes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the main board and restore each color layer from topological cue sprites.

    Input structure:
    - bg is the most common color.
    - The largest foreground component is a square three-layer board, and remote same-color cues may sit outside it.
    - Every outside cue is monochrome and belongs to an L, T, or cross-shaped template family with one or more enclosed holes.

    Rule:
    - Tight-crop the largest board cluster and blank that region out from the original canvas.
    - For each board color, find the unique outside cue of the same color, if it exists.
    - Verify that the cue matches the L/T/cross vocabulary, count its enclosed 4-connected holes, and rotate that color layer by that many quarter-turns.
    - If a board color has no cue, leave that layer unchanged.

    Key insight: The cue's outer silhouette only says "this is a valid marker"; the actual rotation is encoded by the number of enclosed holes inside that silhouette.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster while excluding distant cue carriers ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Decode each same-color cue by template family + hole topology ---
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


# --------------- generator (combo) ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 10
NUM_COLORS = 3

OUTER_DENSITY_LO = 0.17
OUTER_DENSITY_HI = 0.32
MID_DENSITY_LO = 0.30
MID_DENSITY_HI = 0.50
MID_OUTER_CAP = 0.32

LAYER_BAND_WIDTH = 2
MIDDLE_INSET = 2
ASYM_MARKER = (1, 2)
INNER_SPAN_MARGIN = 6

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 2
OUTER_AREA_PAD = 4
GEN_RETRIES = 400


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
        pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
        for r, c in pts:
            if r < span and c < span:
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


def _colored(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), 0)


def _cue_variants(family: str, rot_k: int) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, tuple[int, ...]]] = set()
    mask = CUE_FAMILY_TEMPLATES[family][int(rot_k)]
    bases = [mask, np.fliplr(mask)]
    random.shuffle(bases)
    for base in bases:
        rots = list(range(4))
        random.shuffle(rots)
        for k in rots:
            var = np.rot90(base, k)
            sig = _mask_signature(var)
            if sig in seen:
                continue
            seen.add(sig)
            variants.append(var.copy())
    random.shuffle(variants)
    return variants


def _assign_cue_families(cue_items: list[tuple[int, int]]) -> list[dict[str, int | str]]:
    """
    Assign cue families independently of rotation so all semantic cue branches stay visible.

    When there are multiple cues, prefer distinct families within the same sample; this keeps
    L/T/cross all active across generated examples instead of letting the smallest templates
    dominate placement success.
    """
    if not cue_items:
        return []

    families = list(CUE_FAMILY_TEMPLATES.keys())
    random.shuffle(families)
    chosen = families[: min(len(cue_items), len(families))]
    if len(chosen) < len(cue_items):
        extra = [random.choice(families) for _ in range(len(cue_items) - len(chosen))]
        chosen.extend(extra)
    random.shuffle(chosen)

    specs: list[dict[str, int | str]] = []
    for (col, rot_k), family in zip(cue_items, chosen):
        specs.append({"color": int(col), "rot_k": int(rot_k), "family": str(family)})
    return specs


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev (L-inf) gap distance between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _placement_candidates(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    h, w = patch.shape
    mask = patch != 0
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
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
    random.shuffle(candidates)
    return candidates


def _place_cues(
    canvas: np.ndarray,
    cue_specs: list[dict[str, int | str]],
    board_bbox: tuple[int, int, int, int],
) -> list[dict] | None:
    specs = []
    for item in cue_specs:
        family = str(item["family"])
        rot_k = int(item["rot_k"])
        variants = _cue_variants(family, rot_k)
        max_dim = max(max(mask.shape) for mask in variants)
        specs.append(
            {
                "color": int(item["color"]),
                "rot_k": rot_k,
                "family": family,
                "variants": variants,
                "max_dim": int(max_dim),
            }
        )
    specs.sort(key=lambda item: item["max_dim"], reverse=True)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    placements: list[dict] = []

    def rec(idx: int) -> bool:
        if idx == len(specs):
            return True
        spec = specs[idx]
        for mask in spec["variants"]:
            patch = _colored(mask, int(spec["color"]))
            positions = _placement_candidates(canvas, patch, board_bbox, cue_bboxes)
            for r, c, bb in positions:
                paste(canvas, patch, int(r), int(c), transparent=0)
                cue_bboxes.append(bb)
                placements.append(
                    {
                        "color": int(spec["color"]),
                        "rot_k": int(spec["rot_k"]),
                        "family": str(spec["family"]),
                        "bbox": bb,
                        "area": int(mask.sum()),
                    }
                )
                if rec(idx + 1):
                    return True
                placements.pop()
                cue_bboxes.pop()
                canvas[int(r) : int(r) + patch.shape[0], int(c) : int(c) + patch.shape[1]][patch != 0] = 0
        return False

    return placements if rec(0) else None


def _board_anchor(n: int) -> tuple[int, int]:
    """Pick a board top-left position at one of the four canvas corners."""
    spare = CANVAS_SIZE - n
    return random.choice([(0, 0), (0, spare), (spare, 0), (spare, spare)])


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    c_frame, c_mid, c_inner = colors
    rot_ks = _pick_rotations()
    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)

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

    cue_items = [(int(col), int(k)) for col, k in zip(colors, rot_ks) if k != 0]
    random.shuffle(cue_items)
    cue_specs = _assign_cue_families(cue_items)
    placements = _place_cues(canvas, cue_specs, board_bbox)
    if placements is None:
        return None

    if placements and int((out_board != 0).sum()) <= max(int(item["area"]) for item in placements):
        return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    for item in placements:
        if _bbox_chebyshev(item["bbox"], board_bbox) < CUE_BOARD_DIST:
            return None
        fam, holes = _classify_cue_patch(_component_patch(
            {
                "coords": np.argwhere(_colored(CUE_FAMILY_TEMPLATES[item["family"]][item["rot_k"]], item["color"]) != 0),
                "bbox": (0, 0, CUE_FAMILY_TEMPLATES[item["family"]][item["rot_k"]].shape[0] - 1, CUE_FAMILY_TEMPLATES[item["family"]][item["rot_k"]].shape[1] - 1),
            }
        ))
        if fam != item["family"] or holes != item["rot_k"]:
            return None

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

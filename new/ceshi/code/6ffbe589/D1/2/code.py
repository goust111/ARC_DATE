# Variant of 6ffbe589
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: s.color == d.color，提示精灵与板内颜色层通过颜色一致性配对... → 尺寸匹配：提示精灵面积与颜色层面积排序对应
#
# Design analysis for the D1 variant
# ==================================
# a) concretization_and_trace:
#    - Board geometry stays the same: a largest square multi-color board crop with 3 concentric color
#      layers, each layer represented by all pixels of one board color.
#    - The crucial visible change is outside the board: cue blobs no longer use board colors as identity.
#      Their colors are chosen from a disjoint palette subset, so the only usable relation is size order.
#    - Concrete trace:
#      board-layer areas = [5, 16, 27] after sorting small->large;
#      cue areas = [2, 7, 12] after sorting small->large.
#      Old logic: try color-match; this fails because cue colors are unrelated to board colors.
#      New logic: pair 5<->2, 16<->7, 27<->12, then rotate those three layers by
#      ((-2) % 4), ((-7) % 4), ((-12) % 4) quarter-turns respectively.
# b) complexity_calibration:
#    - To avoid ambiguity, all three board layer areas are forced distinct, and all three cue areas are
#      forced distinct as well.
#    - We keep exactly three cues, one per layer. This removes the under-constrained "which subset rotates?"
#      branch that would otherwise make size-order matching hard to infer from 2-4 examples.
#    - Cue sizes still live in a small integer family and still encode rotation by area mod 4, so the task
#      stays close to the source puzzle instead of becoming a generic assignment problem.
# c) implicit_assumption_audit_for_main:
#    - Original line pattern `find cue with same color as layer` relied on outside cue colors matching board
#      colors. That assumption is invalid here and is replaced by global sorting on total layer area.
#    - Sorting now assumes layer areas are unique and cue areas are unique. `main()` asserts both; the
#      generator guarantees them.
#    - The rotation helper still assumes the cropped board is square. The board generator still produces a
#      square NxN board, so that old assumption remains valid.
#    - Pairing by sorted order assumes cue count == layer count. The generator always places 3 cues.
# d) placement_legality:
#    - The board remains the unique largest 8-connected foreground component.
#    - Cue blobs are placed outside the board crop margin (Chebyshev distance >= 4), so the crop helper
#      excludes them exactly as before.
#    - Cue blobs are 8-connected, non-overlapping, and mutually separated.
# e) visual_observability:
#    - Humans can see that cue colors no longer match any board color, so color matching is visibly ruled out.
#    - The three board layers have obviously different scales (inner/middle/outer), and cue blobs are also
#      visibly size-ordered, making the new matching cue available directly in the pixels.
#    - At least one rotation is non-zero, so each sample exhibits an actual transformation.
# f) visual_cleanliness:
#    - The board still occupies one coherent region; cues stay detached in the background.
#    - Using only 3 cue blobs and disjoint colors keeps the scene readable.
# g) color_change_inference:
#    - There is no cue-to-board recoloring rule. Board colors are preserved exactly.
#    - Cue colors are intentionally non-informative and disjoint from board colors, so they cannot act as a
#      hidden matching key.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board and rotate its color layers by pairing them with size-ranked outside cues.

    Input structure:
    - bg = most common color.
    - A main multi-color board cluster (largest foreground region) with three differently sized color layers.
    - Three small monochrome cue sprites outside the board; their colors do not identify the target layer.

    Rule:
    - Tight-crop around the board cluster.
    - Sort board color layers by in-crop pixel count from smallest to largest.
    - Sort outside cue sprites by pixel count from smallest to largest.
    - Pair the two sorted lists and rotate each matched layer clockwise by (cue pixel count mod 4) × 90°.

    Key insight: Outside cue color is irrelevant; cue area rank tells you which board layer receives that cue's rotation code.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the "board" cluster (drop distant sample sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Pair board layers and outside cues by sorted area, then rotate ---
    layers = _sorted_layer_records(crop, int(bg))
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    cues = _sorted_outside_cues(outside_grid, int(bg))
    assert len(cues) == len(layers), "expected exactly one outside cue per board layer"

    for layer, cue in zip(layers, cues):
        cue_size = int(cue["area"])
        rot_k = (-cue_size) % 4
        if rot_k:
            out = rotate_color_layer(out, int(layer["color"]), int(rot_k), bg=int(bg))

    return out


# --------------- generator (att2 D1-size-ordering) ---------------

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

CUE_GRID = 4
CUE_AREA_MAX = 12
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 3
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


def _cue_patch(color: int, area: int) -> np.ndarray:
    """Grow an 8-connected monochromatic blob of given area in a CUE_GRID*CUE_GRID workspace, then crop."""
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < area:
        frontier = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[r, c] = color
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


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


def _choose_sorted_cue_areas(rot_ks_in_layer_order: list[int]) -> list[int] | None:
    """Choose strictly increasing cue areas whose mod-4 classes encode the requested rotations."""
    by_residue = {res: [a for a in range(1, CUE_AREA_MAX + 1) if a % 4 == res] for res in range(4)}
    residues = [(-int(k)) % 4 for k in rot_ks_in_layer_order]
    chosen: list[int] = []

    def dfs(i: int, prev: int) -> list[int] | None:
        if i == len(residues):
            return list(chosen)
        candidates = by_residue[int(residues[i])][:]
        random.shuffle(candidates)
        for area in candidates:
            if int(area) <= int(prev):
                continue
            chosen.append(int(area))
            result = dfs(i + 1, int(area))
            if result is not None:
                return result
            chosen.pop()
        return None

    return dfs(0, 0)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    palette = random.sample(range(1, 10), 2 * NUM_COLORS)
    colors = [int(c) for c in palette[:NUM_COLORS]]
    cue_colors = [int(c) for c in palette[NUM_COLORS:]]
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

    layer_specs = []
    for col, k in zip(colors, rot_ks):
        layer_specs.append({"color": int(col), "rot_k": int(k), "area": int(np.sum(out_board == int(col)))})
    layer_areas = [int(spec["area"]) for spec in layer_specs]
    if len(set(layer_areas)) != NUM_COLORS:
        return None
    sorted_layers = sorted(layer_specs, key=lambda spec: (int(spec["area"]), int(spec["color"])))

    cue_areas = _choose_sorted_cue_areas([int(spec["rot_k"]) for spec in sorted_layers])
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

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = list(zip(random.sample(cue_colors, len(cue_colors)), cue_areas))
    random.shuffle(cue_items)
    for cue_col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(int(cue_col), int(area)), board_bbox, cue_bboxes):
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

    if set(colors) & set(cue_colors):
        return None
    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 D1-size-ordering input")

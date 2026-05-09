# Variant of 6ffbe589
# Dimension: D14 — 符号编码：面积到参数的映射 (symbol_instructions)
# Change: cue_size mod 4 直接映射到旋转次数，线性取模关系... → 提示精灵的朝向（长轴方向）编码旋转方向

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the D14 variant
# ----------------------------------
# a) Concrete behavior:
#    The board stays the same as the original task: one largest multicolor square crop
#    made of three non-overlapping concentric color layers. The change is only in the
#    outside cue sprites. Instead of arbitrary small blobs whose area modulo 4 selects
#    the rotation, every cue is now a thin monochrome strip with a visible long-axis:
#    horizontal 1x4 -> rotate 90 degrees, vertical 4x1 -> rotate 180 degrees,
#    diagonal 4-on-a-diagonal -> rotate 270 degrees. No cue still means no rotation.
#    Small trace example: originally a color with cue area 3 produced rot_k = 1 in
#    the solver. In the variant, that same color must instead have a horizontal strip,
#    and the solver must ignore area and read orientation -> rot_k = 1.
# b) Complexity calibration:
#    Only three cue types are used, each visually distinct and repeated across samples.
#    Their area is deliberately fixed at 4 so the new rule does not collapse back into
#    the old one. This also avoids ambiguous cases such as tiny squares or compact blobs.
# c) Old assumptions that break:
#    - `cue_size = area` is no longer meaningful; area must be ignored.
#    - A cue is no longer "any monochrome component of that color"; it must satisfy one
#      of three strip templates, otherwise the solver should reject the input.
#    - Because diagonal cues are now valid, bbox shape alone is insufficient unless the
#      solver also checks the occupied cells form one of the two diagonals.
# d) Placement legality:
#    Cue placement still respects non-overlap, canvas bounds, distance from the board,
#    and distance from other cues. Fixed cue area 4 keeps every cue far smaller than the
#    board, so the board remains the unique largest component.
# e) Visual observability:
#    The new symbol is directly visible: line direction. Since all non-zero cues have the
#    same area, a human cannot explain the transformation with the old size rule. The
#    board layers that rotate are forced to be rotationally asymmetric whenever needed,
#    so the rotation is visible.
# f) Visual cleanliness:
#    Cues remain outside the board with the same spacing policy as before, so the crop is
#    visually clean and the three layers remain easy to separate by color.
# g) New solver assumptions and generator guarantees:
#    - Each rotating color has at most one outside cue of the same color.
#    - Every cue is exactly one of: horizontal strip, vertical strip, or diagonal strip.
#    - Rotating colors use asymmetric masks; inactive colors may stay unchanged.
#    The generator explicitly constructs cues and layer masks to satisfy all of these.


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


def cue_rotation_from_component(comp: dict) -> int:
    """Decode a cue strip orientation into the number of CCW quarter turns."""
    coords = np.array(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}

    if h == 1 and w >= 3 and len(rel) == w:
        return 1
    if w == 1 and h >= 3 and len(rel) == h:
        return 2
    if h == w and h >= 3 and len(rel) == h:
        main_diag = {(i, i) for i in range(h)}
        anti_diag = {(i, h - 1 - i) for i in range(h)}
        if rel == main_diag or rel == anti_diag:
            return 3

    raise AssertionError("expected cue to be a horizontal, vertical, or diagonal strip")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a board cluster and rotate its color layers based on outside cue strips.

    Input structure:
    - bg = most common color.
    - A main multi-color board cluster (largest foreground region).
    - Small monochrome cue strips outside the board for the colors that rotate.

    Rule:
    - Tight-crop around the board cluster.
    - For each color layer inside the crop, find its same-color cue outside.
    - Decode the cue strip orientation: horizontal -> 90 degrees, vertical -> 180 degrees,
      diagonal -> 270 degrees, then rotate that color layer by that many quarter turns.
    - No outside cue -> no rotation.

    Key insight: Cue strip orientation, not cue area, is the symbol that encodes the rotation amount.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the "board" cluster (drop distant sample sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) For each layer, rotate according to its unique outside cue orientation ---
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
        rot_k = cue_rotation_from_component(outside_comps[0]) if outside_comps else 0
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator ---------------

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

CUE_LENGTH = 4
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
    """Pick layer rotations with at least two active cues and at least two cue types."""
    while True:
        rot_ks = [0, 0, 0]
        active = random.sample(range(3), random.choice([2, 3]))
        for idx in active:
            rot_ks[idx] = random.randint(1, 3)
        if len({k for k in rot_ks if k != 0}) >= 2:
            return rot_ks


def _cue_patch(color: int, rot_k: int) -> np.ndarray:
    """Build a fixed-area strip cue whose orientation encodes the rotation."""
    col = int(color)
    if rot_k == 1:
        return np.full((1, CUE_LENGTH), col, dtype=int)
    if rot_k == 2:
        return np.full((CUE_LENGTH, 1), col, dtype=int)
    if rot_k == 3:
        arr = np.zeros((CUE_LENGTH, CUE_LENGTH), dtype=int)
        if random.randrange(2):
            for i in range(CUE_LENGTH):
                arr[i, i] = col
        else:
            for i in range(CUE_LENGTH):
                arr[i, CUE_LENGTH - 1 - i] = col
        return arr
    raise AssertionError("cue patch requested for zero rotation")


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

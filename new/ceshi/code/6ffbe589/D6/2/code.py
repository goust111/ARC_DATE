# Variant of 6ffbe589
# Dimension: D6 — 网格输出形状 (grid_shape)
# Change: tight-crop，输出为板的外接矩形裁剪... → 分层输出：每个颜色层独立输出为一个子网格

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# Design analysis for the D6 variant:
# a) Concreteization and trace:
#    - Input stays the same as the source task: one large square board plus 1..3 remote cue sprites.
#    - The board still contains three concentric color layers that may need independent 90-degree rotations.
#    - The old output was one tight crop of the whole repaired board.
#    - The new output is a gallery: after repairing the board, isolate each color layer, tight-crop that
#      single-color layer, then concatenate those crops left-to-right with one background column between frames.
#    - Small trace:
#      old flow: crop whole board -> rotate blue/red/yellow masks in-place -> return NxN repaired crop.
#      new flow: crop whole board -> rotate blue/red/yellow masks in-place -> extract outer/middle/inner
#      crops separately -> pack them into [outer | middle | inner].
#      The rotation rule is unchanged; only the final representation changes.
# b) Complexity calibration:
#    - The variant should be visibly different from tight-crop output on nearly every sample.
#    - Order is fixed by visible geometry (largest bbox to smallest bbox), not by hidden color ids.
#    - We keep exactly three layers and one fixed packing direction (horizontal) to avoid excessive freedom.
# c) Implicit-assumption audit for main():
#    - crop_components_near_largest(...) still assumes the board is the unique largest foreground component.
#      Generator keeps that invariant.
#    - outside_grid[...] = bg still assumes all cues are outside the board crop. Generator enforces spacing.
#    - Rotating layers in a shared square crop is still valid because the board crop remains square.
#    - The old final "return out" depended on D6=tight-crop. This is the only semantic assumption that changes.
#      We now call a gallery builder instead.
# d) Placement legality:
#    - Input placement rules are unchanged: board and cues cannot overlap, must fit in 20x20, and must respect
#      Chebyshev gaps so cues are never accidentally included in the board crop.
#    - Output packing is always legal because frames are built after tight-cropping each color layer, then pasted
#      into a fresh background canvas sized to max frame height and total frame width.
# e) Visual observability:
#    - A human can see that the output is not a single board anymore; it is a left-to-right decomposition.
#    - The frame order is inferable from containment/nesting size: outer layer first, then middle, then inner.
#    - No color-id-based ordering or hidden palette convention is used.
# f) Visual cleanliness:
#    - Each subgrid contains exactly one color on background, so there is no overlap ambiguity.
#    - One blank background column separates neighboring frames, making the decomposition easy to parse.
# g) Color-change inference:
#    - There is no new color remapping. Colors keep their identity from input board to output frames.
# h) Generator anti-degeneration check:
#    - Oracle validation uses the new gallery output, not the old board crop.
#    - Extra post-test statistics compare gallery outputs against the old tight-crop style to confirm that
#      generated samples actually exercise the new D6 behavior.
# i) New assumptions introduced by the gallery step:
#    - Each layer must remain non-empty after rotation.
#    - Layer spans must be distinct enough that "outer -> inner" ordering by bbox area is well-defined.
#      Generator guarantees concentric layers with strictly decreasing bbox sizes.


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

    h, w = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(h - 1), int(mr1 + margin))
    ec1 = min(int(w - 1), int(mc1 + margin))

    keep_mask = np.zeros((h, w), dtype=bool)
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


def _crop_single_color(grid: np.ndarray, color: int, *, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Tight-crop one color layer while preserving that color on background."""
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"expected color {color} to appear in the repaired board"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    frame = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    frame[sub == int(color)] = int(color)
    return frame, (int(r0), int(c0), int(r1), int(c1))


def _pack_frames_horiz(frames: list[np.ndarray], *, bg: int, gap: int = 1) -> np.ndarray:
    """Pack frames left-to-right with a fixed background gap."""
    assert frames, "expected at least one frame to pack"
    height = max(int(frame.shape[0]) for frame in frames)
    width = sum(int(frame.shape[1]) for frame in frames) + int(gap) * (len(frames) - 1)
    out = np.full((height, width), int(bg), dtype=frames[0].dtype)

    col = 0
    for frame in frames:
        h, w = map(int, frame.shape)
        out[:h, col : col + w] = frame
        col += w + int(gap)
    return out


def build_layer_gallery(board: np.ndarray, *, bg: int | None = None) -> np.ndarray:
    """
    Convert a repaired board into a frame-separated gallery of tight-cropped color layers.

    Layers are ordered by visible nesting scale: largest bbox first, smallest bbox last.
    """
    if bg is None:
        bg = get_bg(board)
    bg = int(bg)

    colors = colors_present(board, exclude={int(bg)})
    assert colors, "expected the repaired board to contain foreground colors"

    layers: list[tuple[int, int, np.ndarray]] = []
    bbox_areas: list[int] = []
    for col in colors:
        frame, (r0, c0, r1, c1) = _crop_single_color(board, int(col), bg=int(bg))
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        bbox_areas.append(bbox_area)
        layers.append((bbox_area, int(np.sum(board == int(col))), frame))

    assert len(set(bbox_areas)) == len(bbox_areas), "expected concentric layers with distinct spans"
    layers.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return _pack_frames_horiz([frame for _, _, frame in layers], bg=int(bg), gap=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a square board from outside cue sprites, then display its color layers as separate frames.

    Input structure:
    - The background is the most frequent color.
    - One large multicolor square board is the unique largest foreground component.
    - Small monochrome cue blobs sit away from the board; a cue color matches one board layer.

    Rule:
    - Crop around the board cluster and ignore the remote cue blobs.
    - For each board color, use the same-color cue's pixel count modulo 4 to determine the layer's 90-degree
      clockwise rotation; missing cue means that layer stays as-is.
    - After all layer rotations are applied on the board, tight-crop each color layer separately and concatenate
      those layer crops from outermost to innermost with a one-column background gap.

    Key insight: The cues still control per-color rotation, but the final answer is no longer the repaired board
    itself; it is a frame-separated decomposition of that repaired board into its three nested color layers.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster while excluding distant cue sprites. ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Rotate each color layer according to its same-color outside cue. ---
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
        cue_size = int(outside_comps[0]["area"]) if outside_comps else 0
        rot_k = (-cue_size) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    # --- STEP 3) Separate the repaired board into one cropped frame per color layer. ---
    return build_layer_gallery(out, bg=int(bg))


# --------------- generator (D6 variant) ---------------

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

CUE_GRID = 3
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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
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
    out_board[outer_mask] = colors[0]
    out_board[mid_mask] = colors[1]
    out_board[inner_mask] = colors[2]
    expected = build_layer_gallery(out_board, bg=0)

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
    cue_items = [(col, random.choice([1, 5]) if (-k) % 4 == 1 else (-k) % 4) for col, k in zip(colors, rot_ks) if k != 0]
    random.shuffle(cue_items)
    for col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(col, area), board_bbox, cue_bboxes):
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

    if not np.array_equal(main(canvas), expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 D6 input")

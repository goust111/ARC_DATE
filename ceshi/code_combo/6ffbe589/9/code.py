# Combo variant of 6ffbe589 (index 9)
# combo_spec: D6/2+D7/2
# Name: 分层输出+交替旋转方向

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concreteization and trace
# - Input keeps the same object inventory as the source task: one large square
#   board and detached monochrome cue blobs.
# - The board still has exactly three concentric color layers. They must first
#   be repaired inside one shared square crop, because cue matching is by color
#   and rotation happens layer-by-layer on the same board canvas.
# - Fusion changes two stages at once:
#   1. rotation direction is no longer global; it alternates by visible depth:
#      outer CW, middle CCW, inner CW.
#   2. final output is no longer the repaired board crop; it is a gallery made
#      from three separately cropped layer frames packed left-to-right.
# - Small trace:
#   old task: crop board -> each cue area gives CW turns -> return repaired NxN.
#   combo:    crop board -> order colors by bbox span -> same cue area gives turn
#             amount, depth gives direction -> repair board -> crop each color ->
#             pack [outer | middle | inner].
# - The important interaction is that frame separation exposes the alternating
#   direction rule: the viewer can inspect each repaired layer in isolation.
#
# b) Complexity calibration
# - If the middle layer used 0 or 2 turns, CW and CCW would look identical, so
#   D7 would be visually weak.
# - The generator therefore gives every layer a cue and restricts turn amounts
#   to odd values {1, 3}. This forces the middle layer to differ from the
#   original all-CW interpretation on every valid sample.
# - Packing order is fixed by visible geometry, not color id, and remains only
#   three frames, so the rule stays inferable rather than combinatorial.
#
# c) Implicit-assumption audit for main()
# - crop_components_near_largest(...) assumes the board is the unique largest
#   foreground component. Generator preserves that.
# - outside_grid[...] = bg assumes cues lie fully outside the board crop.
#   Generator enforces Chebyshev spacing from the board.
# - Layer order can no longer come from colors_present(...), because sorted
#   numeric color ids are unrelated to visual depth. We must rank colors by bbox
#   area after cropping the board.
# - rotate_color_layer(...) assumes the working crop is square so np.rot90 keeps
#   shape. The board crop is always an NxN square by construction.
# - The old final "return out" is invalid because D6 changes the output format.
#   We must convert the repaired board into a layer gallery instead.
#
# d) Placement legality
# - The board and cue patches are placed without overlap on a 20x20 canvas.
# - Cue bboxes must stay at Chebyshev distance >= 4 from the board and from one
#   another, so the margin-3 crop never absorbs them.
# - The three board masks remain disjoint and strictly nested, so rotating one
#   color mask at a time has unambiguous ownership.
#
# e) Visual observability
# - From 2-4 examples, a human can see three detached output frames ordered by
#   size, which supports the D6 part.
# - Because all three layers carry cues and all turn amounts are odd, the middle
#   frame always turns opposite to outer and inner under the same cue-reading
#   rule, which supports the D7 part.
# - No hidden factor such as numeric color ordering is needed to infer either
#   the frame order or the direction rule.
#
# f) Visual cleanliness
# - Each output frame contains exactly one color on background, so there is no
#   overlap ambiguity inside the gallery.
# - A single blank column separates frames, which keeps the decomposition legible.
# - The input remains uncluttered because cue blobs are remote and sparse.
#
# g) Change inferability
# - There is no color remapping. Color only links a cue blob to one board layer.
# - Depth is inferred from nested bbox span: largest crop is outer, smallest is
#   inner. That structural cue is stable across all generated examples.
#
# h) Anti-degeneration plan
# - Oracle validation uses the fused gallery output, not the repaired board crop.
# - Generator forces odd turn amounts for all three layers, so the fused solver
#   output differs from a D6-only "all CW then gallery" baseline on every sample.
# - Because the gallery width is the sum of three crops plus gaps, it also
#   differs from the original tight-crop format on every sample.
#
# i) New assumptions introduced and how the generator guarantees them
# - Each layer must stay non-empty after rotation: guaranteed because each layer
#   is a rotated mask, never deleted.
# - The three layer bbox spans must be distinct for visible ordering:
#   guaranteed by full-board outer mask, inset middle mask, and tiny centered
#   inner sprite.
# - There is at most one outside cue per board color: guaranteed by one cue
#   patch per layer.
# -----------------------------------------------------------------------------


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


def _color_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"expected color {color} to appear in the board"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return tuple(map(int, (r0, c0, r1, c1)))


def _color_bbox_area(grid: np.ndarray, color: int) -> int:
    r0, c0, r1, c1 = _color_bbox(grid, int(color))
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _layer_order(crop: np.ndarray, bg: int) -> list[int]:
    colors = colors_present(crop, exclude={int(bg)})
    assert len(colors) == 3, "expected exactly three board colors"

    ranked = sorted(((_color_bbox_area(crop, col), int(col)) for col in colors), reverse=True)
    ordered = [col for _area, col in ranked]
    areas = [_color_bbox_area(crop, col) for col in ordered]
    assert len(set(areas)) == len(areas), "expected distinct bbox spans for outer/middle/inner layers"
    return ordered


def _rot_k_from_layer_and_cue(layer_idx: int, cue_size: int) -> int:
    turns = int(cue_size) % 4
    if turns == 0:
        return 0
    if layer_idx % 2 == 0:
        return (-turns) % 4
    return turns


def _crop_single_color(grid: np.ndarray, color: int, *, bg: int) -> np.ndarray:
    """Tight-crop one color layer while preserving that color on background."""
    r0, c0, r1, c1 = _color_bbox(grid, int(color))
    frame = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    frame[sub == int(color)] = int(color)
    return frame


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

    ordered_colors = _layer_order(board, bg)
    frames = [_crop_single_color(board, col, bg=int(bg)) for col in ordered_colors]
    return _pack_frames_horiz(frames, bg=int(bg), gap=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a square board from outside cue sprites, then display its layers as separate frames.

    Input structure:
    - The background is the most frequent color.
    - One large multicolor square board is the unique largest foreground component.
    - Three detached monochrome cue blobs sit outside the board; each cue color matches one board layer.

    Rule:
    - Crop around the board cluster and rank its three colors by visible depth: outer, middle, inner.
    - For each layer, read the same-color cue's pixel count modulo 4 as the turn amount.
    - Apply alternating directions by layer depth: outer clockwise, middle counterclockwise, inner clockwise.
    - After repairing the board, tight-crop each layer separately and concatenate the three crops from
      outermost to innermost with a one-column background gap.

    Key insight: Cue size only tells how many quarter-turns to apply; the direction comes from nested layer
    depth, and the answer is a frame-separated decomposition rather than the repaired full board crop.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster and recover the visible outer->inner order. ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    ordered_colors = _layer_order(crop, bg)
    out = crop.copy()

    # --- STEP 2) Use same-color outside cues for turn amount, then layer depth for direction. ---
    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    for layer_idx, col in enumerate(ordered_colors):
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
        rot_k = _rot_k_from_layer_and_cue(layer_idx, cue_size)
        if rot_k:
            out = rotate_color_layer(out, int(col), rot_k, bg=int(bg))

    # --- STEP 3) Split the repaired board into three ordered layer frames. ---
    return build_layer_gallery(out, bg=int(bg))


# --------------- generator (combo variant) ---------------

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


def _valid_inner_spans(n: int) -> list[int]:
    """Return valid asymmetric inner-sprite span sizes for a board of side n."""
    max_span = n - INNER_SPAN_MARGIN
    return [s for s in range(1, max_span + 1) if (s % 2) == (n % 2) and s > 1]


def _make_inner_mask(n: int) -> np.ndarray:
    """Build the small central sprite layer."""
    mask = np.zeros((n, n), dtype=bool)
    span = random.choice(_valid_inner_spans(n))
    top = (n - span) // 2

    if span == 2:
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
    elif span == 4:
        local = np.zeros((4, 4), dtype=bool)
        for r, c in [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]:
            local[r, c] = True
        local = np.rot90(local, random.randrange(4))
    else:
        local = np.zeros((5, 5), dtype=bool)
        for r, c in [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]:
            local[r, c] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))

    mask[top : top + local.shape[0], top : top + local.shape[1]] = local
    return mask


def _pick_turn_amounts() -> list[int]:
    """Pick odd visible quarter-turn amounts for outer/middle/inner."""
    return [random.choice([1, 3]) for _ in range(3)]


def _cue_area_for_turn(turns: int) -> int:
    turns = int(turns) % 4
    if turns == 1:
        return random.choice([1, 5])
    if turns == 3:
        return 3
    raise ValueError("combo generator only uses odd turn amounts")


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
    turns = _pick_turn_amounts()
    solver_ks = [_rot_k_from_layer_and_cue(i, turn) for i, turn in enumerate(turns)]

    n = random.choice(list(range(BOARD_N_MIN, BOARD_N_MAX + 1)))

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n)
    if not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    for col, mask in zip(colors, [outer_mask, mid_mask, inner_mask]):
        out_board[mask] = col
    expected = build_layer_gallery(out_board, bg=0)

    in_board = np.zeros((n, n), dtype=int)
    for col, solver_k in zip(colors, solver_ks):
        inv = np.rot90(out_board == col, (-solver_k) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = col

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(col, _cue_area_for_turn(turn)) for col, turn in zip(colors, turns)]
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
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

# Combo variant of 6ffbe589 (index 23)
# combo_spec: D3/3+D7/2
# Name: 条件性变换+交替方向

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# -----------------------------------------------------------------------------
# Design analysis for combo D3/3 + D7/2
#
# a) Concrete behavior and trace
# - The board is still the unique largest foreground component and still contains
#   3 concentric colors: outer frame, middle inset layer, inner centered sprite.
# - Cues are still detached same-color monochrome sprites; cue area mod 4 still
#   supplies a small transform parameter t in {0,1,2,3}.
# - The fused change is not "all layers rotate". Instead, visible depth selects
#   a transform family and direction together:
#     layer_0 (outer): rotation cycle in CW direction
#     layer_1 (middle): reflection cycle whose 1-step transform is horizontal flip
#     layer_2 (inner): reflection cycle whose 1-step transform is vertical flip
# - To keep cue_size mod 4 meaningful on all layers, each family is implemented
#   as a 4-state cycle:
#     outer : [id, rot90 CW, rot180, rot90 CCW]
#     middle: [id, hflip,    rot180, vflip   ]
#     inner : [id, vflip,    rot180, hflip   ]
# - Small trace with cue area = 3:
#     original task: every layer does rot90 CW once.
#     fused task:
#       outer -> rot90 CCW
#       middle -> vertical flip
#       inner -> horizontal flip
#   The same cue numeral is therefore interpreted through the layer-specific
#   family recovered from geometry.
#
# b) Complexity calibration
# - If reflective layers only used "flip or not", D7/2 would disappear on those
#   layers. The 4-state reflection cycles above make t=1 and t=3 visibly
#   different: the axis swaps between horizontal and vertical.
# - The generator restricts turns to odd values {1,3}; this avoids the direction-
#   blind 180-degree case and keeps every cue visually consequential.
# - Every sample carries all three cues, and the two reflective layers always
#   split into one t=1 case and one t=3 case. That gives balanced branch support
#   while still making the alternating-direction effect observable in each task.
#
# c) Old-assumption audit
# - The original main() assumed color iteration order was enough because every
#   layer shared the same CW-rotation rule. That fails here.
# - The fused solver must first infer outer/middle/inner by bbox span; numeric
#   color ids are not semantically ordered and cannot be used.
# - The old "rot_k = (-cue_size) % 4" line is only valid for the outer layer.
#   Middle/inner now map the same t to different reflection-axis outcomes.
#
# d) Placement legality
# - The generator keeps the board as the unique largest 8-connected component.
# - Cue sprites remain 8-connected, non-overlapping, and at Chebyshev distance
#   >= 4 from the board and from one another, so the crop helper still isolates
#   the board cleanly.
# - Output is constructed first, then each layer is independently inverse-
#   transformed inside the fixed n x n crop. Overlaps after inversion are tested
#   and rejected, so the input board remains a valid 3-color partition.
#
# e) Visual observability
# - Humans can read layer depth directly from geometry: full-board frame, inset
#   middle, tiny center sprite.
# - The generator enforces asymmetry against rotations and both flips for every
#   layer, so a quarter-turn, hflip, and vflip all look different on the grid.
# - Because odd cue sizes are used on every layer, the output differs from both
#   the original all-rotation task and from a weakened "middle always hflip /
#   inner always vflip" interpretation.
#
# f) Visual cleanliness
# - The three layers remain disjoint concentric masks with distinct bbox scales,
#   so the recovered depth order is stable and visually clean.
# - Detached cues stay outside the crop margin; they do not mingle with the
#   board and do not create ambiguous secondary structures near the target area.
#
# g) Change inferability
# - No rule depends on hidden properties such as numeric color magnitude.
# - Color only links a cue to its matching board layer. Transform family and
#   direction come from visible depth. Cue area mod 4 supplies the step count.
# - The repeated outer/middle/inner structure gives enough structural support for
#   the solver to infer the layer-to-family correspondence from a few examples.
#
# 9) New solver assumptions and generator guarantees
# - Assumption: each board color occupies exactly one depth level with a distinct
#   bbox span. Guaranteed by full-board / inset / small-center construction.
# - Assumption: each color has at most one outside cue. Guaranteed by one cue per
#   layer in the generator.
# - Assumption: every layer is asymmetric under the transforms that may apply to
#   it. Guaranteed by rejecting masks that match any non-identity rotation,
#   horizontal flip, or vertical flip.
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


def _apply_turn_cycle(mask: np.ndarray, layer_idx: int, turns: int) -> np.ndarray:
    turns = int(turns) % 4
    if turns == 0:
        return mask.copy()
    if int(layer_idx) == 0:
        return np.rot90(mask, (-turns) % 4)
    if int(layer_idx) == 1:
        cycle = [mask, np.fliplr(mask), np.rot90(mask, 2), np.flipud(mask)]
        return cycle[turns].copy()
    assert int(layer_idx) == 2, f"unexpected layer index {layer_idx}"
    cycle = [mask, np.flipud(mask), np.rot90(mask, 2), np.fliplr(mask)]
    return cycle[turns].copy()


def _apply_inverse_turn_cycle(mask: np.ndarray, layer_idx: int, turns: int) -> np.ndarray:
    turns = int(turns) % 4
    if turns == 0:
        return mask.copy()
    if int(layer_idx) == 0:
        return np.rot90(mask, turns)
    return _apply_turn_cycle(mask, int(layer_idx), turns)


def transform_color_layer(
    grid: np.ndarray,
    color: int,
    layer_idx: int,
    cue_size: int,
    *,
    bg: int | None = None,
) -> np.ndarray:
    """Apply the fused layer-specific transform family for one color mask."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    col = int(color)
    turns = int(cue_size) % 4
    if turns == 0:
        return grid

    mask = grid == col
    tmask = _apply_turn_cycle(mask, int(layer_idx), turns)
    assert tmask.shape == mask.shape, "expected transforms to preserve crop shape"

    out = grid.copy()
    out[out == col] = bg
    out[tmask] = col
    return out


def _color_bbox_area(grid: np.ndarray, color: int) -> int:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"expected color {color} to appear in the crop"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _layer_order(crop: np.ndarray, bg: int) -> list[int]:
    colors = colors_present(crop, exclude={int(bg)})
    assert len(colors) == 3, "expected exactly 3 board colors"

    ranked = sorted((-_color_bbox_area(crop, col), int(col)) for col in colors)
    ordered = [col for _neg_area, col in ranked]
    areas = [_color_bbox_area(crop, col) for col in ordered]
    assert len(set(areas)) == len(areas), "expected distinct layer bbox spans"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and transform each color layer from its outside cue sprite.

    Input structure:
    - bg is the dominant color.
    - The largest foreground component is a square board with 3 concentric color layers.
    - Small monochrome cue sprites appear outside the board, one in the matching color of each layer.

    Rule:
    - Tight-crop around the board cluster.
    - Recover layer depth from bbox span: largest visible color = outer, then middle, then inner.
    - For each layer, read the same-color cue's area mod 4 as a step count in a layer-specific transform cycle.
    - Apply the depth-specific family: outer follows the CW rotation cycle, middle follows the horizontal-flip-first cycle, inner follows the vertical-flip-first cycle.

    Key insight: Cue size no longer means "rotate every color the same way"; it tells how many steps to move along a transform family chosen by layer depth.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster and recover outer -> inner order from geometry ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    ordered_colors = _layer_order(crop, bg)
    out = crop.copy()

    # --- STEP 2) Match same-color outside cues, then apply each layer's transform cycle ---
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
        out = transform_color_layer(out, int(col), int(layer_idx), cue_size, bg=int(bg))

    return out


# --------------- generator (combo D3/3 + D7/2) ---------------

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

GEN_RETRIES = 400


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    """Return the 4 positions in the 90-degree rotation orbit of (r, c) in an n*n grid."""
    pts = set()
    rr, cc = r, c
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _is_fused_asym(mask: np.ndarray) -> bool:
    """True if mask changes under every non-identity transform used by this combo."""
    return (
        not np.array_equal(mask, np.rot90(mask, 1))
        and not np.array_equal(mask, np.rot90(mask, 2))
        and not np.array_equal(mask, np.rot90(mask, 3))
        and not np.array_equal(mask, np.fliplr(mask))
        and not np.array_equal(mask, np.flipud(mask))
    )


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
        if int(mask.sum()) >= target:
            return True
        mask[r, c] = True
    return int(mask.sum()) >= target


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    extra_lo_terms: list[int] = (),
) -> np.ndarray | None:
    """Build an n*n boolean mask with perimeter bias and asymmetry under rotation and flips."""
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
    if not _is_fused_asym(mask):
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
    if not _is_fused_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int) -> np.ndarray | None:
    """Build the outer frame layer spanning the full n*n grid."""
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[min_area])


def _valid_inner_spans(n: int) -> list[int]:
    """Return inner-sprite span sizes that can support visible flips/rotations."""
    max_span = n - INNER_SPAN_MARGIN
    return [s for s in range(2, max_span + 1) if (s % 2) == (n % 2)]


def _make_inner_mask(n: int) -> np.ndarray | None:
    """Build a small central sprite that is asymmetric under both flips and rotations."""
    variants = [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0]], dtype=bool),
        np.array([[0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 1], [0, 1, 0, 0, 0]], dtype=bool),
    ]

    spans = _valid_inner_spans(n)
    random.shuffle(spans)
    for span in spans:
        candidates = [v for v in variants if v.shape == (span, span)]
        random.shuffle(candidates)
        for local in candidates:
            oriented = np.rot90(local, random.randrange(4))
            if random.randrange(2):
                oriented = np.fliplr(oriented)
            mask = np.zeros((n, n), dtype=bool)
            top = (n - span) // 2
            mask[top : top + span, top : top + span] = oriented
            if _is_fused_asym(mask):
                return mask
    return None


def _pick_turns() -> list[int]:
    """Pick odd step counts so direction is visible and reflective branches stay balanced."""
    outer = random.choice([1, 3])
    middle, inner = random.choice([(1, 3), (3, 1)])
    return [outer, middle, inner]


def _cue_area_for_turn(turns: int) -> int:
    turns = int(turns) % 4
    if turns == 1:
        return random.choice([1, 5])
    if turns == 3:
        return 3
    if turns == 2:
        return 2
    raise ValueError("turn amount 0 should be represented by omitting the cue")


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


def _original_like_output(grid: np.ndarray) -> np.ndarray:
    """Reference for anti-degeneracy checks: original all-layers-CW interpretation."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    colors = colors_present(crop, exclude={int(bg)})
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
        cue_size = int(outside_comps[0]["area"]) if outside_comps else 0
        turns = int(cue_size) % 4
        if turns == 0:
            continue
        mask = out == int(col)
        rmask = np.rot90(mask, (-turns) % 4)
        out[out == int(col)] = int(bg)
        out[rmask] = int(col)
    return out


def _weakened_conditional_output(grid: np.ndarray) -> np.ndarray:
    """Reference for anti-degeneracy checks: ignore reflective direction and always use the 1-step axis."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    ordered_colors = _layer_order(crop, bg)
    out = crop.copy()

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
        cue_size = int(outside_comps[0]["area"]) if outside_comps else 0
        turns = int(cue_size) % 4
        if turns == 0:
            continue
        if layer_idx == 0:
            out = transform_color_layer(out, int(col), 0, turns, bg=int(bg))
            continue
        if layer_idx == 1:
            mapped = 1 if turns % 2 else 0
        else:
            mapped = 1 if turns % 2 else 0
        out = transform_color_layer(out, int(col), int(layer_idx), mapped, bg=int(bg))
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    turns = _pick_turns()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n)]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n)
    if inner_mask is None:
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    for col, mask in zip(colors, [outer_mask, mid_mask, inner_mask]):
        out_board[mask] = int(col)

    in_board = np.zeros((n, n), dtype=int)
    for layer_idx, (col, turns_i) in enumerate(zip(colors, turns)):
        inv = _apply_inverse_turn_cycle(out_board == int(col), int(layer_idx), int(turns_i))
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(int(col), _cue_area_for_turn(int(turns_i))) for col, turns_i in zip(colors, turns)]
    random.shuffle(cue_items)
    for col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(int(col), int(area)), board_bbox, cue_bboxes):
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

    solver_out = main(canvas)
    if not np.array_equal(solver_out, out_board):
        return None
    if np.array_equal(solver_out, _original_like_output(canvas)):
        return None
    if np.array_equal(solver_out, _weakened_conditional_output(canvas)):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo D3/3+D7/2 input")

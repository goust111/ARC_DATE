# Variant of 6ffbe589
# Dimension: D7 — 旋转方向 (transform)
# Change: 固定顺时针旋转(CW)，rot_k = (-cue_size) % 4... → 每层旋转方向交替：外层CW、中层CCW、内层CW

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# -----------------------------------------------------------------------------
# Design analysis for the D7 variant
#
# a) Concrete behavior and trace
# - The board is still a square crop with 3 concentric color layers:
#   outer frame, middle inset ring, inner centered sprite.
# - Cue sprites still sit outside the board and still use cue area mod 4 to
#   encode only the turn amount t in quarter-turns.
# - The changed behavior is the direction rule:
#   layer_0 (outermost) uses CW, layer_1 uses CCW, layer_2 uses CW.
# - Small trace:
#   original task: middle cue area=1 -> t=1 -> rotate middle layer CW once.
#   new task:      middle cue area=1 -> t=1 -> rotate middle layer CCW once.
#   The cue stays the same; only the layer-dependent direction changes.
#
# b) Complexity calibration
# - The direction rule would be invisible if the middle layer used only 0 or 2
#   quarter-turns, because CW and CCW agree in those cases.
# - To make D7 visible and repeatedly supported across examples, the generator
#   gives every layer a cue and forces all turn amounts to be odd (1 or 3).
# - This keeps the rule simple: cue area says "how much", layer depth says
#   "which direction".
#
# c) Old-assumption audit
# - Old main() assumed "same cue interpretation for every color":
#   rot_k = (-cue_size) % 4, regardless of which layer the color belonged to.
# - In the variant this is invalid for the middle layer. Therefore the solver
#   must infer the layer order from geometry, not from color id iteration.
# - colors_present(...) sorts by numeric color id; that ordering is unrelated to
#   visible structure and is therefore unusable for D7.
#
# d) Placement legality
# - The board remains the unique largest foreground component.
# - Cue sprites remain 8-connected, non-overlapping, and at Chebyshev distance
#   >= 4 from the board and from each other, so crop_components_near_largest()
#   excludes them reliably.
# - The 3 board layers remain disjoint and strictly nested, so rotating one mask
#   per color does not create ambiguous ownership.
#
# e) Visual observability
# - The only changed semantic is direction, so the generator must expose it.
# - Every sample therefore contains all 3 cues and an odd-turn middle cue, so
#   the new solver output differs from the original all-CW rule on the same
#   input. The human can see the middle layer turns opposite to the outer/inner
#   layers even when cue sizes are interpreted the same way.
#
# f) Visual cleanliness
# - Layer ordering is inferred from bbox span: largest bbox = outer, then middle,
#   then inner. The generator preserves strict concentric nesting so this is
#   visually clean and solver-safe.
# - Cues stay detached from the board, avoiding crop pollution or stacked marks.
#
# g) Color-rule audit
# - No rule uses numeric color ids. Color only matches cue to layer membership.
# - Layer depth is recovered from visible geometry (bbox size), which is stable
#   across examples and readable by a human.
#
# 9) New solver assumptions and generator guarantees
# - Assumption: each board color occupies one visible depth level with a unique
#   bbox span. Guaranteed by the concentric construction.
# - Assumption: the outer layer has the largest bbox, the inner layer the
#   smallest. Guaranteed by full-board / inset / centered-span masks.
# - Assumption: if a cue exists for a color, there is at most one outside cue of
#   that color. Guaranteed by one cue per layer.
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


def _color_bbox_area(grid: np.ndarray, color: int) -> int:
    coords = np.argwhere(grid == int(color))
    assert coords.size > 0, f"expected color {color} to appear in the crop"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _layer_order(crop: np.ndarray, bg: int) -> list[int]:
    colors = colors_present(crop, exclude={int(bg)})
    assert colors, "expected the cropped board to contain foreground colors"

    ranked = sorted(
        ((-_color_bbox_area(crop, col), int(col)) for col in colors),
        key=lambda item: item,
    )
    ordered = [col for _neg_area, col in ranked]

    areas = [_color_bbox_area(crop, col) for col in ordered]
    assert len(set(areas)) == len(areas), "expected layer bbox spans to be distinct"
    return ordered


def _rot_k_from_layer_and_cue(layer_idx: int, cue_size: int) -> int:
    turns = int(cue_size) % 4
    if turns == 0:
        return 0
    if layer_idx % 2 == 0:
        return (-turns) % 4
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and rotate each color layer using outside cue sprites.

    Input structure:
    - bg is the dominant color.
    - The largest foreground component is a square board with 3 concentric color layers.
    - Small monochrome cue sprites appear outside the board, one per board color.

    Rule:
    - Tight-crop around the board cluster.
    - Order the board colors by visible layer depth: largest bbox = outer, then middle, then inner.
    - For each layer, read the matching cue's pixel count mod 4 as the turn amount.
    - Apply alternating directions by depth: outer CW, middle CCW, inner CW. No cue means no turn.

    Key insight: Cue size still tells how far to rotate, but the direction must be inferred from layer depth rather than from color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster and recover the visible outer->inner order ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    ordered_colors = _layer_order(crop, bg)
    out = crop.copy()

    # --- STEP 2) Use same-color outside cues for turn amount, then layer depth for direction ---
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

    return out


# --------------- generator (D7 variant) ---------------

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


def _pick_turn_amounts() -> list[int]:
    """Pick visible quarter-turn amounts for outer/middle/inner; all are odd to expose direction."""
    return [random.choice([1, 3]) for _ in range(3)]


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


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    turns = _pick_turn_amounts()
    solver_ks = [_rot_k_from_layer_and_cue(i, turn) for i, turn in enumerate(turns)]

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, True)]
    n = random.choice(valid_ns)

    # Build concentric layer masks. All layers rotate, so all must be asymmetric.
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

    # Compose output board and invert layer-specific rotations for the input board.
    out_board = np.zeros((n, n), dtype=int)
    for col, mask in zip(colors, [outer_mask, mid_mask, inner_mask]):
        out_board[mask] = col

    in_board = np.zeros((n, n), dtype=int)
    for col, solver_k in zip(colors, solver_ks):
        inv = np.rot90(out_board == col, (-solver_k) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = col

    # Place board and all three cue sprites on canvas.
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

    # Validate: board is unique largest component and oracle check passes.
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
    raise RuntimeError("failed to generate a valid 6ffbe589 D7-variant input")

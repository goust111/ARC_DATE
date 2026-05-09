# Combo variant of 6ffbe589 (index 14)
# combo_spec: D3/2+D14/1
# Name: 平移变换+朝向编码

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the combo variant
# ------------------------------------
# a) Concrete behavior:
#    The original task cropped the largest square board and rotated each same-color layer
#    by an amount derived from the outside cue area. Here that logic is replaced by a
#    two-part code. Cue position gives direction; cue orientation gives distance.
#    Example trace: if a red cue sits above the board and is a horizontal 1x4 strip,
#    the red layer shifts up by 1 cell with wraparound. If a blue cue is on the right
#    and is a vertical 4x1 strip, the blue layer shifts right by 2. A diagonal 4-cell
#    strip means distance 3. No cue still means no movement.
# b) Complexity calibration:
#    Distances are restricted to {1,2,3} and directions to the four cardinals, so the
#    rule is visible but not overly free-form. Every non-zero cue uses the same area 4;
#    otherwise the generator could accidentally collapse back to the original size-based
#    rule. At least two cues are active and they do not all share the same direction or
#    distance, so both halves of the code appear repeatedly in each sample.
# c) Broken old assumptions:
#    - `cue_size mod 4` is now irrelevant and must not be read.
#    - The transform is no longer `np.rot90`; it is cyclic translation by a signed
#      vector. Therefore the solver cannot safely update the grid in place layer-by-layer,
#      because one translated layer may temporarily move onto cells still occupied by an
#      unshifted layer. The solver must transform masks independently and re-compose.
#    - Cue decoding now depends on both geometry and relative placement. The generator
#      must keep cues in exclusive top/bottom/left/right side bands; corner placements
#      would make the direction ambiguous.
# d) Placement legality:
#    Cue patches are placed fully inside the canvas, at Chebyshev distance >= 4 from the
#    board and from each other, and on a single cardinal side while overlapping the board
#    span on the orthogonal axis. This preserves an unambiguous "above/below/left/right"
#    reading and prevents cue-board merging.
# e) Visual observability:
#    Input boards are built by inverse-translating a clean concentric output board. The
#    output therefore "snaps back" into alignment, making the sliding rule easier to see
#    than if both input and output were messy. Because cue area is constant, a human must
#    rely on long-axis orientation rather than pixel count.
# f) Visual cleanliness:
#    The board still occupies one square crop, cues stay outside that crop, and layers
#    remain color-separated. The generator rejects any sample where inverse translations
#    create layer overlaps or where active shifts are visually invisible.
# g) New solver assumptions and generator guarantees:
#    - Each color has at most one same-color cue outside the board.
#    - Every active cue is exactly one of: horizontal strip -> 1, vertical strip -> 2,
#      diagonal strip -> 3.
#    - Cue side is exclusive and therefore decodes a unique direction.
#    - The translated output masks are pairwise disjoint after applying the decoded
#      shifts, so re-composition is deterministic.
# h) Review of `main()` assumptions:
#    - Cropping still assumes the outer layer yields a unique largest foreground component;
#      the generator keeps the outer band connected and spanning the full board bbox.
#    - Cue lookup assumes the board crop can be blanked out before finding outside cues;
#      cue-board separation >= 4 preserves this.
#    - The final overlap assertion assumes independent translations do not collide in the
#      intended output; generation checks this before accepting a sample.


SIDE_TO_SHIFT = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


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


def shift_mask(mask: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Translate a boolean mask cyclically inside its fixed board window."""
    dr, dc = map(int, shift)
    if dr == 0 and dc == 0:
        return mask.astype(bool)
    return np.roll(mask.astype(bool), shift=(dr, dc), axis=(0, 1))


def cue_distance_from_component(comp: dict) -> int:
    """Decode strip orientation into a cyclic translation distance in {1, 2, 3}."""
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


def cue_direction_from_component(comp: dict, board_bbox: tuple[int, int, int, int]) -> str:
    """Decode the cue's cardinal side relative to the board."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    br0, bc0, br1, bc1 = map(int, board_bbox)

    if r1 < br0:
        return "up"
    if r0 > br1:
        return "down"
    if c1 < bc0:
        return "left"
    if c0 > bc1:
        return "right"

    raise AssertionError("expected cue to lie fully outside the board on one cardinal side")


def cue_shift_from_component(comp: dict, board_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Combine cue side and cue orientation into a signed translation vector."""
    dist = int(cue_distance_from_component(comp))
    side = cue_direction_from_component(comp, board_bbox)
    ur, uc = SIDE_TO_SHIFT[side]
    return int(ur * dist), int(uc * dist)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and slide its color layers based on outside cue strips.

    Input structure:
    - bg = most common color.
    - One largest multi-color square board cluster.
    - Small same-color cue strips outside the board; each strip sits on one cardinal side.

    Rule:
    - Tight-crop around the board cluster.
    - For each board color, find its unique outside cue if it exists.
    - Cue side encodes translation direction (above/below/left/right), and cue orientation
      encodes distance (horizontal=1, vertical=2, diagonal=3).
    - Cyclically shift that color layer inside the crop by the decoded vector.
    - No cue -> that layer stays in place.

    Key insight: The cue uses two independent visible channels at once: location chooses the
    shift direction, while long-axis orientation chooses how far the layer wraps.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster and discard distant cue strips ---
    crop, board_bbox, bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Decode each cue into a shift vector and re-compose translated layers ---
    colors = colors_present(crop, exclude={int(bg)})
    assert colors, "expected the cropped board to contain foreground colors"

    outside_grid = g.copy()
    r0, c0, r1, c1 = board_bbox
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    shifted_masks: dict[int, np.ndarray] = {}
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
        shift = cue_shift_from_component(outside_comps[0], board_bbox) if outside_comps else (0, 0)
        shifted_masks[int(col)] = shift_mask(crop == int(col), shift)

    out = np.full(crop.shape, int(bg), dtype=int)
    occupied = np.zeros(crop.shape, dtype=bool)
    for col in colors:
        mask = shifted_masks[int(col)]
        assert not np.any(occupied & mask), "decoded translated layers must remain disjoint"
        out[mask] = int(col)
        occupied |= mask

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
CUE_CLEARANCE = CUE_BOARD_DIST + CUE_LENGTH - 1
OUTER_AREA_PAD = 4

GEN_RETRIES = 600
DISTANCE_PAIRS = ((1, 2), (1, 3), (2, 3))
_GEN_CALL_COUNT = 0


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


def _is_shift_visible(mask: np.ndarray, shift: tuple[int, int]) -> bool:
    """True if a cyclic shift visibly changes the mask."""
    dr, dc = map(int, shift)
    if dr == 0 and dc == 0:
        return True
    return not np.array_equal(mask.astype(bool), shift_mask(mask.astype(bool), shift))


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
    """Return valid inner-sprite span sizes for a board of side n."""
    max_span = n - INNER_SPAN_MARGIN
    return [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]


def _make_inner_mask(n: int) -> np.ndarray:
    """Build the small central sprite layer."""
    mask = np.zeros((n, n), dtype=bool)
    span = random.choice(_valid_inner_spans(n))
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


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev (L-inf) gap distance between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _available_sides(board_bbox: tuple[int, int, int, int]) -> list[str]:
    """Return the board sides that have enough room for any cue patch."""
    r0, c0, r1, c1 = map(int, board_bbox)
    sides = []
    if r0 >= CUE_CLEARANCE:
        sides.append("up")
    if CANVAS_SIZE - 1 - r1 >= CUE_CLEARANCE:
        sides.append("down")
    if c0 >= CUE_CLEARANCE:
        sides.append("left")
    if CANVAS_SIZE - 1 - c1 >= CUE_CLEARANCE:
        sides.append("right")
    return sides


def _board_anchor(n: int) -> tuple[int, int]:
    """Pick a board top-left position with room on at least two cue sides."""
    candidates = []
    for r in range(CANVAS_SIZE - n + 1):
        for c in range(CANVAS_SIZE - n + 1):
            bbox = (r, c, r + n - 1, c + n - 1)
            if len(_available_sides(bbox)) >= 2:
                candidates.append((r, c))
    assert candidates, "expected at least one board anchor with two cue sides"
    return random.choice(candidates)


def _pick_shift_specs(
    available_sides: list[str],
    *,
    target_distances: tuple[int, int] | None = None,
) -> list[tuple[str, int] | None]:
    """Assign exactly two active cues with distinct directions and distances."""
    active_count = 2
    active_idx = random.sample(range(NUM_COLORS), active_count)
    sides = random.sample(available_sides, 2)
    if target_distances is None:
        distances = random.sample([1, 2, 3], 2)
    else:
        distances = list(target_distances)
        random.shuffle(distances)

    specs: list[tuple[str, int] | None] = [None, None, None]
    for idx, side, dist in zip(active_idx, sides, distances):
        specs[int(idx)] = (str(side), int(dist))
    return specs


def _cue_patch(color: int, distance: int) -> np.ndarray:
    """Build a fixed-area strip cue whose orientation encodes shift distance."""
    col = int(color)
    if distance == 1:
        return np.full((1, CUE_LENGTH), col, dtype=int)
    if distance == 2:
        return np.full((CUE_LENGTH, 1), col, dtype=int)
    if distance == 3:
        arr = np.zeros((CUE_LENGTH, CUE_LENGTH), dtype=int)
        if random.randrange(2):
            for i in range(CUE_LENGTH):
                arr[i, i] = col
        else:
            for i in range(CUE_LENGTH):
                arr[i, CUE_LENGTH - 1 - i] = col
        return arr
    raise AssertionError("expected distance in {1,2,3}")


def _place_cue_on_side(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
    side: str,
) -> bool:
    """Place a cue patch on one exclusive side of the board."""
    h, w = patch.shape
    mask = patch != 0
    br0, bc0, br1, bc1 = map(int, board_bbox)
    candidates = []

    if side == "up":
        r_lo, r_hi = 0, br0 - CUE_BOARD_DIST - h + 1
        c_lo, c_hi = max(0, bc0 - w + 1), min(CANVAS_SIZE - w, bc1)
    elif side == "down":
        r_lo, r_hi = br1 + CUE_BOARD_DIST, CANVAS_SIZE - h
        c_lo, c_hi = max(0, bc0 - w + 1), min(CANVAS_SIZE - w, bc1)
    elif side == "left":
        r_lo, r_hi = max(0, br0 - h + 1), min(CANVAS_SIZE - h, br1)
        c_lo, c_hi = 0, bc0 - CUE_BOARD_DIST - w + 1
    elif side == "right":
        r_lo, r_hi = max(0, br0 - h + 1), min(CANVAS_SIZE - h, br1)
        c_lo, c_hi = bc1 + CUE_BOARD_DIST, CANVAS_SIZE - w
    else:
        raise AssertionError("unknown side")

    if r_lo > r_hi or c_lo > c_hi:
        return False

    for r in range(int(r_lo), int(r_hi) + 1):
        for c in range(int(c_lo), int(c_hi) + 1):
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
    paste(canvas, patch, int(r), int(c), transparent=0)
    cue_bboxes.append(tuple(map(int, bb)))
    return True


def _gen_attempt(*, target_distances: tuple[int, int] | None = None) -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    c_frame, c_mid, c_inner = colors
    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)

    r0, c0 = _board_anchor(n)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)
    available_sides = _available_sides(board_bbox)
    if len(available_sides) < 2:
        return None
    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n)
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    layer_masks = [outer_mask, mid_mask, inner_mask]
    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = c_frame
    out_board[mid_mask] = c_mid
    out_board[inner_mask] = c_inner

    for _ in range(24):
        cue_specs = _pick_shift_specs(available_sides, target_distances=target_distances)

        shifts: list[tuple[int, int]] = []
        visible = True
        for mask, spec in zip(layer_masks, cue_specs):
            if spec is None:
                shifts.append((0, 0))
                continue
            side, dist = spec
            ur, uc = SIDE_TO_SHIFT[str(side)]
            shift = (int(ur * dist), int(uc * dist))
            if not _is_shift_visible(mask, shift):
                visible = False
                break
            shifts.append(shift)
        if not visible:
            continue

        in_board = np.zeros((n, n), dtype=int)
        valid = True
        for col, mask, shift in zip(colors, layer_masks, shifts):
            inv_mask = shift_mask(mask, (-shift[0], -shift[1]))
            if np.any((in_board != 0) & inv_mask):
                valid = False
                break
            in_board[inv_mask] = int(col)
        if not valid or np.array_equal(in_board, out_board):
            continue

        canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
        paste(canvas, in_board, r0, c0, transparent=0)

        cue_bboxes: list[tuple[int, int, int, int]] = []
        cue_items = [(col, spec) for col, spec in zip(colors, cue_specs) if spec is not None]
        random.shuffle(cue_items)
        placed = True
        for col, spec in cue_items:
            side, dist = spec
            patch = _cue_patch(int(col), int(dist))
            if not _place_cue_on_side(canvas, patch, board_bbox, cue_bboxes, str(side)):
                placed = False
                break
        if not placed:
            continue

        comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
        areas = [int(comp["area"]) for comp in comps]
        if not areas or areas.count(max(areas)) != 1:
            continue
        largest = comps[areas.index(max(areas))]
        if tuple(map(int, largest["bbox"])) != board_bbox:
            continue
        if any(_bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST for bb in cue_bboxes):
            continue
        if not np.array_equal(main(canvas), out_board):
            continue
        return canvas

    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    target_distances = DISTANCE_PAIRS[_GEN_CALL_COUNT % len(DISTANCE_PAIRS)]
    _GEN_CALL_COUNT += 1
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(target_distances=target_distances)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

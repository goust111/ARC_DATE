# Combo variant of 6ffbe589 (index 11)
# combo_spec: D2/2+D7/1+D14/2
# Name: 距离编码+位置方向+方位融合

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import colors_present, find_components, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant D2/2 + D7/1 + D14/2
#
# a) Concrete behavior trace
#   original:
#     - crop the unique largest square board
#     - match outside cues to board layers by same color
#     - use cue area mod 4 to get a fixed clockwise rotation amount
#   combo:
#     - keep the same crop and same color matching
#     - ignore cue area entirely by making cues single pixels
#     - read the cue's Manhattan distance to the board center and reduce mod 4;
#       this is the distance-coded additive quarter-turn amount
#     - read the cue's board-relative side twice:
#         * as D14/2's side code: top/right/bottom/left -> 0/1/2/3 turns
#         * as D7/1's direction bit: top/right mean clockwise, bottom/left mean
#           counterclockwise
#     - add the side code and the distance code mod 4 to get the unsigned turn
#       count, then apply the side-selected direction
#   small example:
#     - suppose a blue cue is 9 Manhattan steps from the board center and sits
#       below the board; distance gives 1, "below" contributes side-code 2 and
#       CCW direction, so (1 + 2) mod 4 = 3 and the layer rotates by three CCW
#       quarter-turns (`np.rot90(..., 3)`)
#     - suppose a red cue is 10 steps away and sits to the right; distance gives
#       2, "right" contributes side-code 1 and CW direction, so (2 + 1) mod 4
#       = 3 and the layer rotates by three CW quarter-turns
#   difference:
#     - D2/2 moves one part of the amount signal from blob size to spatial radius
#     - D14/2 restores board-relative side as an explicit symbolic amount code
#     - D7/1 makes the same side category also decide the rotation direction
#       The fused rule is therefore:
#       final magnitude = distance code + side code (mod 4), sign = side bucket.
#
# b) Complexity calibration
#   - The task keeps the original 3-layer square board and one same-color cue per
#     rotating layer, so there is no new matching ambiguity.
#   - Every sample contains one same-side/different-distance cue pair and one
#     same-distance/different-side cue pair. That gives local evidence for both
#     additive factors without needing hidden bookkeeping.
#   - Cue area is fixed to one cell, so the old area rule is visibly unsupported.
#
# c) Hidden-assumption audit for `main()`
#   - `crop_components_near_largest(...)` still assumes all cues stay outside the
#     board's margin-3 expansion; the generator preserves Chebyshev gap >= 4.
#   - Per-color cue lookup still assumes at most one outside cue per board color;
#     the generator creates exactly one cue for each of the 3 board colors.
#   - Distance decoding adds a new assumption that the board center and cue
#     centroid produce an integral Manhattan distance. The generator restricts the
#     board size to odd N and uses 1x1 cues, so this is guaranteed.
#   - Side decoding adds a new assumption that each cue is aligned to exactly one
#     side corridor, not diagonally ambiguous. The generator exposes all four
#     side corridors by keeping the board near the canvas center.
#   - Sequential layer rotation remains safe because colors are disjoint and
#     `rotate_color_layer(...)` clears only the active color before repainting it.
#
# d) Placement legality
#   - The board remains the unique largest connected component.
#   - Boards use the odd 11x11 size and are anchored in one of four centered
#     positions, leaving all four side corridors legal inside the 20x20 canvas.
#   - Cue-cue Chebyshev distance and cue-board distance are checked explicitly, so
#     no cue overlaps the board or another cue.
#
# e) Visual observability
#   - Size cannot be the rule because every cue is a single cell.
#   - Side cannot be the whole rule because two cues can share a side but differ
#     in distance and therefore rotate by different amounts.
#   - Distance cannot be the whole rule because two cues can share a distance
#     residue but live on different sides, so the side code changes the result.
#
# f) Visual cleanliness
#   - The crop is still one clean square board with 3 non-overlapping layers.
#   - Cues live in sparse side corridors rather than in diagonal clutter, so both
#     the radial and azimuthal signals are easy to read by eye.
#
# g) Quantitative anti-degeneration target
#   - Every sample contains:
#       1) a same-side/different-distance cue pair, proving D2/2 matters
#       2) a same-distance/different-side cue pair, proving D14/2 matters
#       3) at least one CW-side cue and one CCW-side cue, proving D7/1 matters
#   - The generator checks these conditions from the rendered input itself before
#     accepting a sample.
#
# h) New solver assumptions after fusion
#   - Board side length must be odd for clean distance decoding.
#   - Each cue must occupy a legal top/right/bottom/left corridor.
#   - Every board color must have exactly one outside cue in generated samples.
#   `generate_input()` constructs and verifies all three conditions.
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


def _bbox_overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _cue_side(cue_bbox: tuple[int, int, int, int], board_bbox: tuple[int, int, int, int]) -> str:
    qr0, qc0, qr1, qc1 = map(int, cue_bbox)
    br0, bc0, br1, bc1 = map(int, board_bbox)

    if qr1 < br0 and _bbox_overlap_1d(qc0, qc1, bc0, bc1):
        return "top"
    if qr0 > br1 and _bbox_overlap_1d(qc0, qc1, bc0, bc1):
        return "bottom"
    if qc1 < bc0 and _bbox_overlap_1d(qr0, qr1, br0, br1):
        return "left"
    if qc0 > bc1 and _bbox_overlap_1d(qr0, qr1, br0, br1):
        return "right"
    raise AssertionError("expected each cue to align with exactly one board side")


def _side_is_clockwise(side: str) -> bool:
    assert side in {"top", "right", "bottom", "left"}
    return side in {"top", "right"}


SIDE_TO_BASE_TURNS = {
    "top": 0,
    "right": 1,
    "bottom": 2,
    "left": 3,
}


def _fused_turns(distance_turns: int, side: str) -> int:
    return (int(distance_turns) + int(SIDE_TO_BASE_TURNS[str(side)])) % 4


def _signed_rot_k(turns: int, side: str) -> int:
    t = int(turns) % 4
    if t == 0:
        return 0
    return (-t) % 4 if _side_is_clockwise(side) else t


def _cue_rotation_from_geometry(cue_comp: dict, board_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int, str]:
    board_center = _board_center_from_bbox(board_bbox)
    cue_center = _coords_centroid(cue_comp["coords"])
    dist = _integral_manhattan_distance(cue_center, board_center)
    side = _cue_side(tuple(map(int, cue_comp["bbox"])), board_bbox)
    distance_turns = dist % 4
    base_turns = int(SIDE_TO_BASE_TURNS[str(side)])
    fused_turns = _fused_turns(distance_turns, side)
    return distance_turns, base_turns, fused_turns, _signed_rot_k(fused_turns, side), side


def _extract_cue_features(grid: np.ndarray) -> tuple[list[dict], np.ndarray, tuple[int, int, int, int], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    board_bbox = (int(r0), int(c0), int(r1), int(c1))

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    features = []
    for col in colors_present(crop, exclude={int(bg)}):
        comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        assert len(comps) <= 1, f"expected at most one outside cue for color {col}"
        if not comps:
            continue
        distance_turns, base_turns, fused_turns, rot_k, side = _cue_rotation_from_geometry(comps[0], board_bbox)
        features.append(
            {
                "color": int(col),
                "distance_turns": int(distance_turns),
                "base_turns": int(base_turns),
                "fused_turns": int(fused_turns),
                "rot_k": int(rot_k),
                "side": str(side),
            }
        )
    return features, crop, board_bbox, int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board and decode each layer rotation from the polar geometry of its cue.

    Input structure:
    - The background is the most common color.
    - The largest foreground component is a square multi-color board with 3 concentric layers.
    - Small same-color cue sprites sit outside the board, each occupying one clear board side corridor.

    Rule:
    - Crop around the board component and ignore distant cue pixels.
    - For each board color, locate its same-color cue outside the crop.
    - Convert the cue side into D14/2's symbolic base amount: top/right/bottom/left -> 0/1/2/3 quarter turns.
    - Add the cue centroid's Manhattan-distance residue to that base amount, modulo 4.
    - Use the cue's side again to choose the direction: top/right = clockwise, bottom/left = counterclockwise.
    - Apply that signed fused rotation to the matching color layer. If a color has no cue, leave that layer unchanged.

    Key insight:
    - Each cue uses board-relative side in two ways and radius in one way:
      side gives a symbolic base amount, distance adds an offset, and side also chooses CW vs CCW.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster (drop distant cue sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    board_bbox = (int(r0), int(c0), int(r1), int(c1))

    # --- STEP 2) Rotate each color layer using cue side for base+direction and cue distance for additive amount ---
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
        _distance_turns, _base_turns, _fused_turns, rot_k, _side = _cue_rotation_from_geometry(outside_comps[0], board_bbox)
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator ---------------

CANVAS_SIZE = 20
BOARD_N_CHOICES = (11,)
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

GEN_RETRIES = 2000


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    pts = set()
    rr, cc = r, c
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _is_rot_asym(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + phase) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < width
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
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
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[min_area])


def _valid_inner_spans(n: int) -> list[int]:
    max_span = n - INNER_SPAN_MARGIN
    return [s for s in range(1, max_span + 1) if (s % 2) == (n % 2) and s > 1]


def _make_inner_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    span = random.choice(_valid_inner_spans(n))
    top = (n - span) // 2

    if span == 3:
        variants = [
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ]
        local = random.choice(variants)
        local = np.rot90(local, random.randrange(4))
    else:
        local = np.zeros((span, span), dtype=bool)
        pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
        for r, c in pts:
            local[r, c] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))
    mask[top : top + local.shape[0], top : top + local.shape[1]] = local
    return mask

def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _center_anchor_choices(n: int) -> list[tuple[int, int]]:
    lo = CUE_BOARD_DIST
    hi = CANVAS_SIZE - n - CUE_BOARD_DIST
    return [(int(r0), int(c0)) for r0 in range(int(lo), int(hi) + 1) for c0 in range(int(lo), int(hi) + 1)]


def _sample_cue_program() -> list[tuple[str, int]]:
    all_sides = ["top", "right", "bottom", "left"]
    while True:
        repeat_side = random.choice(all_sides)
        other_side_pool = [side for side in all_sides if _side_is_clockwise(side) != _side_is_clockwise(repeat_side)]
        other_side = random.choice(other_side_pool)

        distance_a = random.randrange(4)
        distance_b = random.choice([d for d in range(4) if d != distance_a])
        program = [
            (str(repeat_side), int(distance_a)),
            (str(repeat_side), int(distance_b)),
            (str(other_side), int(distance_a)),
        ]
        fused_turns = [_fused_turns(distance_turns, side) for side, distance_turns in program]
        if len(set(fused_turns)) < 2:
            continue
        if not any(fused != distance_turns for (side, distance_turns), fused in zip(program, fused_turns)):
            continue
        if not any(fused != SIDE_TO_BASE_TURNS[side] for side, fused in zip([side for side, _ in program], fused_turns)):
            continue
        if sum(int(fused != 0) for fused in fused_turns) < 2:
            continue
        if not any(_side_is_clockwise(side) for side, _ in program):
            continue
        if not any(not _side_is_clockwise(side) for side, _ in program):
            continue
        if not any(_side_is_clockwise(side) and fused != 0 for (side, _), fused in zip(program, fused_turns)):
            continue
        if not any((not _side_is_clockwise(side)) and fused != 0 for (side, _), fused in zip(program, fused_turns)):
            continue
        return program


def _place_single_cell_cue(
    canvas: np.ndarray,
    color: int,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
    side: str,
    distance_turns: int,
) -> bool:
    br0, bc0, br1, bc1 = map(int, board_bbox)
    board_center = _board_center_from_bbox(board_bbox)
    candidates = []

    if side == "top":
        r_vals = range(0, br0 - CUE_BOARD_DIST + 1)
        c_vals = range(bc0, bc1 + 1)
    elif side == "bottom":
        r_vals = range(br1 + CUE_BOARD_DIST, CANVAS_SIZE)
        c_vals = range(bc0, bc1 + 1)
    elif side == "left":
        c_vals = range(0, bc0 - CUE_BOARD_DIST + 1)
        r_vals = range(br0, br1 + 1)
    elif side == "right":
        c_vals = range(bc1 + CUE_BOARD_DIST, CANVAS_SIZE)
        r_vals = range(br0, br1 + 1)
    else:
        raise ValueError(f"unknown side {side}")

    for r in r_vals:
        for c in c_vals:
            if canvas[int(r), int(c)] != 0:
                continue
            bb = (int(r), int(c), int(r), int(c))
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            if _cue_side(bb, board_bbox) != side:
                continue
            dist = _integral_manhattan_distance((float(r), float(c)), board_center)
            if dist % 4 != int(distance_turns) % 4:
                continue
            candidates.append((int(r), int(c), bb))

    if not candidates:
        return False

    r, c, bb = random.choice(candidates)
    canvas[int(r), int(c)] = int(color)
    cue_bboxes.append(bb)
    return True


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(range(1, 10), NUM_COLORS)
    n = random.choice(BOARD_N_CHOICES)
    cue_program = _sample_cue_program()

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
        out_board[mask] = int(col)

    anchors = _center_anchor_choices(n)
    if not anchors:
        return None
    r0, c0 = random.choice(anchors)
    board_bbox = (int(r0), int(c0), int(r0 + n - 1), int(c0 + n - 1))
    cue_specs = list(zip(colors, cue_program))
    random.shuffle(cue_specs)

    in_board = np.zeros((n, n), dtype=int)
    for col, (side, distance_turns) in cue_specs:
        fused_turns = _fused_turns(int(distance_turns), str(side))
        rot_k = _signed_rot_k(int(fused_turns), str(side))
        inv = np.rot90(out_board == int(col), (-rot_k) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, int(r0), int(c0), transparent=0)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    for col, (side, distance_turns) in cue_specs:
        if not _place_single_cell_cue(
            canvas,
            int(col),
            board_bbox,
            cue_bboxes,
            str(side),
            int(distance_turns),
        ):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None

    features, _crop, _bbox, _bg = _extract_cue_features(canvas)
    if len(features) != NUM_COLORS:
        return None
    if not any(_side_is_clockwise(str(f["side"])) for f in features):
        return None
    if not any(not _side_is_clockwise(str(f["side"])) for f in features):
        return None
    if not any(int(f["fused_turns"]) != int(f["distance_turns"]) for f in features):
        return None
    if not any(int(f["fused_turns"]) != int(f["base_turns"]) for f in features):
        return None
    same_side_diff_distance = any(
        str(a["side"]) == str(b["side"]) and int(a["distance_turns"]) != int(b["distance_turns"])
        for idx, a in enumerate(features)
        for b in features[idx + 1 :]
    )
    if not same_side_diff_distance:
        return None
    same_distance_diff_side = any(
        int(a["distance_turns"]) == int(b["distance_turns"]) and str(a["side"]) != str(b["side"])
        for idx, a in enumerate(features)
        for b in features[idx + 1 :]
    )
    if not same_distance_diff_side:
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

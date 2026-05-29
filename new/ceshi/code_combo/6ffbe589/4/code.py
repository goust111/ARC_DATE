# Combo variant of 6ffbe589 (index 4)
# combo_spec: D1/3+D14/2
# Name: 空间对齐匹配+位置编码旋转

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the combo variant (D1/3 + D14/2)
#
# a) Concrete behavior trace
#   original:
#     - crop the largest board component
#     - match each outside cue to a board layer by color equality
#     - use cue area mod 4 to decide the rotation
#   combo:
#     - crop the same board
#     - ignore cue color and cue size entirely
#     - match each cue to the board layer whose centroid is best aligned in row
#       or column
#     - read the rotation from which side of the board the cue occupies:
#       above -> 0 turns, right -> 90 CW, below -> 180, left -> 270 CW
#   small example:
#     - suppose the three input-layer centroids are near
#       outer=(3.8, 7.1), mid=(6.2, 4.0), inner=(7.6, 8.4)
#     - a cue on the right side at row 6 best matches the middle layer by row
#       alignment and rotates only that layer by 90 CW
#     - another cue above the board near column 8 matches the inner layer by
#       column alignment but encodes 0 rotation, so the inner layer stays put
#   difference:
#     - the original solver would fail because cue colors are unrelated
#     - the original area code is gone; only geometry matters now
#
# b) Complexity calibration
#   - The board still contains exactly 3 independently controlled layers, so the
#     task remains structured rather than arbitrary.
#   - Each sample uses 3 cues, which gives repeated evidence that color/size are
#     irrelevant and that alignment + side jointly determine the action.
#   - The generator restricts cues to two exposed side bands of the board. This
#     keeps the 4-way direction code visible while respecting the 20x20 canvas.
#
# c) Broken old assumptions
#   - `colors=[col]` for cue lookup is invalid because cue colors are decoys.
#   - `cue area mod 4` is invalid because area no longer carries information.
#   - "no cue means no rotation" is no longer the only way to keep a layer
#     unchanged; a cue above the board now explicitly means rotate by 0.
#   - Because matching is geometric, the generator must enforce separated layer
#     centroids so that row/column alignment is not ambiguous.
#
# d) Placement legality
#   - The board remains the unique largest component.
#   - Cues are single-cell monochrome sprites placed in legal strips at
#     Chebyshev distance >= 4 from the board.
#   - Cue-cue distance is also enforced, and cues never overlap the board or one
#     another.
#
# e) Visual observability
#   - Cue colors are sampled outside the board palette, making color matching
#     visibly wrong.
#   - Cue size is fixed to one cell, so size cannot plausibly be the signal.
#   - The visible regularities are: which cue lines up with which centroid, and
#     which side band that cue belongs to.
#
# f) Visual cleanliness
#   - The board stays as one clean square crop with 3 non-overlapping layers.
#   - Cues live in sparse edge strips instead of diagonal clutter, so side and
#     alignment cues are easy to read.
#
# g) Inferability of the change
#   - No hidden numeric property like color id or blob area participates.
#   - The same cue side can appear for multiple layers in one sample, and the
#     same layer family can be controlled from different sides across samples.
#   - Across a few examples, the consistent explanation is spatial: alignment
#     chooses the target layer, and side chooses the turn amount.
#
# h) Quantitative anti-degeneration target
#   - Every cue color is absent from the board palette.
#   - At least one cue per sample is on a non-top side, so the output differs
#     from the input crop.
#   - Therefore the original color+area solver leaves the crop unchanged, while
#     this combo solver rotates at least one asymmetric layer.
#
# i) New solver assumptions to audit
#   - Each cue must have a unique best layer under alignment scoring.
#   - Every cue must sit clearly above/right/below/left of the board, not in an
#     ambiguous diagonal corner zone.
#   - The cropped board must remain square so rotating a layer with `np.rot90`
#     preserves shape.
#   The generator explicitly enforces these conditions.
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


def _coords_center(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _alignment_cost(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    dr = abs(float(a[0]) - float(b[0]))
    dc = abs(float(a[1]) - float(b[1]))
    return min(dr, dc), dr + dc


def _cue_direction(
    cue_bbox: tuple[int, int, int, int],
    board_bbox: tuple[int, int, int, int],
    cue_center: tuple[float, float],
) -> str:
    br0, bc0, br1, bc1 = map(int, board_bbox)
    r0, c0, r1, c1 = map(int, cue_bbox)

    col_overlap = not (c1 < bc0 or c0 > bc1)
    row_overlap = not (r1 < br0 or r0 > br1)
    if r1 < br0 and col_overlap:
        return "top"
    if c0 > bc1 and row_overlap:
        return "right"
    if r0 > br1 and col_overlap:
        return "bottom"
    if c1 < bc0 and row_overlap:
        return "left"

    board_center = ((br0 + br1) / 2.0, (bc0 + bc1) / 2.0)
    dr = float(cue_center[0]) - float(board_center[0])
    dc = float(cue_center[1]) - float(board_center[1])
    if abs(dr) >= abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _assign_cues_by_alignment(
    cue_infos: list[dict],
    layer_centers: dict[int, tuple[float, float]],
) -> dict[int, dict]:
    colors = list(layer_centers.keys())
    if not cue_infos:
        return {}
    assert len(cue_infos) <= len(colors), "expected no more cues than layers"

    best_perm: tuple[int, ...] | None = None
    best_key: tuple[float, float, float] | None = None
    for perm in itertools.permutations(range(len(colors)), len(cue_infos)):
        primary = 0.0
        secondary = 0.0
        worst_primary = 0.0
        for cue_idx, color_idx in enumerate(perm):
            p, s = _alignment_cost(cue_infos[cue_idx]["center"], layer_centers[colors[color_idx]])
            primary += p
            secondary += s
            worst_primary = max(worst_primary, p)
        key = (round(primary, 6), round(worst_primary, 6), round(secondary, 6))
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm

    assert best_perm is not None
    return {int(colors[color_idx]): cue_infos[cue_idx] for cue_idx, color_idx in enumerate(best_perm)}


DIRECTION_TO_ROT = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the main board and rotate its color layers from the geometry of outside cues.

    Input structure:
    - Background is the most frequent color.
    - One large square board cluster contains 3 non-overlapping color layers.
    - Small monochrome cue sprites lie outside the board; their colors and sizes are distractors.

    Rule:
    - Crop around the largest board component and ignore distant cue pixels.
    - Compute each board layer's centroid and match every outside cue to the layer with the closest row-or-column alignment.
    - Read the rotation from the cue's side relative to the board: above=0, right=90 CW, below=180, left=270 CW.
    - Apply that quarter-turn amount to the matched layer only.

    Key insight: One spatial relation chooses the destination layer (alignment), and another spatial relation chooses the turn amount (which side of the board the cue occupies).
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster (drop distant cue sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Match outside cues by alignment, then rotate by cue side ---
    colors = colors_present(crop, exclude={int(bg)})
    assert colors, "expected the cropped board to contain foreground colors"
    out = crop.copy()

    layer_centers = {
        int(col): _coords_center(np.argwhere(crop == int(col)))
        for col in colors
    }

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    outside_comps = find_components(
        outside_grid,
        bg=int(bg),
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )

    cue_infos = []
    for comp in outside_comps:
        center = _coords_center(comp["coords"])
        cue_infos.append(
            {
                "center": center,
                "direction": _cue_direction(tuple(map(int, comp["bbox"])), (int(r0), int(c0), int(r1), int(c1)), center),
            }
        )

    assigned = _assign_cues_by_alignment(cue_infos, layer_centers)
    for col in colors:
        cue = assigned.get(int(col))
        if cue is None:
            continue
        rot_k = int(DIRECTION_TO_ROT[str(cue["direction"])])
        if rot_k:
            out = rotate_color_layer(out, int(col), rot_k, bg=int(bg))

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
INNER_SPAN_MARGIN = 8
INNER_SPAN_MIN = 3
OUTER_AREA_PAD = 4

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 2
ALIGN_MARGIN = 0.5
MAX_CUE_CANDIDATES = 12

GEN_RETRIES = 2500
BIAS_NAMES = ("tl", "tr", "bl", "br")

PROFILE_TO_SIDES = {
    "bl": ("top", "right"),
    "br": ("top", "left"),
    "tl": ("bottom", "right"),
    "tr": ("bottom", "left"),
}


def _bias_point(n: int, bias: str) -> tuple[int, int]:
    edge = max(1, min(int(n) - 2, 2))
    if bias == "tl":
        return edge - 1, edge
    if bias == "tr":
        return edge - 1, int(n) - 1 - edge
    if bias == "bl":
        return int(n) - edge, edge - 1
    assert bias == "br"
    return int(n) - edge, int(n) - 1 - edge


def _flip_to_bias(mask: np.ndarray, bias: str) -> np.ndarray:
    if bias == "tl":
        return mask
    if bias == "tr":
        return np.fliplr(mask)
    if bias == "bl":
        return np.flipud(mask)
    assert bias == "br"
    return np.flipud(np.fliplr(mask))


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
    for r, c in candidates:
        if int(mask.sum()) >= int(target):
            return True
        mask[r, c] = True
    return int(mask.sum()) >= int(target)


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    bias: str,
    *,
    extra_lo_terms: list[int] | None = None,
) -> np.ndarray | None:
    mask = _checker_perimeter(n, random.randrange(2))
    br, bc = _bias_point(n, bias)

    orbit = _rot_orbit(br, bc, n)
    for p in orbit - {(br, bc)}:
        mask[p] = False
    mask[br, bc] = True

    forced = [(br, bc)]
    for nr, nc in sorted(_band_cells(n, LAYER_BAND_WIDTH), key=lambda rc: abs(rc[0] - br) + abs(rc[1] - bc)):
        if (nr, nc) in forced or (nr, nc) in orbit - {(br, bc)}:
            continue
        forced.append((nr, nc))
        if len(forced) == 4:
            break
    for rr, cc in forced:
        mask[rr, cc] = True

    lo_terms = [int(np.ceil(density_lo * n * n)), int(mask.sum())]
    if extra_lo_terms:
        lo_terms.extend(int(x) for x in extra_lo_terms)
    lo = max(lo_terms)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    blocked = orbit - {(br, bc)}
    candidates = [(r, c) for r, c in _band_cells(n, LAYER_BAND_WIDTH) if (r, c) not in blocked and not mask[r, c]]
    random.shuffle(candidates)
    candidates.sort(key=lambda rc: abs(rc[0] - br) + abs(rc[1] - bc))
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int, bias: str) -> np.ndarray | None:
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(m, MID_DENSITY_LO, min(MID_DENSITY_HI, hi_cap / (m * m)), bias)
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int, bias: str) -> np.ndarray | None:
    mask = _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, bias, extra_lo_terms=[min_area])
    if mask is None:
        return None
    edge_pts = [(0, n // 2), (n - 1, n // 2), (n // 2, 0), (n // 2, n - 1)]
    for r, c in edge_pts:
        mask[int(r), int(c)] = True
    if not _is_rot_asym(mask):
        return None
    return mask


def _valid_inner_spans(n: int) -> list[int]:
    max_span = n - INNER_SPAN_MARGIN
    return [s for s in range(INNER_SPAN_MIN, max_span + 1) if (s % 2) == (n % 2)]


def _inner_local(span: int, bias: str) -> np.ndarray:
    if span == 3:
        base = np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=bool,
        )
    elif span == 4:
        base = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            dtype=bool,
        )
    else:
        base = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=bool,
        )
    return _flip_to_bias(base, bias)


def _make_inner_mask(n: int, bias: str) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    span = random.choice(_valid_inner_spans(n))
    local = _inner_local(span, bias)
    top = (n - span) // 2
    mask[top : top + span, top : top + span] = local
    return mask


def _center_dict(board: np.ndarray, colors: list[int]) -> dict[int, tuple[float, float]]:
    return {int(col): _coords_center(np.argwhere(board == int(col))) for col in colors}


def _axis_gap(
    target: tuple[float, float],
    others: list[tuple[float, float]],
    direction: str,
) -> float:
    if not others:
        return 99.0
    axis = 1 if direction in ("top", "bottom") else 0
    return min(abs(float(target[axis]) - float(other[axis])) for other in others)


def _board_anchor(n: int, profile: str) -> tuple[int, int]:
    if profile == "bl":
        return CANVAS_SIZE - n, 0
    if profile == "br":
        return CANVAS_SIZE - n, CANVAS_SIZE - n
    if profile == "tl":
        return 0, 0
    assert profile == "tr"
    return 0, CANVAS_SIZE - n


def _pick_directions(side_options: tuple[str, str]) -> list[str]:
    # Every sample should exhibit both exposed side bands so the direction code
    # is observable from a small set of examples.
    dirs = [str(side_options[0]), str(side_options[1]), str(random.choice(side_options))]
    random.shuffle(dirs)
    return dirs


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _direction_candidates(
    board_bbox: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, board_bbox)
    cells: list[tuple[int, int]] = []
    if direction == "top":
        for r in range(0, r0 - CUE_BOARD_DIST + 1):
            for c in range(c0, c1 + 1):
                cells.append((int(r), int(c)))
    elif direction == "bottom":
        for r in range(r1 + CUE_BOARD_DIST, CANVAS_SIZE):
            for c in range(c0, c1 + 1):
                cells.append((int(r), int(c)))
    elif direction == "left":
        for r in range(r0, r1 + 1):
            for c in range(0, c0 - CUE_BOARD_DIST + 1):
                cells.append((int(r), int(c)))
    else:
        assert direction == "right"
        for r in range(r0, r1 + 1):
            for c in range(c1 + CUE_BOARD_DIST, CANVAS_SIZE):
                cells.append((int(r), int(c)))
    return cells


def _cue_candidate_positions(
    board_bbox: tuple[int, int, int, int],
    *,
    target_center: tuple[float, float],
    other_centers: list[tuple[float, float]],
    direction: str,
) -> list[tuple[int, int]]:
    scored: list[tuple[float, float, float, int, int]] = []
    for r, c in _direction_candidates(board_bbox, direction):
        bb = (int(r), int(c), int(r), int(c))
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            continue
        cue_center = (float(r), float(c))
        target_score, _ = _alignment_cost(cue_center, target_center)
        other_best = min((_alignment_cost(cue_center, other)[0] for other in other_centers), default=99.0)
        margin = other_best - target_score
        if margin < ALIGN_MARGIN:
            continue
        scored.append((round(margin, 6), round(-target_score, 6), random.random(), int(r), int(c)))

    scored.sort(reverse=True)
    return [(int(r), int(c)) for _m, _s, _rnd, r, c in scored[:MAX_CUE_CANDIDATES]]


def _search_cue_layout(
    canvas: np.ndarray,
    cue_specs: list[dict],
    idx: int,
    cue_bboxes: list[tuple[int, int, int, int]],
    out_board: np.ndarray,
) -> bool:
    if idx >= len(cue_specs):
        try:
            return np.array_equal(main(canvas), out_board)
        except AssertionError:
            return False

    spec = cue_specs[idx]
    for r, c in spec["candidates"]:
        bb = (int(r), int(c), int(r), int(c))
        if int(canvas[int(r), int(c)]) != 0:
            continue
        if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
            continue

        canvas[int(r), int(c)] = int(spec["color"])
        cue_bboxes.append(bb)
        if _search_cue_layout(canvas, cue_specs, idx + 1, cue_bboxes, out_board):
            return True
        cue_bboxes.pop()
        canvas[int(r), int(c)] = 0
    return False


def _has_clear_alignment(grid: np.ndarray) -> bool:
    bg = int(get_bg(grid))
    try:
        crop, (r0, c0, r1, c1), bg = crop_components_near_largest(grid, bg=bg, margin=3, connectivity=8)
    except AssertionError:
        return False

    colors = colors_present(crop, exclude={int(bg)})
    layer_centers = {
        int(col): _coords_center(np.argwhere(crop == int(col)))
        for col in colors
    }

    outside_grid = np.array(grid, copy=True)
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    outside_comps = find_components(
        outside_grid,
        bg=int(bg),
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )

    for comp in outside_comps:
        cue_center = _coords_center(comp["coords"])
        primary_costs = sorted(
            _alignment_cost(cue_center, layer_centers[int(col)])[0]
            for col in colors
        )
        if len(primary_costs) >= 2 and primary_costs[1] - primary_costs[0] < ALIGN_MARGIN:
            return False
    return True


def _baseline_original_like(grid: np.ndarray) -> np.ndarray:
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
        rot_k = (-cue_size) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))
    return out


def _gen_attempt() -> np.ndarray | None:
    board_colors = random.sample(range(1, 10), NUM_COLORS)
    biases = random.sample(BIAS_NAMES, 3)
    profile = random.choice(tuple(PROFILE_TO_SIDES.keys()))
    side_options = PROFILE_TO_SIDES[profile]
    cue_dirs = _pick_directions(side_options)

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n)]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n, biases[1])
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, biases[2])
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD, biases[0])
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = board_colors[0]
    out_board[mid_mask] = board_colors[1]
    out_board[inner_mask] = board_colors[2]

    in_board = np.zeros((n, n), dtype=int)
    rot_args = [int(DIRECTION_TO_ROT[d]) for d in cue_dirs]
    for col, rot_arg in zip(board_colors, rot_args):
        inv = np.rot90(out_board == int(col), (-int(rot_arg)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    input_centers = _center_dict(in_board, board_colors)

    r0, c0 = _board_anchor(n, profile)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (int(r0), int(c0), int(r0 + n - 1), int(c0 + n - 1))
    try:
        _crop, crop_bbox, _ = crop_components_near_largest(canvas, bg=0, margin=3, connectivity=8)
    except AssertionError:
        return None
    if tuple(map(int, crop_bbox)) != board_bbox:
        return None

    global_centers = {
        int(col): (input_centers[int(col)][0] + r0, input_centers[int(col)][1] + c0)
        for col in board_colors
    }

    cue_colors = random.sample([col for col in range(1, 10) if col not in board_colors], NUM_COLORS)
    cue_specs = []
    for idx in range(NUM_COLORS):
        target_col = int(board_colors[idx])
        others = [global_centers[col] for col in board_colors if int(col) != target_col]
        candidates = _cue_candidate_positions(
            board_bbox,
            target_center=global_centers[target_col],
            other_centers=others,
            direction=str(cue_dirs[idx]),
        )
        if not candidates:
            return None
        cue_specs.append(
            {
                "color": int(cue_colors[idx]),
                "candidates": candidates,
                "gap": _axis_gap(global_centers[target_col], others, str(cue_dirs[idx])),
            }
        )
    cue_specs.sort(key=lambda spec: (len(spec["candidates"]), spec["gap"]))
    cue_bboxes: list[tuple[int, int, int, int]] = []
    if not _search_cue_layout(canvas, cue_specs, 0, cue_bboxes, out_board):
        return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    for bb in cue_bboxes:
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            return None

    solved = main(canvas)
    if not np.array_equal(solved, out_board):
        return None
    if not _has_clear_alignment(canvas):
        return None
    if np.array_equal(_baseline_original_like(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

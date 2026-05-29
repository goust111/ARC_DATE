# Variant of 6ffbe589
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: s.color == d.color，提示精灵与板内颜色层通过颜色一致性配对... → 空间对齐匹配：提示精灵的行/列位置与颜色层中心对齐

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# -----------------------------------------------------------------------------
# Design analysis for the D1 variant
#
# a) Concrete behavior trace
#   original:
#     - crop the largest board component
#     - for each board color, search outside cues of the same color
#     - cue area mod 4 decides that color layer's rotation
#   variant:
#     - keep the same crop + per-layer rotation mechanism
#     - cue colors are no longer informative and are chosen from colors absent
#       from the board
#     - each board layer is generated with a distinct mass centroid
#     - every cue is placed so its own center is aligned to one target layer
#       along either a row or a column
#     - solver matches each cue to the layer with the smallest row/column
#       center discrepancy, then uses cue area mod 4 as before
#   small example:
#     - suppose outer/mid/inner input-layer centroids are at
#       (4.2, 3.1), (5.0, 8.0), (7.4, 4.6)
#     - a 3-pixel cue centered near row 5.0 but with unrelated color 9
#       matches the middle layer because |cue_r - 5.0| is smallest
#     - original logic would look for color 9 inside the board and fail
#
# b) Complexity calibration
#   - The board still has exactly 3 independently-rotated layers.
#   - Cues remain 1..3 blobs, so the search space is still small enough for
#     humans and for a brute-force assignment over <= 3! matches.
#   - To avoid ambiguity, generator enforces centroid separation and places cues
#     only when the nearest row/column match has a clear margin over the others.
#
# c) Old assumptions that break under the variant
#   - `colors=[col]` in outside cue search assumed cue color == layer color.
#     This must be removed because cue colors are intentionally unrelated.
#   - `len(outside_comps) <= 1` per board color assumed color partitions the
#     cues. After the variant, cues must be collected globally and assigned.
#   - The original generator could let all layer centroids nearly coincide
#     because color already disambiguated them. That is no longer safe, so the
#     generator now biases each layer toward a different corner and validates the
#     resulting input-layer centroids before placing cues.
#
# d) Placement legality
#   - Board placement is unchanged: it must be the unique largest component.
#   - Cues still stay at Chebyshev distance >= 4 from the board and from each
#     other, so the crop helper continues to isolate the board cleanly.
#   - Cue placement searches only legal locations whose row/column alignment
#     makes the intended target strictly better than all competing layers.
#
# e) Visual observability
#   - Cue colors are decoupled from board colors, so color matching is visibly
#     wrong on the examples.
#   - The three layers keep asymmetric internal structure, so rotations remain
#     visible after matching.
#   - The cue-to-layer relation is exposed by literal horizontal/vertical
#     alignment between the cue and the biased "center of mass" of a layer.
#
# f) Visual cleanliness
#   - The board remains a single square crop with three non-overlapping layers.
#   - Cues are sparse, monochrome, and well separated from the board and from
#     one another, so the alignment signal is readable instead of cluttered.
#
# g) Color inferability
#   - No color mapping is part of the rule anymore. Cue colors are distractors.
#   - Colors are still limited to 3 board colors plus a few cue colors, and all
#     behavior is recoverable from geometry alone.
#
# h) Quantitative anti-degeneration target
#   - Every generated cue color is outside the board palette.
#   - With at least one cue present, the original color-matching solver would
#     leave all layers unpaired, while this variant still rotates by geometry.
#
# i) New solver assumptions to audit
#   - Each rotating cue must have a unique best board layer under row/column
#     alignment scoring.
#   - Each cue blob is treated as a single monochrome component.
#   - The board crop remains square so per-layer `np.rot90` preserves shape.
#   The generator enforces all three conditions explicitly.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and rotate its color layers using spatially aligned outside cues.

    Input structure:
    - Background is the most frequent color.
    - One large square board cluster contains 3 non-overlapping color layers.
    - Small monochrome cue sprites sit outside the board, and their colors are not the matching signal.

    Rule:
    - Crop around the main board component and ignore distant sprites.
    - Compute the center of mass of each board color layer and the center of each cue sprite.
    - Match every cue to the layer whose row or column center is closest, then rotate that layer clockwise by `(cue area mod 4) * 90`.
    - Layers with no assigned cue stay unchanged.

    Key insight: The cue's placement, not its color, determines which layer it controls; area still encodes the quarter-turn count.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster (drop distant cue sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Match outside cues to layer centroids by row/column alignment ---
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
    cue_infos = [
        {
            "area": int(comp["area"]),
            "center": _coords_center(comp["coords"]),
        }
        for comp in outside_comps
    ]

    assigned = _assign_cues_by_alignment(cue_infos, layer_centers)
    for col in colors:
        cue = assigned.get(int(col))
        cue_size = int(cue["area"]) if cue is not None else 0
        rot_k = (-cue_size) % 4
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
INNER_SPAN_MARGIN = 8
INNER_SPAN_MIN = 3
OUTER_AREA_PAD = 4

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 3
ALIGN_EPS = 1.1
ALIGN_MARGIN = 0.0
MIN_AXIS_GAP = 0.4

GEN_RETRIES = 2000
BIAS_NAMES = ("tl", "tr", "bl", "br")


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


def _pick_rotations() -> list[int]:
    rot_ks = [random.randrange(4) for _ in range(3)]
    if not any(rot_ks):
        rot_ks[random.randrange(3)] = random.randint(1, 3)
    return rot_ks


def _cue_patch(color: int, area: int) -> np.ndarray:
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
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _board_anchor(n: int) -> tuple[int, int]:
    spare = CANVAS_SIZE - n
    return random.randint(0, spare), random.randint(0, spare)


def _center_dict(board: np.ndarray, colors: list[int]) -> dict[int, tuple[float, float]]:
    return {int(col): _coords_center(np.argwhere(board == int(col))) for col in colors}


def _best_axis_gap(target: tuple[float, float], others: list[tuple[float, float]]) -> float:
    if not others:
        return 99.0
    row_gap = min(abs(target[0] - other[0]) for other in others)
    col_gap = min(abs(target[1] - other[1]) for other in others)
    return max(row_gap, col_gap)


def _place_aligned_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    *,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
    target_center: tuple[float, float],
    other_centers: list[tuple[float, float]],
) -> bool:
    h, w = patch.shape
    mask = patch != 0
    local_center = _coords_center(np.argwhere(mask))

    row_gap = min(abs(target_center[0] - other[0]) for other in other_centers) if other_centers else 99.0
    col_gap = min(abs(target_center[1] - other[1]) for other in other_centers) if other_centers else 99.0
    axis_order = sorted([("row", row_gap), ("col", col_gap)], key=lambda x: x[1], reverse=True)

    best_candidate: tuple[float, float, int, int, tuple[int, int, int, int]] | None = None
    for axis, gap in axis_order:
        if gap < MIN_AXIS_GAP:
            continue
        for r in range(CANVAS_SIZE - h + 1):
            for c in range(CANVAS_SIZE - w + 1):
                if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                    continue
                bb = (r, c, r + h - 1, c + w - 1)
                if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                    continue
                if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                    continue

                cue_center = (r + local_center[0], c + local_center[1])
                if axis == "row" and abs(cue_center[0] - target_center[0]) > ALIGN_EPS:
                    continue
                if axis == "col" and abs(cue_center[1] - target_center[1]) > ALIGN_EPS:
                    continue

                target_score, _ = _alignment_cost(cue_center, target_center)
                other_best = min((_alignment_cost(cue_center, other)[0] for other in other_centers), default=99.0)
                margin = other_best - target_score
                if margin < ALIGN_MARGIN:
                    continue

                candidate = (margin, -target_score, r, c, bb)
                if best_candidate is None or candidate > best_candidate:
                    best_candidate = candidate
    if best_candidate is None:
        return False

    _margin, _neg_score, r, c, bb = best_candidate
    paste(canvas, patch, int(r), int(c), transparent=0)
    cue_bboxes.append(bb)
    return True


def _gen_attempt() -> np.ndarray | None:
    board_colors = random.sample(range(1, 10), NUM_COLORS)
    c_frame, c_mid, c_inner = board_colors
    rot_ks = _pick_rotations()
    biases = random.sample(BIAS_NAMES, 3)

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
    out_board[outer_mask] = c_frame
    out_board[mid_mask] = c_mid
    out_board[inner_mask] = c_inner

    in_board = np.zeros((n, n), dtype=int)
    for col, k in zip(board_colors, rot_ks):
        inv = np.rot90(out_board == col, (-k) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = col

    input_centers = _center_dict(in_board, board_colors)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    global_centers = {
        int(col): (input_centers[int(col)][0] + r0, input_centers[int(col)][1] + c0)
        for col in board_colors
    }

    rotating_idxs = [i for i, k in enumerate(rot_ks) if k != 0]
    remaining_colors = [col for col in range(1, 10) if col not in board_colors]
    cue_colors = random.sample(remaining_colors, len(rotating_idxs))

    cue_specs = []
    for cue_color, idx in zip(cue_colors, rotating_idxs):
        area_mod = (-rot_ks[idx]) % 4
        area = random.choice([1, 5]) if area_mod == 1 else area_mod
        target_col = board_colors[idx]
        others = [global_centers[col] for col in board_colors if col != target_col]
        cue_specs.append((target_col, cue_color, area, _best_axis_gap(global_centers[target_col], others)))
    cue_specs.sort(key=lambda item: item[3])

    cue_bboxes: list[tuple[int, int, int, int]] = []
    for target_col, cue_color, area, _gap in cue_specs:
        patch = _cue_patch(int(cue_color), int(area))
        others = [global_centers[col] for col in board_colors if col != target_col]
        if not _place_aligned_cue(
            canvas,
            patch,
            board_bbox=board_bbox,
            cue_bboxes=cue_bboxes,
            target_center=global_centers[target_col],
            other_centers=others,
        ):
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
    raise RuntimeError("failed to generate a valid 6ffbe589 D1 variant input")

# Combo variant of 6ffbe589 (index 22)
# combo_spec: D4/3+D14/0
# Name: L/T形精灵+图例区域

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# combo_spec:
#   preserved_backbone:
#     - Keep the original 20x20 canvas, square board crop, three centered color
#       layers, and output-first construction.
#     - Keep same-color addressing: a cue still controls the board layer of the
#       same color, and missing cue still means that layer is already aligned.
#   fused_change:
#     - D4/3 replaces area-based cue decoding with a fixed cue vocabulary:
#         L -> 1 quarter-turn, T -> 2 quarter-turns, cross -> 3 quarter-turns.
#     - D14/0 moves the cues into a dedicated edge legend: the top strip is split
#       into three fixed lanes, and lane position also encodes 1/2/3 turns.
#     - Fusion rule: each outside cue must satisfy both channels at once. The
#       shape family and the legend lane must agree on the same rotation amount.
#   concrete_arc_realization:
#     - The board stays as the unique largest foreground component, flush to the
#       bottom edge so the top strip remains visually reserved for the legend.
#     - The legend occupies the top 3 rows and uses three wide, separated slots:
#         left slot  -> rotate by 1
#         middle slot -> rotate by 2
#         right slot -> rotate by 3
#     - Each non-zero instruction is a monochrome L/T/cross patch of the target
#       board color, pasted into the corresponding slot. No zero-rotation cue is
#       drawn, so absence still means "leave unchanged".
#   behavior_trace:
#     - Example: suppose the red inner layer in the desired output is the input
#       layer rotated one quarter-turn backward.
#     - Original task: a red blob of area 1 or 5 would sit somewhere outside the
#       board, and the solver would compute (-area) % 4 == 1.
#     - D4 alone: that blob becomes a red L template; the solver classifies the
#       silhouette as L and rotates the red layer once.
#     - Fused task: the red L is also placed in the left legend lane. The solver
#       first reads "left lane -> 1", independently reads "L -> 1", verifies the
#       two channels match, and only then rotates the red layer once.
#     - Difference: the control channel is no longer arithmetic over free-form
#       blob size. It is a structured legend with redundant semantic encoding.
#   complexity_calibration:
#     - Only three non-zero instruction types exist, matching the three visible
#       cue families and three legend lanes. This keeps the hypothesis space small.
#     - Non-zero rotations are sampled without duplication, so each lane holds at
#       most one cue. That avoids packing ambiguity in the short top strip and
#       makes the legend layout visually crisp.
#     - Every sample contains at least one L or T cue. Those both have area 4, so
#       the old size-mod-4 rule fails on every generated sample, not just on average.
#   implicit_assumption_review_for_main:
#     - crop_components_near_largest(...) assumes remote cues do not intersect the
#       board's margin-3 expansion. Generator guarantees board top >= 7 and keeps
#       the legend in rows 0..2, so the expanded crop excludes the legend.
#     - colors_present(crop, exclude={bg}) assumes the crop still contains exactly
#       the board colors. Generator keeps the board as the unique largest component.
#     - find_components(..., monochromatic=True) assumes each cue stays isolated.
#       The three legend lanes are separated by wide blank gaps, so cue components
#       do not touch one another even under 8-connectivity.
#     - _decode_cue_rotation(...) assumes each cue matches exactly one template
#       family and exactly one legend lane. Generator emits only those templates at
#       fixed lane anchors, and each lane holds at most one cue.
#     - rotate_color_layer(...) assumes the crop is square. Generator still builds
#       an NxN board and the crop helper returns that bbox.
#   placement_legality:
#     - Board and legend never overlap and all cue patches fit inside the 20x20
#       canvas.
#     - Each cue patch is pasted into its lane without touching another cue.
#     - The board remains much larger than any cue, so the largest-component crop
#       remains stable.
#   visual_observability:
#     - Across examples, the same three top-lane positions recur, so a human can
#       infer that the edge strip is a legend rather than random clutter.
#     - The same three silhouettes recur, so the second channel is also learnable.
#     - Because L and T both have area 4, the new behavior is visibly different
#       from the original size-count rule.
#   cleanliness:
#     - The board occupies the bottom of the canvas and the legend occupies the top
#       strip, so the two semantic zones are clearly separated.
#     - No extra framing color is introduced; the puzzle remains focused on shape,
#       position, and layer rotation only.
#   new_solver_assumptions_to_audit:
#     - Solver assumes all outside components belong to the legend. Generator emits
#       no distractor sprites outside the board.
#     - Solver assumes any color with a cue should show a visible rotation. The
#       generator retains asymmetry checks for all rotated layers.
# ---------------------------------------------------------------------------


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


CUE_TEMPLATES: dict[int, np.ndarray] = {
    1: np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    2: np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    3: np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
}

LANE_STARTS: dict[int, int] = {1: 0, 2: 8, 3: 16}
LEGEND_HEIGHT = 3


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    arr = np.array(mask, dtype=np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), tuple(int(x) for x in arr.ravel())


def _template_signatures(mask: np.ndarray) -> set[tuple[int, int, tuple[int, ...]]]:
    sigs: set[tuple[int, int, tuple[int, ...]]] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            sigs.add(_mask_signature(np.rot90(base, k)))
    return sigs


CUE_SIGNATURES = {rot_k: _template_signatures(mask) for rot_k, mask in CUE_TEMPLATES.items()}
LANE_BY_START = {int(v): int(k) for k, v in LANE_STARTS.items()}


def _component_patch(comp: dict) -> np.ndarray:
    coords = np.array(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _decode_shape_rotation(comp: dict) -> int:
    sig = _mask_signature(_component_patch(comp))
    matches = [rot_k for rot_k, sigs in CUE_SIGNATURES.items() if sig in sigs]
    assert len(matches) == 1, "expected cue to match exactly one template family"
    return int(matches[0])


def _decode_lane_rotation(comp: dict) -> int:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    assert 0 <= r0 <= r1 < LEGEND_HEIGHT, "expected cue to lie inside the legend strip"
    assert c1 < 20, "expected cue bbox to stay within the canvas"
    assert c0 in LANE_BY_START, "expected cue to start at a known legend lane"
    return int(LANE_BY_START[int(c0)])


def _decode_cue_rotation(comp: dict) -> int:
    lane_rot = _decode_lane_rotation(comp)
    shape_rot = _decode_shape_rotation(comp)
    assert lane_rot == shape_rot, "expected legend lane and cue shape to agree"
    return int(lane_rot)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and restore its color layers by reading a top-edge legend.

    Input structure:
    - bg is the most common color.
    - The largest foreground component is a square multi-color board made of three centered layers.
    - Above the board, a dedicated top-strip legend contains same-color cue sprites placed in one of three fixed lanes and drawn as L/T/cross templates.

    Rule:
    - Tight-crop the board cluster and ignore the remote legend strip.
    - For each board color, look for the unique outside cue of that same color.
    - Decode the cue twice: its lane says 1/2/3 quarter-turns, and its shape says L/T/cross = 1/2/3 quarter-turns.
    - If both channels agree, rotate that color layer by the decoded amount; if no cue exists, leave the layer unchanged.

    Key insight: Rotation is redundantly encoded by both legend position and cue silhouette, so the solver must treat the top strip as a legend and cross-check the shape class inside it.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the board cluster while excluding the top legend strip ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) For each color layer, decode its legend cue by lane and shape ---
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
        rot_k = _decode_cue_rotation(outside_comps[0]) if outside_comps else 0
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator ---------------

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

OUTER_AREA_PAD = 4
GEN_RETRIES = 300


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


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray:
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
    active_count = random.randint(1, 3)
    active_idxs = random.sample(range(NUM_COLORS), active_count)
    first_rot = random.choice([1, 2])
    remaining_rots = [rot for rot in (1, 2, 3) if rot != first_rot]
    active_rots = [first_rot]
    if active_count > 1:
        active_rots.extend(random.sample(remaining_rots, active_count - 1))
    random.shuffle(active_rots)
    rot_ks = [0, 0, 0]
    for idx, rot_k in zip(active_idxs, active_rots):
        rot_ks[int(idx)] = int(rot_k)
    return rot_ks


def _cue_patch(color: int, rot_k: int) -> np.ndarray:
    template = CUE_TEMPLATES[int(rot_k)]
    patch = np.where(template, int(color), 0)
    return np.rot90(patch, random.randrange(4))


def _board_anchor(n: int) -> tuple[int, int]:
    spare = CANVAS_SIZE - n
    return spare, random.randint(0, spare)


def _place_legend_cues(canvas: np.ndarray, cue_items: list[tuple[int, int]]) -> bool:
    for col, rot_k in cue_items:
        patch = _cue_patch(int(col), int(rot_k))
        h, w = map(int, patch.shape)
        if h > LEGEND_HEIGHT:
            return False
        slot_c = int(LANE_STARTS[int(rot_k)])
        slot_w = 3
        if w > slot_w or slot_c + w > CANVAS_SIZE:
            return False
        max_r = LEGEND_HEIGHT - h
        r = random.randint(0, max_r)
        if np.any(canvas[r : r + h, slot_c : slot_c + w] != 0):
            return False
        paste(canvas, patch, r, slot_c, transparent=0)
    return True


def _gen_attempt() -> np.ndarray | None:
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

    cue_items = [(col, k) for col, k in zip(colors, rot_ks) if k != 0]
    random.shuffle(cue_items)
    if not _place_legend_cues(canvas, cue_items):
        return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None

    legend_comps = find_components(canvas[:LEGEND_HEIGHT], bg=0, connectivity=8, monochromatic=True, return_format="dict")
    if len(legend_comps) != len(cue_items):
        return None
    for comp in legend_comps:
        if _decode_cue_rotation(comp) not in (1, 2, 3):
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

# Combo variant of 6ffbe589 (index 3)
# combo_spec: D2/0+D4/1
# Name: 宽度编码+线段精灵

"""
Design analysis for the combo variant
=====================================

combo_rule:
  preserved_core:
    - The largest foreground component is still the square three-layer board.
    - Output is still the cropped board with each color layer restored independently.
    - Cue colors still match board-layer colors one-to-one.
  fused_change:
    - D4 turns every cue into a solid horizontal or vertical line segment.
    - D2 changes the decoded quantity from cue area to cue bbox width.
    - Combined effect: orientation matters only because it changes what "width" means.
      A horizontal line has width = its length, while a vertical line always has width = 1.
    - Therefore:
      * horizontal length 2 -> rot_k = 2
      * horizontal length 3 -> rot_k = 1
      * horizontal length 5 -> rot_k = 3
      * any vertical length 2/3/4 -> rot_k = 3 because bbox width stays 1
  behavior_trace:
    - Original task example:
      a red 3-cell cue would be read by area, so red rotates by one quarter-turn.
    - Combo example:
      a red vertical 4x1 cue has area 4 and length 4, but bbox width 1.
      Old area/length reading would produce rot_k = 0.
      New fused reading produces rot_k = 3, so the red layer rotates 90 degrees clockwise.
    - Meanwhile a blue horizontal 1x2 cue still gives width 2 -> rot_k = 2.
    - The difference is not "vertical cue means another operation";
      the difference is that verticality changes the measured span.
  complexity_calibration:
    - Cue geometry is restricted to a single clean family: solid axis-aligned line segments.
    - Horizontal widths are restricted to {2, 3, 5}; vertical lengths to {2, 3, 4}.
    - Every generated sample contains at least one vertical cue whose area-based decoding
      disagrees with width-based decoding, so the fusion does not collapse back to the original task.
    - When multiple cues are present, the generator mixes at least one vertical and one horizontal cue
      so both branches of the rule are visible in the same sample set.
  implicit_assumption_review:
    - crop_components_near_largest assumes the board is the unique largest foreground component.
      The generator keeps the board large and all cues small and well separated.
    - The per-color outside-cue lookup assumes at most one cue per board color.
      The generator places at most one same-color line per layer.
    - rotate_color_layer assumes the crop is square.
      The board remains an NxN square, so this stays valid.
    - Width decoding only makes sense if cues are true solid line segments.
      The solver now asserts line-ness explicitly, and the generator constructs only such cues.
  placement_legality:
    - Cue patches are tight-cropped, stay inside the 20x20 canvas, do not overlap,
      and keep Chebyshev distance >= 4 from the board and from each other.
    - Board anchoring leaves enough side-band to host the required line orientations.
  visual_observability:
    - Vertical lines of different lengths producing the same 90-degree CW restoration
      make the width rule visible to a human observer.
    - Horizontal lines provide the other non-zero rotation bins without introducing many categories.
    - Every cued layer is forced to be rotationally asymmetric, so a wrong decode is visible.
  cleanliness_and_inference:
    - The scene still has one dominant board plus a few small same-color cues.
    - No hidden use of color ids, ordering, or extra marker colors is introduced.
    - Orientation/width can be inferred directly from visible shape, not from metadata.
  main_assumption_audit:
    - _decode_line_cue_rot_k assumes (h == 1) xor (w == 1), area == max(h, w), and length >= 2.
    - main assumes colors without cues should remain unchanged.
    - main assumes same-color matching is unique and all cues lie fully outside the crop.
    - generate_input() enforces each of those assumptions explicitly.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT_DIR = Path(__file__).resolve().parents[4]
_ARC_UTILS_DIR = _ROOT_DIR / "arc2_solver_select"
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg, paste


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
    """Rotate one color mask inside the fixed square crop."""
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


def _decode_line_cue_rot_k(comp: dict, *, use_bbox_width: bool) -> int:
    """Decode one outside line cue by bbox width (combo rule) or by area (legacy regression check)."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(comp["area"])
    assert (h == 1) ^ (w == 1), "expected a horizontal or vertical line cue"

    length = int(max(h, w))
    assert length >= 2, "expected cue orientation to be visually observable"
    assert area == length, "expected the cue to be a solid line segment"

    cue_value = int(w) if use_bbox_width else int(area)
    return (-cue_value) % 4


def _solve_with_metric(grid: np.ndarray, *, use_bbox_width: bool) -> np.ndarray:
    """Shared solver core used by the combo solver and generator-side regression checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

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
        rot_k = _decode_line_cue_rot_k(outside_comps[0], use_bbox_width=use_bbox_width)
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and restore each color layer using same-color line cues read by width.

    Input structure:
    - The background is the most frequent color.
    - The largest foreground component is a square board made of three centered non-overlapping color layers.
    - Outside the board, some of those colors appear again as separate monochrome straight line segments.

    Rule:
    - Crop around the large board component and ignore the distant cue segments.
    - For each board color, find its same-color cue outside the crop if one exists.
    - Require that every cue is a solid horizontal or vertical line, then read its bounding-box width.
    - Apply rot_k = (-width) % 4 quarter-turns to that color layer inside the cropped square.
    - Colors with no cue stay unchanged.

    Key insight: The cue is still matched by color, but the transform is encoded by horizontal span; a vertical segment matters because its width stays 1 no matter how long the line is.
    """
    return _solve_with_metric(grid, use_bbox_width=True)


# --------------- generator (att2) ---------------

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

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4
EDGE_BAND_FOR_LINES = 5

GEN_RETRIES = 500

HORIZONTAL_WIDTH_BY_ROT = {1: 3, 2: 2, 3: 5}
VERTICAL_LENGTHS = (2, 3, 4)


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
    candidates = [
        (r, c)
        for r, c in _band_cells(n, LAYER_BAND_WIDTH)
        if (r, c) not in blocked and not mask[r, c]
    ]
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


def _pick_cue_specs() -> list[dict[str, int | bool] | None]:
    """
    Pick cue specs for the three layers.

    Guarantees:
    - at least one cue exists
    - at least one vertical cue exists
    - every vertical cue is informative under width-vs-area decoding
    - when multiple cues exist, at least one horizontal cue also exists
    """
    cue_count = random.choice([1, 2, 2, 3])
    active = random.sample(range(NUM_COLORS), cue_count)
    specs: list[dict[str, int | bool] | None] = [None] * NUM_COLORS

    vertical_idx = random.choice(active)
    specs[vertical_idx] = {
        "rot_k": 3,
        "vertical": True,
        "line_len": random.choice(VERTICAL_LENGTHS),
    }

    remaining = [idx for idx in active if idx != vertical_idx]
    if remaining:
        horizontal_idx = random.choice(remaining)
        horizontal_rot = random.choice([1, 2, 3])
        specs[horizontal_idx] = {
            "rot_k": int(horizontal_rot),
            "vertical": False,
            "line_len": int(HORIZONTAL_WIDTH_BY_ROT[int(horizontal_rot)]),
        }
        remaining.remove(horizontal_idx)

    for idx in remaining:
        if random.randrange(2):
            specs[idx] = {
                "rot_k": 3,
                "vertical": True,
                "line_len": random.choice(VERTICAL_LENGTHS),
            }
        else:
            horizontal_rot = random.choice([1, 2, 3])
            specs[idx] = {
                "rot_k": int(horizontal_rot),
                "vertical": False,
                "line_len": int(HORIZONTAL_WIDTH_BY_ROT[int(horizontal_rot)]),
            }

    return specs


def _cue_patch(color: int, *, vertical: bool, line_len: int) -> np.ndarray:
    """Return a tight-cropped solid line cue with the requested orientation and length."""
    length = int(line_len)
    assert length >= 2, "expected visible line cues"
    if vertical:
        return np.full((length, 1), int(color), dtype=int)
    return np.full((1, length), int(color), dtype=int)


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


def _board_anchor(n: int, cue_specs: list[dict[str, int | bool] | None]) -> tuple[int, int]:
    """
    Pick a board top-left position with enough free side-band to host the required line orientations.
    Horizontal cues benefit from a top/bottom band; vertical cues benefit from a left/right band.
    """
    spare = CANVAS_SIZE - n
    need_row_band = any(spec is not None and not bool(spec["vertical"]) for spec in cue_specs)
    need_col_band = any(spec is not None and bool(spec["vertical"]) for spec in cue_specs)

    row_candidates = [
        r for r in range(spare + 1) if (not need_row_band) or max(int(r), int(spare - r)) >= EDGE_BAND_FOR_LINES
    ]
    col_candidates = [
        c for c in range(spare + 1) if (not need_col_band) or max(int(c), int(spare - c)) >= EDGE_BAND_FOR_LINES
    ]
    assert row_candidates and col_candidates, "expected at least one legal board anchor"
    return random.choice(row_candidates), random.choice(col_candidates)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    cue_specs = _pick_cue_specs()
    require_asym = [spec is not None for spec in cue_specs]

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, require_asym[2])]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=require_asym[2])
    if require_asym[2] and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    masks = [outer_mask, mid_mask, inner_mask]
    for mask, needs_asym in zip(masks, require_asym):
        if needs_asym and not _is_rot_asym(mask):
            return None

    out_board = np.zeros((n, n), dtype=int)
    for col, mask in zip(colors, masks):
        out_board[mask] = int(col)

    in_board = np.zeros((n, n), dtype=int)
    for col, mask, spec in zip(colors, masks, cue_specs):
        rot_k = int(spec["rot_k"]) if spec is not None else 0
        inv = np.rot90(mask, (-rot_k) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    r0, c0 = _board_anchor(n, cue_specs)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_items = [
        (int(col), bool(spec["vertical"]), int(spec["line_len"]))
        for col, spec in zip(colors, cue_specs)
        if spec is not None
    ]
    if not cue_items:
        return None

    cue_bboxes: list[tuple[int, int, int, int]] = []
    random.shuffle(cue_items)
    for col, vertical, line_len in cue_items:
        patch = _cue_patch(col, vertical=vertical, line_len=line_len)
        if not _place_cue(canvas, patch, board_bbox, cue_bboxes):
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

    variant_out = main(canvas)
    if not np.array_equal(variant_out, out_board):
        return None

    legacy_out = _solve_with_metric(canvas, use_bbox_width=False)
    if np.array_equal(legacy_out, out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

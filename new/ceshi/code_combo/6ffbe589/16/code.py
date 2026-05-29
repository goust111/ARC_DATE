# Combo variant of 6ffbe589 (index 16)
# combo_spec: D1/1+D4/0
# Name: 边缘颜色匹配+对角像素提示

from __future__ import annotations

import random
import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
# ---------------------------------------------------------------------------
# Concrete fused behavior:
# - The board still contains three concentric, independently rotating layers.
# - The crop also contains three fixed edge feature bars:
#   top bar    -> identifies the outer layer
#   right bar  -> identifies the middle layer
#   bottom bar -> identifies the inner layer
# - The colors of those bars are NOT layer colors. External cues reuse the bar
#   colors, so the mapping chain is:
#     outside diagonal-pair color -> matching edge bar -> target layer -> rotate
# - The diagonal cue distance (1..3) gives the clockwise turn count.
#
# Small example trace:
# - Suppose the top edge bar is color 7, right bar is color 4, bottom bar is
#   color 9, while the actual layer colors are 2/6/8.
# - Outside the board we see two color-4 pixels on a diagonal with gap 2.
# - Color 4 matches the right-edge bar, so it refers to the middle layer.
# - Gap 2 means rotate that middle layer clockwise by 180 degrees.
# - This differs from the original task because color 4 does not match the
#   middle layer color directly; the solver must first decode the edge feature.
#
# Complexity / observability calibration:
# - Only three edge bar slots exist and their side positions are fixed across
#   samples, so humans can align "top/right/bottom marker" with
#   outer/middle/inner roles after a few examples.
# - Cue gaps are restricted to 1, 2, 3 cells, keeping the rotation alphabet
#   small and visually measurable.
# - Edge bar colors are forced to be distinct from all layer colors, preventing
#   collapse back to the original same-color matching rule.
#
# Implicit-assumption audit:
# - Cropping still assumes the intended board bbox is the unique largest
#   foreground cluster neighborhood. Generator preserves that.
# - Layer classification no longer relies on cue colors. Instead:
#   * outer layer = only non-feature layer color touching the crop border
#   * inner layer = smaller bbox among the two non-feature, non-outer colors
#   * middle layer = the remaining one
#   Generator preserves these geometry cues.
# - Fixed edge bars would collide with a rotated outer layer unless the outer
#   mask reserves the full 4-side orbit of those bar positions. Generator clears
#   all four side slots from the outer layer, so bars stay fixed in every input.
#
# Placement legality / cleanliness:
# - Cue pixels are placed in empty background, their pair bbox stays at least
#   Chebyshev distance 4 from the board bbox and from other cue bboxes.
# - Each cue color appears either 0 or exactly 2 times outside the board.
# - The left-edge slot is kept blank only to complete the outer-layer reserved
#   rotation orbit; it is not itself a feature color, which keeps the visual
#   grammar simple instead of adding a fourth mapping case.


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """Crop the component neighborhood around the unique largest foreground part."""
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
    assert keep_coords.size > 0, "expected the expanded crop to retain foreground"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def rotate_color_layer_cw(grid: np.ndarray, color: int, cw_turns: int, *, bg: int | None = None) -> np.ndarray:
    """Rotate one color mask clockwise inside a fixed square crop."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    turns = int(cw_turns) % 4
    if turns == 0:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    rotated = np.rot90(mask, (-turns) % 4)
    assert rotated.shape == mask.shape, "expected a square crop for layer rotation"

    out = grid.copy()
    out[out == col] = bg
    out[rotated == 1] = col
    return out


def _edge_slots(n: int) -> dict[str, list[tuple[int, int]]]:
    """Fixed 2-cell bars on the board edges; left slot is reserved but uncolored."""
    assert n >= 6, "board too small for fused edge slots"
    a = int(n // 2 - 1)
    b = int(n // 2)
    return {
        "outer": [(0, a), (0, b)],
        "middle": [(a, n - 1), (b, n - 1)],
        "inner": [(n - 1, a), (n - 1, b)],
        "blank": [(a, 0), (b, 0)],
    }


def _read_feature_colors(crop: np.ndarray, bg: int) -> dict[str, int]:
    """Read the three edge feature colors from their fixed slots."""
    n = int(crop.shape[0])
    assert crop.shape == (n, n), "expected square crop"
    slots = _edge_slots(n)

    feature_colors: dict[str, int] = {}
    for role in ("outer", "middle", "inner"):
        vals = {int(crop[r, c]) for r, c in slots[role]}
        assert len(vals) == 1, f"expected uniform feature bar for {role}"
        col = vals.pop()
        assert col != int(bg), f"expected non-bg feature bar for {role}"
        feature_colors[role] = int(col)

    blank_vals = {int(crop[r, c]) for r, c in slots["blank"]}
    assert blank_vals == {int(bg)}, "expected reserved left slot to stay blank"
    return feature_colors


def _bbox_area_for_color(grid: np.ndarray, color: int) -> int:
    pts = np.argwhere(grid == int(color))
    assert pts.size > 0, f"expected color {color} to exist"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _infer_layer_colors(crop: np.ndarray, bg: int, feature_colors: dict[str, int]) -> dict[str, int]:
    """Recover outer/middle/inner layer colors from geometry after removing feature colors."""
    feature_set = set(map(int, feature_colors.values()))
    layer_colors = [int(c) for c in colors_present(crop, exclude={int(bg)}) if int(c) not in feature_set]
    assert len(layer_colors) == 3, "expected exactly three rotating layer colors"

    n = int(crop.shape[0])
    border_mask = np.zeros((n, n), dtype=bool)
    border_mask[0, :] = True
    border_mask[n - 1, :] = True
    border_mask[:, 0] = True
    border_mask[:, n - 1] = True
    border_hits = {int(col): int(((crop == int(col)) & border_mask).sum()) for col in layer_colors}

    outer = max(layer_colors, key=lambda col: border_hits[int(col)])
    assert border_hits[int(outer)] > 0, "expected an outer layer touching the border"

    rem = [int(col) for col in layer_colors if int(col) != int(outer)]
    inner = min(rem, key=lambda col: _bbox_area_for_color(crop, int(col)))
    middle = rem[0] if rem[1] == inner else rem[1]

    return {"outer": int(outer), "middle": int(middle), "inner": int(inner)}


def _find_outside_diagonal_gaps(outside_grid: np.ndarray, cue_colors: set[int]) -> dict[int, int]:
    """For each cue color, read the diagonal-pair gap if that cue exists."""
    gaps: dict[int, int] = {}
    for col in sorted(set(map(int, cue_colors))):
        pts = np.argwhere(outside_grid == int(col))
        if pts.size == 0:
            continue
        assert int(pts.shape[0]) == 2, f"expected 0 or 2 outside cue pixels for color {col}"
        (r0, c0), (r1, c1) = map(tuple, pts.tolist())
        dr = abs(int(r0) - int(r1))
        dc = abs(int(c0) - int(c1))
        assert dr == dc and 1 <= dr <= 3, f"expected diagonal cue gap in 1..3 for color {col}"
        gaps[int(col)] = int(dr)
    return gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the main board and rotate its three layers from edge-matched diagonal hints.

    Input structure:
    - bg is the most common color.
    - One square board neighborhood is the unique largest foreground structure.
    - The board contains three concentric rotating layers plus three fixed edge bars:
      top bar tags the outer layer, right bar tags the middle layer, bottom bar
      tags the inner layer.
    - Outside the board, optional cues appear as pairs of same-color pixels placed
      on a diagonal; the cue colors match the edge bars, not the layer colors.

    Rule:
    - Tight-crop the board neighborhood.
    - Read the three edge-bar colors and infer which actual layer color is
      outer/middle/inner from geometry.
    - For each cue color found outside, match it to its edge bar and measure the
      diagonal gap between the two cue pixels.
    - Gap 1/2/3 means rotate that target layer clockwise by 90/180/270 degrees.
      Missing cue means that layer stays unchanged.

    Key insight: The outside color is only an intermediate key; first use it to
    identify a specific edge bar, then use the bar's side position to select the
    layer, and only then turn that layer by the cue's diagonal spacing.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    assert crop.shape[0] == crop.shape[1], "expected square board crop"

    feature_colors = _read_feature_colors(crop, int(bg))
    layer_colors = _infer_layer_colors(crop, int(bg), feature_colors)

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    cue_gaps = _find_outside_diagonal_gaps(outside_grid, set(feature_colors.values()))

    out = crop.copy()
    for role in ("outer", "middle", "inner"):
        cue_color = int(feature_colors[role])
        if cue_color not in cue_gaps:
            continue
        out = rotate_color_layer_cw(out, int(layer_colors[role]), int(cue_gaps[cue_color]), bg=int(bg))
    return out


# --------------- generator ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_LAYERS = 3

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

GEN_RETRIES = 400
TEMPLATE_RETRIES = 24

# Cycle through 2-cue and 3-cue cases so generator coverage stays balanced and
# each sample shows enough fusion evidence for humans to infer the rule.
TURN_ROLE_TEMPLATES = (
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
_turn_template_cursor = random.randrange(len(TURN_ROLE_TEMPLATES))


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    """Return the 4 positions in the 90-degree rotation orbit of (r, c)."""
    pts = set()
    rr, cc = int(r), int(c)
    for _ in range(4):
        pts.add((int(rr), int(cc)))
        rr, cc = int(n - 1 - cc), int(rr)
    return pts


def _is_rot_asym(mask: np.ndarray) -> bool:
    """True if mask changes under every non-zero quarter turn."""
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + int(phase)) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < int(width)
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    random.shuffle(candidates)
    for r, c in candidates:
        if int(mask.sum()) >= int(target):
            return True
        mask[int(r), int(c)] = True
    return int(mask.sum()) >= int(target)


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    *,
    extra_lo_terms: list[int] | None = None,
) -> np.ndarray | None:
    """Build an asymmetric perimeter-heavy boolean mask."""
    if extra_lo_terms is None:
        extra_lo_terms = []

    mask = _checker_perimeter(n, random.randrange(2))

    orbit = _rot_orbit(*ASYM_MARKER, n)
    for p in orbit - {ASYM_MARKER}:
        mask[p] = False
    mask[ASYM_MARKER] = True

    lo = max(int(np.ceil(float(density_lo) * n * n)), int(mask.sum()), *[int(v) for v in extra_lo_terms])
    hi = int(np.floor(float(density_hi) * n * n))
    if lo > hi:
        return None
    target = random.randint(int(lo), int(hi))

    blocked = orbit - {ASYM_MARKER}
    candidates = [(r, c) for r, c in _band_cells(n, LAYER_BAND_WIDTH) if (r, c) not in blocked and not mask[r, c]]
    if not _fill_to_count(mask, candidates, int(target)):
        return None
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int) -> np.ndarray | None:
    m = int(n - 2 * MIDDLE_INSET)
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
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[int(min_area)])


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    max_span = int(n - INNER_SPAN_MARGIN)
    spans = [int(s) for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [int(s) for s in spans if int(s) > 1]
    else:
        # Avoid 1x1 inner layers even when no rotation is applied: they hide the
        # inner role too much and skew acceptance toward zero-turn cases.
        spans = [int(s) for s in spans if int(s) > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    span = int(random.choice(_valid_inner_spans(n, require_asym)))
    top = int((n - span) // 2)

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
            local[int(r), int(c)] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))
    mask[top : top + local.shape[0], top : top + local.shape[1]] = local
    return mask


def _sample_cw_turns(template: tuple[int, int, int]) -> list[int]:
    """Sample clockwise rotations from a fixed active-role template."""
    turns = [0, 0, 0]
    for idx, active in enumerate(template):
        if active:
            turns[idx] = random.randint(1, 3)
    return [int(v) for v in turns]


def _reserve_outer_feature_slots(mask: np.ndarray) -> np.ndarray:
    """Clear the full 4-side orbit of the feature bars so fixed bars never collide."""
    out = mask.copy()
    slots = _edge_slots(int(mask.shape[0]))
    reserve = set()
    for name in ("outer", "middle", "inner", "blank"):
        reserve.update((int(r), int(c)) for r, c in slots[name])
    for r, c in reserve:
        out[int(r), int(c)] = False
    return out


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return int(max(dr, dc))


def _place_diagonal_cue(
    canvas: np.ndarray,
    color: int,
    gap: int,
    *,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Place a same-color diagonal pixel pair with the given gap."""
    H, W = canvas.shape
    color = int(color)
    gap = int(gap)
    assert 1 <= gap <= 3

    candidates: list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]] = []
    for slope in (-1, 1):
        for r0 in range(H - gap):
            for c0 in range(W):
                r1 = int(r0 + gap)
                c1 = int(c0 + slope * gap)
                if not (0 <= c1 < W):
                    continue
                pts = [(int(r0), int(c0)), (int(r1), int(c1))]
                if any(int(canvas[r, c]) != 0 for r, c in pts):
                    continue
                rr = [int(r) for r, _ in pts]
                cc = [int(c) for _, c in pts]
                bb = (min(rr), min(cc), max(rr), max(cc))
                sub = canvas[bb[0] : bb[2] + 1, bb[1] : bb[3] + 1]
                if np.any(sub != 0):
                    continue
                if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                    continue
                if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                    continue
                candidates.append((pts, bb))
    if not candidates:
        return False

    pts, bb = random.choice(candidates)
    for r, c in pts:
        canvas[int(r), int(c)] = int(color)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int) -> tuple[int, int]:
    spare = int(CANVAS_SIZE - n)
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


def _gen_attempt(cw_turns: list[int] | None = None) -> np.ndarray | None:
    layer_colors = random.sample(range(1, 10), 6)
    c_outer, c_middle, c_inner, f_outer, f_middle, f_inner = map(int, layer_colors)
    if cw_turns is None:
        cw_turns = _sample_cw_turns(random.choice(TURN_ROLE_TEMPLATES))

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, cw_turns[2] > 0)]
    n = int(random.choice(valid_ns))

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=cw_turns[2] > 0)
    if cw_turns[2] > 0 and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None
    outer_mask = _reserve_outer_feature_slots(outer_mask)
    if cw_turns[0] > 0 and not _is_rot_asym(outer_mask):
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = int(c_outer)
    out_board[mid_mask] = int(c_middle)
    out_board[inner_mask] = int(c_inner)

    slots = _edge_slots(n)
    feature_by_role = {"outer": int(f_outer), "middle": int(f_middle), "inner": int(f_inner)}
    for role, col in feature_by_role.items():
        for r, c in slots[role]:
            assert out_board[int(r), int(c)] == 0
            out_board[int(r), int(c)] = int(col)
    for r, c in slots["blank"]:
        assert out_board[int(r), int(c)] == 0

    in_board = np.zeros((n, n), dtype=int)
    for role, col in feature_by_role.items():
        for r, c in slots[role]:
            in_board[int(r), int(c)] = int(col)

    for col, turns in zip((c_outer, c_middle, c_inner), cw_turns):
        rotated = np.rot90(out_board == int(col), int(turns))
        if np.any((in_board != 0) & rotated):
            return None
        in_board[rotated] = int(col)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, int(r0), int(c0), transparent=0)
    board_bbox = (int(r0), int(c0), int(r0 + n - 1), int(c0 + n - 1))

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [
        (int(feature_by_role["outer"]), int(cw_turns[0])),
        (int(feature_by_role["middle"]), int(cw_turns[1])),
        (int(feature_by_role["inner"]), int(cw_turns[2])),
    ]
    random.shuffle(cue_items)
    for cue_color, turns in cue_items:
        if int(turns) == 0:
            continue
        if not _place_diagonal_cue(canvas, int(cue_color), int(turns), board_bbox=board_bbox, cue_bboxes=cue_bboxes):
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

    try:
        solved = main(canvas)
    except Exception:
        return None
    if not np.array_equal(solved, out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    global _turn_template_cursor

    for _ in range(GEN_RETRIES):
        template = TURN_ROLE_TEMPLATES[int(_turn_template_cursor) % len(TURN_ROLE_TEMPLATES)]
        _turn_template_cursor += 1
        for _ in range(TEMPLATE_RETRIES):
            grid = _gen_attempt(cw_turns=_sample_cw_turns(template))
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid fused 6ffbe589 input")

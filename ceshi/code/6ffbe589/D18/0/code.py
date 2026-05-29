# Variant of 6ffbe589
# Dimension: D18 — 提示精灵数量与板内颜色的对应关系 (meta)
# Change: 1:1对应，每种需旋转的颜色恰好有一个提示精灵，且最多3个... → 多对一：多个提示精灵对应同一颜色层

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# Design analysis for the D18 variant
# a) Concrete behavior:
#    - The board stays a square 3-layer construction exactly like the source task.
#    - The visible change is outside the board: a board color may now appear on 2-3
#      separate monochrome cue blobs instead of exactly one.
#    - Solver behavior changes only at cue aggregation time: for one color, collect
#      every outside component of that color, sum their areas, and use (sum mod 4)
#      as the rotation code. Example: a middle layer with two same-color cues of
#      areas 2 and 3 now rotates by (-5) % 4 = 3 quarter-turns, whereas the source
#      logic would have looked for a single cue and would fail on this input.
# b) Complexity calibration:
#    - The board still has only 3 colors, so the new reasoning load comes from
#      grouping repeated colors, not from introducing more object types.
#    - At least one rotating color is forced to have multiple cues; some samples may
#      split two colors, but total cue count is capped by construction to keep the
#      scene readable and placement feasible.
#    - Multi-cue area splits are chosen so that at least one split color genuinely
#      needs aggregation: its total area mod 4 differs from every individual cue's
#      area mod 4. This prevents a visual multi-cue arrangement from secretly
#      collapsing back to a single-cue explanation.
# c) Main() assumptions affected by the variant:
#    - Original code assumed len(outside_comps) <= 1 per color. That no longer holds.
#    - The safe replacement is to sum all same-color outside component areas; the
#      rest of the pipeline (crop board, rotate layer mask inside the crop) remains
#      valid because cue placement still keeps every cue outside the crop margin.
# d) Placement legality:
#    - Board/cue non-overlap and in-bounds constraints stay unchanged.
#    - Cue-to-board distance remains >= 4 so the crop helper cannot swallow cues.
#    - Cue-to-cue distance is relaxed from 4 to 2 to make room for extra blobs while
#      still guaranteeing separate 8-connected components.
# e) Visual observability:
#    - Humans can see repeated same-color blobs outside the board and must total
#      their pixel counts. Because the split color is guaranteed in every sample,
#      D18 is visible from a small set of training pairs.
# f) Visual cleanliness:
#    - Extra cues are still isolated and monochrome, and the board is unchanged, so
#      the image stays legible rather than introducing overlapping annotations.
# g) Color reasoning:
#    - No hidden color-id ordering is used. Matching is purely by visible color
#      equality between board layers and outside cue blobs.
# 9) New implicit assumptions introduced by the new solver:
#    - Same-color cues must remain disconnected components so aggregation is over a
#      list of blobs rather than one merged object; the generator validates this.
#    - At least one color should actually have >1 cue, and aggregation should matter;
#      the generator checks both properties before accepting a sample.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board region and rotate each color layer using aggregated outside cues.

    Input structure:
    - Background is the most common color.
    - The largest foreground component is the multi-color square board to be cropped.
    - Outside the board are small monochrome cue blobs, and one board color may own
      several disconnected cue blobs of that same color.

    Rule:
    - Crop the board cluster while excluding distant cue blobs.
    - For each color present inside the crop, collect every outside cue blob of the
      same color and add their pixel counts.
    - Rotate that color layer by 90-degree steps according to the summed area mod 4.
    - Colors with no outside cues stay in place.

    Key insight: Rotation is no longer encoded by a single cue size; it is encoded by
    the total area of all same-color outside cues taken together.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the "board" cluster (drop distant sample sprites) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) For each layer, rotate according to the total size of all outside cues ---
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
        cue_area_total = sum(int(comp["area"]) for comp in outside_comps)
        rot_k = (-cue_area_total) % 4
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
ASYM_MARKER = (1, 2)
INNER_SPAN_MARGIN = 8

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 2
OUTER_AREA_PAD = 4

GEN_RETRIES = 300

SINGLE_CUE_AREAS = {
    1: (1, 5),
    2: (2,),
    3: (3,),
}

MULTI_CUE_AREAS = {
    1: ((2, 3), (1, 2, 2)),
    2: ((1, 1), (3, 3), (1, 1, 4)),
    3: ((1, 2), (1, 1, 1), (2, 5)),
}


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


def _pick_rotations() -> list[int]:
    """Pick rotation amounts for 3 color layers; at least one must be non-zero."""
    rot_ks = [random.randrange(4) for _ in range(3)]
    if not any(rot_ks):
        rot_ks[random.randrange(3)] = random.randint(1, 3)
    return rot_ks


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


def _single_cue_area(total_mod: int) -> int:
    """Pick a single cue area whose mod-4 code is `total_mod`."""
    return int(random.choice(SINGLE_CUE_AREAS[int(total_mod)]))


def _multi_cue_areas(total_mod: int, *, allow_three: bool) -> tuple[int, ...]:
    """Pick a multi-cue area split whose total mod-4 code is `total_mod`."""
    options = [
        areas
        for areas in MULTI_CUE_AREAS[int(total_mod)]
        if (allow_three or len(areas) == 2)
        and all((int(area) % 4) != int(total_mod) for area in areas)
    ]
    choice = tuple(int(a) for a in random.choice(options))
    total = sum(choice) % 4
    assert total == int(total_mod)
    assert len(choice) >= 2
    return choice


def _build_cue_plan(colors: list[int], rot_ks: list[int]) -> tuple[list[tuple[int, int]], dict[int, list[int]]]:
    """Plan cue areas per color, forcing at least one color to use multiple cues."""
    rotating = [idx for idx, k in enumerate(rot_ks) if int(k) != 0]
    assert rotating, "expected at least one rotating layer"

    if len(rotating) == 1:
        multi_indices = set(rotating)
    elif len(rotating) == 2:
        multi_indices = {random.choice(rotating)}
        if random.randrange(3) == 0:
            multi_indices = set(rotating)
    else:
        multi_indices = {random.choice(rotating)}

    cue_items: list[tuple[int, int]] = []
    cue_groups: dict[int, list[int]] = defaultdict(list)
    for idx in rotating:
        color = int(colors[idx])
        target_mod = (-int(rot_ks[idx])) % 4
        assert target_mod in {1, 2, 3}
        if idx in multi_indices:
            allow_three = len(rotating) == 1 and random.randrange(3) == 0
            areas = _multi_cue_areas(target_mod, allow_three=allow_three)
        else:
            areas = (_single_cue_area(target_mod),)
        for area in areas:
            cue_items.append((color, int(area)))
            cue_groups[color].append(int(area))

    assert any(len(areas) > 1 for areas in cue_groups.values())
    return cue_items, cue_groups


def _validate_cue_layout(
    canvas: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    colors: list[int],
    rot_ks: list[int],
    planned_groups: dict[int, list[int]],
) -> bool:
    """Check that cue components stayed separated and that the D18 behavior is present."""
    outside_grid = canvas.copy()
    r0, c0, r1, c1 = board_bbox
    outside_grid[r0 : r1 + 1, c0 : c1 + 1] = 0

    multi_color_count = 0
    aggregation_needed = False
    for color, k in zip(colors, rot_ks):
        comps = find_components(
            outside_grid,
            bg=0,
            connectivity=8,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        areas = sorted(int(comp["area"]) for comp in comps)
        expected = sorted(int(a) for a in planned_groups.get(int(color), []))
        if areas != expected:
            return False
        if int(k) == 0 and areas:
            return False
        if len(areas) > 1:
            multi_color_count += 1
            total_mod = sum(areas) % 4
            if all((area % 4) != total_mod for area in areas):
                aggregation_needed = True

    return multi_color_count >= 1 and aggregation_needed


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
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
    out_board[outer_mask] = int(colors[0])
    out_board[mid_mask] = int(colors[1])
    out_board[inner_mask] = int(colors[2])

    in_board = np.zeros((n, n), dtype=int)
    for col, k in zip(colors, rot_ks):
        inv = np.rot90(out_board == int(col), (-int(k)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_items, cue_groups = _build_cue_plan(colors, rot_ks)
    cue_bboxes: list[tuple[int, int, int, int]] = []
    for col, area in sorted(cue_items, key=lambda item: (-item[1], item[0])):
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

    if not _validate_cue_layout(canvas, board_bbox, colors, rot_ks, cue_groups):
        return None
    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 D18 input")

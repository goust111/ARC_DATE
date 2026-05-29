# Combo variant of 6ffbe589 (index 5)
# combo_spec: D8/1+D18/1
# Name: 无提示删除层+干扰精灵

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
# - The input still contains one large square board with 3 disjoint concentric color layers,
#   plus several small monochrome sprites outside the board.
# - Valid outside sprites are still same-color cues: their area mod 4 encodes the clockwise
#   rotation for that board color.
# - New fused rule:
#   1) outside sprites whose colors do not appear on the board are distractors and must be ignored;
#   2) for a board color with no matching outside cue at all, the whole layer is deleted.
# - Example trace:
#   board colors = {2 outer, 4 middle, 7 inner}; outside sprites = {color 2, area 3},
#   {color 9, area 5}. Original task would rotate only color 2 and leave colors 4 and 7 unchanged.
#   D18-only would still leave colors 4 and 7 unchanged while ignoring color 9.
#   D8-only would delete colors 4 and 7 but would never include color 9 as noise.
#   Combo version must do both at once: ignore color 9, rotate color 2 by `(-3) % 4 = 1`,
#   and erase colors 4 and 7 because they have no matching cues.
# - The visible difference from the original is therefore selective disappearance of board layers
#   despite the presence of multiple board-external sprites, some of which are red herrings.
#
# b) Complexity calibration:
# - The board still has exactly 3 colors and the cue semantics are unchanged, so the task stays
#   close to the source puzzle rather than becoming an open-ended object-matching problem.
# - The generator restricts cue-presence patterns to four repeated cases:
#   keep only outer, keep only middle, keep outer+inner, keep middle+inner.
#   This guarantees at least one deleted layer and usually deletes a visually large band.
# - Distractors are limited to 1-2 extra colors absent from the board. That is enough to force
#   filtering without flooding the canvas with too many alternatives.
#
# c) Hidden assumptions review:
# - `crop_components_near_largest(...)` assumes the full input board is the unique largest
#   foreground component. Still valid because the generator always draws the intact 3-layer board
#   in the input; deletion happens only in the output.
# - `colors_present(crop)` assumes every board color exists in the cropped input board. Still true.
# - The original solver treated missing cue as `cue_size=0 -> rot_k=0`. That assumption is no
#   longer sound: cue absence now means erase, not identity.
# - Filtering by color is now mandatory because outside monochrome components can exist whose
#   colors are not on the board at all. The solver must collect outside components and keep only
#   those with colors in the cropped board color set.
# - The solver still assumes at most one valid cue per board color. The generator enforces that by
#   creating at most one matching cue for each kept layer, even though multiple distractors may exist.
#
# d) Placement legality:
# - The board is placed on the 20x20 canvas exactly as before, flush with one canvas edge.
# - Every valid cue and distractor is an 8-connected monochrome blob of area 1..5.
# - All outside sprites obey a Chebyshev gap from the board bbox and from each other, so they
#   neither overlap nor merge, and none intrudes into the board crop.
#
# e) Visual observability:
# - Each sample contains at least one missing board-color cue, so the output always deletes at
#   least one layer relative to the input crop. This prevents degeneration to the original task.
# - Each sample also contains at least one distractor sprite with a color absent from the board,
#   so the solver cannot simply treat every outside sprite as meaningful.
# - Keeping deletion focused on large outer/middle layers in many cases makes the rule visible in
#   only a few examples; otherwise the effect could hide in a tiny center sprite.
#
# f) Visual cleanliness:
# - The board remains the only large connected object.
# - Outside sprites are sparse, separated, and monochrome; their only visual burden is cue/noise
#   discrimination by color membership, which is readable because distractor colors never appear
#   inside the cropped board.
#
# g) Change inferability:
# - No behavior depends on numeric color ids or ordering. A human can infer the rule from visible
#   structure alone: colors appearing both inside the board and outside as isolated blobs are
#   candidates for transformation; colors that appear only outside are irrelevant; board colors
#   with no valid external partner disappear in the output.


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


def erase_color_layer(grid: np.ndarray, color: int, *, bg: int | None = None) -> np.ndarray:
    """Delete one board color layer from the cropped board."""
    if bg is None:
        bg = get_bg(grid)
    out = grid.copy()
    out[out == int(color)] = int(bg)
    return out


def _matching_outside_cue_sizes(
    outside_grid: np.ndarray,
    board_colors: list[int],
    *,
    bg: int,
) -> dict[int, int]:
    """Collect all outside monochrome sprites and keep only cues whose colors appear on the board."""
    outside_comps = find_components(
        outside_grid,
        bg=int(bg),
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )

    board_color_set = {int(col) for col in board_colors}
    cue_sizes: dict[int, int] = {}
    for comp in outside_comps:
        col = int(comp["color"])
        if col not in board_color_set:
            continue
        assert col not in cue_sizes, f"expected at most one valid outside cue for color {col}"
        cue_sizes[col] = int(comp["area"])
    return cue_sizes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the main board, ignore off-board distractors, and rotate or erase each color layer.

    Input structure:
    - The background is the dominant color on a 20x20 canvas.
    - One large square multi-color board is the unique largest foreground component.
    - Several small monochrome sprites lie outside the board; some share a board color and some
      use colors absent from the board entirely.

    Rule:
    - Crop around the large board cluster and determine which colors actually belong to that board.
    - Inspect all outside monochrome sprites, but treat only same-color board matches as valid cues.
    - For a board color with a valid cue, rotate that color layer clockwise by `cue_area mod 4`
      quarter-turns.
    - For a board color with no valid matching cue, erase that whole layer to background.

    Key insight:
    Outside sprites serve two jobs at once: color membership tells you whether a sprite matters at
    all, and the surviving matched sprite's size mod 4 tells you how to rotate its layer.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    board_colors = colors_present(crop, exclude={int(bg)})
    assert board_colors, "expected the cropped board to contain foreground colors"
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    cue_sizes = _matching_outside_cue_sizes(outside_grid, board_colors, bg=int(bg))

    for col in board_colors:
        if int(col) not in cue_sizes:
            out = erase_color_layer(out, int(col), bg=int(bg))
            continue
        rot_k = (-int(cue_sizes[int(col)])) % 4
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator (combo) ---------------

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
CUE_INTER_DIST = 3
OUTER_AREA_PAD = 4
DISTRACTOR_COUNT_OPTIONS = (1, 2)
DISTRACTOR_AREA_OPTIONS = (1, 2, 3, 4, 5)

GEN_RETRIES = 500
KEEP_LAYER_PATTERNS = ((0,), (1,), (0, 2), (1, 2))


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


def _pick_keep_plan() -> tuple[tuple[int, ...], list[int]]:
    """Choose which layers receive valid cues and their non-zero output rotations."""
    keep_layers = random.choice(KEEP_LAYER_PATTERNS)
    rot_ks = [0, 0, 0]
    for idx in keep_layers:
        rot_ks[int(idx)] = random.randint(1, 3)
    return keep_layers, rot_ks


def _cue_area_from_rot(rot_k: int) -> int:
    """Map a non-zero clockwise rotation amount to a cue area in 1..5 with matching inverse mod."""
    mod = (-int(rot_k)) % 4
    assert mod in (1, 2, 3), "only layers with valid cues should request a cue area"
    if mod == 1:
        return random.choice([1, 5])
    return mod


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
        arr[r, c] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev gap between two axis-aligned bounding boxes."""
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
    """Place a cue-like patch on the canvas while respecting spacing constraints."""
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
    """Single attempt to generate a valid combo input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    keep_layers, rot_ks = _pick_keep_plan()
    keep_set = {int(idx) for idx in keep_layers}

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, 2 in keep_set)]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=2 in keep_set)
    if 2 in keep_set and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    layer_masks = [outer_mask, mid_mask, inner_mask]

    # Output-first: only layers with matching cues survive in the target board.
    out_board = np.zeros((n, n), dtype=int)
    for idx, (col, mask) in enumerate(zip(colors, layer_masks)):
        if idx in keep_set:
            out_board[mask] = int(col)

    # The input board remains complete; surviving layers are inverse-rotated, deleted layers stay unrotated.
    in_board = np.zeros((n, n), dtype=int)
    for idx, (col, mask) in enumerate(zip(colors, layer_masks)):
        placed = np.rot90(mask, (-rot_ks[idx]) % 4) if idx in keep_set else mask
        if np.any((in_board != 0) & placed):
            return None
        in_board[placed] = int(col)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(int(colors[idx]), _cue_area_from_rot(rot_ks[idx])) for idx in keep_layers]
    random.shuffle(cue_items)
    for col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(col, area), board_bbox, cue_bboxes):
            return None

    distractor_pool = [col for col in range(1, 10) if col not in colors]
    random.shuffle(distractor_pool)
    distractor_count = min(random.choice(DISTRACTOR_COUNT_OPTIONS), len(distractor_pool))
    distractor_colors = distractor_pool[:distractor_count]
    for col in distractor_colors:
        area = random.choice(DISTRACTOR_AREA_OPTIONS)
        if not _place_cue(canvas, _cue_patch(int(col), int(area)), board_bbox, cue_bboxes):
            return None

    # Validate largest-component assumptions and fused behavior.
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

    outside_grid = canvas.copy()
    outside_grid[r0 : r0 + n, c0 : c0 + n] = 0
    outside_mono = find_components(outside_grid, bg=0, connectivity=8, monochromatic=True, return_format="dict")
    board_color_set = set(colors_present(in_board, exclude={0}))
    matched = [comp for comp in outside_mono if int(comp["color"]) in board_color_set]
    distractors = [comp for comp in outside_mono if int(comp["color"]) not in board_color_set]
    if len(matched) != len(keep_layers):
        return None
    if len(distractors) != distractor_count or len(distractors) < 1:
        return None
    if len(matched) >= len(board_color_set):
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

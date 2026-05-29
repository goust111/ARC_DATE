# Combo variant of 6ffbe589 (index 24)
# combo_spec: D2/0+D8/1+D11/1
# Name: 宽度编码+无提示删除+角落放置

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for combo D2/0 + D8/1 + D11/1
# ------------------------------------------------
# a) Concrete behavior and trace:
# - Keep the original 20x20 scene type: one large square board made from 3 disjoint
#   concentric color layers, plus a few same-color monochrome cue sprites outside it.
# - D11 changes where cues live: every cue bbox is now flush to a canvas corner instead
#   of being scattered in free space.
# - D2 changes what a cue means: rotation is decoded from cue bbox width, not from filled
#   area. We keep rot_k = (-width) % 4, so usable widths are 1/2/3 for rotations 3/2/1.
# - D8 changes the default branch: if a board color has no cue in any corner, that whole
#   layer is removed from the output instead of being preserved.
# - Example trace:
#   suppose the outer blue layer has a top-right cue whose bbox width is 3 and area is 5,
#   the middle red layer has no cue, and the inner green layer has a bottom-left cue whose
#   width is 1 and area is 2. Original logic would read blue by area 5 -> rot_k 3 and keep
#   red unchanged; the combo reads blue by width 3 -> rot_k 1, deletes red entirely, and
#   reads green by width 1 -> rot_k 3. Same-color matching is unchanged, but both the cue
#   statistic and the missing-cue branch change at once.
#
# b) Complexity calibration:
# - Cue widths are limited to {1,2,3}; that keeps the coding alphabet small and repeated.
# - The generator only uses four keep/delete patterns: keep outer, keep middle,
#   keep outer+inner, keep middle+inner. This guarantees at least one deletion and keeps
#   the visual cases recurring across examples.
# - Cue templates are drawn from a tiny per-width catalog where area % 4 intentionally
#   disagrees with width % 4, so width really matters and the task does not silently fall
#   back to the original area-based rule.
#
# c) Hidden assumption review:
# - The original crop helper expanded the largest component by margin 3 to pull in nearby
#   board fragments while excluding far cues. With corner-fixed cues this becomes fragile if
#   the board approaches a cue corner, so the solver now crops the exact bbox of the unique
#   largest component. This is safe because the outer layer always spans the whole board bbox
#   and the inner layers lie strictly inside that bbox.
# - `main()` now assumes every real cue is flush to one of the four grid corners. The
#   generator guarantees that by anchoring cue patches exactly to those corners.
# - The missing-cue branch must be tested before decoding any rotation value, because "no
#   cue" no longer means rot_k = 0; it means erase the layer.
# - Independent color-layer rotation is still valid because the canonical board layers are
#   disjoint, and the generator rejects any inverse-rotated input board with overlaps.
#
# d) Placement legality:
# - Cues use distinct corners, so cue-cue overlap is impossible.
# - The board anchor is chosen only from positions whose bbox stays at Chebyshev gap >= 2
#   from every used corner cue bbox, preventing 8-connected fusion with the board.
# - All objects are pasted in-bounds, and the board remains the unique largest foreground
#   component.
#
# e) Visual observability:
# - Corner anchoring is obvious because cues touch the canvas edges at literal corners.
# - At least one layer is deleted in every sample, and it is always an outer or middle
#   layer, so the output visibly loses a substantial band rather than only a tiny center.
# - Width-coded cues visibly disagree with area-coded interpretation because the cue shapes
#   are selected from a catalog with mismatched area and width residues.
#
# f) Visual cleanliness:
# - Only 1 or 2 cues are present, each isolated in its own corner.
# - No new marker colors are introduced; cues keep the same-color matching relation from
#   the base task.
# - Output remains a clean square crop with zero overdraw between layers.
#
# g) Inferability:
# - The solver never depends on numeric color ids; colors are paired only by identity.
# - Corner location is a coarse structural hint ("look at corners for cues"), while width is
#   the fine-grained code and cue absence is the keep/delete switch. These three signals are
#   all human-visible from a few input/output pairs.


def crop_board_bbox(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Crop the exact bbox of the unique largest non-background component.

    The board's outer layer spans the full board bbox, so taking this bbox keeps the whole
    board while avoiding corner cues.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(
        grid,
        bg=bg,
        connectivity=int(connectivity),
        monochromatic=False,
        return_format="dict",
    )
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    r0, c0, r1, c1 = map(int, largest[0]["bbox"])
    crop = np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int, copy=True)
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    """Rotate one color mask inside a fixed-size square crop."""
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
    """Delete all pixels of one color layer inside the crop."""
    if bg is None:
        bg = get_bg(grid)
    out = grid.copy()
    out[out == int(color)] = int(bg)
    return out


def _cue_rot_k_from_comp(comp: dict, *, use_bbox_width: bool) -> int:
    """Decode rotation from cue bbox width (combo rule) or cue area (ablation checks)."""
    if use_bbox_width:
        _r0, c0, _r1, c1 = map(int, comp["bbox"])
        cue_value = int(c1 - c0 + 1)
    else:
        cue_value = int(comp["area"])
    return (-cue_value) % 4


def _is_corner_anchored(comp: dict, shape: tuple[int, int]) -> bool:
    """True iff a component bbox is flush to one of the four canvas corners."""
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _solve_with_options(
    grid: np.ndarray,
    *,
    use_bbox_width: bool,
    delete_missing: bool,
) -> np.ndarray:
    """Shared solver core used by the combo solver and generator-side ablation checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_board_bbox(g, bg=bg, connectivity=8)
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
        corner_comps = [comp for comp in outside_comps if _is_corner_anchored(comp, g.shape)]
        assert len(corner_comps) <= 1, f"expected at most one corner cue for color {col}"
        if not corner_comps:
            if delete_missing:
                out = erase_color_layer(out, int(col), bg=int(bg))
            continue
        rot_k = _cue_rot_k_from_comp(corner_comps[0], use_bbox_width=use_bbox_width)
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the main board and transform each color layer from same-color corner cues.

    Input structure:
    - The dominant color is background.
    - One large square board contains three disjoint concentric color layers and is the
      unique largest foreground component.
    - A subset of board colors also appears as small monochrome cue sprites, and every cue
      is flush to one of the four canvas corners.

    Rule:
    - Crop the board by taking the bbox of the largest component.
    - For each color inside that crop, search for a same-color cue outside the crop that is
      anchored to a corner.
    - If the cue exists, rotate that color layer by `(-cue_width) % 4` quarter-turns, where
      `cue_width` is the cue bbox width.
    - If the cue does not exist, erase that entire color layer to background.

    Key insight: Corner placement tells you where to look, cue width tells you how much to
    rotate, and cue absence means the matching board layer should disappear.
    """
    return _solve_with_options(grid, use_bbox_width=True, delete_missing=True)


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

OUTER_AREA_PAD = 4
CUE_BOARD_DIST = 2
GEN_RETRIES = 400

KEEP_LAYER_PATTERNS = [(0,), (1,), (0, 2), (1, 2)]
CORNER_NAMES = ("tl", "tr", "bl", "br")


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


def _pick_keep_plan() -> tuple[tuple[int, ...], list[int]]:
    """Choose which layers receive corner cues and their non-zero rotation amounts."""
    keep_layers = random.choice(KEEP_LAYER_PATTERNS)
    rot_ks = [0, 0, 0]
    for idx in keep_layers:
        rot_ks[int(idx)] = random.randint(1, 3)
    return keep_layers, rot_ks


def _cue_patch(color: int, width: int) -> np.ndarray:
    """
    Build a monochrome cue whose bbox width encodes rotation.

    Every template is chosen so area % 4 differs from width % 4.
    """
    catalogs = {
        1: [
            np.array([[1], [1]], dtype=int),
            np.array([[1], [1], [1]], dtype=int),
        ],
        2: [
            np.array([[1, 1], [1, 0]], dtype=int),
            np.array([[1, 1], [1, 1]], dtype=int),
            np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
        ],
        3: [
            np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
            np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
            np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
        ],
    }
    patch = random.choice(catalogs[int(width)]).copy()
    if random.randrange(2):
        patch = np.fliplr(patch)
    if random.randrange(2):
        patch = np.flipud(patch)
    return patch * int(color)


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev gap distance between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _anchored_bbox(shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    """BBox of a patch when it is pasted flush to the chosen canvas corner."""
    h, w = map(int, shape)
    if corner == "tl":
        return 0, 0, h - 1, w - 1
    if corner == "tr":
        return 0, CANVAS_SIZE - w, h - 1, CANVAS_SIZE - 1
    if corner == "bl":
        return CANVAS_SIZE - h, 0, CANVAS_SIZE - 1, w - 1
    if corner == "br":
        return CANVAS_SIZE - h, CANVAS_SIZE - w, CANVAS_SIZE - 1, CANVAS_SIZE - 1
    raise ValueError(f"unknown corner {corner}")


def _paste_corner(canvas: np.ndarray, patch: np.ndarray, corner: str) -> tuple[int, int, int, int]:
    """Paste a cue patch flush to the chosen corner and return its bbox."""
    h, w = map(int, patch.shape)
    if corner == "tl":
        r, c = 0, 0
    elif corner == "tr":
        r, c = 0, CANVAS_SIZE - w
    elif corner == "bl":
        r, c = CANVAS_SIZE - h, 0
    elif corner == "br":
        r, c = CANVAS_SIZE - h, CANVAS_SIZE - w
    else:
        raise ValueError(f"unknown corner {corner}")
    paste(canvas, patch, int(r), int(c), transparent=0)
    return _anchored_bbox((h, w), corner)


def _cue_width_from_rot(rot_k: int) -> int:
    """Map a non-zero rotation amount to the unique cue width that encodes it."""
    width = (-int(rot_k)) % 4
    assert width in (1, 2, 3), "only rotating colors receive cues in this combo variant"
    return int(width)


def _board_anchor(n: int, cue_bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int] | None:
    """Pick a board top-left whose bbox stays separated from all used corner cues."""
    candidates = []
    for r in range(CANVAS_SIZE - n + 1):
        for c in range(CANVAS_SIZE - n + 1):
            bb = (r, c, r + n - 1, c + n - 1)
            if any(_bbox_chebyshev(bb, cue_bb) < CUE_BOARD_DIST for cue_bb in cue_bboxes):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid combo input. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    keep_layers, rot_ks = _pick_keep_plan()
    keep_set = set(int(idx) for idx in keep_layers)

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

    out_board = np.zeros((n, n), dtype=int)
    for idx, (col, mask) in enumerate(zip(colors, layer_masks)):
        if idx in keep_set:
            out_board[mask] = int(col)

    in_board = np.zeros((n, n), dtype=int)
    for idx, (col, mask) in enumerate(zip(colors, layer_masks)):
        placed = np.rot90(mask, (-rot_ks[idx]) % 4) if idx in keep_set else mask
        if np.any((in_board != 0) & placed):
            return None
        in_board[placed] = int(col)

    chosen_corners = random.sample(CORNER_NAMES, len(keep_layers))
    cue_specs = []
    for idx, corner in zip(keep_layers, chosen_corners):
        width = _cue_width_from_rot(rot_ks[int(idx)])
        patch = _cue_patch(int(colors[int(idx)]), int(width))
        cue_specs.append((int(colors[int(idx)]), int(width), corner, patch, _anchored_bbox(patch.shape, corner)))

    cue_bboxes = [bb for _col, _width, _corner, _patch, bb in cue_specs]
    anchor = _board_anchor(n, cue_bboxes)
    if anchor is None:
        return None
    r0, c0 = anchor
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    for _col, _width, corner, patch, cue_bb in cue_specs:
        if _bbox_chebyshev(board_bbox, cue_bb) < CUE_BOARD_DIST:
            return None
        _paste_corner(canvas, patch, corner)

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None

    combo_out = main(canvas)
    if not np.array_equal(combo_out, out_board):
        return None

    area_delete_out = _solve_with_options(canvas, use_bbox_width=False, delete_missing=True)
    if np.array_equal(area_delete_out, out_board):
        return None

    width_keep_out = _solve_with_options(canvas, use_bbox_width=True, delete_missing=False)
    if np.array_equal(width_keep_out, out_board):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

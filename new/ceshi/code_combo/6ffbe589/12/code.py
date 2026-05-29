# Combo variant of 6ffbe589 (index 12)
# combo_spec: D4/2+D8/2
# Name: 板内嵌入提示+无提示变色

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The original task has one square board plus same-color cue blobs outside it. Here all outer
#    cue blobs are removed. Instead, the board reserves four tiny corner pads inside its own bbox.
#    Three of those pads may host a single corner marker pixel, and the marker uses the same color
#    as the layer it controls. The marker's corner encodes rotation:
#      - top-left corner marker    -> rotate that color layer 90 deg clockwise
#      - top-right corner marker   -> rotate that color layer 180 deg
#      - bottom-right corner marker -> rotate that color layer 270 deg clockwise
#    The fourth pad (bottom-left) is always empty and acts as a repeated "this is a cue well"
#    visual template. Example trace: suppose frame color 2 is marked at top-right, middle color 7
#    is unmarked, inner color 4 is marked at bottom-right. Old logic: look outside for color 2/7/4,
#    rotate marked colors, leave unmarked color 7 unchanged. New logic: read the in-board corner
#    markers instead, rotate color 2 by 180 deg, recolor the unmarked color-7 layer to grey, and
#    rotate color 4 by 270 deg. The cue pixels themselves disappear in the output.
# b) Complexity calibration:
#    The cue alphabet stays small: exactly three usable cue corners, one per non-zero quarter-turn.
#    To keep the rule legible, each sample contains exactly two marked colors and one unmarked color,
#    so both branches "has cue -> rotate" and "no cue -> turn grey" are always visible together.
#    Board colors never use grey=5, so the recolor branch is unambiguous.
# c) Implicit-assumption audit for main():
#    - crop_components_near_largest still assumes the board region is anchored by one unique largest
#      foreground component. The generator keeps the perimeter-heavy outer layer as the unique
#      largest structure, while corner markers are only single pixels inside the same bbox.
#    - The original per-color search in outside_grid is invalid because there are no external cues.
#      The new solver instead assumes a fixed set of reserved corner pads exists inside the crop.
#    - Because cue pixels reuse the target layer's color, the solver must strip those reserved pad
#      cells before rotating masks; otherwise a marker pixel would be rotated as part of the layer.
#    - A color can appear in at most one cue corner. The generator guarantees this by activating
#      exactly two layers and assigning them distinct non-zero rotations from {1,2,3}.
#    - The solver assumes the non-tip cells in every corner pad stay background. The generator clears
#      those pad cells from every layer mask before any rotation is applied.
# d) Placement legality:
#    Nothing is placed outside the board any more. The generator only needs to place one board on
#    the 20x20 canvas without overlap or overflow. The internal cue pads live inside the board bbox,
#    so there are no external spacing constraints to satisfy.
# e) Visual observability:
#    The same-color corner marker preserves the original "match by color" affordance, while the
#    fixed cue corners visibly replace the old external sprites. The missing cue branch is also
#    easy to notice because exactly one board color loses its original color and becomes grey.
# f) Visual cleanliness:
#    The only extra structure is four small corner notches that stay background in the true board.
#    Input samples fill at most two notch tips with cue pixels. This keeps the board readable and
#    avoids clutter from many scattered auxiliary sprites.
# g) Color-change inferability:
#    Recoloring is tied to a structural condition that humans can see: the layer's color appears on
#    the board but nowhere in the cue corners. The output target color is always the single fixed
#    grey color, so the recolor rule can be inferred from a few examples.

GREY_COLOR = 5
NUM_COLORS = 3


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
    """Rotate one color layer inside a fixed-size square crop."""
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


def recolor_color_layer(grid: np.ndarray, color: int, new_color: int) -> np.ndarray:
    """Replace one color layer while keeping its geometry fixed."""
    out = grid.copy()
    out[out == int(color)] = int(new_color)
    return out


def _cue_pad_cells(n: int) -> dict[int, list[tuple[int, int]]]:
    """Reserved cue-pad cells keyed by their encoded rotation; key 0 is the always-empty pad."""
    return {
        1: [(0, 0), (0, 1), (1, 0)],
        2: [(0, n - 1), (0, n - 2), (1, n - 1)],
        3: [(n - 1, n - 1), (n - 2, n - 1), (n - 1, n - 2)],
        0: [(n - 1, 0), (n - 2, 0), (n - 1, 1)],
    }


def _reserved_pad_cells(n: int) -> list[tuple[int, int]]:
    """All cells that must stay background in the clean board."""
    cells: list[tuple[int, int]] = []
    for pts in _cue_pad_cells(int(n)).values():
        cells.extend(pts)
    return cells


def _strip_cue_pads(grid: np.ndarray, bg: int) -> np.ndarray:
    """Blank every reserved cue-pad cell."""
    out = grid.copy()
    for r, c in _reserved_pad_cells(out.shape[0]):
        out[int(r), int(c)] = int(bg)
    return out


def _decode_corner_cues(crop: np.ndarray, bg: int) -> tuple[np.ndarray, dict[int, int]]:
    """
    Remove internal cue pads and return (clean_board, cue_by_color).

    cue_by_color maps board color -> rot_k in {1,2,3}.
    """
    assert crop.shape[0] == crop.shape[1], "expected the cropped board to stay square"
    n = int(crop.shape[0])
    pads = _cue_pad_cells(n)

    cue_by_color: dict[int, int] = {}
    for rot_k in (1, 2, 3):
        r, c = pads[rot_k][0]
        col = int(crop[r, c])
        if col == int(bg):
            continue
        assert col not in cue_by_color, "expected at most one cue corner per board color"
        cue_by_color[col] = int(rot_k)

    for rot_k, pts in pads.items():
        for rr, cc in pts[1:]:
            assert int(crop[rr, cc]) == int(bg), f"expected only the pad tip to be usable for cue {rot_k}"
    for rr, cc in pads[0]:
        assert int(crop[rr, cc]) == int(bg), "expected the bottom-left reference pad to stay empty"

    return _strip_cue_pads(crop, int(bg)), cue_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster, then rotate or recolor layers from corner markers hidden inside the board.

    Input structure:
    - The background is the most common color.
    - One square multi-color board is still the dominant foreground object.
    - There are no detached outside cues; instead, the board reserves four tiny corner pads.
    - Up to two pad tips contain a single pixel whose color matches a board layer.

    Rule:
    - Crop around the board cluster.
    - Read the marker color, if any, from the top-left, top-right, and bottom-right cue pads.
    - Those three pad locations encode 90 deg, 180 deg, and 270 deg clockwise rotation for the matching color layer.
    - Remove all cue-pad pixels from the crop, then process each board color independently.
    - If a color has a corner marker, rotate that layer by the encoded amount.
    - If a color has no marker anywhere, leave its shape in place but recolor the whole layer grey.

    Key insight: The old same-color cue matching still exists, but the clue has moved inside the board:
    the marker's color tells which layer to affect, while its corner position tells how far to rotate.
    Missing any such marker now means "turn this untouched layer grey".
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, _bbox, bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    board, cue_by_color = _decode_corner_cues(crop, int(bg))

    colors = colors_present(board, exclude={int(bg)})
    assert len(colors) == NUM_COLORS, "expected the cropped board to contain exactly three board colors"
    out = board.copy()

    for col in colors:
        rot_k = int(cue_by_color.get(int(col), 0))
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))
        else:
            out = recolor_color_layer(out, int(col), GREY_COLOR)

    return out


# --------------- generator (combo variant) ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13

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
GEN_RETRIES = 400


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


def _clear_reserved_pads(mask: np.ndarray) -> np.ndarray:
    """Force every cue-pad cell to background in the clean board."""
    out = mask.copy()
    for r, c in _reserved_pad_cells(out.shape[0]):
        out[int(r), int(c)] = False
    return out


def _pick_rotations() -> list[int]:
    """Pick exactly two distinct non-zero rotations and one uncued layer."""
    active_layers = random.sample(range(NUM_COLORS), 2)
    active_rots = random.sample([1, 2, 3], 2)
    rot_ks = [0, 0, 0]
    for idx, rot_k in zip(active_layers, active_rots):
        rot_ks[int(idx)] = int(rot_k)
    assert any(k == 0 for k in rot_ks) and any(k != 0 for k in rot_ks)
    return rot_ks


def _cue_tip_for_rotation(n: int, rot_k: int) -> tuple[int, int]:
    """Return the single occupied cue-tip cell for a non-zero rotation."""
    return _cue_pad_cells(int(n))[int(rot_k)][0]


def _board_anchor(n: int) -> tuple[int, int]:
    """Pick a random board top-left position anywhere the square fits."""
    spare = CANVAS_SIZE - n
    return random.randint(0, spare), random.randint(0, spare)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid combo-variant input grid. Returns None on failure."""
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], NUM_COLORS)
    rot_ks = _pick_rotations()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, rot_ks[2] != 0)]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=rot_ks[2] != 0)
    if rot_ks[2] != 0 and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    layer_masks = [
        _clear_reserved_pads(outer_mask),
        _clear_reserved_pads(mid_mask),
        _clear_reserved_pads(inner_mask),
    ]
    if any(not _is_rot_asym(mask) for mask, k in zip(layer_masks, rot_ks) if k != 0):
        return None

    outer_mask, mid_mask, inner_mask = layer_masks
    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    in_board = np.zeros((n, n), dtype=int)
    cue_tips: list[tuple[int, int, int]] = []

    for col, mask, k in zip(colors, layer_masks, rot_ks):
        if k == 0:
            out_board[mask] = GREY_COLOR
            if np.any((in_board != 0) & mask):
                return None
            in_board[mask] = int(col)
            continue

        out_board[mask] = int(col)
        inv = np.rot90(mask, (-int(k)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)
        cue_tips.append((*_cue_tip_for_rotation(n, int(k)), int(col)))

    for r, c, col in cue_tips:
        if in_board[int(r), int(c)] != 0:
            return None
        in_board[int(r), int(c)] = int(col)

    clean_input = _strip_cue_pads(in_board, 0)
    if colors_present(clean_input, exclude={0}) != colors_present(in_board, exclude={0}):
        return None

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None

    actual = main(canvas)
    if not np.array_equal(actual, out_board):
        return None
    if GREY_COLOR not in colors_present(actual, exclude={0}):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

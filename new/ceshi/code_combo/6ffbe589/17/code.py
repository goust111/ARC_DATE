# Combo variant of 6ffbe589 (index 17)
# combo_spec: D6/0+D19/0
# Name: 保持原始尺寸+保留提示精灵

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for combo D6/0 + D19/0
# a) Concrete behavior trace:
#    In the original task, a 20x20 input contains one large square board plus small same-color cue blobs outside it.
#    The solver crops to the board bbox, so the output is only the board and the cues disappear automatically.
#    Here the same cue-driven color-layer rotation still happens, but only inside the board bbox; the output stays
#    20x20 and the cue blobs remain visible at their original coordinates. A small example:
#    - original: board at rows 7..16, cue at (1,1), output is a 10x10 crop with the cue gone.
#    - combo: board still at rows 7..16, cue still at (1,1), output is 20x20 and only the board area changes.
# b) Complexity calibration:
#    Keep the original 3 board colors and at most one cue per color. This preserves a small, readable control set:
#    humans can match colors directly, and at least one cue still changes the board so the transform is visible.
# c) Hidden-assumption review:
#    The original main() assumed "crop == final output" and "outside objects can be discarded". Both are false here.
#    We therefore separate board localization from output construction, and we overwrite only the board bbox in a
#    full-canvas copy. The remaining assumptions still needed are: the board is the unique largest component, its
#    bbox is square, and each board color has at most one same-color outside cue. The generator enforces these.
# d) Placement legality:
#    The board and cues are placed without overlap; cues stay at Chebyshev distance >= 4 from the board and from each
#    other, so they cannot merge into the main component and cannot invalidate the largest-component board detection.
# e) Visual observability:
#    Across examples, the preserved cue blobs are an explicit reminder that they still matter. The board stays put,
#    so solvers can compare input/output in-place and notice that only same-color layers rotate while cues persist.
# f) Visual cleanliness:
#    Only one region mutates: the square board. Cues remain detached in empty space, so the output is not cluttered.
# g) Inferability:
#    No hidden color ranking is used. The cue-to-layer mapping is direct same-color matching, and the only change from
#    the original task is trajectory/output format: the same local rule now acts in-place on the 20x20 canvas.


def largest_component_bbox(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 8,
) -> tuple[int, int, int, int]:
    """Return the bbox of the unique largest non-bg component."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    return tuple(map(int, largest[0]["bbox"]))


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    """Rotate one color's occupancy mask inside a fixed square board."""
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    if k == 0:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    rmask = np.rot90(mask, k)
    assert rmask.shape == mask.shape, "expected a square board for layer rotation"

    out = grid.copy()
    out[out == col] = bg
    out[rmask == 1] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate color layers inside the main board while keeping the original 20x20 scene intact.

    Input structure:
    - Background is the most common color.
    - One square multicolor board is the unique largest foreground component.
    - Small monochrome cue blobs sit outside the board; each cue color matches a board layer color.

    Rule:
    - Keep the full canvas size and preserve every outside cue in place.
    - Locate the board bbox, then for each board color read the area of its same-color outside cue.
    - Rotate that color layer inside the board by (cue area mod 4) quarter-turns clockwise; no cue means no change.

    Key insight:
    - The cues still encode per-color rotations exactly as before, but only the board region is rewritten; the rest of
      the scene is copied through unchanged, so the output is the transformed board embedded back into the source grid.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    r0, c0, r1, c1 = largest_component_bbox(g, bg=bg, connectivity=8)
    board = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
    assert board.shape[0] == board.shape[1], "expected the main board bbox to be square"

    colors = colors_present(board, exclude={int(bg)})
    assert colors, "expected the board to contain foreground colors"

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    rotated = board.copy()
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
        cue_size = int(outside_comps[0]["area"]) if outside_comps else 0
        rot_k = (-cue_size) % 4
        if rot_k:
            rotated = rotate_color_layer(rotated, int(col), int(rot_k), bg=int(bg))

    out = g.copy()
    out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = rotated
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

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4

GEN_RETRIES = 300


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    """Return the 4-cell rotation orbit of (r, c) in an n*n grid."""
    pts = set()
    rr, cc = int(r), int(c)
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _is_rot_asym(mask: np.ndarray) -> bool:
    """True iff mask changes under every non-trivial 90-degree rotation."""
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    """Checkerboard subset of the perimeter of an n*n board."""
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + int(phase)) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    """All cells within `width` of at least one border in an n*n board."""
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < int(width)
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    """Randomly turn on candidate cells until the target area is reached."""
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
    extra_lo_terms: list[int] = (),
) -> np.ndarray | None:
    """Build a rotationally-asymmetric layer concentrated near the border."""
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
    """Build the middle concentric layer inset from the outer frame."""
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(m, MID_DENSITY_LO, min(MID_DENSITY_HI, hi_cap / (m * m)))
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET:MIDDLE_INSET + m, MIDDLE_INSET:MIDDLE_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int) -> np.ndarray | None:
    """Build the outer frame layer spanning the full board."""
    return _make_asym_layer(n, OUTER_DENSITY_LO, OUTER_DENSITY_HI, extra_lo_terms=[int(min_area)])


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    """Allowed side lengths for the centered inner sprite."""
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_asym: bool) -> np.ndarray:
    """Build the small central layer."""
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
            local[int(r), int(c)] = True
        if random.randrange(2):
            local = np.fliplr(local)
        local = np.rot90(local, random.randrange(4))

    mask[top:top + local.shape[0], top:top + local.shape[1]] = local
    return mask


def _pick_rotations() -> list[int]:
    """Pick per-color output rotations with balanced coverage of 1/2/3 active cues."""
    active_count = random.randint(1, 3)
    active_idxs = set(random.sample(range(3), active_count))
    rot_ks = [random.randint(1, 3) if idx in active_idxs else 0 for idx in range(3)]
    return rot_ks


def _cue_patch(color: int, area: int) -> np.ndarray:
    """Grow a small 8-connected cue blob of the requested area."""
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < int(area):
        frontier = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[int(r), int(c)] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev gap distance between two inclusive bboxes."""
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Place a cue patch without overlap and with required separation."""
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r:r + h, c:c + w] != 0) & mask):
                continue
            bb = (int(r), int(c), int(r + h - 1), int(c + w - 1))
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            candidates.append((int(r), int(c), bb))
    if not candidates:
        return False
    r, c, bb = random.choice(candidates)
    paste(canvas, patch, int(r), int(c), transparent=0)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int) -> tuple[int, int]:
    """Place the board flush against one canvas edge."""
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
    """Single output-first attempt; returns an input grid or None on failure."""
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
    for col, mask in zip(colors, [outer_mask, mid_mask, inner_mask]):
        out_board[mask] = int(col)

    in_board = np.zeros((n, n), dtype=int)
    for col, k in zip(colors, rot_ks):
        inv = np.rot90(out_board == int(col), (-int(k)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    r0, c0 = _board_anchor(n)
    board_bbox = (int(r0), int(c0), int(r0 + n - 1), int(c0 + n - 1))
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, int(r0), int(c0), transparent=0)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [
        (int(col), random.choice([1, 5]) if (-int(k)) % 4 == 1 else (-int(k)) % 4)
        for col, k in zip(colors, rot_ks)
        if int(k) != 0
    ]
    random.shuffle(cue_items)
    for col, area in cue_items:
        if not _place_cue(canvas, _cue_patch(int(col), int(area)), board_bbox, cue_bboxes):
            return None

    out_canvas = canvas.copy()
    out_canvas[int(r0) : int(r0) + n, int(c0) : int(c0) + n] = out_board

    # Generator guarantees for the new solver assumptions.
    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    if (largest["bbox"][2] - largest["bbox"][0]) != (largest["bbox"][3] - largest["bbox"][1]):
        return None
    for bb in cue_bboxes:
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            return None
    outside_grid = canvas.copy()
    outside_grid[int(r0) : int(r0) + n, int(c0) : int(c0) + n] = 0
    for col in colors:
        outside = find_components(
            outside_grid,
            bg=0,
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        if len(outside) > 1:
            return None

    if not np.array_equal(main(canvas), out_canvas):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")

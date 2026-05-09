# Variant of 6ffbe589
# Dimension: D7 — 旋转方向 (transform)
# Change: 固定顺时针旋转(CW)，rot_k = (-cue_size) % 4... → 方向由提示精灵位置决定

# Design analysis for this D7 variant
# - Concrete behavior:
#   The board is still a cropped square with three concentric color layers. Each rotating color still has
#   one same-color cue outside the board, and cue area still encodes the number of quarter turns.
#   The new observable change is that cue side now matters:
#   top/right cue -> rotate that color clockwise; bottom/left cue -> rotate it counterclockwise.
#   Example with cue area 2: original task always applies 180 degrees so direction is irrelevant, but
#   with cue area 1 or 3 the input/output pair now differs visibly depending on whether the cue is above
#   or below the board. Therefore the generator should favor quarter-turn magnitudes where direction is
#   visible, while still allowing all non-zero magnitudes.
# - Complexity calibration:
#   Direction is limited to four visible side categories, but they collapse into two behavioral groups
#   (CW vs CCW). This keeps the rule learnable while still forcing the solver to combine two cues:
#   cue size for angle and cue position for direction. We keep one cue per rotating color to avoid
#   matching ambiguity, and we preserve the original three-layer structure so the task does not drift.
# - Implicit-assumption audit of main():
#   1) Cropping near the largest component still assumes every cue is at least margin+1 away from the
#      board; generator keeps Chebyshev gap >= 4 while crop margin stays 3.
#   2) The original code only needed cue area. The new code also needs a unique side classification.
#      Diagonal cues would make side ambiguous, so the generator must place each cue wholly above,
#      below, left, or right of the board while overlapping the orthogonal span of the board.
#   3) Sequential per-color rotation remains safe only if inverse-rotated input layers are disjoint;
#      generator still constructs output first, then applies the exact inverse transform per color.
# - Placement legality:
#   A 20x20 canvas with a 10..13 square board cannot reliably support cues on all four sides with the
#   required gap. To keep the original size range, the board is anchored in a canvas corner. This leaves
#   exactly two long cue corridors on the opposite sides, enough for up to three cues without overlap.
# - Visual observability and cleanliness:
#   Side-specific corridors avoid diagonal placements and keep cues visually separate from the board.
#   Rotating layers remain rotationally asymmetric whenever they get a cue, so CW vs CCW is visible.
#   Colors are unchanged; only geometry changes, so there is no hidden color-id logic.
# - New assumptions after the change:
#   main() now assumes each cue is side-aligned and not diagonal. generate_input() guarantees this with
#   a side-constrained placer and a final oracle check. It also balances cue usage across the two
#   available sides when multiple colors rotate, reducing accidental degeneration into one-side-only data.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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


def _bbox_overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    """True iff two inclusive 1D intervals overlap."""
    return not (a1 < b0 or b1 < a0)


def _cue_side(cue_bbox: tuple[int, int, int, int], board_bbox: tuple[int, int, int, int]) -> str:
    """Classify a cue as top/bottom/left/right relative to the board, rejecting diagonal ambiguity."""
    qr0, qc0, qr1, qc1 = map(int, cue_bbox)
    br0, bc0, br1, bc1 = map(int, board_bbox)

    if qr1 < br0 and _bbox_overlap_1d(qc0, qc1, bc0, bc1):
        return "top"
    if qr0 > br1 and _bbox_overlap_1d(qc0, qc1, bc0, bc1):
        return "bottom"
    if qc1 < bc0 and _bbox_overlap_1d(qr0, qr1, br0, br1):
        return "left"
    if qc0 > bc1 and _bbox_overlap_1d(qr0, qr1, br0, br1):
        return "right"
    raise AssertionError("expected each cue to be aligned to exactly one board side")


def _side_is_clockwise(side: str) -> bool:
    """Top/right encode clockwise; bottom/left encode counterclockwise."""
    assert side in {"top", "right", "bottom", "left"}
    return side in {"top", "right"}


def _rotation_from_cue(cue_comp: dict, board_bbox: tuple[int, int, int, int]) -> int:
    """Return np.rot90 quarter-turn count for a cue component."""
    cue_size = int(cue_comp["area"]) % 4
    if cue_size == 0:
        return 0
    side = _cue_side(tuple(map(int, cue_comp["bbox"])), board_bbox)
    return (-cue_size) % 4 if _side_is_clockwise(side) else cue_size


def main(grid: np.ndarray) -> np.ndarray:
    """
    Isolate the board and rotate each color layer using cue size plus cue side.

    Input structure:
    - The background is the most common color in the grid.
    - The largest foreground component is a square multi-color board made of concentric layers.
    - Small monochrome cues of matching colors sit outside the board, aligned to one board side.

    Rule:
    - Crop the board cluster away from the distant cues.
    - For each color present in the board crop, locate the outside cue of that same color.
    - The cue area modulo 4 gives the number of quarter turns.
    - A cue above or to the right means rotate clockwise; a cue below or to the left means rotate
      counterclockwise. If a color has no cue, that layer stays fixed.

    Key insight:
    - Each cue carries two independent pieces of information: size sets the angle, and side sets the
      direction.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    board_bbox = (int(r0), int(c0), int(r1), int(c1))

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
        rot_k = _rotation_from_cue(outside_comps[0], board_bbox) if outside_comps else 0
        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


# --------------- generator (D7 variant) ---------------

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


def _make_asym_layer(n: int, density_lo: float, density_hi: float, extra_lo_terms: list[int] = ()) -> np.ndarray | None:
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


def _pick_turns() -> list[int]:
    """Pick non-directional quarter-turn counts for 3 layers; at least one must be non-zero."""
    turns = [random.randrange(4) for _ in range(3)]
    if not any(turns):
        turns[random.randrange(3)] = random.randint(1, 3)
    return turns


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


def _corner_anchor_and_sides(n: int) -> tuple[tuple[int, int], tuple[str, str]]:
    """Pick a corner anchor and return the two side corridors available for cues."""
    spare = CANVAS_SIZE - n
    corner = random.randrange(4)
    if corner == 0:
        return (0, 0), ("bottom", "right")
    if corner == 1:
        return (0, spare), ("bottom", "left")
    if corner == 2:
        return (spare, 0), ("top", "right")
    return (spare, spare), ("top", "left")


def _place_cue_on_side(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
    side: str,
) -> bool:
    """Place a cue patch on a specified board side with orthogonal-span overlap."""
    h, w = patch.shape
    mask = patch != 0
    br0, bc0, br1, bc1 = map(int, board_bbox)
    candidates = []

    if side == "top":
        r_vals = range(0, br0 - h - CUE_BOARD_DIST + 1)
        c_min = max(0, bc0 - w + 1)
        c_max = min(CANVAS_SIZE - w, bc1)
        c_vals = range(c_min, c_max + 1)
    elif side == "bottom":
        r_vals = range(br1 + CUE_BOARD_DIST, CANVAS_SIZE - h + 1)
        c_min = max(0, bc0 - w + 1)
        c_max = min(CANVAS_SIZE - w, bc1)
        c_vals = range(c_min, c_max + 1)
    elif side == "left":
        c_vals = range(0, bc0 - w - CUE_BOARD_DIST + 1)
        r_min = max(0, br0 - h + 1)
        r_max = min(CANVAS_SIZE - h, br1)
        r_vals = range(r_min, r_max + 1)
    elif side == "right":
        c_vals = range(bc1 + CUE_BOARD_DIST, CANVAS_SIZE - w + 1)
        r_min = max(0, br0 - h + 1)
        r_max = min(CANVAS_SIZE - h, br1)
        r_vals = range(r_min, r_max + 1)
    else:
        raise ValueError(f"unknown side {side}")

    for r in r_vals:
        for c in c_vals:
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            try:
                if _cue_side(bb, board_bbox) != side:
                    continue
            except AssertionError:
                continue
            candidates.append((r, c, bb))

    if not candidates:
        return False

    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _area_for_turns(turns: int) -> int:
    """Choose a cue area whose modulo-4 class encodes the requested turn count."""
    t = int(turns) % 4
    assert t != 0
    if t == 1:
        return random.choice([1, 5])
    return t


def _assign_cue_sides(rotating_cols: list[int], available_sides: tuple[str, str]) -> dict[int, str]:
    """Distribute cues across the two legal sides, encouraging both sides when multiple cues exist."""
    cols = list(rotating_cols)
    random.shuffle(cols)
    side_a, side_b = available_sides
    assignments: dict[int, str] = {}

    if len(cols) == 1:
        assignments[cols[0]] = random.choice([side_a, side_b])
        return assignments

    mixed_dirs = _side_is_clockwise(side_a) != _side_is_clockwise(side_b)
    seeded_sides = [side_a, side_b] if mixed_dirs else random.sample([side_a, side_b], 2)
    assignments[cols[0]] = seeded_sides[0]
    assignments[cols[1]] = seeded_sides[1]
    for idx, col in enumerate(cols[2:]):
        assignments[col] = [side_a, side_b][idx % 2]
    return assignments


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    turns = _pick_turns()

    valid_ns = [n for n in range(BOARD_N_MIN, BOARD_N_MAX + 1) if _valid_inner_spans(n, turns[2] > 0)]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n)
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, require_asym=turns[2] > 0)
    if turns[2] > 0 and not _is_rot_asym(inner_mask):
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD)
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    for col, mask in zip(colors, [outer_mask, mid_mask, inner_mask]):
        out_board[mask] = col

    (r0, c0), available_sides = _corner_anchor_and_sides(n)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    rotating_cols = [col for col, t in zip(colors, turns) if t != 0]
    side_by_color = _assign_cue_sides(rotating_cols, available_sides)
    signed_rotations = {}
    for col, t in zip(colors, turns):
        if t == 0:
            signed_rotations[col] = 0
            continue
        side = side_by_color[col]
        signed_rotations[col] = (-t) % 4 if _side_is_clockwise(side) else t

    in_board = np.zeros((n, n), dtype=int)
    for col in colors:
        k = int(signed_rotations[col]) % 4
        inv = np.rot90(out_board == col, (-k) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = col

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_specs = []
    for col, t in zip(colors, turns):
        if t == 0:
            continue
        cue_specs.append((col, _area_for_turns(t), side_by_color[col]))
    random.shuffle(cue_specs)

    for col, area, side in cue_specs:
        if not _place_cue_on_side(canvas, _cue_patch(col, area), board_bbox, cue_bboxes, side):
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
        if _cue_side(bb, board_bbox) not in available_sides:
            return None

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 D7 input")

# Variant of aa4ec2a5
# Dimension: D4 — 轮廓颜色规则 (transform)
# Change: 所有对象（无论是否有洞）统一使用红色轮廓... → 轮廓颜色由对象类别决定

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the D4 variant
# ----------------------------------
# spec:
#   preserved_input:
#     grid: 15-28 rows x 15-27 cols, uniform yellow background (4)
#     objects: 3-4 blue 4-connected components, with 1-2 hollow and at least 1 solid
#     spacing: pairwise Chebyshev distance >= 3, border clearance >= 1
#   changed_transform:
#     hollow_component:
#       recolor component body to teal (8)
#       recolor enclosed hole cells to magenta (6)
#       draw 1-pixel 8-neighborhood outline in red (2)
#     solid_component:
#       keep component body blue (1)
#       draw 1-pixel 8-neighborhood outline in orange (7)
#   behavior_trace:
#     example: a solid 2x2 blue block used to get a red ring in the original task;
#       now it keeps the same geometry but its ring is orange.
#       A hollow rectangle with one enclosed hole still becomes teal with a magenta
#       cavity, and its ring stays red. The geometric processing is unchanged; only
#       the component-class -> outline-color mapping changes.
#   complexity_calibration:
#     - The rule only uses two outline colors, one per visually detectable class.
#     - Every generated grid contains both hollow and solid components, so both
#       outline colors repeat within a single example instead of appearing as a
#       one-off coincidence.
#   implicit_assumption_review:
#     - Hole detection via bbox-local 4-connected background regions remains valid.
#     - The outline must still be computed from the "solidified" hollow shape that
#       includes the hole cells; otherwise the ring would leak into the cavity.
#     - The original `out[outline & (out == bg)] = 2` was harmless with one shared
#       outline color. After introducing two colors, overlapping outlines would make
#       write order observable. We therefore preserve and rely on the generator's
#       Chebyshev >= 3 spacing so outline masks stay disjoint.
#   placement_legality:
#     - The original placement rule already leaves at least one empty cell between
#       distinct 8-neighborhood outlines, so red and orange borders never collide.
#     - Border clearance >= 1 is still enough because an outline may touch the grid
#       edge but is clipped safely by `_outline_mask`.
#   visual_observability:
#     - Hollow objects are identifiable from their teal body and magenta hole, which
#       provides the structural cue explaining why their border is red.
#     - Solid objects stay blue, so the contrasting orange border is easy to compare
#       against the red border in the same output.
#   cleanliness:
#     - No new overlays are introduced inside objects; only the border color changes.
#     - The palette remains small and structured: blue/yellow input, then blue/teal/
#       magenta interiors plus orange/red outlines in the output.


RED_OUTLINE = 2
ORANGE_OUTLINE = 7
MAGENTA_HOLE = 6
TEAL_HOLLOW = 8


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == sub.shape[0] - 1)
            | (hole[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = hole.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw class-dependent outlines around components and mark enclosed holes.

    Input structure:
    - A uniform yellow background contains several blue 4-connected foreground objects.
    - Some objects enclose 4-connected background holes strictly inside their bbox.

    Rule:
    - For every foreground component, compute a one-cell 8-neighborhood outline on background cells.
    - If the component contains enclosed holes, recolor its body to teal (8), recolor each hole
      to magenta (6), and draw its outline in red (2).
    - If the component has no enclosed holes, leave it blue (1) and draw its outline in orange (7).

    Key insight:
    - The border color is determined by the same hollow-vs-solid classification that governs the
      interior recoloring, and hollow objects must be outlined as if their holes were filled in.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        outline_color = ORANGE_OUTLINE
        if holes:
            outline_color = RED_OUTLINE
            out[mask] = TEAL_HOLLOW
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA_HOLE
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = outline_color

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

# Component counts
COMP_COUNT_RANGE = (3, 4)  # total components per grid
HOLLOW_COUNT_RANGE = (1, 2)  # how many are hollow (capped by total - 1)

# Polyomino construction (solid shapes)
POLY_NUM_RECTS_RANGE = (2, 3)  # overlapping rectangles per polyomino
POLY_RECT_DIM_RANGE = (2, 7)  # h/w range for each rectangle
POLY_CANVAS_PAD = 4  # extra canvas margin during construction
POLY_PLACEMENT_RETRIES = 50  # retries for overlapping subsequent rects
POLY_GEN_RETRIES = 30  # retries to make one valid polyomino

# Small solid shapes
SMALL_SHAPE_PROB = 0.2  # probability of choosing a small (1-4 cell) solid
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Hollow component construction
HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]  # allowed rectangular hole areas
HOLE_DIM_MAX = 4  # max dimension of hole rectangle
HOLLOW_PAD_RANGE = (1, 3)  # padding around hole on each side
HOLLOW_MAX_NOTCHES = 2  # 0-2 corner notches
HOLLOW_EXTENSION_PROB = 0.5  # probability of adding an L/T extension
HOLLOW_EXT_H_RANGE = (1, 4)  # height range for extension
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)  # width range for left/right extensions
HOLLOW_HOLE_SEARCH_RETRIES = 200  # retries to find a valid hole position
HOLLOW_GEN_RETRIES = 50  # retries to build one hollow component
HOLLOW_MIN_AREA = 10  # minimum foreground cells for hollow component

# Placement
CHEBYSHEV_MIN = 3  # minimum Chebyshev distance between any two component cells
BORDER_MIN = 1  # minimum clearance from grid edge
PLACEMENT_RETRIES = 300  # random placement attempts per component

# Top-level generation
GEN_OUTER_RETRIES = 300  # retries for the whole generation


# ── Helpers ──────────────────────────────────────────────────────────────────

def _crop_mask(mask):
    """Crop a boolean mask to its tight bounding box. Returns None if empty."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_interior_hole(mask):
    """Return True if the boolean mask contains at least one 4-connected interior bg region."""
    h, w = mask.shape
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not touches_edge:
            return True
    return False


def _rect_dims(area):
    """Return (h, w) for a rectangle with given area, randomized among valid factorizations."""
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= HOLE_DIM_MAX and area // i <= HOLE_DIM_MAX
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
    """Return True if any cell in (rows_a, cols_a) is within CHEBYSHEV_MIN of any cell in pm
    placed at (pr, pc). Uses a bbox pre-filter for speed."""
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _variant_visible(out: np.ndarray) -> bool:
    """Ensure each sample exhibits both outline classes and both interior classes."""
    colors = {int(v) for v in out.flat}
    return {
        BG_COLOR,
        FG_COLOR,
        RED_OUTLINE,
        ORANGE_OUTLINE,
        MAGENTA_HOLE,
        TEAL_HOLLOW,
    }.issubset(colors)


# ── Shape generators ─────────────────────────────────────────────────────────

def _make_polyomino():
    """Generate an irregular polyomino from overlapping axis-aligned rectangles.
    Returns a boolean mask (tight bounding box), or None on failure."""
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE))
        for _ in range(num_rects)
    ]

    canvas_h = sum(r[0] for r in rects) + POLY_CANVAS_PAD
    canvas_w = sum(r[1] for r in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for i in range(1, num_rects):
        rh, rw = rects[i]
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), r0 + rects[0][0] + 2)
            nc = random.randint(max(0, c0 - rw + 1), c0 + rects[0][1] + 2)
            if nr + rh > canvas_h or nc + rw > canvas_w or nr < 0 or nc < 0:
                continue
            if mask[nr : nr + rh, nc : nc + rw].any():
                mask[nr : nr + rh, nc : nc + rw] = True
                break

    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _extend_mask(mask, base_h, base_w):
    """Optionally extend mask with a rectangle on one side to create L/T shapes.
    Returns the (potentially enlarged) mask."""
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)

    if side in ("top", "bottom"):
        ext_w = min(random.randint(2, max(2, base_w)), base_w)
        offset = random.randint(0, base_w - ext_w)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[0:ext_h, offset : offset + ext_w] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h : base_h + ext_h, offset : offset + ext_w] = True
    else:
        ext_h_actual = min(ext_h, base_h)
        ext_w_actual = random.randint(*HOLLOW_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, 0:ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _carve_hole(mask, hh, hw):
    """Try to carve a hh x hw rectangular hole in mask's interior.
    Returns cropped mask with hole, or None on failure."""
    mh, mw = mask.shape
    for _ in range(HOLLOW_HOLE_SEARCH_RETRIES):
        hr = random.randint(1, mh - hh - 1)
        hc = random.randint(1, mw - hw - 1)
        if not mask[hr : hr + hh, hc : hc + hw].all():
            continue
        if (
            not mask[hr - 1, hc : hc + hw].any()
            or not mask[hr + hh, hc : hc + hw].any()
            or not mask[hr : hr + hh, hc - 1].any()
            or not mask[hr : hr + hh, hc + hw].any()
        ):
            continue

        carved = mask.copy()
        carved[hr : hr + hh, hc : hc + hw] = False
        if not is_contiguous(carved.astype(int), background=0, connectivity=4):
            continue
        cropped = _crop_mask(carved)
        if cropped is not None and _has_interior_hole(cropped):
            return cropped
    return None


def _make_hollow_component(target_hole_area):
    """Build a hollow component with a rectangular hole of the given area.
    Returns a boolean mask (tight bounding box), or None on failure."""
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, HOLLOW_MAX_NOTCHES)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        nh = random.randint(
            1, max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1)
        )
        nw = random.randint(
            1, max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1)
        )
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w)
    return _carve_hole(mask, hh, hw)


def _make_small_shape():
    """Return a small (1-4 cell) solid polyomino mask."""
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return np.ones((1, 2), dtype=bool) if random.random() < 0.5 else np.ones((2, 1), dtype=bool)
    if size == 3:
        return random.choice(
            [
                np.array([[1, 1], [1, 0]], dtype=bool),
                np.array([[1, 0], [1, 1]], dtype=bool),
                np.array([[1, 1, 1]], dtype=bool),
                np.array([[1], [1], [1]], dtype=bool),
            ]
        )
    return np.ones((2, 2), dtype=bool)


# ── Orchestration ────────────────────────────────────────────────────────────

def _gen_attempt():
    """Single attempt to generate a valid input grid."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    components = []
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and mask.sum() >= HOLLOW_MIN_AREA:
                    components.append(mask)
                    found = True
                    break
            if not found:
                return None
        elif random.random() < SMALL_SHAPE_PROB:
            components.append(_make_small_shape())
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append(mask)
                    break
            else:
                return None

    if len(components) != num_components:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, pm in placed:
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    if not _variant_visible(out):
        return None

    return grid


def generate_input():
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

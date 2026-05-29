# Combo variant of aa4ec2a5 (index 1)
# combo_spec: D1/0+D4/0
# Name: 4邻域轮廓+按类别着色轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   source_task:
#     - Input remains a yellow background (4) with 3-4 blue 4-connected objects.
#     - Exactly 1-2 objects are hollow in the structural sense: inside the component bbox there is
#       a 4-connected background region that does not touch that bbox boundary.
#     - Hollow objects are still recolored to teal (8) with magenta (6) holes, and holes still
#       count as filled when computing the exterior outline.
#   fused_changes:
#     - D1/0 changes the outline geometry from the full 8-neighborhood ring to a 4-neighborhood
#       orthogonal ring.
#     - D4/0 changes the outline color rule from "all red" to "red iff hollow, orange iff solid".
#     - The final rule is therefore not just "use a different color" or "drop diagonal corners":
#       it is "classify each object by hole status, then assign both a contour color and the
#       orthogonal contour geometry from that class-independent neighborhood rule".
#   concrete_trace:
#     - Take a 2x2 solid blue block. In the source task its output keeps the blue block and paints
#       the whole surrounding 4x4 frame red, including the four diagonal outer corners.
#       In this combo task the same block keeps its blue body but gets an orange orthogonal ring,
#       so those four diagonal corner cells stay yellow.
#     - Take a hollow rectangle with one interior rectangular hole. In the source task its body
#       becomes teal, the hole becomes magenta, and the outside ring is fully red in 8-neighborhood
#       style. In this combo task the teal/magenta recoloring stays, the ring stays red because the
#       object is hollow, but the diagonal exterior corner cells are no longer painted.
#     - Therefore the solver must infer two linked differences at once:
#       1. outline growth uses only up/down/left/right adjacency;
#       2. red vs orange depends on whether the object contains an enclosed hole.
#   complexity_calibration:
#     - The color rule uses only two outline colors, which keeps the mapping learnable from a few
#       examples instead of turning into unconstrained recoloring.
#     - Each generated sample contains both hollow and solid components, so both outline colors
#       appear repeatedly in the same grid.
#     - The generator also rejects samples where the orthogonal outline accidentally looks identical
#       to the old 8-neighborhood outline, so the geometry change remains visually obvious.
#   main_assumption_audit:
#     - `find_components(... connectivity=4, monochromatic=True)` still holds because inputs remain
#       blue 4-connected components on a dominant yellow background.
#     - `_interior_holes()` still works unchanged because hole definition is structural and does not
#       depend on outline color or neighborhood choice.
#     - The old assumption "all outlines share one color" is no longer valid; color is now chosen
#       per component from its hole classification.
#     - The old assumption "outline is built from all 8 offsets" is also invalid; diagonal writes
#       would violate the new task, so the outline helper must use only 4 directions.
#     - Writing with `out[outline & (out == bg)] = color` remains valid only if outlines never
#       overlap. The generator keeps the stronger original Chebyshev spacing so write order does not
#       become observable.
#   placement_legality:
#     - Original placement already guarantees component cells are pairwise Chebyshev-separated by at
#       least 3 and remain at least 1 cell from the border.
#     - This is stronger than the combo task needs, but it safely prevents both 4-neighborhood
#       outline collisions and any ambiguity between red and orange outline regions.
#     - Border clearance 1 is still sufficient because the 1-pixel orthogonal outline may touch the
#       canvas edge but `_outline_mask` clips out-of-bounds coordinates.
#   visual_observability:
#     - Hollow objects advertise their class directly through teal shell + magenta cavity, so the
#       reason for their red outline is human-visible.
#     - Solid objects stay blue and get orange outlines, creating an immediate same-grid comparison.
#     - Orthogonal outlines are also human-visible because diagonal corner cells around rectangles,
#       elbows, and endpoints stay yellow instead of being colored.
#   cleanliness:
#     - The palette stays compact: yellow background, blue solids, teal/magenta hollow interiors,
#       red hollow outlines, orange solid outlines.
#     - There is no multi-layer overlap inside one local patch because outlines write only onto
#       background and component spacing prevents border collisions.


RED_OUTLINE = 2
MAGENTA_HOLE = 6
ORANGE_OUTLINE = 7
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


def _outline_mask(mask: np.ndarray, *, orthogonal_only: bool) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] if orthogonal_only else [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for dr, dc in directions:
        rr = rows + dr
        cc = cols + dc
        ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _transform(
    grid: np.ndarray, *, orthogonal_only: bool, class_colored: bool
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        outline_color = RED_OUTLINE
        if holes:
            out[mask] = TEAL_HOLLOW
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA_HOLE
                solid[hole[:, 0], hole[:, 1]] = True
        elif class_colored:
            outline_color = ORANGE_OUTLINE

        outline = _outline_mask(solid, orthogonal_only=orthogonal_only)
        out[outline & (out == bg)] = outline_color

    return out


def _orthogonal_delta_count(grid: np.ndarray) -> int:
    orthogonal = _transform(grid, orthogonal_only=True, class_colored=True)
    diagonal = _transform(grid, orthogonal_only=False, class_colored=True)
    return int(np.count_nonzero(orthogonal != diagonal))


def _legacy_delta_count(grid: np.ndarray) -> int:
    combo = _transform(grid, orthogonal_only=True, class_colored=True)
    legacy = _transform(grid, orthogonal_only=False, class_colored=False)
    return int(np.count_nonzero(combo != legacy))


def _variant_visible(out: np.ndarray) -> bool:
    colors = {int(v) for v in out.flat}
    return {
        BG_COLOR,
        FG_COLOR,
        RED_OUTLINE,
        ORANGE_OUTLINE,
        MAGENTA_HOLE,
        TEAL_HOLLOW,
    }.issubset(colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects and draw class-dependent orthogonal outlines around every component.

    Input structure:
    - A uniform yellow background contains several blue 4-connected foreground components.
    - Some components enclose 4-connected background holes strictly inside their bounding boxes.

    Rule:
    - For each foreground component, build a one-cell outline only through the 4-neighborhood.
    - If a component encloses at least one hole, recolor its body to teal (8), recolor each hole
      to magenta (6), and draw its outline in red (2).
    - If a component has no enclosed hole, keep it blue (1) and draw its outline in orange (7).

    Key insight: Hole status controls the outline color, but every outline uses the same
    orthogonal growth rule; hollow components must still be outlined as if their holes were filled.
    """
    return _transform(grid, orthogonal_only=True, class_colored=True)


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

# Component counts
COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

# Polyomino construction (solid shapes)
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Small solid shapes
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Hollow component construction
HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

# Placement
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

# Top-level generation
GEN_OUTER_RETRIES = 300

# Non-degeneracy thresholds
MIN_ORTHOGONAL_DELTA = 4


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_interior_hole(mask: np.ndarray) -> bool:
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


def _rect_dims(area: int) -> tuple[int, int] | None:
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


def _chebyshev_conflict(
    rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray
) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
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


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
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


def _carve_hole(mask: np.ndarray, hh: int, hw: int) -> np.ndarray | None:
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


def _make_hollow_component(target_hole_area: int) -> np.ndarray | None:
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


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        if random.random() < 0.5:
            return np.ones((1, 2), dtype=bool)
        return np.ones((2, 1), dtype=bool)
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


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(
        HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1)
    )

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
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
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))

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
    if _orthogonal_delta_count(grid) < MIN_ORTHOGONAL_DELTA:
        return None
    if _legacy_delta_count(grid) <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

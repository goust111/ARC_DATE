# Combo variant of aa4ec2a5 (index 36)
# combo_spec: D12/0+D15/1
# Name: 减少颜色种类+仅角落像素轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The source task does two things to each blue component: hollow components are
#    recolored with two interior colors (teal shell + magenta hole fill), and all
#    components receive a full red 8-neighborhood ring. This combo keeps the hole
#    detection logic but merges the two hollow-related colors into one and also
#    sparsifies the outline.
#    Small trace:
#      - Original 3x3 ring with a 1-cell center hole:
#          input: blue ring around a yellow center.
#          output: ring becomes teal, center becomes magenta, and the whole
#          3x3-plus-halo gets a continuous red border.
#      - Combo version:
#          input: unchanged.
#          output: both ring cells and the enclosed center become teal, so the
#          hollow object turns into one filled teal mass; red marks appear only at
#          diagonal corner positions around that filled mass.
#      - Difference:
#          1. Hollow interiors are still special, but no longer color-distinct
#             from their enclosing object.
#          2. Orthogonally adjacent outline pixels disappear; only diagonal corner
#             pixels remain.
# b) Complexity calibration:
#    The task still contains 3-4 components with 1-2 hollow ones, so samples
#    simultaneously show blue solids, teal formerly-hollow objects, and sparse red
#    corner marks. That keeps the color simplification visible without collapsing
#    the task into a trivial "all holes just vanish" rule, because only components
#    that truly enclose background regions get recolored.
# c) Original-code assumptions affected by the combo:
#      - `_interior_holes(...)` still relies on 4-connected interior background
#        regions inside a component bbox. This remains valid and is now even more
#        important because color can no longer distinguish shell vs filled hole.
#      - `solid[hole] = True` is still mandatory. Without it, former hole corners
#        would generate interior red marks, contradicting the intended "fill first,
#        then outline the merged mass" behavior.
#      - `out[mask] = 8` and `out[hole] = 8` intentionally assign the same color.
#        The solver therefore assumes spatial containment, not output color, is
#        what identifies a hole-bearing component.
#      - `_corner_outline_mask(...)` assumes red marks should be background cells
#        with diagonal contact but no orthogonal contact to the merged mass. That
#        is a stricter subset of the original outline and does not alter component
#        detection on the input.
# d) Placement legality:
#    The original placement rule Chebyshev >= 3 is still sufficient. Corner-only
#    outlines occupy fewer cells than the original halo, so any valid original
#    placement stays non-overlapping here. Border clearance >= 1 also remains
#    enough because the new outline reaches at most one cell outward.
# e) Visual observability:
#    A solver can infer the combo from a few examples because two stable signals
#    repeat together:
#      - hollow inputs lose their internal yellow hole and become solid teal;
#      - red marks appear only at diagonal corner positions, never along straight
#        side-adjacent border runs.
#    The generator preserves both hollow and solid objects so the conditional
#    recoloring is observable within the same sample.
# f) Visual cleanliness:
#    Output uses only four visible roles: yellow background, blue solid objects,
#    teal filled formerly-hollow objects, and red corner marks. Since red is still
#    restricted to background cells and components are well separated, the grid
#    stays legible despite mixing recoloring with outlining.
# g) Change inferability:
#    Merging magenta into teal reduces the color vocabulary rather than expanding
#    it, which helps inference. The lost color distinction is compensated by a
#    strong structural cue: only enclosed background regions inside a blue object
#    get filled teal together with that object, while exterior background remains
#    yellow and may carry red corner marks.
# h) Generator anti-degeneracy checks applied below:
#      - At least one hollow component is always generated, so the color-merging
#        rule is exercised on every sample.
#      - Generated samples are filtered so the corner-only red count is strictly
#        smaller than the full-outline red count, guaranteeing D15 is active.
# i) New solver assumptions guaranteed by generation:
#      - Background remains the majority color so `get_bg(...)` recovers yellow.
#      - All input foreground components are monochromatic blue, so
#        `find_components(..., monochromatic=True)` stays valid.
#      - Hollow components are truly enclosed and remain large enough for hole
#        filling to be visually noticeable.
#      - There is at least one solid component as a contrast class against the
#        teal hollow-derived components.


TEAL_FILL = 8
OUTLINE_COLOR = 2


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


def _corner_outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    diagonal = np.zeros_like(mask, dtype=bool)
    orthogonal = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)

    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr = rows + dr
        cc = cols + dc
        ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        diagonal[rr[ok], cc[ok]] = True

    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = rows + dr
        cc = cols + dc
        ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        orthogonal[rr[ok], cc[ok]] = True

    outline = diagonal & ~orthogonal
    outline &= ~mask
    return outline


def _full_outline_mask(mask: np.ndarray) -> np.ndarray:
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
    Fill hole-bearing objects with one shared interior color, then add red corner-only marks.

    Input structure:
    - A mostly uniform yellow background contains several 4-connected blue foreground objects.
    - Some blue objects enclose fully bounded 4-connected background holes inside their bbox.

    Rule:
    - For every blue component, compute its enclosed holes, if any.
    - If a component has holes, recolor both the component cells and those enclosed hole cells to teal (8).
    - Around each resulting solid mass, paint only background cells that touch it diagonally but not orthogonally in red (2).

    Key insight: Hollow objects must still be detected from spatial enclosure, but after
    detection the shell and its interior hole fill collapse to the same teal color;
    the red annotation is also reduced from a full halo to corner-only diagonal marks.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = TEAL_FILL
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = TEAL_FILL
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _corner_outline_mask(solid)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


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
        return (
            np.ones((1, 2), dtype=bool)
            if random.random() < 0.5
            else np.ones((2, 1), dtype=bool)
        )
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


def _component_hole_count(grid: np.ndarray) -> int:
    bg = get_bg(grid)
    count = 0
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        if _interior_holes(mask, coords):
            count += 1
    return count


def _outline_counts(grid: np.ndarray) -> tuple[int, int]:
    bg = get_bg(grid)
    corner_count = 0
    full_count = 0
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        corner_count += int(_corner_outline_mask(solid).sum())
        full_count += int(_full_outline_mask(solid).sum())
    return corner_count, full_count


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
    placed: list[tuple[int, int, np.ndarray]] = []

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

    hollow_count = _component_hole_count(grid)
    if hollow_count < 1 or hollow_count >= num_components:
        return None

    corner_count, full_count = _outline_counts(grid)
    if corner_count <= 0 or corner_count >= full_count:
        return None

    if 6 in set(map(int, out.flatten())):
        return None
    if TEAL_FILL not in set(map(int, out.flatten())):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

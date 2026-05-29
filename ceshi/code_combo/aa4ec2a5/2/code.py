# Combo variant of aa4ec2a5 (index 2)
# combo_spec: D1/2+D5/0
# Name: 内轮廓+洞保持背景色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# source_task: aa4ec2a5
# combo_spec: D1/2 + D5/0
# a) concretization_and_trace:
#    - Input remains the original family: a yellow background with 3-4 blue 4-connected components,
#      including 1-2 hollow ones that enclose rectangular background holes.
#    - Original behavior on a hollow object was: recolor the shell teal, recolor the hole magenta,
#      then paint a red halo outside the object. The object itself kept its full size.
#    - This combo changes both altered stages together:
#      1. the contour flips inward, so red now overwrites object-side boundary pixels rather than
#         surrounding background cells;
#      2. the hole is no longer recolored, so it stays yellow like the global background.
#    - Concrete trace on a thick hollow rectangle:
#      original -> teal shell + magenta hole + red exterior halo;
#      combo -> outer shell cells become red, inner shell cells stay teal, and the enclosed hole
#      stays yellow. The visible structure becomes "red inner frame / teal body / yellow cavity".
#    - Concrete trace on a solid 3x3 block:
#      original -> a blue 3x3 block with a red ring outside it;
#      combo -> the outer 8 object cells become red and only the center stays blue.
#      That makes the contour effect a shrink, not a surrounding halo.
# b) complexity_calibration:
#    - Keeping the hole yellow removes the old magenta cue, so the generator must not rely on tiny or
#      overly thin hollow shapes; otherwise the hollow rule would be easy to miss or collapse to all-red.
#    - To keep the fused rule readable from a few examples, generation now requires:
#      at least one hollow component with both a preserved hole and a surviving teal shell core, plus
#      at least one solid component with a surviving blue core after the inner contour.
#    - This preserves repeated evidence for both dimensions: inward red erosion and hole preservation.
# c) implicit_assumption_audit_for_main:
#    - _interior_holes still assumes holes are detectable as 4-connected background regions strictly
#      inside the component bbox; the original generator already guarantees that, so this remains valid.
#    - The original _outline_mask assumed red is written only onto background cells. That assumption is
#      invalid here because the contour is now internal, so the helper must return object cells instead.
#    - Hollow objects must still be temporarily solidified for contour detection by filling their holes
#      in a boolean mask. Without that step, shell cells touching the yellow cavity would turn red and
#      create a second inner ring around the hole, which contradicts the intended combo behavior.
#    - Even though holes are filled in the temporary solid mask, final red writes must be intersected
#      with the original component mask, otherwise the preserved yellow hole would be overwritten.
#    - Because holes stay background-colored, main() must detect components from the input grid before
#      any recoloring; iterating over input components directly preserves that assumption.
# d) placement_legality:
#    - Input placement constraints stay compatible with the variant: components are still 4-connected,
#      mutually separated by Chebyshev >= 3, and at least one cell from the border.
#    - Since red is drawn inside objects rather than outside them, the original spacing is more than
#      sufficient to prevent interactions between transformed components.
#    - Hollow construction still guarantees fully enclosed rectangular holes, so preserved yellow holes
#      remain unambiguous cavities rather than leaks to the exterior.
# e) visual_observability:
#    - A solver should be able to infer the rule by seeing that red appears on object edges rather than
#      around them, while yellow holes remain untouched inside teal hollow objects.
#    - The generator therefore forces at least one visible three-layer hollow example and at least two
#      components with surviving non-red cores overall, so the inward erosion is not mistaken for a
#      blanket recolor to red.
# f) visual_cleanliness:
#    - The combo actually simplifies local color clutter compared with the source task: magenta is gone,
#      leaving red for inner boundaries, teal for hollow-object bodies, blue for solid interiors, and
#      yellow for both exterior background and preserved cavities.
#    - To avoid outputs dominated by degenerate all-red thin shapes, generation filters out cases with
#      too few surviving cores.
# g) color_change_inference:
#    - Color changes remain structural rather than arbitrary:
#      blue -> red only on solid-object boundary cells,
#      blue -> teal on hollow-object non-boundary cells,
#      yellow stays yellow inside holes.
#    - The preserved hole is inferable because it is enclosed by recolored shell pixels, not because it
#      receives a unique color. The generator explicitly supplies samples where that enclosure is visible.


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


def _inner_outline_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        if not padded[r : r + 3, c : c + 3].all():
            outline[r, c] = True
    return outline


def _component_visibility_stats(grid: np.ndarray, bg: int) -> list[dict[str, int | bool]]:
    stats: list[dict[str, int | bool]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        hole_cells = 0
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
            hole_cells += len(hole)

        inner = _inner_outline_mask(solid) & mask
        core = mask & ~inner
        stats.append(
            {
                "has_hole": bool(holes),
                "hole_cells": int(hole_cells),
                "inner_cells": int(np.count_nonzero(inner)),
                "core_cells": int(np.count_nonzero(core)),
            }
        )
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects to teal, keep their holes as background, and paint red on the inside edge.

    Input structure:
    - A uniform yellow background contains several blue 4-connected components.
    - Some components enclose 4-connected background holes that are fully inside the component bbox.

    Rule:
    - For every component, paint its one-cell 8-neighborhood inner contour red (2): object cells that
      touch the exterior of the component's solid shape become red.
    - If a component contains enclosed holes, recolor the component itself to teal (8) first.
    - Those holes are not recolored; they remain background-colored while the red inner contour is
      applied only to original object cells.

    Key insight: Hollow objects are still treated as temporarily solid for contour detection, so the
    red erosion follows only the outer perimeter; the preserved hole stays yellow and does not gain
    its own red rim.
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
            out[mask] = 8
            for hole in holes:
                solid[hole[:, 0], hole[:, 1]] = True

        inner = _inner_outline_mask(solid) & mask
        out[inner] = 2

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
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

# Variant visibility
MIN_CORE_COMPONENTS = 2
MIN_SOLID_CORE_COMPONENTS = 1
MIN_HOLLOW_VISIBLE_COMPONENTS = 1

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


def _visibility_requirements_met(grid: np.ndarray) -> bool:
    stats = _component_visibility_stats(grid, BG_COLOR)
    core_components = sum(int(s["core_cells"] > 0) for s in stats)
    solid_core_components = sum(
        int((not s["has_hole"]) and (s["core_cells"] > 0)) for s in stats
    )
    hollow_visible_components = sum(
        int(s["has_hole"] and (s["core_cells"] > 0) and (s["hole_cells"] > 0)) for s in stats
    )
    return (
        core_components >= MIN_CORE_COMPONENTS
        and solid_core_components >= MIN_SOLID_CORE_COMPONENTS
        and hollow_visible_components >= MIN_HOLLOW_VISIBLE_COMPONENTS
    )


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

    if not _visibility_requirements_met(grid):
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

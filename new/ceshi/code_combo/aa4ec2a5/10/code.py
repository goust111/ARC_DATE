# Combo variant of aa4ec2a5 (index 10)
# combo_spec: D5/3+D15/0
# Name: 洞的独立轮廓+间断点外轮廓

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Keep the original object detection: blue 4-connected components sit on a yellow background.
#    - Hollow components still recolor their body to teal and their enclosed hole area to magenta.
#    - New inner-hole rule: before finishing the hole, overwrite the hole's 4-neighbor inner rim
#      with yellow, leaving only the strict interior of the hole magenta.
#    - New outer-outline rule: compute the usual 8-neighbor outer border on the hole-filled object,
#      but paint only checkerboard-parity cells in red so the outside border becomes dotted.
#    - Example trace: a 5x5 ring with a 3x3 hole used to become teal shell + full magenta 3x3
#      hole + full red outside outline. Now it becomes teal shell + yellow 8-cell hole rim +
#      magenta center cell + red dotted outer outline. The outer/inner borders differ by both
#      location and drawing style.
# b) Complexity calibration:
#    - The dotted outer border uses one global parity rule, so the sampling pattern repeats across
#      all objects and is visible from a few examples.
#    - The inner border is deterministic and local to holes, so the combined task introduces two
#      boundary systems without creating per-object exceptions or hidden ordering.
#    - Hole sizes are restricted to 3x3, 3x4, or 4x4 so the yellow rim and magenta core are both
#      visible; otherwise the hole-border rule could collapse into a full recolor.
# c) Implicit assumptions review:
#    - `_interior_holes(...)` still relies on holes being 4-connected background regions strictly
#      inside the component bbox; the generator preserves that.
#    - The original outer outline logic assumed every candidate outline cell should be painted.
#      That assumption is intentionally replaced by `_dashed_outline_mask(...)`.
#    - `solid |= hole_mask` remains essential: the red outer border must be computed on the
#      hole-filled body so it never appears inside the cavity.
#    - Writing the yellow hole rim before the red outline is safe because the red pass only paints
#      background cells in the external outline mask, never cells inside holes.
# d) Placement legality:
#    - The original Chebyshev >= 3 spacing already prevents overlap for a full solid outer outline.
#      The new outer border is sparser, and the yellow hole rim lives strictly inside objects, so
#      the same placement constraints remain sufficient.
# e) Visual observability:
#    - Every generated puzzle keeps 1-2 hollow components, and each hollow component has a visible
#      yellow ring plus magenta core. This makes the extra hole rule readable in 2-4 examples.
#    - The red dotted border is also forced to have enough cells and enough drop from the full
#      outline count, so the D15 change is visually obvious rather than incidental.
# f) Visual cleanliness:
#    - The two outline systems occupy disjoint regions: red dots only outside components on
#      background cells, yellow rim only inside holes. This avoids ambiguous overlaps.
# g) Color inference:
#    - The palette remains structured: blue = unchanged solid object, teal = hollow object body,
#      magenta = hole core, yellow = hole rim, red = outer dotted outline.
#    - Yellow matches the background color numerically, but it appears as a closed ring inside
#      hollow objects, so its role is still spatially inferable instead of arbitrary recoloring.
# h) New solver assumptions and generator guarantees:
#    - `main()` now assumes each generated hole is large enough to leave at least one magenta core
#      cell after the yellow rim is drawn.
#    - The generator checks both the dotted-outline metrics and the hole-rim/core metrics, so it
#      rejects samples that would degenerate back toward the original task.


BG_COLOR = 4
FG_COLOR = 1
OUTLINE_COLOR = 2
HOLE_INNER_OUTLINE_COLOR = 4
HOLE_FILL_COLOR = 6
HOLLOW_BODY_COLOR = 8
OUTLINE_KEEP_PARITY = 0


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


def _dashed_outline_mask(mask: np.ndarray) -> np.ndarray:
    outline = _outline_mask(mask)
    parity = (np.indices(mask.shape).sum(axis=0) % 2) == OUTLINE_KEEP_PARITY
    return outline & parity


def _inner_perimeter(mask: np.ndarray) -> np.ndarray:
    """Return True cells that touch the exterior by 4-neighborhood."""
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = rows + dr
        cc = cols + dc
        out_of_bounds = (rr < 0) | (rr >= h) | (cc < 0) | (cc >= w)
        if np.any(out_of_bounds):
            border[rows[out_of_bounds], cols[out_of_bounds]] = True
        ok = ~out_of_bounds
        if np.any(ok):
            exposed = ~mask[rr[ok], cc[ok]]
            border[rows[ok][exposed], cols[ok][exposed]] = True
    return border


def _component_hole_masks(shape: tuple[int, int], coords: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = _coords_mask(shape, coords)
    hole_masks = [_coords_mask(shape, hole) for hole in _interior_holes(mask, coords)]
    return mask, hole_masks


def _variant_structure_masks(grid: np.ndarray) -> dict[str, np.ndarray | int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dashed_outline = np.zeros(g.shape, dtype=bool)
    hole_rim = np.zeros(g.shape, dtype=bool)
    hole_core = np.zeros(g.shape, dtype=bool)
    full_outline_cells = 0
    hollow_components = 0

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        mask, hole_masks = _component_hole_masks(g.shape, comp["coords"])
        solid = mask.copy()

        if hole_masks:
            hollow_components += 1
        for hole_mask in hole_masks:
            rim = _inner_perimeter(hole_mask)
            hole_rim |= rim
            hole_core |= hole_mask & ~rim
            solid |= hole_mask

        full_outline_cells += int(_outline_mask(solid).sum())
        dashed_outline |= _dashed_outline_mask(solid)

    return {
        "dashed_outline": dashed_outline,
        "hole_rim": hole_rim,
        "hole_core": hole_core,
        "full_outline_cells": full_outline_cells,
        "hollow_components": hollow_components,
    }


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    masks = _variant_structure_masks(grid)
    return {
        "full_outline_cells": int(masks["full_outline_cells"]),
        "dashed_outline_cells": int(np.count_nonzero(masks["dashed_outline"])),
        "hole_rim_cells": int(np.count_nonzero(masks["hole_rim"])),
        "hole_core_cells": int(np.count_nonzero(masks["hole_core"])),
        "hollow_components": int(masks["hollow_components"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a dotted red outer border for every object, and give each enclosed hole its own yellow rim.

    Input structure:
    - A mostly uniform yellow background contains several blue 4-connected components.
    - Some components are hollow: they contain one enclosed 4-connected background hole inside the
      component's bounding box.

    Rule:
    - Every component gets an outer border based on its one-cell 8-neighborhood, but only every
      other outline cell is painted red (2) using a fixed checkerboard parity.
    - If a component contains enclosed holes, recolor the component itself to teal (8).
    - Recolor each hole to magenta (6), then overwrite the hole's one-cell inner rim with yellow
      (4), creating a solid inner border plus a remaining magenta core.

    Key insight:
    - Hollow objects require two separate boundary computations: the yellow hole rim is drawn on
      the hole cells themselves, while the red dotted exterior border is drawn from the hole-filled
      object so no red cells appear inside the cavity.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask, hole_masks = _component_hole_masks(g.shape, coords)

        solid = mask.copy()
        if hole_masks:
            out[mask] = HOLLOW_BODY_COLOR
            for hole_mask in hole_masks:
                out[hole_mask] = HOLE_FILL_COLOR
                out[_inner_perimeter(hole_mask)] = HOLE_INNER_OUTLINE_COLOR
                solid |= hole_mask

        outline = _dashed_outline_mask(solid)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Restrict holes so the yellow rim never consumes the entire cavity.
HOLE_VALID_AREAS = [9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

GEN_OUTER_RETRIES = 300

MIN_DASHED_OUTLINE_CELLS = 12
MIN_OUTLINE_CELL_DROP = 6
MIN_HOLE_RIM_CELLS = 8
MIN_HOLE_CORE_CELLS = 1


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


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
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
            1,
            max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1),
        )
        nw = random.randint(
            1,
            max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1),
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
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

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

    masks = _variant_structure_masks(grid)
    metrics = _variant_metrics(grid)

    if metrics["hollow_components"] < 1:
        return None
    if metrics["dashed_outline_cells"] < MIN_DASHED_OUTLINE_CELLS:
        return None
    if metrics["full_outline_cells"] - metrics["dashed_outline_cells"] < MIN_OUTLINE_CELL_DROP:
        return None
    if metrics["hole_rim_cells"] < MIN_HOLE_RIM_CELLS:
        return None
    if metrics["hole_core_cells"] < MIN_HOLE_CORE_CELLS:
        return None
    if int(np.count_nonzero(out == OUTLINE_COLOR)) != metrics["dashed_outline_cells"]:
        return None
    if int(np.count_nonzero(out == HOLE_FILL_COLOR)) != metrics["hole_core_cells"]:
        return None
    if not np.all(out[masks["hole_rim"]] == HOLE_INNER_OUTLINE_COLOR):
        return None
    if not np.all(out[masks["hole_core"]] == HOLE_FILL_COLOR):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

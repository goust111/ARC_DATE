# Combo variant of aa4ec2a5 (index 26)
# combo_spec: D3/0+D10/0
# Name: 8连通封闭洞+8连通对象
#
# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Source task:
#      * blue foreground was split into 4-connected objects;
#      * inside each object's bbox, a background island counted as a hole if it was 4-closed;
#      * hollow objects turned teal, their holes turned magenta, and every object got an 8-neighbor red outline.
#    - Fused task:
#      * foreground objects are now 8-connected, so diagonal-touching blue parts merge into one object;
#      * hole closure is also 8-connected, so a cavity that only diagonally leaks to bbox-edge background
#        is no longer a true hole.
#    - Small trace:
#      * diagonal merge example: two 2x2 blue blocks touching only at one corner were two objects before,
#        but they become one object now, so the output has one shared outline pattern instead of two.
#      * diagonal leak example:
#            1 1 1
#            1 0 1
#            1 1 0
#        Inside this bbox the center 0 is a 4-hole, but under 8-connectivity it reaches the edge
#        through the bottom-right 0 diagonally, so the fused solver keeps the object blue instead of
#        recoloring it teal/magenta.
# b) Complexity calibration:
#    - The generator always includes both signal types:
#      * at least one diagonally merged 8-object;
#      * at least one "almost hollow" object whose cavity is 4-closed but 8-open;
#      * at least one genuinely 8-closed hollow object that still triggers teal/magenta recoloring.
#    - This keeps the difference visually obvious without introducing many unrelated freedoms.
# c) Hidden assumption review against the original main():
#    - `find_components(... connectivity=4)` assumed diagonal contact does not merge objects; this is false now.
#    - `_interior_holes(... connectivity=4)` assumed 4-closed background islands are holes; this is false now.
#    - The outline logic itself is still valid because the original task already used an 8-neighborhood halo.
#    - Recoloring still depends only on hole existence, but hole existence must now be tested on the
#      8-connected background graph inside the merged 8-object bbox.
# d) Placement legality:
#    - Components are still placed with pairwise Chebyshev distance >= 3 and border clearance >= 1.
#      Because outlines are only 1 pixel thick in the 8-neighborhood, this keeps outlines disjoint.
#    - Every generated mask is validated as a single 8-component before placement.
# e) Visual observability:
#    - A solver looking at 2-4 examples can see that some diagonal-touching blue pieces behave as one object.
#    - The leaky near-ring objects look almost hollow in the input but stay blue in the output, while true
#      8-closed rings turn teal with magenta holes. That contrast exposes the 8-hole rule directly.
# f) Visual cleanliness:
#    - The generator keeps only 3-4 objects, uses one foreground color in the input, and preserves generous
#      spacing so the red outlines and teal/magenta interiors do not clutter each other.
# g) Color inferability:
#    - No new colors are introduced beyond the original transform palette: blue input objects, red outlines,
#      teal hollow bodies, magenta holes, yellow background.
# h) Generator-side new assumptions:
#    - At least one component must satisfy `holes_4 > holes_8`, otherwise the hole-rule change is invisible.
#    - At least one grid must satisfy `num_components_4 > num_components_8`, otherwise the object-rule change
#      is invisible.
#    - The generator enforces both conditions and also checks that the fused output differs from the legacy
#      4-connected output on the same input, preventing silent degeneration back to the source task.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray, connectivity: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in components_mask(~sub, connectivity=connectivity):
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


def _apply_rule(grid: np.ndarray, object_connectivity: int, hole_connectivity: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=object_connectivity, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords, connectivity=hole_connectivity)

        solid = mask.copy()
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _apply_rule(grid, object_connectivity=4, hole_connectivity=4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red halos around merged 8-connected blue objects, and only recolor cavities that stay
    fully sealed under 8-connectivity.

    Input structure:
    - A yellow background contains several blue foreground shapes.
    - Some shapes are diagonally stitched together, so one visual object may consist of multiple
      4-connected chunks that still form a single 8-connected component.
    - Some shapes contain true 8-closed background holes, while others contain diagonal leaks that
      would look hollow under 4-connectivity but are open under 8-connectivity.

    Rule:
    - Treat each blue 8-connected component as one object.
    - Inside that object's bbox, find background regions using 8-connectivity.
    - If such a background region does not touch the bbox edge, recolor the object to teal (8),
      recolor the hole to magenta (6), then draw a red (2) one-pixel 8-neighborhood outline around
      the object-plus-hole-solid.
    - If no 8-closed hole exists, keep the object blue and only add the red outline.

    Key insight:
    The same 8-neighbor notion governs both which blue pixels belong to one object and whether an
    apparent cavity is truly sealed, so diagonal contacts can both merge objects and destroy holes.
    """
    return _apply_rule(grid, object_connectivity=8, hole_connectivity=8)


# --------------- generator (combo) ---------------

GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
MANDATORY_COMPONENT_KINDS = ("true_hollow", "leaky_hollow", "diag_chain")

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_EXT_H_RANGE = (1, 3)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_MIN_AREA = 10
HOLLOW_GEN_RETRIES = 60

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 40
SMALL_SHAPE_PROB = 0.2

DIAG_BLOCK_COUNT_RANGE = (2, 3)
DIAG_BLOCK_DIM_RANGE = (2, 4)
DIAG_GEN_RETRIES = 30

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 400
GEN_OUTER_RETRIES = 500


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


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


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randint(0, 3)
    out = np.rot90(out, k)
    return out.copy()


def _local_hole_count(mask: np.ndarray, connectivity: int) -> int:
    count = 0
    for hole in components_mask(~mask, connectivity=connectivity):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == mask.shape[0] - 1)
            | (hole[:, 1] == mask.shape[1] - 1)
        )
        if not bool(touches_edge):
            count += 1
    return count


def _extend_mask(mask: np.ndarray) -> np.ndarray:
    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)
    base_h, base_w = mask.shape

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
        return new_mask

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


def _make_true_hollow_component() -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        dims = _rect_dims(random.choice(HOLE_VALID_AREAS))
        if dims is None:
            continue
        hh, hw = dims
        pad_top = random.randint(*HOLLOW_PAD_RANGE)
        pad_bot = random.randint(*HOLLOW_PAD_RANGE)
        pad_left = random.randint(*HOLLOW_PAD_RANGE)
        pad_right = random.randint(*HOLLOW_PAD_RANGE)

        mask = np.ones((hh + pad_top + pad_bot, hw + pad_left + pad_right), dtype=bool)
        mask[pad_top : pad_top + hh, pad_left : pad_left + hw] = False
        if random.random() < 0.55:
            mask = _extend_mask(mask)
        mask = _random_transform(mask)

        if mask.sum() < HOLLOW_MIN_AREA:
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=8):
            continue
        if _local_hole_count(mask, connectivity=8) < 1:
            continue
        return mask
    return None


def _make_leaky_hollow_component() -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        dims = _rect_dims(random.choice(HOLE_VALID_AREAS))
        if dims is None:
            continue
        hh, hw = dims
        pad_top = random.randint(*HOLLOW_PAD_RANGE)
        pad_bot = random.randint(*HOLLOW_PAD_RANGE)
        pad_left = random.randint(*HOLLOW_PAD_RANGE)
        pad_right = random.randint(*HOLLOW_PAD_RANGE)

        mask = np.ones((hh + pad_top + pad_bot, hw + pad_left + pad_right), dtype=bool)
        mask[pad_top : pad_top + hh, pad_left : pad_left + hw] = False

        corner = random.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            mask[:pad_top, :pad_left] = False
        elif corner == "tr":
            mask[:pad_top, pad_left + hw :] = False
        elif corner == "bl":
            mask[pad_top + hh :, :pad_left] = False
        else:
            mask[pad_top + hh :, pad_left + hw :] = False

        mask = _random_transform(mask)
        if mask.sum() < HOLLOW_MIN_AREA:
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=8):
            continue
        if _local_hole_count(mask, connectivity=4) < 1:
            continue
        if _local_hole_count(mask, connectivity=8) != 0:
            continue
        return mask
    return None


def _make_diagonal_chain() -> np.ndarray | None:
    for _ in range(DIAG_GEN_RETRIES):
        num_blocks = random.randint(*DIAG_BLOCK_COUNT_RANGE)
        dims = [
            (
                random.randint(*DIAG_BLOCK_DIM_RANGE),
                random.randint(*DIAG_BLOCK_DIM_RANGE),
            )
            for _ in range(num_blocks)
        ]
        total_h = sum(h for h, _ in dims) + num_blocks - 1
        total_w = sum(w for _, w in dims) + num_blocks - 1
        mask = np.zeros((total_h, total_w), dtype=bool)

        r = 0
        c = 0
        for h, w in dims:
            mask[r : r + h, c : c + w] = True
            r += h
            c += w

        mask = _crop_mask(mask)
        if mask is None:
            continue
        mask = _random_transform(mask)
        if not is_contiguous(mask.astype(int), background=0, connectivity=8):
            continue
        if is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        return mask
    return None


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (
            random.randint(*POLY_RECT_DIM_RANGE),
            random.randint(*POLY_RECT_DIM_RANGE),
        )
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
    if mask is None:
        return None
    mask = _random_transform(mask)
    if not is_contiguous(mask.astype(int), background=0, connectivity=8):
        return None
    return mask


def _make_small_shape() -> np.ndarray:
    choice = random.choice(
        [
            np.ones((1, 1), dtype=bool),
            np.ones((1, 2), dtype=bool),
            np.ones((2, 1), dtype=bool),
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.array([[1, 0], [1, 1]], dtype=bool),
            np.array([[1, 1, 1]], dtype=bool),
            np.array([[1], [1], [1]], dtype=bool),
            np.ones((2, 2), dtype=bool),
        ]
    )
    return choice.copy()


def _make_extra_solid() -> np.ndarray | None:
    if random.random() < SMALL_SHAPE_PROB:
        return _random_transform(_make_small_shape())
    for _ in range(POLY_GEN_RETRIES):
        mask = _make_polyomino()
        if mask is not None:
            return mask
    return None


def _chebyshev_conflict(
    rows_a: np.ndarray,
    cols_a: np.ndarray,
    pr: int,
    pc: int,
    pm: np.ndarray,
) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _build_component(kind: str) -> np.ndarray | None:
    if kind == "true_hollow":
        return _make_true_hollow_component()
    if kind == "leaky_hollow":
        return _make_leaky_hollow_component()
    if kind == "diag_chain":
        return _make_diagonal_chain()
    return _make_extra_solid()


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps4 = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    comps8 = find_components(g, bg=bg, connectivity=8, monochromatic=True)

    holes4_total = 0
    holes8_total = 0
    true_hollow_count = 0
    leaky_count = 0
    for comp in comps8:
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes4 = len(_interior_holes(mask, coords, connectivity=4))
        holes8 = len(_interior_holes(mask, coords, connectivity=8))
        holes4_total += holes4
        holes8_total += holes8
        if holes8 > 0:
            true_hollow_count += 1
        if holes4 > holes8:
            leaky_count += 1

    fused_out = main(g)
    legacy_out = _legacy_main(g)
    return {
        "components_4": len(comps4),
        "components_8": len(comps8),
        "component_merge_delta": len(comps4) - len(comps8),
        "holes_4": holes4_total,
        "holes_8": holes8_total,
        "hole_delta": holes4_total - holes8_total,
        "true_hollow_count": true_hollow_count,
        "leaky_count": leaky_count,
        "legacy_diff_cells": int(np.count_nonzero(fused_out != legacy_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    num_components = random.randint(*COMP_COUNT_RANGE)
    kinds = list(MANDATORY_COMPONENT_KINDS)
    while len(kinds) < num_components:
        kinds.append("solid")
    random.shuffle(kinds)

    components: list[np.ndarray] = []
    for kind in kinds:
        mask = _build_component(kind)
        if mask is None:
            return None
        components.append(mask)

    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []

    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))
    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        if mh > H - 2 * BORDER_MIN or mw > W - 2 * BORDER_MIN:
            return None

        placed_ok = False
        rows_local, cols_local = np.where(mask)
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, H - mh - BORDER_MIN)
            c = random.randint(BORDER_MIN, W - mw - BORDER_MIN)

            rows_a = rows_local + r
            cols_a = cols_local + c
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

    metrics = _fusion_metrics(grid)
    if metrics["components_8"] != num_components:
        return None
    if metrics["component_merge_delta"] < 1:
        return None
    if metrics["hole_delta"] < 1:
        return None
    if metrics["true_hollow_count"] < 1:
        return None
    if metrics["leaky_count"] < 1:
        return None
    if metrics["legacy_diff_cells"] < 1:
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

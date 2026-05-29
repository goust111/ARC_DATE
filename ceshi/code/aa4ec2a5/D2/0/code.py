# Variant of aa4ec2a5
# Dimension: D2 — 洞检测条件与分类规则 (transform)
# Change: 若蓝色对象内部包含被完全封闭的背景色区域（洞），则将该对象重着色为青色，洞重着色为品红色；无洞的对象保持蓝色... → 按洞的数量分级着色

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis
# variant_spec:
#   core_change:
#     original: any object with >=1 enclosed hole used one fixed recolor pair (body=8, holes=6)
#     variant: enclosed-hole objects split into three visible tiers by the exact hole count
#       - 1 hole  -> body 8, holes 6
#       - 2 holes -> body 3, holes 4
#       - 3+ holes -> body 7, holes 5
#   concretization_and_trace:
#     input still consists of blue 4-connected components on a uniform background, but some
#     components now contain several separate enclosed background pockets instead of exactly one.
#     Example: a ring with two disconnected rectangular voids was previously recolored the same
#     way as a single-hole ring (teal body, magenta holes). In this variant the same object must
#     first be recognized as having exactly two interior background components, then recolored
#     green with yellow holes, and only after that receive the red 8-neighborhood outline.
#   complexity_calibration:
#     the rule is not a free-form mapping over many counts; it has only three tiers (1 / 2 / 3+),
#     which keeps the hypothesis space small enough for ARC, while still being strictly richer
#     than the original binary hollow-vs-solid split.
#     the generator therefore guarantees at least two hollow objects per sample and forces at
#     least two distinct tiers to appear, so the count-based behavior is visually inferable.
#   implicit_assumption_audit:
#     - _interior_holes() in the original solver already returns a list of all enclosed bg
#       components, so its logic remains valid; only the downstream use of "if holes" was too
#       coarse for the new task.
#     - the original unconditional recolor `out[mask] = 8` depended on every hollow object
#       sharing one palette. That assumption no longer holds and is replaced by a tier lookup.
#     - the original background color was yellow=4. That becomes invalid here because the
#       two-hole tier needs yellow holes. If the background stayed 4, those holes would become
#       invisible and semantically merge with the exterior. The generator therefore moves the
#       background to black=0 while leaving the input objects blue.
#     - outlining still depends on holes being filled into the temporary solid mask; otherwise
#       each internal hole would incorrectly receive its own red border. That assumption is still
#       correct and is preserved unchanged.
#   placement_legality:
#     multi-hole components are generated as one connected outer shell with 1-cell minimum walls
#     between holes and between every hole and the exterior boundary. Optional outer extensions
#     are added only by attaching foreground rectangles, so they cannot change the hole count.
#     objects are placed with Chebyshev distance >= 3, which keeps the red outlines disjoint.
#   visual_observability_and_cleanliness:
#     solids remain blue, so "no-hole" objects are still easy to spot.
#     hollow tiers use only three body/hole palette pairs, with the holes sitting inside the
#     same object that changes color, giving a direct structural cue for the mapping.
#     the generator limits the hole count to 1, 2, 3, or 4 so the 3+ tier is visible without
#     producing oversized cluttered objects.


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


def _count_interior_holes(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0
    return len(_interior_holes(mask, coords))


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


def _hole_palette(num_holes: int) -> tuple[int, int] | None:
    if num_holes <= 0:
        return None
    if num_holes == 1:
        return 8, 6
    if num_holes == 2:
        return 3, 4
    return 7, 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around objects, and recolor hollow objects by the number of enclosed holes.

    Input structure:
    - A uniform background contains several blue 4-connected foreground components.
    - Some components are solid, while others surround one or more fully enclosed 4-connected
      background regions inside their bounding boxes.

    Rule:
    - For every foreground component, draw a one-cell 8-neighborhood outline in red (2).
    - Solid components stay blue (1).
    - Hollow components are recolored before outlining according to their hole count:
      1 hole -> teal body (8) + magenta holes (6),
      2 holes -> green body (3) + yellow holes (4),
      3 or more holes -> orange body (7) + gray holes (5).

    Key insight: The same hole-detection step as the original task still identifies enclosed
    regions, but now the solver must count how many separate holes belong to each object and map
    that count to one of three palette tiers before treating those holes as part of the solid
    mask for the outer outline.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        palette = _hole_palette(len(holes))

        solid = mask.copy()
        if palette is not None:
            body_color, hole_color = palette
            out[mask] = body_color
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = hole_color
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (variant) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
BG_COLOR = 0
FG_COLOR = 1

# Component counts
COMP_COUNT_RANGE = (4, 5)
HOLLOW_COUNT_RANGE = (2, 3)

# Polyomino construction (solid shapes)
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Small solid shapes
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Hollow component construction
HOLLOW_GEN_RETRIES = 80
HOLLOW_MIN_AREA = 12
HOLLOW_OUTER_MARGIN_RANGE = (1, 2)
HOLLOW_WALL_RANGE = (1, 2)
HOLLOW_MAX_EXTENSIONS = 2
HOLLOW_EXTENSION_PROB = 0.75
HOLLOW_EXT_H_RANGE = (1, 3)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_OPTIONS_SMALL = [(1, 2), (2, 1), (2, 2)]
HOLLOW_HOLE_OPTIONS_LARGE = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]

# Placement
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 400

# Top-level generation
GEN_OUTER_RETRIES = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino():
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


def _make_small_shape():
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


def _extend_mask(mask: np.ndarray) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    base_h, base_w = mask.shape
    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)

    if side in ("top", "bottom"):
        ext_w = random.randint(max(2, base_w // 2), base_w)
        offset = random.randint(0, base_w - ext_w)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[:ext_h, offset : offset + ext_w] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h:, offset : offset + ext_w] = True
        return new_mask

    ext_h_actual = random.randint(max(2, base_h // 2), base_h)
    ext_w_actual = random.randint(*HOLLOW_EXT_SIDE_W_RANGE)
    offset = random.randint(0, base_h - ext_h_actual)
    new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
    if side == "left":
        new_mask[:, ext_w_actual:] = mask
        new_mask[offset : offset + ext_h_actual, :ext_w_actual] = True
    else:
        new_mask[:, :base_w] = mask
        new_mask[offset : offset + ext_h_actual, base_w:] = True
    return new_mask


def _hole_dims_for_count(num_holes: int) -> list[tuple[int, int]]:
    options = HOLLOW_HOLE_OPTIONS_LARGE if num_holes == 1 else HOLLOW_HOLE_OPTIONS_SMALL
    return [random.choice(options) for _ in range(num_holes)]


def _layout_cols(num_holes: int) -> int:
    if num_holes == 1:
        return 1
    if num_holes == 2:
        return random.choice([1, 2])
    if num_holes == 3:
        return random.choice([2, 3])
    return 2


def _make_hollow_component(num_holes: int) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        hole_dims = _hole_dims_for_count(num_holes)
        n_cols = _layout_cols(num_holes)
        n_rows = math.ceil(num_holes / n_cols)

        row_heights = [0] * n_rows
        col_widths = [0] * n_cols
        for idx, (hh, hw) in enumerate(hole_dims):
            rr = idx // n_cols
            cc = idx % n_cols
            row_heights[rr] = max(row_heights[rr], hh)
            col_widths[cc] = max(col_widths[cc], hw)

        top = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
        bottom = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
        left = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
        right = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
        row_gaps = [random.randint(*HOLLOW_WALL_RANGE) for _ in range(max(0, n_rows - 1))]
        col_gaps = [random.randint(*HOLLOW_WALL_RANGE) for _ in range(max(0, n_cols - 1))]

        base_h = top + sum(row_heights) + sum(row_gaps) + bottom
        base_w = left + sum(col_widths) + sum(col_gaps) + right

        mask = np.ones((base_h, base_w), dtype=bool)
        cursor_r = top
        idx = 0
        for rr in range(n_rows):
            cursor_c = left
            for cc in range(n_cols):
                if idx >= num_holes:
                    break
                hh, hw = hole_dims[idx]
                slot_h = row_heights[rr]
                slot_w = col_widths[cc]
                hole_r = cursor_r + random.randint(0, slot_h - hh)
                hole_c = cursor_c + random.randint(0, slot_w - hw)
                mask[hole_r : hole_r + hh, hole_c : hole_c + hw] = False
                cursor_c += slot_w
                if cc < len(col_gaps):
                    cursor_c += col_gaps[cc]
                idx += 1
            cursor_r += row_heights[rr]
            if rr < len(row_gaps):
                cursor_r += row_gaps[rr]

        for _ in range(random.randint(0, HOLLOW_MAX_EXTENSIONS)):
            mask = _extend_mask(mask)

        cropped = _crop_mask(mask)
        if cropped is None:
            continue
        if not is_contiguous(cropped.astype(int), background=0, connectivity=4):
            continue
        if _count_interior_holes(cropped) != num_holes:
            continue
        if cropped.sum() < HOLLOW_MIN_AREA:
            continue
        return cropped
    return None


def _sample_hole_targets(num_hollow: int) -> list[int]:
    if num_hollow == 2:
        pair = random.choice([(1, 2), (1, 3), (2, 3)])
        return list(pair)
    return [1, 2, random.choice([3, 4])]


def _gen_attempt():
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))
    hollow_targets = _sample_hole_targets(num_hollow)
    random.shuffle(hollow_targets)

    components: list[np.ndarray] = []
    hollow_hole_counts: list[int] = []

    for holes_needed in hollow_targets:
        mask = _make_hollow_component(holes_needed)
        if mask is None:
            return None
        components.append(mask)
        hollow_hole_counts.append(holes_needed)

    while len(components) < num_components:
        if random.random() < SMALL_SHAPE_PROB:
            components.append(_make_small_shape())
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append(mask)
                    break
            else:
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

    comp_hole_counts: list[int] = []
    for comp in find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True):
        mask = _coords_mask(grid.shape, comp["coords"])
        comp_hole_counts.append(_count_interior_holes(mask))

    if sum(count > 0 for count in comp_hole_counts) < 2:
        return None
    if len({min(count, 3) for count in comp_hole_counts if count > 0}) < 2:
        return None
    if not any(count >= 2 for count in comp_hole_counts):
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input():
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

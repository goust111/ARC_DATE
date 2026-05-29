# Combo variant of aa4ec2a5 (index 33)
# combo_spec: D2/1+D4/2
# Name: 按洞面积决定着色+轮廓颜色与洞颜色互补

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis
# ---------------
# source_task:
#   - baseline is original aa4ec2a5: hollow objects become teal/magenta and every outline is red
#   - D2/1 changes the hollow recolor trigger from boolean "has hole" to measured hole area
#   - D4/2 changes the hollow outline color so it depends on the hole color instead of being fixed
#
# concretization_and_trace:
#   input:
#     - yellow background (4)
#     - 3-4 blue 4-connected components
#     - exactly 2 hollow components are generated here so both hole-area branches are visible
#     - one hollow component has a small rectangular hole of area 2 or 3
#     - one hollow component has a large rectangular hole of area 4, 6, 8, 9, 12, or 16
#     - at least one additional solid component remains as the "no-hole" control case
#   output:
#     - solid components stay blue (1) and keep the original red outline (2)
#     - small-hole objects: shell teal (8), hole magenta (6), outline green (3)
#     - large-hole objects: shell green (3), hole yellow (4), outline blue (1)
#     - holes are merged into the solid mask before outlining, so there is never a ring drawn inside a cavity
#   trace_example:
#     - original logic: a 1x3 hole and a 2x2 hole both become shell=8, hole=6, outline=2
#     - fused logic step 1: detect enclosed holes and sum their area
#     - fused logic step 2: area 3 -> shell=8 and hole=6, then hole color 6 implies complement outline 3
#     - fused logic step 3: area 4 -> shell=3 and hole=4, then hole color 4 implies complement outline 1
#     - difference: the outline is no longer chosen directly from geometry; it is chained through the interior recolor
#
# complexity_calibration:
#   - only two hole-area classes are used, keeping the classification learnable
#   - every generated sample contains both small-hole and large-hole hollow objects, so both branches can be compared side by side
#   - a solid blue object is always present, preserving the original red-outline behavior as a visible control
#   - although yellow holes match the background color, the surrounding green shell makes the enclosed area still readable
#
# implicit_assumption_review_for_main:
#   - find_components(... monochromatic=True) assumes the input objects are monochrome blue; the generator preserves that
#   - _interior_holes must run on the original blue mask; recoloring happens only after hole detection
#   - the old "if holes: out[mask] = 8" and "out[hole] = 6" are no longer safe because hollow objects split into two classes
#   - outline color cannot be decided before hole recoloring, because it is derived from the chosen hole color
#   - drawing the outline only on current background cells remains safe even when large holes use yellow (4), because hole cells are
#     added into the solid mask before outline extraction, preventing accidental inner outlines
#
# placement_legality:
#   - generation remains mask-first, then placement on the full grid
#   - components are placed with Chebyshev distance >= 3 and at least one-cell border clearance
#   - this keeps outer outlines non-overlapping and avoids ambiguity when validating outline colors
#   - each hollow component is checked to contain exactly one enclosed hole, and the final assembled grid is revalidated
#
# visual_observability:
#   - the two hollow classes differ in both shell color and outline color, so the change from the original task is obvious
#   - the chained rule is inferable because the same hole-area split that changes shell/hole recolor also predicts the outline color
#   - repeated support exists within one sample: red outlines mark no-hole controls, green outlines mark magenta-hole objects,
#     and blue outlines mark yellow-hole objects
#
# visual_cleanliness_and_color_inference:
#   - outputs use a controlled palette with structured reuse: blue/red for solid controls, teal-magenta-green for small holes,
#     and green-yellow-blue for large holes
#   - no hidden ordering, component ids, or off-grid metadata are used; only visible hole area and visible hole color drive the transform
#   - the generator rejects samples unless all three visible behavior families (solid, small-hole, large-hole) actually appear
#
# post_change_assumption_audit:
#   - main() now assumes the complement mapping is only queried for hollow objects; generate_input() guarantees exactly two hollow objects
#   - main() assumes each hollow object's total hole area falls into one of the two intended buckets; generator restricts hole areas accordingly
#   - validation in _fusion_signature confirms that the produced outputs really exhibit the non-red hollow outlines required by the combo


SMALL_HOLE_MAX_AREA = 3

SMALL_SHELL_COLOR = 8
SMALL_HOLE_COLOR = 6
LARGE_SHELL_COLOR = 3
LARGE_HOLE_COLOR = 4

DEFAULT_OUTLINE_COLOR = 2
COMPLEMENT_OUTLINE_BY_HOLE_COLOR = {
    SMALL_HOLE_COLOR: 3,
    LARGE_HOLE_COLOR: 1,
}


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


def _hole_style(total_hole_area: int) -> tuple[int, int, int]:
    if total_hole_area <= SMALL_HOLE_MAX_AREA:
        hole_color = SMALL_HOLE_COLOR
        return SMALL_SHELL_COLOR, hole_color, COMPLEMENT_OUTLINE_BY_HOLE_COLOR[hole_color]
    hole_color = LARGE_HOLE_COLOR
    return LARGE_SHELL_COLOR, hole_color, COMPLEMENT_OUTLINE_BY_HOLE_COLOR[hole_color]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects by enclosed-hole area, then derive each hollow outline from the hole color.

    Input structure:
    - A mostly uniform background contains several 4-connected blue foreground objects.
    - Some objects enclose one or more 4-connected background holes entirely inside their bounding boxes.

    Rule:
    - Every foreground component gets a one-cell 8-neighborhood outline drawn only on background cells.
    - Components without holes stay blue (1) and use the original red outline (2).
    - For hollow components, first sum the enclosed hole area: total area <= 3 gives teal shell (8) with magenta holes (6),
      while total area > 3 gives green shell (3) with yellow holes (4).
    - The hollow object's outline color is then chosen as the complement partner of that hole color: magenta holes imply
      green outlines (3), and yellow holes imply blue outlines (1).

    Key insight:
    - The transform is chained: detect holes, measure their total area, recolor shell and hole, then infer the outline color
      from the hole recolor. Holes are treated as filled when extracting the outer outline, so no inner cavity ring is drawn.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        outline_color = DEFAULT_OUTLINE_COLOR
        if holes:
            total_hole_area = sum(int(hole.shape[0]) for hole in holes)
            shell_color, hole_color, outline_color = _hole_style(total_hole_area)
            out[mask] = shell_color
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = hole_color
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = outline_color

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT = 2

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

SMALL_HOLE_AREAS = [2, 3]
LARGE_HOLE_AREAS = [4, 6, 8, 9, 12, 16]
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


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _local_holes(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    holes: list[np.ndarray] = []
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            holes.append(bg_reg)
    return holes


def _total_hole_area(mask: np.ndarray) -> int:
    return sum(int(hole.shape[0]) for hole in _local_holes(mask))


def _hole_class(mask: np.ndarray) -> str | None:
    total = _total_hole_area(mask)
    if total == 0:
        return None
    return "small" if total <= SMALL_HOLE_MAX_AREA else "large"


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
        if cropped is None:
            continue
        if len(_local_holes(cropped)) != 1:
            continue
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
    carved = _carve_hole(mask, hh, hw)
    if carved is None:
        return None
    if _total_hole_area(carved) != target_hole_area:
        return None
    return carved


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


def _fusion_signature(grid: np.ndarray) -> dict[str, int] | None:
    out = main(grid)
    counts = {"solid": 0, "small": 0, "large": 0}

    for comp in find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()

        if not holes:
            outline = _outline_mask(solid)
            outline_colors = set(map(int, out[outline & (grid == BG_COLOR)]))
            if outline_colors != {DEFAULT_OUTLINE_COLOR}:
                return None
            if set(map(int, out[mask])) != {FG_COLOR}:
                return None
            counts["solid"] += 1
            continue

        total_hole_area = sum(int(hole.shape[0]) for hole in holes)
        shell_color, hole_color, outline_color = _hole_style(total_hole_area)
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
            if set(map(int, out[hole[:, 0], hole[:, 1]])) != {hole_color}:
                return None
        if set(map(int, out[mask])) != {shell_color}:
            return None

        outline = _outline_mask(solid)
        outline_colors = set(map(int, out[outline & (grid == BG_COLOR)]))
        if outline_colors != {outline_color}:
            return None

        counts["small" if total_hole_area <= SMALL_HOLE_MAX_AREA else "large"] += 1

    return counts


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    if num_components <= HOLLOW_COUNT:
        return None

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = indices[:HOLLOW_COUNT]

    hollow_targets = {
        hollow_indices[0]: "small",
        hollow_indices[1]: "large",
    }
    if random.random() < 0.5:
        hollow_targets[hollow_indices[0]], hollow_targets[hollow_indices[1]] = (
            hollow_targets[hollow_indices[1]],
            hollow_targets[hollow_indices[0]],
        )

    components: list[np.ndarray] = []
    for i in range(num_components):
        target_class = hollow_targets.get(i)
        if target_class is not None:
            area_choices = SMALL_HOLE_AREAS if target_class == "small" else LARGE_HOLE_AREAS
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(area_choices))
                if mask is None or int(mask.sum()) < HOLLOW_MIN_AREA:
                    continue
                if _hole_class(mask) != target_class:
                    continue
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

    signature = _fusion_signature(grid)
    if signature is None:
        return None
    if signature != {"solid": num_components - HOLLOW_COUNT, "small": 1, "large": 1}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

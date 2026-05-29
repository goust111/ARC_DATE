# Combo variant of aa4ec2a5 (index 23)
# combo_spec: D3/1+D12/1
# Name: 多颜色对象+输入已有多种颜色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis:
# spec:
#   fused_rule:
#     keep:
#       - 4-connected monochromatic objects on a uniform background
#       - hollow-object detection by enclosed 4-connected background regions inside the bbox
#       - hole cells filled before outline computation so interior borders never become outlines
#     change:
#       - input foreground now uses three source colors instead of one shared blue
#       - each source color has its own outline color
#       - each source color also has its own hollow-shell recolor target
#       - hole fill stays globally magenta, so the new learnable parts are exactly two finite maps:
#         source->outline and source->hollow-shell
#   behavior_trace:
#     original:
#       - every object was blue
#       - every outline was red
#       - every hollow shell became teal
#     fused_example:
#       - a blue hollow object becomes teal with a red outline
#       - a green solid object stays green but gets a black outline
#       - an orange hollow object becomes maroon with a green outline
#       - if blue also appears as a solid object in the same grid, it keeps blue but still gets
#         the same red outline as the blue hollow object; this proves outline color depends on
#         source color, not on hollow-vs-solid status
#   complexity_calibration:
#     - exactly three source colors are used: blue(1), green(3), orange(7)
#     - exactly four components are generated: two hollow and two solid
#     - one source color is forced to appear once hollow and once solid, giving repeated support
#       for the outline map and showing that shell recoloring triggers only for hollow objects
#     - the third source color still appears at least once, so every sample shows all three
#       outline colors without exploding the palette
#   implicit_assumptions_audit:
#     - `find_components(..., monochromatic=True)` is still valid because each object remains
#       single-color even though different objects can now use different colors
#     - the old unconditional `out[mask] = 8` is invalid; hollow recoloring must become a
#       per-component lookup keyed by the component's input color
#     - the old unconditional `out[outline & (out == bg)] = 2` is also invalid; outline color
#       must be looked up per source color
#     - the old assumption that write order is irrelevant still depends on non-overlapping
#       outline masks, so Chebyshev spacing >= 3 must be preserved by the generator
#   placement_legality:
#     - objects remain at least one cell from the border and pairwise Chebyshev-separated by 3
#     - this guarantees outlines stay clipped safely at the frame and never overwrite each other
#   visual_observability:
#     - every input shows all three source colors, so the output also shows all three outline colors
#     - two hollow objects per sample show at least two source->shell mappings directly
#     - the repeated source color appearing in both hollow and solid form prevents the false rule
#       "outline color depends on whether the object has a hole"
#     - the ARC palette is too small to keep the two mapping tables completely disjoint; one
#       outline color overlaps a source color, but the spatial roles remain clean because outlines
#       live only outside objects while source colors live on object interiors in the input
#   cleanliness:
#     - no legends or extra markers are introduced
#     - output structure stays readable: shell color on hollow bodies, magenta in holes, and a
#       one-pixel exterior outline around each object
#   anti_degeneracy_metric:
#     - generated samples are rejected unless the combo output differs from both single-alt
#       baselines:
#         1. D3-only style: source-specific hollow shell, but all outlines forced red
#         2. D12-only style: source-specific outlines, but all hollow shells forced teal
#     - this guarantees both mapping dimensions are behaviorally active on the same input
#   new_solver_assumptions_checked_by_generator:
#     - all foreground colors must belong to the source-color mapping domain
#     - at least one hollow object must use a non-blue source color, otherwise the shell-map
#       variation would not be visible
#     - at least one non-blue object must exist at all times, otherwise the outline-map variation
#       would collapse to the base task


BG_COLOR = 4
SRC_COLORS = (1, 3, 7)
HOLE_COLOR = 6
HOLLOW_RECOLOR_MAP = {
    1: 8,  # blue -> teal
    3: 5,  # green -> gray
    7: 9,  # orange -> maroon
}
OUTLINE_COLOR_MAP = {
    1: 2,  # blue -> red outline
    3: 0,  # green -> black outline
    7: 3,  # orange -> green outline
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


def _transform_with_maps(
    grid: np.ndarray,
    outline_color_map: dict[int, int],
    hollow_recolor_map: dict[int, int],
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        src_color = int(comp["color"])
        if src_color not in outline_color_map or src_color not in hollow_recolor_map:
            raise ValueError(f"unsupported source color: {src_color}")

        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = hollow_recolor_map[src_color]
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = HOLE_COLOR
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = outline_color_map[src_color]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects and outline every object using source-color-specific mappings.

    Input structure:
    - A uniform yellow background contains several separated 4-connected monochromatic objects.
    - Objects can be blue, green, or orange, and some of them enclose 4-connected background holes.

    Rule:
    - For every object, draw a one-cell outline on its 8-neighborhood exterior, but choose the
      outline color from a fixed mapping keyed by the object's input color.
    - Solid objects keep their original color.
    - Hollow objects recolor their shell by another fixed source-color mapping, fill enclosed
      holes with magenta (6), then receive the same source-dependent outline.

    Key insight: The geometry is still entirely local to each monochromatic component, but the
    solver must now learn two coordinated color tables from examples: source->outline for all
    objects, and source->hollow-shell for the subset that actually contains holes.
    """
    return _transform_with_maps(grid, OUTLINE_COLOR_MAP, HOLLOW_RECOLOR_MAP)


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

# Component counts
COMP_COUNT = 4
HOLLOW_COUNT = 2

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


def _component_specs() -> list[dict[str, object]]:
    hollow_colors = random.sample(SRC_COLORS, HOLLOW_COUNT)
    repeated_color = random.choice(hollow_colors)
    leftover_color = next(color for color in SRC_COLORS if color not in hollow_colors)

    specs = [
        {"color": hollow_colors[0], "is_hollow": True, "allow_small": False},
        {"color": hollow_colors[1], "is_hollow": True, "allow_small": False},
        {"color": repeated_color, "is_hollow": False, "allow_small": False},
        {"color": leftover_color, "is_hollow": False, "allow_small": True},
    ]
    random.shuffle(specs)
    return specs


def _single_alt_d3_output(grid: np.ndarray) -> np.ndarray:
    return _transform_with_maps(grid, {color: 2 for color in SRC_COLORS}, HOLLOW_RECOLOR_MAP)


def _single_alt_d12_output(grid: np.ndarray) -> np.ndarray:
    return _transform_with_maps(grid, OUTLINE_COLOR_MAP, {color: 8 for color in SRC_COLORS})


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    component_specs = _component_specs()
    components = []
    for spec in component_specs:
        if spec["is_hollow"]:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
                    components.append({"mask": mask, **spec})
                    found = True
                    break
            if not found:
                return None
        elif spec["allow_small"] and random.random() < SMALL_SHAPE_PROB:
            components.append({"mask": _make_small_shape(), **spec})
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append({"mask": mask, **spec})
                    break
            else:
                return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    placed = []

    order = sorted(range(len(components)), key=lambda i: -int(components[i]["mask"].sum()))

    for idx in order:
        component = components[idx]
        mask = component["mask"]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, h - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, w - mw - BORDER_MIN))
            if r + mh > h - BORDER_MIN or c + mw > w - BORDER_MIN:
                continue

            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, placed_mask, _placed_color, _placed_hollow in placed:
                row_gap = max(0, max(r - (pr + placed_mask.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + placed_mask.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, placed_mask):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask, int(component["color"]), bool(component["is_hollow"])))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask, color, _is_hollow in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = color

    if get_bg(grid) != BG_COLOR:
        return None

    present_colors = {int(v) for v in np.unique(grid) if int(v) != BG_COLOR}
    if present_colors != set(SRC_COLORS):
        return None

    hollow_colors = {color for _r, _c, _mask, color, is_hollow in placed if is_hollow}
    solid_colors = {color for _r, _c, _mask, color, is_hollow in placed if not is_hollow}
    if len(hollow_colors) != HOLLOW_COUNT:
        return None
    if not (hollow_colors & solid_colors):
        return None
    if len(solid_colors) != 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    outline_cells = (grid == BG_COLOR) & (out != BG_COLOR) & (out != HOLE_COLOR)
    outline_colors_used = {int(v) for v in out[outline_cells].tolist()}
    if outline_colors_used != {OUTLINE_COLOR_MAP[color] for color in SRC_COLORS}:
        return None

    combo_vs_d3 = not np.array_equal(out, _single_alt_d3_output(grid))
    combo_vs_d12 = not np.array_equal(out, _single_alt_d12_output(grid))
    if not combo_vs_d3 or not combo_vs_d12:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

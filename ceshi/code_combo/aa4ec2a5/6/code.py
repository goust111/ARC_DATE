# Combo variant of aa4ec2a5 (index 6)
# combo_spec: D3/1+D4/0
# Name: 多颜色对象+轮廓颜色按类别分配

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis:
# a) concretization_and_trace:
#    - Original behavior: every foreground object is blue, every outline is red, and only hollow
#      objects change inside the object itself (shell -> teal, hole -> magenta).
#    - Fused behavior: objects may start as blue(1), green(3), or orange(7). For a green hollow
#      object, the old solver would have produced a teal shell with a red outline if such an input
#      were allowed; the fused solver instead maps green -> gray(5), fills the hole with magenta(6),
#      and uses a red outline because the object is hollow. A green solid object stays green but now
#      gets an orange outline. The source color controls hollow-shell recoloring, while hole-status
#      controls outline color.
#    - Small example trace:
#      input: blue ring with one hole + green solid L.
#      original: impossible input because the solid is not blue.
#      D3-only: blue ring -> teal shell + magenta hole + red outline; green L stays green + red outline.
#      fused: blue ring -> teal shell + magenta hole + red outline; green L stays green + orange outline.
# b) complexity_calibration:
#    - Use exactly three input colors and exactly three hollow-shell targets, keeping the mapping
#      finite and learnable.
#    - Force four components with exactly two hollow and two solid so every sample shows both outline
#      classes repeatedly.
#    - Force one source color to appear once as hollow and once as solid, proving the shell recolor
#      is not purely color-driven and the outline recolor is not purely source-color-driven.
# c) implicit_assumption_audit:
#    - `find_components(..., monochromatic=True)` still holds because each object remains single-color.
#    - The original `out[mask] = 8` no longer holds; hollow shell recoloring must depend on the
#      component's source color through `HOLLOW_RECOLOR_MAP`.
#    - The original fixed `out[outline & (out == bg)] = 2` no longer holds; the outline color depends
#      on whether `holes` is empty.
#    - Filling holes into `solid` before outlining is still required; otherwise a hollow object would
#      incorrectly receive an inner red ring around the cavity.
#    - Two-color outlines make write order observable if borders touch, so generator spacing must keep
#      outline masks disjoint.
# d) placement_legality:
#    - The generator preserves Chebyshev distance >= 3 and border clearance >= 1, so components do not
#      overlap, outlines stay clipped safely inside the grid, and red/orange borders never collide.
# e) visual_observability:
#    - Every generated sample contains all three input colors, two distinct hollow source colors, and
#      one repeated source color across hollow and solid classes.
#    - Every output contains both outline colors plus magenta holes and at least two distinct hollow
#      shell targets, making both inference layers visible in a single example.
# f) cleanliness:
#    - No extra markers are added beyond shell recolor, hole fill, and one-pixel outlines.
#    - The palette remains structured: yellow background, three input colors, magenta holes, red/orange
#      borders, and hollow shells chosen from teal/gray/maroon.
# g) inferability_of_changes:
#    - The color mapping is limited to three source colors and repeated across samples.
#    - The hollow-vs-solid outline rule is supported within each sample because both classes are always
#      present. This avoids ambiguous one-off recolors that a human could not infer from 2-4 examples.


BG_COLOR = 4
SRC_COLORS = (1, 3, 7)
HOLLOW_RECOLOR_MAP = {
    1: 8,  # blue -> teal
    3: 5,  # green -> gray
    7: 9,  # orange -> maroon
}
RED_OUTLINE = 2
ORANGE_OUTLINE = 7
MAGENTA_HOLE = 6


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
    Recolor hollow multicolor objects by source-color class and outline objects by hole-status class.

    Input structure:
    - A uniform background contains several separated 4-connected monochromatic objects.
    - Objects may use multiple source colors such as blue, green, and orange, and some objects enclose
      4-connected background holes strictly inside their bbox.

    Rule:
    - For every object, compute the one-cell 8-neighborhood outline on background cells only.
    - If an object has enclosed holes, recolor its shell by a fixed source-color mapping, recolor each
      enclosed hole to magenta (6), and draw its outline in red (2).
    - If an object has no enclosed holes, keep its original color and draw its outline in orange (7).

    Key insight:
    - Two independent classifications interact: source color decides the shell color only for hollow
      objects, while hollow-vs-solid decides the outline color for every object. Hollow objects must be
      outlined as if their hole cells were filled into the object first.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        src_color = int(comp["color"])
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        outline_color = ORANGE_OUTLINE
        if holes:
            outline_color = RED_OUTLINE
            out[mask] = HOLLOW_RECOLOR_MAP[src_color]
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA_HOLE
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = outline_color

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

COMP_COUNT = 4
HOLLOW_COUNT = 2

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

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

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

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


def _component_specs() -> list[dict[str, int | bool]]:
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


def _render_components(
    H: int, W: int, components: list[dict[str, object]]
) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray, int, bool]]] | None:
    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray, int, bool]] = []

    order = sorted(range(len(components)), key=lambda i: -int(np.sum(components[i]["mask"])))

    for idx in order:
        component = components[idx]
        mask = component["mask"]
        assert isinstance(mask, np.ndarray)
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
            for pr, pc, placed_mask, _placed_color, _placed_hollow in placed:
                row_gap = max(0, max(r - (pr + placed_mask.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + placed_mask.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, placed_mask):
                    conflict = True
                    break

            if not conflict:
                placed.append(
                    (
                        r,
                        c,
                        mask,
                        int(component["color"]),
                        bool(component["is_hollow"]),
                    )
                )
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask, color, _is_hollow in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = color

    return grid, placed


def _variant_visible(
    grid: np.ndarray, out: np.ndarray, placed: list[tuple[int, int, np.ndarray, int, bool]]
) -> bool:
    if get_bg(grid) != BG_COLOR:
        return False

    input_colors = {int(v) for v in np.unique(grid) if int(v) != BG_COLOR}
    if input_colors != set(SRC_COLORS):
        return False

    hollow_colors = {color for _r, _c, _mask, color, is_hollow in placed if is_hollow}
    solid_colors = {color for _r, _c, _mask, color, is_hollow in placed if not is_hollow}
    if len(hollow_colors) != HOLLOW_COUNT or not (hollow_colors & solid_colors):
        return False

    output_colors = {int(v) for v in out.flat}
    if RED_OUTLINE not in output_colors or ORANGE_OUTLINE not in output_colors:
        return False
    if MAGENTA_HOLE not in output_colors:
        return False

    hollow_targets = {HOLLOW_RECOLOR_MAP[color] for color in hollow_colors}
    visible_targets = output_colors & set(HOLLOW_RECOLOR_MAP.values())
    return hollow_targets.issubset(visible_targets) and len(hollow_targets) == HOLLOW_COUNT


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    component_specs = _component_specs()
    components: list[dict[str, object]] = []
    for spec in component_specs:
        if bool(spec["is_hollow"]):
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
                    components.append({"mask": mask, **spec})
                    found = True
                    break
            if not found:
                return None
        elif bool(spec["allow_small"]) and random.random() < SMALL_SHAPE_PROB:
            components.append({"mask": _make_small_shape(), **spec})
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append({"mask": mask, **spec})
                    break
            else:
                return None

    rendered = _render_components(H, W, components)
    if rendered is None:
        return None
    grid, placed = rendered

    try:
        out = main(grid)
    except Exception:
        return None

    if not _variant_visible(grid, out, placed):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

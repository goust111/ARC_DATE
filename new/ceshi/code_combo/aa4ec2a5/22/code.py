# Combo variant of aa4ec2a5 (index 22)
# combo_spec: D1/0+D4/0+D15/1
# Name: 4邻域轮廓+仅角落像素+按类别着色

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
# -------------------------------------
# a) Concrete behavior trace:
#    - Source task: every component gets a full 8-neighborhood red ring; hollow components
#      are recolored teal with magenta holes before outlining.
#    - Naively intersecting D1 (orthogonal-only) with D15 (corner-only) would erase the
#      outline completely, so the fusion rule must split styles by component class instead.
#    - Final rule used here:
#        * hollow component -> teal body, magenta hole, red corner-only outline
#        * solid component  -> blue body, orange orthogonal-only outline
#    - Small trace:
#        * solid 2x2 block: source paints a red 4x4 frame; combo paints only the eight
#          side-adjacent cells in orange, leaving the four diagonal outer corners yellow.
#        * hollow 4x4 ring with a 2x2 hole: source paints a full outside red ring; combo
#          paints only the four exterior diagonal corner cells in red. The interior hole
#          stays magenta and produces no inner outline because hole cells are merged into
#          the mass before outline extraction.
# b) Complexity calibration:
#    - The variant uses exactly two output outline styles and ties them to an already
#      visible structural class (has hole vs has no hole), so the mapping is inferable.
#    - The generator keeps 3-4 components with 1-2 hollow ones, guaranteeing both classes
#      appear in the same sample and usually with repetition across samples.
#    - Irregular solids, tiny solids, and notched hollow shapes are preserved so the task
#      cannot collapse into a trivial "bbox corners vs bbox sides" shortcut.
# c) Implicit-assumption audit for main():
#    - `find_components(... connectivity=4, monochromatic=True)` still holds because inputs
#      remain blue 4-connected monochrome objects on a uniform background.
#    - `_interior_holes(...)` is still valid because hole definition did not change.
#    - `solid[hole] = True` becomes even more important: without it, hollow components would
#      get false red corner marks inside the cavity, contradicting the intended exterior-only
#      object-class outline.
#    - `out[outline & (out == bg)] = color` still assumes outlines land only on background
#      cells. Both orthogonal-edge and corner-only helpers satisfy that by masking with `~mask`.
#    - Different outline colors make write order observable if outlines can collide, so the
#      old placement separation must be preserved.
# d) Placement legality:
#    - Original Chebyshev distance >= 3 is strong enough for both styles here: orthogonal
#      orange edge marks and diagonal red corner marks remain disjoint across components.
#    - Border clearance >= 1 remains sufficient because outlines are clipped to the grid.
# e) Visual observability:
#    - Hollow objects advertise their class through teal shell + magenta cavity, providing
#      a visible reason why they get sparse red corner dots instead of orange edge pixels.
#    - Solid objects stay blue and receive orange side marks, so the solver can compare two
#      classes within one output and infer both geometry and color simultaneously.
# f) Visual cleanliness:
#    - Only one outline layer is applied per component, and it is always outside the object.
#    - The palette stays compact: yellow bg, blue solids, teal/magenta hollow interiors,
#      orange orthogonal marks, red corner marks.
# g) Generator anti-degeneracy and new solver assumptions:
#    - Every generated sample must show both red and orange outputs and both interior classes.
#    - We also reject samples where the combo output matches either "all components orthogonal"
#      or "all components corner-only" class-colored ablations; that proves both branches are
#      actively used on the sample.
#    - New assumption review: the solid branch assumes at least one solid component exists, and
#      the hollow branch assumes at least one hollow component exists. Generation already enforces
#      1-2 hollow among 3-4 total components, so both assumptions are guaranteed.


RED_OUTLINE = 2
YELLOW_BG = 4
MAGENTA_HOLE = 6
ORANGE_OUTLINE = 7
TEAL_HOLLOW = 8

STYLE_FULL = "full"
STYLE_ORTH = "orth"
STYLE_CORNER = "corner"


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


def _outline_mask(mask: np.ndarray, style: str) -> np.ndarray:
    h, w = mask.shape
    rows, cols = np.where(mask)

    if style == STYLE_FULL:
        outline = np.zeros_like(mask, dtype=bool)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),            (0, 1),
            (1, -1),  (1, 0),   (1, 1),
        ]
        for dr, dc in directions:
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
        outline &= ~mask
        return outline

    if style == STYLE_ORTH:
        outline = np.zeros_like(mask, dtype=bool)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
        outline &= ~mask
        return outline

    if style == STYLE_CORNER:
        diagonal = np.zeros_like(mask, dtype=bool)
        orthogonal = np.zeros_like(mask, dtype=bool)
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

    raise ValueError(f"Unknown outline style: {style}")


def _transform(
    grid: np.ndarray,
    *,
    hollow_style: str,
    solid_style: str,
    hollow_outline_color: int = RED_OUTLINE,
    solid_outline_color: int = ORANGE_OUTLINE,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        outline_style = solid_style
        outline_color = solid_outline_color

        if holes:
            outline_style = hollow_style
            outline_color = hollow_outline_color
            out[mask] = TEAL_HOLLOW
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA_HOLE
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid, outline_style)
        out[outline & (out == bg)] = outline_color

    return out


def _legacy_transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = TEAL_HOLLOW
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA_HOLE
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid, STYLE_FULL)
        out[outline & (out == bg)] = RED_OUTLINE

    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = _transform(grid, hollow_style=STYLE_CORNER, solid_style=STYLE_ORTH)
    all_orth = _transform(grid, hollow_style=STYLE_ORTH, solid_style=STYLE_ORTH)
    all_corner = _transform(grid, hollow_style=STYLE_CORNER, solid_style=STYLE_CORNER)
    legacy = _legacy_transform(grid)
    return {
        "red": int(np.count_nonzero(combo == RED_OUTLINE)),
        "orange": int(np.count_nonzero(combo == ORANGE_OUTLINE)),
        "magenta": int(np.count_nonzero(combo == MAGENTA_HOLE)),
        "teal": int(np.count_nonzero(combo == TEAL_HOLLOW)),
        "delta_vs_all_orth": int(np.count_nonzero(combo != all_orth)),
        "delta_vs_all_corner": int(np.count_nonzero(combo != all_corner)),
        "delta_vs_legacy": int(np.count_nonzero(combo != legacy)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the outline rule by component class: hollow objects get red corner marks, while solid objects get orange orthogonal edge marks.

    Input structure:
    - A uniform yellow background contains several blue 4-connected foreground objects.
    - Some objects contain one or more enclosed 4-connected background holes fully inside their bbox.

    Rule:
    - For each solid component with no enclosed holes, keep it blue and draw only the 4-neighborhood side-adjacent outline pixels in orange (7).
    - For each hollow component, recolor its body to teal (8), recolor each enclosed hole to magenta (6), and draw only true corner pixels in red (2), meaning diagonal neighbors that are not also orthogonally adjacent.
    - In both cases, paint outlines only onto background cells, and treat enclosed hole cells as filled when deciding where the hollow component's exterior outline lies.

    Key insight: The two seemingly incompatible outline variants are resolved by using the component class as the switch: hole-bearing objects advertise themselves with teal/magenta interiors and sparse red corner dots, while hole-free objects stay blue and receive orange orthogonal edge marks.
    """
    return _transform(grid, hollow_style=STYLE_CORNER, solid_style=STYLE_ORTH)


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = YELLOW_BG
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


def _metrics_support_fusion(grid: np.ndarray) -> bool:
    metrics = _behavior_metrics(grid)
    return (
        metrics["red"] > 0
        and metrics["orange"] > 0
        and metrics["magenta"] > 0
        and metrics["teal"] > 0
        and metrics["delta_vs_all_orth"] > 0
        and metrics["delta_vs_all_corner"] > 0
        and metrics["delta_vs_legacy"] > 0
    )


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
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

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

    grid = np.full((h, w), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))

    for idx in order:
        mask = components[idx]
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
    if not _metrics_support_fusion(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

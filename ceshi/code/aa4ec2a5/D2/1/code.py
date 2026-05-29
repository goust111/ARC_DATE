# Variant of aa4ec2a5
# Dimension: D2 — 洞检测条件与分类规则 (transform)
# Change: 若蓝色对象内部包含被完全封闭的背景色区域（洞），则将该对象重着色为青色，洞重着色为品红色；无洞的对象保持蓝色... → 按洞的面积大小决定着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis
# ---------------
# source_task:
#   none: the variant is creative, so the baseline is only the original aa4ec2a5 solver/generator.
#
# concretization_and_trace:
#   input:
#     - yellow background (4)
#     - 3-4 blue 4-connected components
#     - exactly 2 hollow components are generated here so both hole-size branches are visible
#     - one hollow component contains a small enclosed rectangular hole of area 2 or 3
#     - one hollow component contains a large enclosed rectangular hole of area 4, 6, 8, 9, 12, or 16
#     - remaining components are solid blue polyominoes
#   output:
#     - all components get a 1-cell red 8-neighborhood outline on background cells only
#     - solid components stay blue
#     - hollow components are classified by total enclosed-hole area:
#         area <= 3  -> shell teal (8), hole magenta (6)
#         area > 3   -> shell green (3), hole yellow (4)
#     - holes are still merged into the solid mask before outlining, so there is no red ring drawn inside a hole
#   trace_example:
#     - original logic: a 1x3 hole and a 2x2 hole both trigger the same recolor, shell=8 and hole=6
#     - variant logic: the 1x3 hole has total area 3, so shell=8 and hole=6; the 2x2 hole has total area 4, so shell=3 and hole=4
#     - difference: the decision point is no longer boolean "has hole?", but a measured quantity "sum of hole cells"
#
# complexity_calibration:
#   - only two hole-size classes are used, keeping the rule learnable
#   - each sample contains both classes, so the color split is directly comparable within one grid
#   - solid objects remain present, preserving the original "outline-only" branch as a repeated control case
#
# implicit_assumption_review_for_main:
#   - find_components(... monochromatic=True) still assumes each input object is monochrome blue; generator preserves that
#   - _interior_holes computes holes from the original blue mask, not from recolored output; still valid
#   - the old "if holes: out[mask] = 8" relied on all hollow objects sharing one shell color; this is no longer safe
#   - the old "out[hole] = 6" relied on all holes sharing one fill color; this is no longer safe
#   - outline painting still writes only onto current background cells; this remains safe even when large holes are yellow(4),
#     because holes are added into the solid mask before outline extraction, so interior yellow holes are not mistaken for outline targets
#
# placement_legality:
#   - component generation remains mask-first and placement enforces Chebyshev distance >= 3 and border clearance >= 1
#   - this prevents outer red outlines from colliding across components
#   - hollow components are validated before placement, and the full grid is revalidated by calling main(grid)
#
# visual_observability:
#   - small-hole and large-hole objects coexist in every generated sample
#   - the large-hole branch uses a different shell color (green) even though its hole is yellow like the background,
#     so the class change is still clearly visible to a human
#   - hole size remains rectangular and low-entropy, making the threshold "3 vs >3" inferable from few examples
#
# visual_cleanliness_and_color_inference:
#   - colors are limited to five roles in outputs: bg yellow, solid blue, outline red, small-hole teal/magenta, large-hole green/yellow
#   - no sorting or branching depends on hidden ids; only visible enclosed area controls the color choice
#   - generator avoids overcrowding by keeping the original spacing discipline


SMALL_HOLE_MAX_AREA = 3
SMALL_SHELL_COLOR = 8
SMALL_HOLE_COLOR = 6
LARGE_SHELL_COLOR = 3
LARGE_HOLE_COLOR = 4


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


def _hole_colors(total_hole_area: int) -> tuple[int, int]:
    if total_hole_area <= SMALL_HOLE_MAX_AREA:
        return SMALL_SHELL_COLOR, SMALL_HOLE_COLOR
    return LARGE_SHELL_COLOR, LARGE_HOLE_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outer outlines around components, while recoloring hollow ones by hole size.

    Input structure:
    - A mostly uniform background contains several 4-connected blue foreground objects.
    - Some objects enclose one or more 4-connected background holes inside their bounding boxes.

    Rule:
    - Draw a one-cell red (2) outline in the 8-neighborhood around every foreground component.
    - If a component has enclosed holes, sum all enclosed hole cells before recoloring it.
    - Total hole area <= 3 gives teal shell (8) with magenta holes (6); total hole area > 3 gives green shell (3) with yellow holes (4).

    Key insight: The outline still treats holes as filled-in mass, but the interior recolor is no
    longer a simple hollow-vs-solid test; it depends on measuring the enclosed area first.
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
            total_hole_area = sum(int(hole.shape[0]) for hole in holes)
            shell_color, hole_color = _hole_colors(total_hole_area)
            out[mask] = shell_color
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = hole_color
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

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

    actual_classes: list[str] = []
    for comp in find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True):
        mask = _coords_mask(grid.shape, comp["coords"])
        holes = _interior_holes(mask, comp["coords"])
        if holes:
            total = sum(int(hole.shape[0]) for hole in holes)
            actual_classes.append("small" if total <= SMALL_HOLE_MAX_AREA else "large")

    if sorted(actual_classes) != ["large", "small"]:
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

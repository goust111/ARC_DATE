# Combo variant of aa4ec2a5 (index 7)
# combo_spec: D2/1+D3/3
# Name: 嵌套对象+按洞面积着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ---------------
# concretization_and_trace:
#   input:
#     - yellow background (4) with 3-4 separated top-level objects
#     - every top-level object still has a blue (1) outer shell
#     - exactly 2 objects are nested in the generator: each shell encloses one rectangular pocket,
#       and that pocket is filled by a small orange (7) inner object instead of background
#     - one nested object has inner area 2 or 3, the other has inner area 4/6/8/9/12/16
#     - remaining objects are plain solid blue polyominoes
#   output:
#     - every object gets a 1-cell red (2) outline on background cells only
#     - plain solid blue objects stay blue
#     - nested objects are classified by enclosed non-background area, not by empty-hole existence:
#         area <= 3  -> blue shell becomes teal (8), inner object becomes magenta (6)
#         area > 3   -> blue shell becomes green (3), inner object becomes yellow (4)
#     - the outline is computed on the occupied union of shell + inner object, so no red ring is
#       drawn inside the shell
#   trace_example:
#     - original task: a blue shell around a 2x2 background hole becomes teal shell + magenta hole
#     - D3-only variant: the same shell around a colored 2x2 inner object becomes teal shell while
#       the inner object keeps its input color
#     - combo here: the same shell around a colored 2x2 inner object has enclosed area 4, so it
#       becomes green shell + yellow inner object; a 1x3 inner object instead becomes teal + magenta
#     - difference: the enclosed content must first be recognized as a nested object, then measured
#       by area to choose between two recolor branches
#
# complexity_calibration:
#   - the rule only has two visible size buckets (<= 3 vs > 3), which keeps inference tractable
#   - every generated sample contains one object from each bucket, so the color split is directly
#     comparable within a single training example
#   - non-nested solid blue controls remain present, preserving the baseline "outline only" branch
#
# implicit_assumption_review_for_main:
#   - `find_components(... colors=[FG_COLOR])` assumes every transformable outer shell is blue; the
#     generator enforces that and never creates independent blue inner objects
#   - `_interior_regions` uses shell geometry only; this stays valid because nested objects are fully
#     contained inside pockets carved from the blue shell
#   - the original boolean test "has enclosed hole?" is no longer sufficient; the new branch depends
#     on counting enclosed non-background cells
#   - the D3-only assumption "preserve the inner object color" is no longer valid; combo logic must
#     overwrite the inner object with branch-specific colors
#   - outline painting must still target background cells only, because the large branch recolors the
#     inner object to yellow (4), which matches the background value
#
# placement_legality:
#   - placement uses the occupied union of shell and inner object, not just the blue shell, so nested
#     components cannot overlap each other or violate the Chebyshev spacing after insertion
#   - nested shells are produced by carving a single interior pocket from a connected shell mask, then
#     filling that pocket completely with the inner object; this guarantees legal enclosure geometry
#
# visual_observability:
#   - the orange input core makes the D3-style nesting visible before any recoloring happens
#   - both size branches appear in each sample, so the solver can compare a small inner object against
#     a larger one and infer that area, not identity, drives the shell/inner recolor pair
#   - even though the large branch turns the inner object yellow like the background, the shell turns
#     green, so the branch remains visually distinct
#
# visual_cleanliness_and_inference:
#   - there is at most one pocket per nested shell and exactly one inner color in the input, avoiding
#     clutter or ambiguous multi-object interactions inside the shell
#   - the palette stays structured: input uses bg/blue/orange, output adds red plus two recolor pairs
#   - no hidden ids, ordering, or invisible statistics are used; only visible enclosure and visible
#     inner area control the transformation
#
# verification_metric:
#   - generator samples are non-degenerate only if the detected nested class multiset is exactly
#     ["large", "small"]; this proves the key control variable is exercised in every retained sample
#   - comparing combo output against the original and D3-only behaviors on generated samples should
#     yield non-zero differences, because the nested inner objects are real colored regions and their
#     area changes both shell and inner recolor outcomes


BG_COLOR = 4
FG_COLOR = 1
INNER_INPUT_COLOR = 7
OUTLINE_COLOR = 2

SMALL_INNER_MAX_AREA = 3
SMALL_SHELL_COLOR = 8
SMALL_INNER_COLOR = 6
LARGE_SHELL_COLOR = 3
LARGE_INNER_COLOR = 4


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_regions(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    regions: list[np.ndarray] = []
    for region in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (region[:, 0] == 0)
            | (region[:, 1] == 0)
            | (region[:, 0] == sub.shape[0] - 1)
            | (region[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = region.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            regions.append(shifted)
    return regions


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


def _enclosed_nonbg_regions(
    grid: np.ndarray, outer_mask: np.ndarray, coords: np.ndarray, bg: int
) -> list[np.ndarray]:
    enclosed: list[np.ndarray] = []
    for pocket in _interior_regions(outer_mask, coords):
        vals = grid[pocket[:, 0], pocket[:, 1]]
        occupied = pocket[vals != bg]
        if len(occupied) > 0:
            enclosed.append(occupied)
    return enclosed


def _nested_colors(total_inner_area: int) -> tuple[int, int]:
    if total_inner_area <= SMALL_INNER_MAX_AREA:
        return SMALL_SHELL_COLOR, SMALL_INNER_COLOR
    return LARGE_SHELL_COLOR, LARGE_INNER_COLOR


def _nested_class(total_inner_area: int) -> str | None:
    if total_inner_area <= 0:
        return None
    return "small" if total_inner_area <= SMALL_INNER_MAX_AREA else "large"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outer outlines around blue shells, recoloring nested objects by enclosed area.

    Input structure:
    - A mostly uniform background contains several blue 4-connected outer shells, plus some solid blue objects.
    - Some blue shells enclose a fully internal pocket occupied by a smaller non-background inner object instead of background.

    Rule:
    - Draw a one-cell red (2) outline in the 8-neighborhood around every top-level object.
    - For a blue shell that encloses other colored pixels, measure the total enclosed non-background area.
    - Area <= 3 gives teal shell (8) with magenta inner object (6); area > 3 gives green shell (3) with yellow inner object (4).

    Key insight: The shell still defines the region to inspect, but the decisive signal is no longer
    an empty hole. You must detect a nested occupied interior, treat shell + inner object as one
    occupied mass for outlining, and use the inner object's visible area to choose the recolor pair.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[FG_COLOR]):
        coords = comp["coords"]
        outer_mask = _coords_mask(g.shape, coords)
        enclosed = _enclosed_nonbg_regions(g, outer_mask, coords, bg)

        occupied = outer_mask.copy()
        if enclosed:
            total_inner_area = sum(int(region.shape[0]) for region in enclosed)
            shell_color, inner_color = _nested_colors(total_inner_area)
            out[outer_mask] = shell_color
            for region in enclosed:
                out[region[:, 0], region[:, 1]] = inner_color
                occupied[region[:, 0], region[:, 1]] = True

        outline = _outline_mask(occupied)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

COMP_COUNT_RANGE = (3, 4)
NESTED_COUNT = 2

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

SMALL_NESTED_AREAS = [2, 3]
LARGE_NESTED_AREAS = [4, 6, 8, 9, 12, 16]
POCKET_DIM_MAX = 4
NESTED_PAD_RANGE = (1, 3)
NESTED_MAX_NOTCHES = 2
NESTED_EXTENSION_PROB = 0.5
NESTED_EXT_H_RANGE = (1, 4)
NESTED_EXT_SIDE_W_RANGE = (1, 3)
NESTED_POCKET_SEARCH_RETRIES = 200
NESTED_GEN_RETRIES = 50
NESTED_MIN_OUTER_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

GEN_OUTER_RETRIES = 300


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _rect_dims(area: int) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= POCKET_DIM_MAX and area // i <= POCKET_DIM_MAX
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
    if random.random() >= NESTED_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*NESTED_EXT_H_RANGE)

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
        ext_w_actual = random.randint(*NESTED_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, 0:ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _carve_pocket(mask: np.ndarray, hh: int, hw: int) -> np.ndarray | None:
    mh, mw = mask.shape
    for _ in range(NESTED_POCKET_SEARCH_RETRIES):
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
        pockets = _interior_regions(cropped, np.argwhere(cropped))
        if len(pockets) == 1 and int(pockets[0].shape[0]) == hh * hw:
            return cropped
    return None


def _make_nested_shell(target_area: int) -> np.ndarray | None:
    dims = _rect_dims(target_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*NESTED_PAD_RANGE)
    pad_bot = random.randint(*NESTED_PAD_RANGE)
    pad_left = random.randint(*NESTED_PAD_RANGE)
    pad_right = random.randint(*NESTED_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, NESTED_MAX_NOTCHES)
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
    return _carve_pocket(mask, hh, hw)


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


def _build_nested_component(target_class: str) -> dict[str, np.ndarray | str | int] | None:
    area_choices = SMALL_NESTED_AREAS if target_class == "small" else LARGE_NESTED_AREAS
    for _ in range(NESTED_GEN_RETRIES):
        target_area = random.choice(area_choices)
        outer = _make_nested_shell(target_area)
        if outer is None or int(outer.sum()) < NESTED_MIN_OUTER_AREA:
            continue
        pockets = _interior_regions(outer, np.argwhere(outer))
        if len(pockets) != 1:
            continue
        inner = _coords_mask(outer.shape, pockets[0])
        inner_area = int(inner.sum())
        if _nested_class(inner_area) != target_class:
            continue
        occupied = outer | inner
        if not is_contiguous(occupied.astype(int), background=0, connectivity=4):
            continue
        return {
            "occupied": occupied,
            "outer": outer,
            "inner": inner,
            "class": target_class,
            "inner_area": inner_area,
        }
    return None


def _build_solid_component() -> dict[str, np.ndarray | str | int] | None:
    if random.random() < SMALL_SHAPE_PROB:
        outer = _make_small_shape()
    else:
        outer = None
        for _ in range(POLY_GEN_RETRIES):
            outer = _make_polyomino()
            if outer is not None:
                break
        if outer is None:
            return None

    return {
        "occupied": outer.copy(),
        "outer": outer,
        "inner": np.zeros_like(outer, dtype=bool),
        "class": "solid",
        "inner_area": 0,
    }


def _detected_nested_classes(grid: np.ndarray) -> list[str]:
    bg = get_bg(grid)
    classes: list[str] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, colors=[FG_COLOR]):
        outer_mask = _coords_mask(grid.shape, comp["coords"])
        enclosed = _enclosed_nonbg_regions(grid, outer_mask, comp["coords"], bg)
        if enclosed:
            total_inner_area = sum(int(region.shape[0]) for region in enclosed)
            nested_class = _nested_class(total_inner_area)
            if nested_class is not None:
                classes.append(nested_class)
    return classes


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    if num_components <= NESTED_COUNT:
        return None

    indices = list(range(num_components))
    random.shuffle(indices)
    nested_indices = indices[:NESTED_COUNT]

    nested_targets = {
        nested_indices[0]: "small",
        nested_indices[1]: "large",
    }
    if random.random() < 0.5:
        nested_targets[nested_indices[0]], nested_targets[nested_indices[1]] = (
            nested_targets[nested_indices[1]],
            nested_targets[nested_indices[0]],
        )

    components: list[dict[str, np.ndarray | str | int]] = []
    for i in range(num_components):
        target_class = nested_targets.get(i)
        comp = _build_nested_component(target_class) if target_class else _build_solid_component()
        if comp is None:
            return None
        components.append(comp)

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, dict[str, np.ndarray | str | int]]] = []

    order = sorted(range(num_components), key=lambda i: -int(np.sum(components[i]["occupied"])))

    for idx in order:
        comp = components[idx]
        occ = np.asarray(comp["occupied"], dtype=bool)
        mh, mw = occ.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_a, cols_a = np.where(occ)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, prev in placed:
                prev_occ = np.asarray(prev["occupied"], dtype=bool)
                row_gap = max(0, max(r - (pr + prev_occ.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + prev_occ.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, prev_occ):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, comp))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, comp in placed:
        outer = np.asarray(comp["outer"], dtype=bool)
        inner = np.asarray(comp["inner"], dtype=bool)
        outer_rows, outer_cols = np.where(outer)
        inner_rows, inner_cols = np.where(inner)
        grid[outer_rows + r, outer_cols + c] = FG_COLOR
        grid[inner_rows + r, inner_cols + c] = INNER_INPUT_COLOR

    actual_classes = sorted(_detected_nested_classes(grid))
    if actual_classes != ["large", "small"]:
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

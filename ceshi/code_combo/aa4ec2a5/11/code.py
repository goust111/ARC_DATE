# Combo variant of aa4ec2a5 (index 11)
# combo_spec: D2/0+D10/2
# Name: 凹陷也视为洞+按洞数量分级着色

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis
# combo_design:
#   fused_rule:
#     - solver no longer counts only fully enclosed 4-connected background pockets.
#     - deep rectangular indentations that open to a bbox side also count as holes when
#       opening width <= 2 and depth >= 3.
#     - the body/hole recolor is tiered by the total count of such regions:
#       1 -> teal/magenta, 2 -> green/yellow, 3+ -> orange/gray.
#   a) concretization_and_trace:
#     - Example: start from a blue rectangle with one true inner hole and one deep top notch.
#       Original aa4ec2a5 sees only the enclosed hole, so it outputs teal body + magenta hole
#       and leaves the notch as background with red lining inside the notch.
#       D10-only would also fill the notch, but still use the same teal/magenta palette.
#       This combo must count both regions together, recolor the body green, recolor both the
#       inner hole and the notch yellow, then compute the red outline on the mask after both
#       regions are filled in. The visible difference is therefore not just "notch fills now",
#       but also "the notch can push the object into a higher color tier".
#   b) complexity_calibration:
#     - The deep-notch rule is intentionally limited to axis-aligned rectangular cul-de-sacs
#       with width 1-2 and depth 3-5; shallow or wide notches remain negatives.
#     - The count palette is limited to three tiers (1 / 2 / 3+) so the hypothesis space stays
#       small enough for ARC while still being richer than the original binary hollow-vs-solid
#       rule.
#     - Each generated sample contains three special components spanning multiple tiers plus a
#       shallow-or-wide notch negative, so both the relaxed hole definition and the count tiers
#       are supported by repeated evidence inside one grid.
#   c) implicit_assumption_audit_for_main:
#     - `_interior_holes()` alone was sufficient only when every triggering region was closed;
#       that assumption breaks, so bbox-edge notch scanning is added.
#     - `if holes:` assumed a binary recolor decision. The combo needs the exact count, so the
#       branch now depends on a palette lookup from the total number of feature regions.
#     - `solid[region] = True` must now fill both enclosed holes and deep notches; otherwise the
#       outline would incorrectly run inside a counted notch.
#     - The original yellow background is no longer viable because the 2-hole tier uses yellow
#       for counted regions. The generator therefore moves the background to black so yellow
#       holes remain visible and distinguishable from exterior background.
#   d) placement_legality:
#     - Components are placed with Chebyshev distance >= 3 and at least one-cell border
#       clearance, so the 8-neighborhood red outlines do not overlap and no component exits
#       the canvas.
#     - Deep notches are generated away from bbox corners, with disjoint carved rectangles,
#       so the solver's notch scanner sees clean independent runs.
#   e) visual_observability:
#     - Every sample contains at least one deep-only tier-1 object, at least one mixed object
#       whose notch raises the count to tier 2 or 3, and one shallow/wide notch negative that
#       stays blue. A human can therefore compare positive and negative notch cases directly.
#   f) cleanliness:
#     - Special components are simple rectangles with a small number of carved regions, while
#       filler objects remain ordinary solids. This avoids clutter from too many overlapping
#       annotations inside one object.
#   g) color_inference:
#     - Only three body/hole palette pairs are used, and each pair is tied to a visible count of
#       counted regions. The geometry explains the color change; no hidden ordering is used.
#   h) anti-degeneracy_metric:
#     - The generator rejects samples unless the combo output differs from both:
#       1) a count-tier solver that ignores deep notches, and
#       2) a deep-notch-aware solver with only the original binary teal/magenta palette.
#       This guarantees that both dimensions matter on the generated inputs.
#   i) new_solver_assumptions_and_generator_guards:
#     - Notch detection assumes rectangular mouth runs on the tight bbox boundary, not arbitrary
#       concave shapes. The generator explicitly constructs only those notches.
#     - Counted regions are assumed to be disjoint. The generator assigns notches to distinct
#       sides and rejects overlaps by exact recounting after carving.


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _edge_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        if not bool(line[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(line[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _interior_holes_in_sub(sub: np.ndarray) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    h, w = sub.shape
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == h - 1)
            | (hole[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            holes.append(hole)
    return holes


def _deep_notch_regions_in_sub(sub: np.ndarray) -> list[np.ndarray]:
    h, w = sub.shape
    regions: list[np.ndarray] = []

    for c0, c1 in _edge_runs(~sub[0]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            if sub[depth, c0 : c1 + 1].any():
                break
            if not bool(sub[depth, c0 - 1]) or not bool(sub[depth, c1 + 1]):
                break
            depth += 1
        if depth >= 3 and depth < h and sub[depth, c0 : c1 + 1].all():
            regions.append(np.array([(r, c) for r in range(depth) for c in range(c0, c1 + 1)], dtype=int))

    for c0, c1 in _edge_runs(~sub[h - 1]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            r = h - 1 - depth
            if sub[r, c0 : c1 + 1].any():
                break
            if not bool(sub[r, c0 - 1]) or not bool(sub[r, c1 + 1]):
                break
            depth += 1
        close_r = h - 1 - depth
        if depth >= 3 and close_r >= 0 and sub[close_r, c0 : c1 + 1].all():
            regions.append(
                np.array([(r, c) for r in range(h - depth, h) for c in range(c0, c1 + 1)], dtype=int)
            )

    for r0, r1 in _edge_runs(~sub[:, 0]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            if sub[r0 : r1 + 1, depth].any():
                break
            if not bool(sub[r0 - 1, depth]) or not bool(sub[r1 + 1, depth]):
                break
            depth += 1
        if depth >= 3 and depth < w and sub[r0 : r1 + 1, depth].all():
            regions.append(np.array([(r, c) for r in range(r0, r1 + 1) for c in range(depth)], dtype=int))

    for r0, r1 in _edge_runs(~sub[:, w - 1]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            c = w - 1 - depth
            if sub[r0 : r1 + 1, c].any():
                break
            if not bool(sub[r0 - 1, c]) or not bool(sub[r1 + 1, c]):
                break
            depth += 1
        close_c = w - 1 - depth
        if depth >= 3 and close_c >= 0 and sub[r0 : r1 + 1, close_c].all():
            regions.append(
                np.array([(r, c) for r in range(r0, r1 + 1) for c in range(w - depth, w)], dtype=int)
            )

    return regions


def _shift_regions(regions: list[np.ndarray], r0: int, c0: int) -> list[np.ndarray]:
    shifted: list[np.ndarray] = []
    for region in regions:
        here = region.copy()
        here[:, 0] += int(r0)
        here[:, 1] += int(c0)
        shifted.append(here)
    return shifted


def _component_regions(mask: np.ndarray, coords: np.ndarray, *, include_deep: bool) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    enclosed = _shift_regions(_interior_holes_in_sub(sub), r0, c0)
    if not include_deep:
        return enclosed
    deep = _shift_regions(_deep_notch_regions_in_sub(sub), r0, c0)
    return enclosed + deep


def _component_feature_counts(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0, 0
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return len(_interior_holes_in_sub(sub)), len(_deep_notch_regions_in_sub(sub))


def _feature_palette(count: int, *, tiered: bool) -> tuple[int, int] | None:
    if count <= 0:
        return None
    if not tiered:
        return 8, 6
    if count == 1:
        return 8, 6
    if count == 2:
        return 3, 4
    return 7, 5


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


def _render_output(grid: np.ndarray, *, include_deep: bool, tiered: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        regions = _component_regions(mask, coords, include_deep=include_deep)
        palette = _feature_palette(len(regions), tiered=tiered)

        solid = mask.copy()
        if palette is not None:
            body_color, hole_color = palette
            out[mask] = body_color
            for region in regions:
                out[region[:, 0], region[:, 1]] = hole_color
                solid[region[:, 0], region[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red outer outlines, but count both true holes and deep narrow indentations when choosing the recolor tier.

    Input structure:
    - A uniform background contains several blue 4-connected foreground components.
    - Some components contain fully enclosed background holes, some contain deep rectangular
      notches that open to one side of the component's tight bounding box, and some contain only
      shallow or wide concavities that should remain ordinary background.

    Rule:
    - Draw a one-cell 8-neighborhood red outline (2) around every foreground component.
    - For each component, count every enclosed hole plus every qualifying deep indentation
      (opening width <= 2 and depth >= 3).
    - Recolor by that total before outlining:
      1 region -> teal body (8) and magenta counted region(s) (6),
      2 regions -> green body (3) and yellow counted region(s) (4),
      3 or more regions -> orange body (7) and gray counted region(s) (5).
    - Components with zero counted regions stay blue (1).

    Key insight:
    - The solver must first convert each component into a "filled feature mask" where both closed
      holes and deep notches are treated as solid for outlining, and only then map the total
      number of those regions to one of the three color tiers.
    """
    return _render_output(grid, include_deep=True, tiered=True)


# --------------- generator (combo) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (22, 32)
GRID_W_RANGE = (22, 32)
BG_COLOR = 0
FG_COLOR = 1

# Component counts
COMP_COUNT_RANGE = (4, 5)

# Placement
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 400

# Top-level generation
GEN_OUTER_RETRIES = 500

# Solid filler shapes
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 40
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Structured feature components
FEATURE_GEN_RETRIES = 120
FEATURE_MIN_AREA = 14
FEATURE_CORE_RANGE = (3, 6)
FEATURE_MARGIN_SLACK_RANGE = (0, 2)
FEATURE_HOLE_GAP_RANGE = (1, 2)
FEATURE_ROWCOL_SLACK_RANGE = (0, 1)
FEATURE_MAX_DIM = 14
DEEP_NOTCH_DEPTH_RANGE = (3, 5)
DEEP_NOTCH_WIDTH_RANGE = (1, 2)
NOTCH_OFFSET_RETRIES = 60
NEGATIVE_NOTCH_GEN_RETRIES = 80
SHALLOW_NOTCH_DEPTH_RANGE = (1, 2)
WIDE_NOTCH_DEPTH_RANGE = (3, 5)

HOLE_OPTIONS_SINGLE = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (2, 3), (3, 2)]
HOLE_OPTIONS_MULTI = [(1, 2), (2, 1), (2, 2)]


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_special_region(mask: np.ndarray) -> bool:
    enclosed, deep = _component_feature_counts(mask)
    return enclosed > 0 or deep > 0


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
    if _has_special_region(mask):
        return None
    return mask


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


def _make_clean_solid() -> np.ndarray | None:
    if random.random() < SMALL_SHAPE_PROB:
        mask = _make_small_shape()
        return mask if not _has_special_region(mask) else None
    for _ in range(POLY_GEN_RETRIES):
        mask = _make_polyomino()
        if mask is not None:
            return mask
    return None


def _sample_deep_specs(num_deep: int) -> list[dict[str, int | str]]:
    sides = random.sample(["top", "bottom", "left", "right"], k=num_deep)
    specs: list[dict[str, int | str]] = []
    for side in sides:
        specs.append(
            {
                "side": side,
                "width": random.randint(*DEEP_NOTCH_WIDTH_RANGE),
                "depth": random.randint(*DEEP_NOTCH_DEPTH_RANGE),
            }
        )
    return specs


def _notch_region(shape: tuple[int, int], *, side: str, width: int, depth: int, offset: int) -> np.ndarray | None:
    h, w = shape
    if side in ("top", "bottom"):
        if w - width - 2 < 0 or depth >= h:
            return None
        if not (1 <= offset <= w - width - 1):
            return None
        rows = range(0, depth) if side == "top" else range(h - depth, h)
        cols = range(offset, offset + width)
    else:
        if h - width - 2 < 0 or depth >= w:
            return None
        if not (1 <= offset <= h - width - 1):
            return None
        rows = range(offset, offset + width)
        cols = range(0, depth) if side == "left" else range(w - depth, w)
    return np.array([(r, c) for r in rows for c in cols], dtype=int)


def _carve_regions(mask: np.ndarray, regions: list[np.ndarray]) -> np.ndarray:
    carved = mask.copy()
    for region in regions:
        carved[region[:, 0], region[:, 1]] = False
    return carved


def _assign_deep_regions(shape: tuple[int, int], specs: list[dict[str, int | str]]) -> list[np.ndarray] | None:
    if not specs:
        return []

    for _ in range(NOTCH_OFFSET_RETRIES):
        regions: list[np.ndarray] = []
        occupied: set[tuple[int, int]] = set()
        ok = True
        for spec in specs:
            side = str(spec["side"])
            width = int(spec["width"])
            depth = int(spec["depth"])
            limit = shape[1] - width - 1 if side in ("top", "bottom") else shape[0] - width - 1
            if limit < 1:
                ok = False
                break
            offset = random.randint(1, limit)
            region = _notch_region(shape, side=side, width=width, depth=depth, offset=offset)
            if region is None:
                ok = False
                break
            cells = {(int(r), int(c)) for r, c in region}
            if occupied & cells:
                ok = False
                break
            occupied |= cells
            regions.append(region)
        if ok:
            return regions
    return None


def _sample_hole_dims(num_holes: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for i in range(num_holes):
        options = HOLE_OPTIONS_SINGLE if num_holes == 1 and i == 0 else HOLE_OPTIONS_MULTI
        dims.append(random.choice(options))
    return dims


def _make_feature_component(num_holes: int, num_deep: int) -> np.ndarray | None:
    for _ in range(FEATURE_GEN_RETRIES):
        deep_specs = _sample_deep_specs(num_deep)
        top_need = max((int(spec["depth"]) + 1 for spec in deep_specs if spec["side"] == "top"), default=2)
        bottom_need = max((int(spec["depth"]) + 1 for spec in deep_specs if spec["side"] == "bottom"), default=2)
        left_need = max((int(spec["depth"]) + 1 for spec in deep_specs if spec["side"] == "left"), default=2)
        right_need = max((int(spec["depth"]) + 1 for spec in deep_specs if spec["side"] == "right"), default=2)

        top = top_need + random.randint(*FEATURE_MARGIN_SLACK_RANGE)
        bottom = bottom_need + random.randint(*FEATURE_MARGIN_SLACK_RANGE)
        left = left_need + random.randint(*FEATURE_MARGIN_SLACK_RANGE)
        right = right_need + random.randint(*FEATURE_MARGIN_SLACK_RANGE)

        holes = _sample_hole_dims(num_holes)
        if holes:
            n_cols = 1 if num_holes == 1 or random.random() < 0.5 else 2
            n_rows = (num_holes + n_cols - 1) // n_cols
            row_heights = [0] * n_rows
            col_widths = [0] * n_cols
            for idx, (hh, hw) in enumerate(holes):
                rr = idx // n_cols
                cc = idx % n_cols
                row_heights[rr] = max(row_heights[rr], hh + random.randint(*FEATURE_ROWCOL_SLACK_RANGE))
                col_widths[cc] = max(col_widths[cc], hw + random.randint(*FEATURE_ROWCOL_SLACK_RANGE))
            row_gaps = [random.randint(*FEATURE_HOLE_GAP_RANGE) for _ in range(max(0, n_rows - 1))]
            col_gaps = [random.randint(*FEATURE_HOLE_GAP_RANGE) for _ in range(max(0, n_cols - 1))]
            inner_h = sum(row_heights) + sum(row_gaps)
            inner_w = sum(col_widths) + sum(col_gaps)
        else:
            n_cols = 0
            row_heights = []
            col_widths = []
            row_gaps = []
            col_gaps = []
            inner_h = random.randint(*FEATURE_CORE_RANGE)
            inner_w = random.randint(*FEATURE_CORE_RANGE)

        base_h = top + inner_h + bottom
        base_w = left + inner_w + right
        if base_h > FEATURE_MAX_DIM or base_w > FEATURE_MAX_DIM:
            continue

        mask = np.ones((base_h, base_w), dtype=bool)
        if holes:
            cursor_r = top
            idx = 0
            for rr, slot_h in enumerate(row_heights):
                cursor_c = left
                for cc, slot_w in enumerate(col_widths):
                    if idx >= num_holes:
                        break
                    hh, hw = holes[idx]
                    hole_r = cursor_r + random.randint(0, slot_h - hh)
                    hole_c = cursor_c + random.randint(0, slot_w - hw)
                    mask[hole_r : hole_r + hh, hole_c : hole_c + hw] = False
                    cursor_c += slot_w
                    if cc < len(col_gaps):
                        cursor_c += col_gaps[cc]
                    idx += 1
                cursor_r += slot_h
                if rr < len(row_gaps):
                    cursor_r += row_gaps[rr]

        deep_regions = _assign_deep_regions(mask.shape, deep_specs)
        if deep_regions is None:
            continue
        mask = _carve_regions(mask, deep_regions)

        cropped = _crop_mask(mask)
        if cropped is None:
            continue
        if not is_contiguous(cropped.astype(int), background=0, connectivity=4):
            continue
        enclosed, deep = _component_feature_counts(cropped)
        if enclosed != num_holes or deep != num_deep:
            continue
        if int(cropped.sum()) < FEATURE_MIN_AREA:
            continue
        return cropped
    return None


def _make_negative_notch_component() -> np.ndarray | None:
    for _ in range(NEGATIVE_NOTCH_GEN_RETRIES):
        base_h = random.randint(6, 11)
        base_w = random.randint(6, 11)
        mask = np.ones((base_h, base_w), dtype=bool)
        side = random.choice(["top", "bottom", "left", "right"])
        use_shallow = random.random() < 0.5

        if use_shallow:
            width = random.randint(1, 2)
            depth_range = SHALLOW_NOTCH_DEPTH_RANGE
        else:
            width = 3
            depth_range = WIDE_NOTCH_DEPTH_RANGE

        max_depth = base_h - 1 if side in ("top", "bottom") else base_w - 1
        if max_depth < depth_range[0]:
            continue
        depth = random.randint(depth_range[0], min(depth_range[1], max_depth))
        limit = base_w - width - 1 if side in ("top", "bottom") else base_h - width - 1
        if limit < 1:
            continue
        offset = random.randint(1, limit)
        region = _notch_region(mask.shape, side=side, width=width, depth=depth, offset=offset)
        if region is None:
            continue
        carved = _carve_regions(mask, [region])
        cropped = _crop_mask(carved)
        if cropped is None:
            continue
        if not is_contiguous(cropped.astype(int), background=0, connectivity=4):
            continue
        if _component_feature_counts(cropped) != (0, 0):
            continue
        return cropped
    return None


def _component_specs(num_components: int) -> list[tuple[str, tuple[int, int] | None]]:
    specs: list[tuple[str, tuple[int, int] | None]] = [
        ("feature", (0, 1)),
        ("feature", random.choice([(1, 1), (0, 2), (2, 0)])),
        ("feature", random.choice([(1, 2), (2, 1)])),
        ("negative", None),
    ]
    while len(specs) < num_components:
        specs.append(("solid", None))
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    num_components = random.randint(*COMP_COUNT_RANGE)

    components: list[np.ndarray] = []
    for kind, payload in _component_specs(num_components):
        if kind == "feature":
            assert payload is not None
            mask = _make_feature_component(payload[0], payload[1])
        elif kind == "negative":
            mask = _make_negative_notch_component()
        else:
            mask = _make_clean_solid()
        if mask is None:
            return None
        components.append(mask)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        rows_a, cols_a = np.where(mask)

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, h - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, w - mw - BORDER_MIN))
            if r + mh > h - BORDER_MIN or c + mw > w - BORDER_MIN:
                continue

            rows_here, cols_here = rows_a + r, cols_a + c
            conflict = False
            for pr, pc, pm in placed:
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_here, cols_here, pr, pc, pm):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask))
                placed_ok = True
                break

        if not placed_ok:
            return None

    comp_stats: list[tuple[int, int, int]] = []
    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR
        enclosed, deep = _component_feature_counts(mask)
        comp_stats.append((enclosed, deep, enclosed + deep))

    try:
        combo_out = main(grid)
        count_only_out = _render_output(grid, include_deep=False, tiered=True)
        deep_binary_out = _render_output(grid, include_deep=True, tiered=False)
    except Exception:
        return None

    tiers = {min(total, 3) for _, _, total in comp_stats if total > 0}
    if len(tiers) < 2:
        return None
    if not any(deep > 0 for _, deep, _ in comp_stats):
        return None
    if not any(enclosed > 0 and deep > 0 for enclosed, deep, _ in comp_stats):
        return None
    if not any(deep > 0 and total >= 2 for _, deep, total in comp_stats):
        return None
    if np.array_equal(combo_out, count_only_out):
        return None
    if np.array_equal(combo_out, deep_binary_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

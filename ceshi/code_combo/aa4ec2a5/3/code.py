# Combo variant of aa4ec2a5 (index 3)
# combo_spec: D2/0+D5/2
# Name: 按洞数量分级着色+洞按面积分色填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis
# ---------------
# a) Concrete behavior and trace
#    - Input still shows several blue 4-connected objects on one uniform background, but hollow
#      objects may now contain 1, 2, or 3 enclosed rectangular holes rather than only one.
#    - Output applies two interacting judgments:
#        * object body color depends on the number of enclosed holes
#          1 hole -> teal(8), 2 holes -> green(3), 3+ holes -> orange(7)
#        * each hole color depends on its own area
#          1-2 cells -> magenta(6), 3-5 cells -> yellow(4), >5 cells -> green(3)
#    - Trace example:
#        original task: a shell with two holes would be recolored uniformly as teal body with
#        magenta holes, then outlined in red.
#        combo task: the same shell is first recognized as a 2-hole object, so its body becomes
#        green; then each hole is recolored independently by area, e.g. a 2-cell hole becomes
#        magenta while a 4-cell hole becomes yellow; finally the outer red outline is drawn using
#        the object-plus-holes union as the solid mask.
#    - The visible delta is therefore not just "hollow vs solid": the solver must count holes to
#      pick the shell color and measure each hole to pick its fill color.
#
# b) Complexity calibration
#    - The count rule uses only three tiers (1 / 2 / 3+) and the area rule uses only three bands
#      (small / medium / large). That keeps the hypothesis space small enough for ARC.
#    - To avoid visually weak samples, the generator forces at least two hollow count tiers per
#      grid and at least two hole-area bands overall, with one hollow object containing mixed
#      bands whenever possible.
#    - The generator also caps hole counts at 3 and keeps holes rectangular, so scenes remain
#      legible instead of turning into dense clutter.
#
# c) Implicit assumption audit for main()
#    - `_interior_holes(...)` still correctly identifies enclosed 4-connected background regions;
#      the original detection mechanism remains valid.
#    - The original branch `if holes: out[mask] = 8` assumed all hollow objects shared one body
#      color. This is no longer valid and is replaced with a count-tier lookup.
#    - The original `out[hole] = 6` assumed one universal hole color. This breaks under D5/2 and
#      is replaced by a per-hole area mapping.
#    - The outline still must be computed from `solid = mask + holes`; otherwise each interior
#      hole would incorrectly get its own red border. That original assumption still holds.
#    - D2/0's original hole-side palette member (magenta / yellow / gray) cannot remain a fixed
#      object-level choice in the combo, because D5/2 explicitly overrides hole color by area.
#      In the fused rule only the body color comes from hole count; hole cells come from area.
#
# d) Placement legality
#    - Hollow components are generated from one solid rectangle, then multiple interior rectangles
#      are carved out with at least one foreground cell separating every hole from the exterior
#      and from neighboring holes. This guarantees enclosed holes and a 4-connected shell.
#    - Components are placed with Chebyshev distance >= 3 and at least one-cell border clearance,
#      so the red outlines remain disjoint and never need to overwrite another object.
#
# e) Visual observability
#    - Yellow medium holes would be ambiguous on a yellow background, so the generator switches
#      the background away from yellow to maroon(9). This keeps all three hole bands visible.
#    - A second ambiguity appears when a 2-hole object's body is green and a large hole would also
#      be green. The solver still supports that case, but the generator forbids large holes inside
#      2-hole objects so the count signal stays readable from examples.
#    - Each sample therefore exposes both control variables through visible structure: different
#      body colors across hollow objects and different hole colors within or across those objects.
#
# f) Visual cleanliness
#    - Hole counts stay low, holes are rectangular, and spacing between holes is at least one wall
#      cell, so internal markings stay easy to parse.
#    - Solid objects remain present as blue controls, which makes the transition from "no holes"
#      to "counted holes with recoloring" easier to infer.
#
# g) Change inferability
#    - The color mapping is structural rather than arbitrary: body color is tied to how many holes
#      an object encloses, while hole color is tied to the visible size of each enclosed region.
#    - The generator guarantees repeated evidence for both mappings by producing multiple hollow
#      tiers and multiple area bands in one grid. This prevents degeneration into the original
#      task or into a single-rule subset where one of the two variables never matters.


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


def _body_color_for_hole_count(num_holes: int) -> int | None:
    if num_holes <= 0:
        return None
    if num_holes == 1:
        return 8
    if num_holes == 2:
        return 3
    return 7


def _hole_fill_color(area: int) -> int:
    if area <= 2:
        return 6
    if area <= 5:
        return 4
    return 3


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _hole_areas_in_mask(mask: np.ndarray) -> list[int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    return sorted(len(hole) for hole in _interior_holes(mask, coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red outlines around objects while combining hole-count body colors with area-based holes.

    Input structure:
    - A uniform background contains several blue 4-connected foreground components.
    - Some components are solid, while others enclose one or more fully sealed 4-connected
      background regions inside their bounding boxes.

    Rule:
    - Every foreground component gets a one-cell red outline over the 8-neighborhood exterior.
    - Solid components remain blue (1).
    - Hollow component bodies are recolored by the number of enclosed holes:
      1 hole -> teal (8), 2 holes -> green (3), 3 or more holes -> orange (7).
    - Each enclosed hole is filled independently by its own area:
      1-2 cells -> magenta (6), 3-5 cells -> yellow (4), and more than 5 -> green (3).

    Key insight: Solving the task requires two linked measurements on each hollow object:
    count the number of enclosed holes to recolor the shell, then measure every hole's area to
    recolor the hole itself, while still sealing those holes into the temporary solid mask before
    computing the outer red outline.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        body_color = _body_color_for_hole_count(len(holes))

        solid = mask.copy()
        if body_color is not None:
            out[mask] = body_color
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = _hole_fill_color(len(hole))
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (combo) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (20, 31)
GRID_W_RANGE = (20, 30)
BG_COLOR = 9
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
HOLE_DIM_MAX = 5
HOLE_AREA_BANDS = {
    "small": [1, 2],
    "medium": [3, 4, 5],
    "large": [6, 8, 9, 10, 12, 15, 16],
}
HOLLOW_OUTER_MARGIN_RANGE = (1, 3)
HOLLOW_HOLE_GAP_RANGE = (2, 3)
HOLLOW_ROW_GAP_RANGE = (2, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_MAX_EXTENSIONS = 2
HOLLOW_EXTENSION_PROB = 0.6
HOLLOW_EXT_H_RANGE = (1, 3)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_GEN_RETRIES = 80
HOLLOW_MIN_AREA = 12

# Placement
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 400

# Top-level generation
GEN_OUTER_RETRIES = 400


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


def _band_for_area(area: int) -> str:
    if area <= 2:
        return "small"
    if area <= 5:
        return "medium"
    return "large"


def _sample_hole_area(band: str) -> int:
    return random.choice(HOLE_AREA_BANDS[band])


def _areas_for_hole_count(num_holes: int) -> list[int]:
    if num_holes == 1:
        band = random.choice(["medium", "large", "large", "small"])
        return [_sample_hole_area(band)]
    if num_holes == 2:
        pattern = random.choice(
            [
                ("small", "medium"),
                ("small", "medium"),
                ("small", "small"),
                ("medium", "medium"),
            ]
        )
        return [_sample_hole_area(band) for band in pattern]

    pattern = random.choice(
        [
            ("small", "medium", "large"),
            ("small", "large", "large"),
            ("medium", "medium", "large"),
            ("small", "medium", "medium"),
        ]
    )
    return [_sample_hole_area(band) for band in pattern]


def _choose_hole_plans(num_hollow: int) -> list[list[int]]:
    if num_hollow == 2:
        counts = list(random.choice([(1, 2), (1, 3), (2, 3)]))
    else:
        counts = [1, 2, 3]

    plans = [_areas_for_hole_count(count) for count in counts]
    random.shuffle(plans)
    return plans


def _layout_holes(
    hole_dims: list[tuple[int, int]]
) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    dims = hole_dims[:]
    random.shuffle(dims)
    n = len(dims)

    if n == 1:
        rows = [dims]
    elif n == 2:
        rows = [dims] if random.random() < 0.6 else [[dims[0]], [dims[1]]]
    else:
        pattern = random.choice(["one_row", "two_rows"])
        if pattern == "one_row":
            rows = [dims]
        else:
            split = random.choice([1, 2])
            rows = [dims[:split], dims[split:]]

    placements: list[tuple[int, int, int, int]] = []
    row_top = 0
    layout_w = 0
    for row_idx, row in enumerate(rows):
        row_h = max(h for h, _ in row)
        gap = random.randint(*HOLLOW_HOLE_GAP_RANGE)
        row_w = sum(w for _, w in row) + gap * max(0, len(row) - 1)
        layout_w = max(layout_w, row_w)

        col_left = 0
        for hh, hw in row:
            r = row_top + random.randint(0, row_h - hh)
            placements.append((r, col_left, hh, hw))
            col_left += hw + gap

        if row_idx != len(rows) - 1:
            row_top += row_h + random.randint(*HOLLOW_ROW_GAP_RANGE)
        else:
            row_top += row_h

    return row_top, layout_w, placements


def _make_hollow_component(hole_areas: list[int]) -> np.ndarray | None:
    hole_dims = []
    for area in hole_areas:
        dims = _rect_dims(area)
        if dims is None:
            return None
        hole_dims.append(dims)

    layout = _layout_holes(hole_dims)
    if layout is None:
        return None
    layout_h, layout_w, placements = layout

    pad_top = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
    pad_bot = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
    pad_left = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)
    pad_right = random.randint(*HOLLOW_OUTER_MARGIN_RANGE)

    mask = np.ones((layout_h + pad_top + pad_bot, layout_w + pad_left + pad_right), dtype=bool)
    for hr, hc, hh, hw in placements:
        rr = pad_top + hr
        cc = pad_left + hc
        mask[rr : rr + hh, cc : cc + hw] = False

    base_h, base_w = mask.shape
    for _ in range(random.randint(0, HOLLOW_MAX_NOTCHES)):
        corner = random.choice(["tl", "tr", "bl", "br"])
        nh = random.randint(1, max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1))
        nw = random.randint(1, max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1))
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    for _ in range(random.randint(0, HOLLOW_MAX_EXTENSIONS)):
        mask = _extend_mask(mask)

    mask = _crop_mask(mask)
    if mask is None:
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    if mask.sum() < HOLLOW_MIN_AREA:
        return None
    if _hole_areas_in_mask(mask) != sorted(hole_areas):
        return None
    return mask


def _component_hole_stats(grid: np.ndarray) -> list[tuple[int, list[int]]]:
    stats: list[tuple[int, list[int]]] = []
    for comp in find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True):
        mask = _coords_mask(grid.shape, comp["coords"])
        holes = _interior_holes(mask, comp["coords"])
        stats.append((len(holes), sorted(len(hole) for hole in holes)))
    return stats


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))
    hole_plans = _choose_hole_plans(num_hollow)

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    hollow_plan_by_index: dict[int, list[int]] = {}
    for idx, plan in zip(hollow_indices, hole_plans):
        hollow_plan_by_index[idx] = plan

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in hollow_indices:
            plan = hollow_plan_by_index[i]
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(plan)
                if mask is not None:
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
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        if mh > H - 2 * BORDER_MIN or mw > W - 2 * BORDER_MIN:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, H - mh - BORDER_MIN)
            c = random.randint(BORDER_MIN, W - mw - BORDER_MIN)

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

    stats = _component_hole_stats(grid)
    hollow_stats = [(count, areas) for count, areas in stats if count > 0]
    if len(hollow_stats) < 2:
        return None

    count_tiers = {min(count, 3) for count, _ in hollow_stats}
    if len(count_tiers) < 2:
        return None

    area_bands = {_band_for_area(area) for _, areas in hollow_stats for area in areas}
    if len(area_bands) < 2 or area_bands == {"small"}:
        return None

    if not any(len({_band_for_area(area) for area in areas}) >= 2 for _, areas in hollow_stats if len(areas) >= 2):
        return None

    if any(count == 2 and any(area > 5 for area in areas) for count, areas in hollow_stats):
        return None

    if not any(count >= 2 for count, _ in hollow_stats):
        return None

    if not any(area > 2 for _, areas in hollow_stats for area in areas):
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

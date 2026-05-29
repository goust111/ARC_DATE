# Variant of aa4ec2a5
# Dimension: D5 — 洞的处理方式 (transform)
# Change: 被封闭的背景色洞重着色为品红色... → 洞按面积大小用不同颜色填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the D5 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input still contains 3-4 blue 4-connected objects on a yellow background.
#    - Some objects are hollow. Unlike the original generator, a hollow object may now contain
#      1-3 enclosed rectangular holes, and those holes may have different areas.
#    - Output still draws a 1-pixel red 8-neighborhood outline around every object.
#    - Hollow objects are still recolored teal, but each enclosed hole is filled independently:
#      area 1-2 -> magenta(6), area 3-5 -> yellow(4), area >5 -> green(3).
#    - Small trace:
#      original logic: detect a hollow component -> recolor all its fg cells to 8 -> fill every
#      hole with 6 -> compute outline from fg union holes.
#      new logic: detect the same holes -> recolor fg cells to 8 -> for each hole compute
#      len(hole) and fill it with 6/4/3 -> compute the same outline from fg union holes.
#      The only semantic delta in main() is that hole color is no longer object-level constant.
#
# b) Complexity calibration
#    - If every hollow object had only one tiny hole, many samples would still look like the
#      original task because small holes remain magenta. That would hide the variant.
#    - Therefore each generated grid must contain at least two size bands, and at least one of
#      them must be medium or large. This forces visible departures from the original behavior.
#    - The number of size bands stays small (three) and repeated across examples, so the mapping
#      is still learnable from 2-4 ARC examples.
#
# c) Hidden assumptions in the original main()
#    - `holes = _interior_holes(...)` assumes holes are 4-connected bg regions strictly inside
#      the component bbox. This remains valid.
#    - `if holes: out[mask] = 8` assumes hollow-vs-solid is still object-level. This remains valid.
#    - `for hole in holes: out[...] = 6` assumed one universal hole color. This is no longer valid
#      and must be replaced with a per-hole color mapping from hole area.
#    - `solid[hole] = True` assumed holes should be absorbed before outlining. This remains valid,
#      including for yellow medium holes that visually match the background.
#    - `out[outline & (out == bg)] = 2` assumes only untouched background should receive the outer
#      outline. This remains safe because components are placed far apart and outlines never need
#      to overwrite colored holes inside another object.
#
# d) Placement legality
#    - Objects still obey border clearance >= 1 and pairwise Chebyshev distance >= 3, so the red
#      outlines do not collide.
#    - Hollow components are built from a solid rectangle with optional corner notches / side
#      extension, then one or more interior rectangles are carved out as holes. Because the holes
#      stay at least one cell away from the outer boundary and from each other, the shell remains
#      4-connected and solver hole detection is stable.
#
# e) Visual observability
#    - Yellow medium holes are the same color as the background, so a sample containing only
#      yellow holes would be ambiguous. To avoid that, the generator always includes at least two
#      size bands in a grid, usually with one object containing multiple holes of different sizes.
#    - This makes the rule readable as "hole size controls fill color" rather than as an
#      accidental object-specific recoloring.
#
# f) Visual cleanliness
#    - Holes are rectangular and separated by at least one fg corridor, so multiple markings inside
#      one object remain legible.
#    - External outlines stay disjoint because spacing rules are unchanged.
#
# g) Color-change inferability
#    - The mapping uses only three colors and only the visible area of each enclosed region.
#    - No hidden ordering, palette arithmetic, or non-visual metadata is used.
#    - Supporting adjustment: the original generator only allowed hole dimensions up to 4, which
#      cannot realize area 5. This variant raises the hole dimension cap to 5 so the medium band
#      3-5 is fully representable; this is a necessary D5 compatibility change, not a new rule.


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
    Draw a red outline around each object, while classifying enclosed holes by area.

    Input structure:
    - The grid has a dominant yellow background and several blue 4-connected objects.
    - Some objects are solid; some objects enclose one or more 4-connected background holes.

    Rule:
    - Every foreground component gets a one-cell red outline over its 8-neighborhood exterior.
    - A component that contains enclosed holes is recolored teal (8).
    - Each enclosed hole is then filled by its own area: 1-2 cells -> magenta (6),
      3-5 cells -> yellow (4), and more than 5 cells -> green (3).

    Key insight: The outline is computed after treating the holes as sealed into the object, but
    the fill color of each hole depends on that hole's area rather than on the object as a whole.
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
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = _hole_fill_color(len(hole))
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

HOLE_DIM_MAX = 5
HOLE_AREA_BANDS = {
    "small": [1, 2],
    "medium": [3, 4, 5],
    "large": [6, 8, 9, 12, 16],
}
GRID_BAND_PATTERNS = [
    ("small", "medium"),
    ("small", "medium"),
    ("small", "large"),
    ("medium", "large"),
    ("small", "medium", "large"),
]
HOLE_GAP_RANGE = (2, 3)
ROW_GAP_RANGE = (2, 3)
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_GEN_RETRIES = 60
HOLLOW_MIN_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

GEN_OUTER_RETRIES = 300


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


def _sample_hole_area(band: str) -> int:
    return random.choice(HOLE_AREA_BANDS[band])


def _choose_hole_plans(num_hollow: int) -> list[list[int]]:
    bands = list(random.choice(GRID_BAND_PATTERNS))
    areas = [_sample_hole_area(band) for band in bands]

    if num_hollow == 2 and random.random() < 0.4:
        areas.append(_sample_hole_area(random.choice(bands)))

    random.shuffle(areas)
    plans = [[] for _ in range(num_hollow)]

    # Ensure every hollow component gets at least one hole.
    for idx in range(num_hollow):
        plans[idx].append(areas[idx])
    for area in areas[num_hollow:]:
        target = random.randrange(num_hollow)
        if len(plans[target]) >= 3:
            target = min(range(num_hollow), key=lambda i: len(plans[i]))
        plans[target].append(area)

    for plan in plans:
        random.shuffle(plan)
    random.shuffle(plans)
    return plans


def _layout_holes(hole_dims: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
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
        gap = random.randint(*HOLE_GAP_RANGE)
        row_w = sum(w for _, w in row) + gap * max(0, len(row) - 1)
        layout_w = max(layout_w, row_w)

        col_left = 0
        for hh, hw in row:
            r = row_top + random.randint(0, row_h - hh)
            placements.append((r, col_left, hh, hw))
            col_left += hw + gap

        if row_idx != len(rows) - 1:
            row_top += row_h + random.randint(*ROW_GAP_RANGE)
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

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    mask = np.ones((layout_h + pad_top + pad_bot, layout_w + pad_left + pad_right), dtype=bool)
    for hr, hc, hh, hw in placements:
        rr = pad_top + hr
        cc = pad_left + hc
        mask[rr : rr + hh, cc : cc + hw] = False

    num_notches = random.randint(0, HOLLOW_MAX_NOTCHES)
    base_h, base_w = mask.shape
    for _ in range(num_notches):
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

    mask = _extend_mask(mask, *mask.shape)
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

# Combo variant of aa4ec2a5 (index 13)
# combo_spec: D12/2+D13/0
# Name: 图例定义颜色映射+图例指示着色方案

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, is_contiguous, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    The base task only had implicit rules: every foreground component got an
#    8-neighborhood outline, and hollow ones also changed shell/hole colors.
#    Here every input carries one explicit edge legend anchored at the top-left
#    corner, either as a 5x14 top strip or as its 14x5 transpose on the left.
#    The legend fuses two instruction styles in one region:
#    - a 2x3 chained color-pair block encoding object->outline, object->hollow,
#      and hollow->hole;
#    - two miniature result icons: a solid outlined square and a hollow outlined
#      square, showing which palette branch belongs to solid vs hollow geometry.
#    Example trace:
#    - Old pipeline: detect all non-background components -> if holes then paint
#      fixed teal/magenta -> draw fixed red outline.
#    - New pipeline: detect legend orientation -> decode four functional colors
#      from the chained pairs and verify them against the two state icons ->
#      mask away the legend band -> process only source-color components ->
#      solid objects stay in source color, hollow ones use decoded hollow/hole
#      colors, and every object gets the decoded outline color.
#    The geometry rule is unchanged; only the way the palette and the two branch
#    conditions are communicated has changed.
# b) Complexity calibration:
#    The legend is compact and canonical, so the decoding burden is real but not
#    excessive. Only four functional colors vary, and each appears multiple times
#    inside the legend. Every generated grid still contains both solid and hollow
#    objects, so both branches of the recoloring rule are repeatedly evidenced.
# c) Hidden-assumption audit of main():
#    - `get_bg(...)` still assumes the background is the majority color; the
#      generator preserves a large empty field and keeps the legend small.
#    - The original "all non-bg cells are task objects" assumption is broken by
#      the legend, so the solver now blanks out the whole legend+separator band
#      before any component search.
#    - The original hard-coded colors 2/8/6 no longer hold. All recolor targets
#      are decoded from the legend and then verified by canonical icon shapes.
#    - The original solver could scan all colors; the new one must only scan the
#      decoded source object color, because the legend also contains output-role
#      colors.
#    - `_interior_holes(...)` still relies on bbox-local background components;
#      this remains valid because generator places real objects wholly outside the
#      legend band and preserves at least one separator row/column.
# d) Placement legality:
#    The generator still uses 3-4 components with 1-2 hollow ones, Chebyshev
#    distance >= 3, and border clearance >= 1, but it now builds a separate main
#    field below or right of the legend separator so outlines cannot collide with
#    legend cells.
# e) Visual observability:
#    A human sees one direct color-mapping cue (the chained pairs) and two state
#    cues (solid-result icon and hollow-result icon). Across 2-4 samples, the
#    varying palette is visible while the icon geometry stays fixed, making the
#    rule inferable without relying on raw color ids.
# f) Visual cleanliness:
#    The legend is confined to the corner, separated from the task field by one
#    full background row/column, and the main field keeps the original component
#    spacing. No output outline can overwrite the legend.
# g) Color-change inferability:
#    Only four non-background functional roles must be learned: object, outline,
#    hollow shell, and hole fill. The chained pair block exposes their mapping,
#    and the two icons expose which branch uses which target colors.
# h) Post-generation metric to detect degeneration:
#    We will later measure whether (1) every sample decodes a non-base palette,
#    and (2) both solid and hollow branches are active in the main field. If both
#    hold, the combo logic is behaviorally active rather than collapsing to the
#    original fixed-palette task.
# i) New generator obligations:
#    - The legend must be exactly the canonical 5x14 layout or its transpose.
#    - It must be top-left anchored, with a full separator row/column of bg.
#    - All real objects must use the decoded source color and live entirely in
#      the main field outside the masked legend band.


LEGEND_H = 5
LEGEND_W = 14
LEGEND_SEP = 1
ORIENTATIONS = ("top", "left")

# Overall grid bounds (kept within the original task's envelope).
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
MIN_FIELD_DIM = 12

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
SMALL_SHAPE_PROB = 0.25
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


def _build_combo_legend(
    bg: int,
    object_color: int,
    outline_color: int,
    hollow_color: int,
    hole_color: int,
) -> np.ndarray:
    legend = np.full((LEGEND_H, LEGEND_W), bg, dtype=int)

    # Chained color-pair block: object->outline, object->hollow, hollow->hole.
    legend[0, 0:3] = [object_color, object_color, hollow_color]
    legend[1, 0:3] = [outline_color, hollow_color, hole_color]

    solid_out = np.full((4, 4), outline_color, dtype=int)
    solid_out[1:3, 1:3] = object_color
    paste(legend, solid_out, 0, 4)

    hollow_out = np.full((5, 5), outline_color, dtype=int)
    inner = np.full((3, 3), hollow_color, dtype=int)
    inner[1, 1] = hole_color
    hollow_out[1:4, 1:4] = inner
    paste(legend, hollow_out, 0, 9)

    return legend


def _decode_combo_legend(crop: np.ndarray, bg: int) -> dict[str, int]:
    coords = np.argwhere(crop != bg)
    assert coords.size > 0, "empty legend crop"
    assert bbox_coords(coords) == (0, 0, LEGEND_H - 1, LEGEND_W - 1), "legend bbox mismatch"

    assert np.all(crop[2:, 0:3] == bg), "pair block spill"
    assert np.all(crop[:, 3] == bg), "first separator column malformed"
    assert np.all(crop[4, 4:8] == bg), "solid icon spill"
    assert np.all(crop[:, 8] == bg), "second separator column malformed"

    object_color = int(crop[0, 0])
    outline_color = int(crop[1, 0])
    hollow_color = int(crop[0, 2])
    hole_color = int(crop[1, 2])
    assert crop[0, 1] == object_color, "pair block object repetition malformed"
    assert crop[1, 1] == hollow_color, "pair block hollow link malformed"

    roles = {object_color, outline_color, hollow_color, hole_color}
    assert len(roles) == 4 and bg not in roles, "legend colors must be four distinct non-bg roles"

    solid_block = crop[0:4, 4:8]
    assert np.all(solid_block[[0, 3], :] == outline_color), "solid icon top/bottom malformed"
    assert np.all(solid_block[:, [0, 3]] == outline_color), "solid icon sides malformed"
    assert np.all(solid_block[1:3, 1:3] == object_color), "solid icon interior malformed"

    hollow_block = crop[0:5, 9:14]
    assert np.all(hollow_block[[0, 4], :] == outline_color), "hollow icon top/bottom malformed"
    assert np.all(hollow_block[:, [0, 4]] == outline_color), "hollow icon sides malformed"
    inner = hollow_block[1:4, 1:4]
    assert np.all(inner[[0, 2], :] == hollow_color), "hollow icon shell malformed"
    assert np.all(inner[:, [0, 2]] == hollow_color), "hollow icon shell malformed"
    assert inner[1, 1] == hole_color, "hollow icon center malformed"

    return {
        "object_color": object_color,
        "outline_color": outline_color,
        "hollow_color": hollow_color,
        "hole_color": hole_color,
    }


def _legend_info(grid: np.ndarray, bg: int) -> dict[str, object]:
    h, w = grid.shape

    top_ok = (
        h >= LEGEND_H + LEGEND_SEP + MIN_FIELD_DIM
        and w >= LEGEND_W
        and np.all(grid[LEGEND_H, :] == bg)
        and np.all(grid[:LEGEND_H, LEGEND_W:] == bg)
    )
    if top_ok:
        decoded = _decode_combo_legend(grid[:LEGEND_H, :LEGEND_W], bg)
        return {
            **decoded,
            "orientation": "top",
            "main_mask_off": (slice(0, LEGEND_H + LEGEND_SEP), slice(None)),
        }

    left_ok = (
        w >= LEGEND_H + LEGEND_SEP + MIN_FIELD_DIM
        and h >= LEGEND_W
        and np.all(grid[:, LEGEND_H] == bg)
        and np.all(grid[LEGEND_W:, :LEGEND_H] == bg)
    )
    if left_ok:
        decoded = _decode_combo_legend(grid[:LEGEND_W, :LEGEND_H].T, bg)
        return {
            **decoded,
            "orientation": "left",
            "main_mask_off": (slice(None), slice(0, LEGEND_H + LEGEND_SEP)),
        }

    raise AssertionError("legend not found in canonical top or left position")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a fused edge legend, then recolor and outline the real objects with that palette.

    Input structure:
    - A mostly uniform background contains several 4-connected components, all drawn in one shared source color.
    - A compact legend is anchored at the top-left edge as either a top strip or a left strip.
    - Inside the legend, one chained color-pair block defines the palette mapping, and two miniature icons show the solid-object and hollow-object outcome schemes.

    Rule:
    - Detect the legend orientation and decode from it the source object color, outline color, hollow-shell color, and hole-fill color.
    - Ignore the legend band while finding real task objects.
    - For each source-color component, keep solid objects in the source color and draw a one-cell 8-neighborhood outline in the decoded outline color.
    - If a component encloses background holes, recolor its body with the decoded hollow color, fill the holes with the decoded hole color, then outline the solidified mask.

    Key insight: The spatial transformation is still the original "classify by hole presence, then outline" rule; the new challenge is to infer both the color mapping and the solid-vs-hollow recoloring branch from one explicit corner legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _legend_info(g, bg)

    out = g.copy()
    work = g.copy()
    rr, cc = legend["main_mask_off"]
    work[rr, cc] = bg

    object_color = int(legend["object_color"])
    outline_color = int(legend["outline_color"])
    hollow_color = int(legend["hollow_color"])
    hole_color = int(legend["hole_color"])

    for coords in components_mask(work == object_color, connectivity=4):
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = hollow_color
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = hole_color
                solid[hole[:, 0], hole[:, 1]] = True
        else:
            out[mask] = object_color

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = outline_color

    return out


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

    mask = _extend_mask(mask, base_h, base_w)
    return _carve_hole(mask, hh, hw)


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


def _choose_colors() -> tuple[int, int, int, int, int]:
    while True:
        colors = list(range(10))
        random.shuffle(colors)
        bg, object_color, outline_color, hollow_color, hole_color = colors[:5]
        if (object_color, outline_color, hollow_color, hole_color) != (1, 2, 8, 6):
            return bg, object_color, outline_color, hollow_color, hole_color


def _generate_components(field_h: int, field_w: int) -> list[np.ndarray] | None:
    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])
    max_h = field_h - 2 * BORDER_MIN
    max_w = field_w - 2 * BORDER_MIN

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if (
                    mask is not None
                    and int(mask.sum()) >= HOLLOW_MIN_AREA
                    and mask.shape[0] <= max_h
                    and mask.shape[1] <= max_w
                ):
                    components.append(mask)
                    found = True
                    break
            if not found:
                return None
        elif random.random() < SMALL_SHAPE_PROB:
            mask = _make_small_shape()
            if mask.shape[0] > max_h or mask.shape[1] > max_w:
                return None
            components.append(mask)
        else:
            found = False
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None and mask.shape[0] <= max_h and mask.shape[1] <= max_w:
                    components.append(mask)
                    found = True
                    break
            if not found:
                return None

    return components


def _generate_field(field_h: int, field_w: int, object_color: int, bg: int) -> np.ndarray | None:
    components = _generate_components(field_h, field_w)
    if components is None:
        return None

    grid = np.full((field_h, field_w), bg, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(len(components)), key=lambda i: -int(components[i].sum()))

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        rows_local, cols_local = np.where(mask)

        r_hi = field_h - mh - BORDER_MIN
        c_hi = field_w - mw - BORDER_MIN
        if r_hi < BORDER_MIN or c_hi < BORDER_MIN:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, r_hi)
            c = random.randint(BORDER_MIN, c_hi)
            rows_a, cols_a = rows_local + r, cols_local + c

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
        grid[rows + r, cols + c] = object_color

    return grid


def _grid_dims(orientation: str) -> tuple[int, int, int, int]:
    if orientation == "top":
        total_h = random.randint(max(GRID_H_RANGE[0], LEGEND_H + LEGEND_SEP + MIN_FIELD_DIM), GRID_H_RANGE[1])
        total_w = random.randint(max(GRID_W_RANGE[0], LEGEND_W), GRID_W_RANGE[1])
        field_h = total_h - LEGEND_H - LEGEND_SEP
        field_w = total_w
        return total_h, total_w, field_h, field_w

    total_h = random.randint(max(GRID_H_RANGE[0], LEGEND_W), GRID_H_RANGE[1])
    total_w = random.randint(max(GRID_W_RANGE[0], LEGEND_H + LEGEND_SEP + MIN_FIELD_DIM), GRID_W_RANGE[1])
    field_h = total_h
    field_w = total_w - LEGEND_H - LEGEND_SEP
    return total_h, total_w, field_h, field_w


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(ORIENTATIONS)
    bg, object_color, outline_color, hollow_color, hole_color = _choose_colors()
    total_h, total_w, field_h, field_w = _grid_dims(orientation)

    field = _generate_field(field_h, field_w, object_color, bg)
    if field is None:
        return None

    legend = _build_combo_legend(bg, object_color, outline_color, hollow_color, hole_color)
    grid = np.full((total_h, total_w), bg, dtype=int)

    if orientation == "top":
        paste(grid, legend, 0, 0)
        paste(grid, field, LEGEND_H + LEGEND_SEP, 0)
    else:
        paste(grid, legend.T, 0, 0)
        paste(grid, field, 0, LEGEND_H + LEGEND_SEP)

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

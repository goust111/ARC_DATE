# Variant of aa4ec2a5
# Dimension: D13 — 符号编码与图例系统 (symbol_instructions)
# Change: 无显式图例或符号编码，所有规则通过输入输出对隐式传达
#   → 网格边缘图例指示有洞/无洞的着色方案

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, is_contiguous, paste


# Design analysis for the D13 variant
# ==================================
# a) Concrete behavior:
#    The main object logic stays the same: blue 4-connected components on yellow
#    background get a 1-cell red 8-neighborhood outline; hollow ones also recolor
#    their shell and hole. The change is that the palette rule is no longer only
#    implied by input/output examples. Every input now contains a small legend
#    anchored at the top-left corner, either as a 5x17 strip on the top edge or
#    as its transpose on the left edge. The legend contains two miniature
#    before/after samples:
#    - solid blue 2x2 block -> blue 2x2 block with red outline
#    - hollow blue 3x3 ring -> teal ring + magenta center + red outline
#    Example trace:
#    - Original pipeline: find a blue component -> test whether it has holes ->
#      hardcode blue/red or teal/magenta/red output colors.
#    - Variant pipeline: first detect legend orientation -> read the solid fill,
#      hollow fill, hole fill, and outline colors from the legend's example
#      output miniatures -> ignore the legend region -> process the real objects
#      with those decoded colors.
#    The logical difference is therefore localized to "how the palette is
#    obtained", not to component geometry or hole detection.
# b) Complexity calibration:
#    The legend has only two cases and uses exact miniature examples rather than
#    an arbitrary symbol language, so the added decoding burden is visible but
#    not excessive. Both cases appear in every sample, which gives repeated
#    support for the two recoloring branches. The legend orientation varies
#    between top and left so the instruction is not a single frozen bitmap.
# c) Hidden-assumption audit of main():
#    - get_bg(...) still works because yellow remains the dominant color.
#    - The old loop over all non-background components would now be wrong,
#      because the legend itself contains blue/red/teal/magenta cells. The new
#      solver must mask out the legend strip before searching for task objects.
#    - The old hardcoded assignments out[mask] = 8 / out[hole] = 6 / outline=2
#      depended on fixed output colors. In the variant these values must be read
#      from the legend crop instead.
#    - Hole detection via bbox-local background components still remains valid,
#      because the legend is spatially separated by a full background separator
#      row or column and never touches task objects.
# d) Placement legality:
#    The generator still places 3-4 components with Chebyshev distance >= 3 and
#    1-cell border clearance, but it does so inside a main field that excludes
#    the legend strip plus one full separator row/column. This preserves
#    non-overlap, keeps outlines disjoint, and guarantees the solver can safely
#    blank out the legend region without losing object cells.
# e) Visual observability:
#    A human can infer the new rule from 2-3 examples because every input visibly
#    contains the two before/after miniatures. The legend explicitly shows the
#    special hollow-object recoloring, so the variant is easier to infer than the
#    original while still requiring an extra decoding step.
# f) Visual cleanliness:
#    The legend is confined to the corner and isolated by a full yellow separator
#    row or column. No main object overlaps it, so the grid stays easy to parse.
# g) Color-change inferability:
#    Only three output colors are introduced in the legend (red outline, teal
#    hollow shell, magenta hole). Their roles are tied to distinct miniature
#    structures, so the mapping is structurally cued rather than arbitrary.
# h) Variant metric used after testing:
#    1. Inputs must contain a detected legend in top or left orientation.
#    2. The new solver output should differ from the original aa4ec2a5 solver on
#       generated inputs, proving the legend region is behaviorally active.
# i) New solver assumptions that the generator must guarantee:
#    - The legend is exactly the canonical 5x17 strip or its transpose.
#    - The legend is anchored at the top-left corner.
#    - Row 5 is all background for top legends; column 5 is all background for
#      left legends.
#    - All real task objects use the legend's source color and lie entirely
#      outside the legend strip.


LEGEND_H = 5
LEGEND_W = 17
LEGEND_SEP = 1


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


def _build_canonical_legend(
    bg: int,
    source_color: int,
    solid_fill_color: int,
    outline_color: int,
    hollow_fill_color: int,
    hole_fill_color: int,
) -> np.ndarray:
    legend = np.full((LEGEND_H, LEGEND_W), bg, dtype=int)

    legend[1:3, 0:2] = source_color

    solid_out = np.full((4, 4), outline_color, dtype=int)
    solid_out[1:3, 1:3] = solid_fill_color
    paste(legend, solid_out, 0, 3)

    hollow_in = np.full((3, 3), source_color, dtype=int)
    hollow_in[1, 1] = bg
    paste(legend, hollow_in, 1, 8)

    hollow_out = np.full((5, 5), outline_color, dtype=int)
    inner = np.full((3, 3), hollow_fill_color, dtype=int)
    inner[1, 1] = hole_fill_color
    hollow_out[1:4, 1:4] = inner
    paste(legend, hollow_out, 0, 12)

    return legend


def _decode_canonical_legend(crop: np.ndarray, bg: int) -> dict[str, int]:
    coords = np.argwhere(crop != bg)
    assert coords.size > 0, "empty legend crop"
    assert bbox_coords(coords) == (0, 0, LEGEND_H - 1, LEGEND_W - 1), "legend bbox mismatch"

    source_color = int(crop[1, 0])
    assert np.all(crop[1:3, 0:2] == source_color), "solid source icon malformed"

    outline_color = int(crop[0, 3])
    solid_fill_color = int(crop[1, 4])
    solid_block = crop[0:4, 3:7]
    assert np.all(solid_block[[0, 3], :] == outline_color), "solid outline top/bottom malformed"
    assert np.all(solid_block[:, [0, 3]] == outline_color), "solid outline sides malformed"
    assert np.all(solid_block[1:3, 1:3] == solid_fill_color), "solid fill malformed"

    assert crop[2, 9] == bg, "hollow source center must be background"
    assert np.all(crop[1:4, 8:11][[0, 2], :] == source_color), "hollow source ring malformed"
    assert np.all(crop[1:4, 8:11][:, [0, 2]] == source_color), "hollow source ring malformed"

    hollow_out = crop[:, 12:17]
    assert np.all(hollow_out[[0, 4], :] == outline_color), "hollow outline top/bottom malformed"
    assert np.all(hollow_out[:, [0, 4]] == outline_color), "hollow outline sides malformed"
    hollow_fill_color = int(hollow_out[1, 1])
    hole_fill_color = int(hollow_out[2, 2])
    inner = hollow_out[1:4, 1:4]
    assert np.all(inner[[0, 2], :] == hollow_fill_color), "hollow fill ring malformed"
    assert np.all(inner[:, [0, 2]] == hollow_fill_color), "hollow fill ring malformed"
    assert inner[1, 1] == hole_fill_color, "hole fill malformed"

    return {
        "source_color": source_color,
        "solid_fill_color": solid_fill_color,
        "outline_color": outline_color,
        "hollow_fill_color": hollow_fill_color,
        "hole_fill_color": hole_fill_color,
    }


def _legend_info(grid: np.ndarray, bg: int) -> dict[str, object]:
    h, w = grid.shape

    top_ok = (
        h >= LEGEND_H + LEGEND_SEP
        and w >= LEGEND_W
        and np.all(grid[LEGEND_H, :] == bg)
        and np.all(grid[:LEGEND_H, LEGEND_W:] == bg)
    )
    if top_ok:
        decoded = _decode_canonical_legend(grid[:LEGEND_H, :LEGEND_W], bg)
        return {
            **decoded,
            "orientation": "top",
            "main_mask_off": (slice(0, LEGEND_H + LEGEND_SEP), slice(None)),
        }

    left_ok = (
        w >= LEGEND_H + LEGEND_SEP
        and h >= LEGEND_W
        and np.all(grid[:, LEGEND_H] == bg)
        and np.all(grid[LEGEND_W:, :LEGEND_H] == bg)
    )
    if left_ok:
        decoded = _decode_canonical_legend(grid[:LEGEND_W, :LEGEND_H].T, bg)
        return {
            **decoded,
            "orientation": "left",
            "main_mask_off": (slice(None), slice(0, LEGEND_H + LEGEND_SEP)),
        }

    raise AssertionError("legend not found in canonical top or left position")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode an edge legend, then recolor and outline the real blue objects accordingly.

    Input structure:
    - A mostly yellow field contains several blue 4-connected components plus a compact legend
      anchored on the top edge or left edge.
    - The legend shows two miniature before/after cases: a solid blue object and a hollow blue object.

    Rule:
    - Detect the legend strip and read from it the source object color, the solid-object fill color,
      the hollow-object shell color, the hole color, and the outline color.
    - Leave the legend unchanged, but ignore it when searching for task objects.
    - For each real object outside the legend, recolor solid ones with the legend's solid fill color.
    - If an object encloses a background hole, recolor its shell and hole with the legend's hollow colors,
      then add the one-cell 8-neighborhood outline using the legend's outline color.

    Key insight: The geometric transformation is still "detect holes, optionally recolor, then outline",
    but the color rule is now explicit in a miniature edge legend that must be decoded first.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend = _legend_info(g, bg)
    out = g.copy()

    work = g.copy()
    rr, cc = legend["main_mask_off"]
    work[rr, cc] = bg

    fg_color = int(legend["source_color"])
    solid_fill_color = int(legend["solid_fill_color"])
    outline_color = int(legend["outline_color"])
    hollow_fill_color = int(legend["hollow_fill_color"])
    hole_fill_color = int(legend["hole_fill_color"])

    for coords in components_mask(work == fg_color, connectivity=4):
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = hollow_fill_color
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = hole_fill_color
                solid[hole[:, 0], hole[:, 1]] = True
        else:
            out[mask] = solid_fill_color

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = outline_color

    return out


# --------------- generator ---------------

# Canvas / main-field bounds before adding the legend strip.
FIELD_H_RANGE = (15, 28)
FIELD_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1
SOLID_FILL_COLOR = 1
OUTLINE_COLOR = 2
HOLLOW_FILL_COLOR = 8
HOLE_FILL_COLOR = 6

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


def _generate_field(field_h: int, field_w: int) -> np.ndarray | None:
    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and mask.sum() >= HOLLOW_MIN_AREA:
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

    grid = np.full((field_h, field_w), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, field_h - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, field_w - mw - BORDER_MIN))
            if r + mh > field_h - BORDER_MIN or c + mw > field_w - BORDER_MIN:
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

    return grid


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["top", "left"])
    legend = _build_canonical_legend(
        BG_COLOR,
        FG_COLOR,
        SOLID_FILL_COLOR,
        OUTLINE_COLOR,
        HOLLOW_FILL_COLOR,
        HOLE_FILL_COLOR,
    )

    if orientation == "top":
        field_h = random.randint(*FIELD_H_RANGE)
        field_w = random.randint(max(FIELD_W_RANGE[0], LEGEND_W), FIELD_W_RANGE[1])
        field = _generate_field(field_h, field_w)
        if field is None:
            return None
        grid = np.full((field_h + LEGEND_H + LEGEND_SEP, field_w), BG_COLOR, dtype=int)
        paste(grid, legend, 0, 0)
        paste(grid, field, LEGEND_H + LEGEND_SEP, 0)
    else:
        field_h = random.randint(max(FIELD_H_RANGE[0], LEGEND_W), FIELD_H_RANGE[1])
        field_w = random.randint(*FIELD_W_RANGE)
        field = _generate_field(field_h, field_w)
        if field is None:
            return None
        grid = np.full((field_h, field_w + LEGEND_H + LEGEND_SEP), BG_COLOR, dtype=int)
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

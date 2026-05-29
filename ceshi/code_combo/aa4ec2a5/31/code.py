# Combo variant of aa4ec2a5 (index 31)
# combo_spec: D5/3+D9/2
# Name: 洞的独立轮廓+轮廓避让洞区域

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant D5/3 + D9/2
# a) Concrete behavior trace:
#    - Start from the original task: every blue 4-connected object gets a red 8-neighborhood
#      outline on background cells; hollow objects additionally turn teal and their enclosed
#      background holes turn magenta, while the red outline is still computed on the hole-filled
#      silhouette.
#    - D5/3 adds a second, independent outline for each hole. Because the global background is
#      already yellow (4), a literal yellow hole-border would be invisible against the canvas, so
#      the visible substitute color is orange (7): each hole becomes orange on its 4-neighbor rim
#      and magenta in the remaining interior.
#    - D9/2 then changes where the red outline may appear: after computing the usual outer outline
#      of the hole-filled silhouette, delete every red candidate that lies in the hole's own
#      8-neighborhood. On a hollow object with a thin shell and one missing outer corner, the old
#      task would place a red pixel in that notch cavity; the combo version leaves that spot empty
#      because it is diagonally adjacent to the hole.
#    - Net visual result: solids still look like blue + red. Hollow objects now show teal walls,
#      an orange/magenta nested cavity, and a nearby missing red segment where the hole suppresses
#      part of the otherwise valid outer outline.
# b) Complexity calibration:
#    - The generator keeps the palette structured: blue solids, teal hollow bodies, magenta hole
#      interiors, orange hole rims, red external outlines, yellow background.
#    - To avoid a weak D9/2 signal, at least one hollow component per sample is forced to have a
#      guaranteed red-gap event. To avoid a weak D5/3 signal, hole sizes are restricted to 3x3,
#      3x4, or 4x4 so both orange rim and magenta core remain visible.
# c) Implicit assumption review:
#    - `_interior_holes(...)` still assumes holes are 4-connected background regions fully inside a
#      component bbox; that remains valid.
#    - The original "fill holes into `solid` before computing the outer outline" assumption is
#      still required; otherwise the red outline would leak into the cavity and conflict with the
#      new orange inner rim.
#    - The original `out[outline & (out == bg)] = 2` guard still matters. Even after D9/2 removes
#      some red candidates, the surviving red pixels must still be background-only and may not
#      overwrite teal/orange/magenta cells.
#    - The new hole-neighborhood suppression assumes at least one hollow object has an exterior
#      background pixel within Chebyshev distance 1 of a hole cell. The generator now guarantees
#      that by constructing one special hollow object with a diagonally exposed corner notch.
# d) Placement legality:
#    - Components remain pairwise separated by Chebyshev distance >= 3, so ordinary red outlines
#      still do not collide across different objects.
#    - The new orange border is internal to a hollow object, and the D9/2 gap only removes red
#      pixels; it never adds new external occupancy, so placement remains safe under the original
#      spacing rule.
# e) Visual observability:
#    - Every generated grid contains at least one hollow object with an orange rim, magenta core,
#      and a measurable missing-red effect. A human comparing 2-4 examples can see that hollows
#      have two nested boundary systems and that the outer red ring is locally interrupted near the
#      cavity, unlike the original task's uninterrupted outer outline.
# f) Visual cleanliness:
#    - The two outline systems are kept disjoint by construction: orange lives on hole cells,
#      red lives only on background cells, and D9/2 removes ambiguous red cells near the hole.
#      This prevents dense overpainting in one location.
# g) Change inferability:
#    - Color changes are tied to geometry, not arbitrary lookup. Orange always means "hole rim",
#      magenta means "hole interior", teal means "hollow object body", and the red omission occurs
#      exactly where the hole's local neighborhood interferes with the outer-outline candidates.
# h) New solver assumptions checked against the generator:
#    - Assumption: hole rims are large enough to leave non-rim interior cells. Generator enforces
#      hole areas 9/12/16 only.
#    - Assumption: at least one hollow object creates a nonzero red-gap count. Generator forces one
#      special hollow component whose full-outline mask intersects the hole 8-neighborhood.
#    - Assumption: hollow masks remain 4-connected and still contain interior holes after notch and
#      extension edits. Generator verifies contiguity and hole existence before placement.


BG_COLOR = 4
FG_COLOR = 1
OUTLINE_COLOR = 2
HOLE_INNER_OUTLINE_COLOR = 7
HOLE_FILL_COLOR = 6
HOLLOW_BODY_COLOR = 8


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


def _inner_perimeter(mask: np.ndarray) -> np.ndarray:
    """Return the subset of True cells touching non-mask space by 4-neighborhood."""
    h, w = mask.shape
    border = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = rows + dr
        cc = cols + dc
        out_of_bounds = (rr < 0) | (rr >= h) | (cc < 0) | (cc >= w)
        if np.any(out_of_bounds):
            border[rows[out_of_bounds], cols[out_of_bounds]] = True
        ok = ~out_of_bounds
        if np.any(ok):
            exposed = ~mask[rr[ok], cc[ok]]
            border[rows[ok][exposed], cols[ok][exposed]] = True
    return border


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            dilated[rr[ok], cc[ok]] = True
    return dilated


def _component_outline_gap_count(mask: np.ndarray, holes: list[np.ndarray]) -> int:
    solid = mask.copy()
    hole_avoid = np.zeros_like(mask, dtype=bool)
    for hole in holes:
        hole_mask = _coords_mask(mask.shape, hole)
        solid |= hole_mask
        hole_avoid |= _dilate8(hole_mask)
    full_outline = _outline_mask(solid)
    return int(np.count_nonzero(full_outline & hole_avoid))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outer outlines to objects, give holes their own inner rim, and suppress red near holes.

    Input structure:
    - A mostly uniform yellow background contains one or more 4-connected blue foreground objects.
    - Some objects enclose one or more 4-connected background holes inside their bounding boxes.

    Rule:
    - Every foreground component still receives a one-cell 8-neighborhood outer outline in red (2).
    - If a component contains enclosed holes, recolor the object body to teal (8).
    - Recolor each enclosed hole to magenta (6), then recolor the hole's own 4-neighbor inner rim
      to orange (7) so the hole has an independent internal border.
    - For hollow objects only, compute the usual red outer outline on the hole-filled silhouette,
      but skip any red candidate that lies in the hole's 8-neighborhood.

    Key insight:
    - Hollow objects must be processed with two coupled geometric masks: a hole mask that creates
      the orange inner rim and a hole-filled solid mask that creates the red outer outline. The
      final red border is then pruned by the hole's 8-neighborhood, producing a deliberate gap
      near the cavity while preserving the separate inner hole border.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        hole_avoid = np.zeros_like(mask, dtype=bool)

        if holes:
            out[mask] = HOLLOW_BODY_COLOR
            for hole in holes:
                hole_mask = _coords_mask(g.shape, hole)
                out[hole[:, 0], hole[:, 1]] = HOLE_FILL_COLOR
                out[_inner_perimeter(hole_mask)] = HOLE_INNER_OUTLINE_COLOR
                solid |= hole_mask
                hole_avoid |= _dilate8(hole_mask)

        outline = _outline_mask(solid)
        if holes:
            outline &= ~hole_avoid
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (combo) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Large enough holes so the orange rim does not consume the full cavity.
HOLE_VALID_AREAS = [9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_EXTRA_NOTCHES = 1
HOLLOW_EXTENSION_PROB = 0.45
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_GEN_RETRIES = 80
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


def _extend_mask(mask: np.ndarray) -> np.ndarray:
    base_h, base_w = mask.shape
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


def _corner_slices(mask: np.ndarray, corner: str, nh: int, nw: int) -> tuple[slice, slice]:
    h, w = mask.shape
    if corner == "tl":
        return slice(0, nh), slice(0, nw)
    if corner == "tr":
        return slice(0, nh), slice(w - nw, w)
    if corner == "bl":
        return slice(h - nh, h), slice(0, nw)
    return slice(h - nh, h), slice(w - nw, w)


def _corner_pad_limits(
    corner: str, pad_top: int, pad_bot: int, pad_left: int, pad_right: int
) -> tuple[int, int]:
    if corner == "tl":
        return pad_top, pad_left
    if corner == "tr":
        return pad_top, pad_right
    if corner == "bl":
        return pad_bot, pad_left
    return pad_bot, pad_right


def _make_hollow_component(target_hole_area: int, require_gap: bool) -> np.ndarray | None:
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    gap_corners: tuple[str, ...] = ()
    if require_gap:
        gap_corners = random.choice(
            [
                ("tl",),
                ("tr",),
                ("bl",),
                ("br",),
                ("tl", "tr"),
                ("bl", "br"),
                ("tl", "bl"),
                ("tr", "br"),
                ("tl", "br"),
                ("tr", "bl"),
            ]
        )
        if "tl" in gap_corners:
            pad_top = 1
            pad_left = 1
        if "tr" in gap_corners:
            pad_top = 1
            pad_right = 1
        if "bl" in gap_corners:
            pad_bot = 1
            pad_left = 1
        if "br" in gap_corners:
            pad_bot = 1
            pad_right = 1

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)
    mask[pad_top : pad_top + hh, pad_left : pad_left + hw] = False

    for gap_corner in gap_corners:
        rs, cs = _corner_slices(mask, gap_corner, 1, 1)
        mask[rs, cs] = False

    extra_corners = ["tl", "tr", "bl", "br"]
    random.shuffle(extra_corners)
    extra_notches = random.randint(0, HOLLOW_MAX_EXTRA_NOTCHES)
    applied = 0
    for corner in extra_corners:
        if corner in gap_corners:
            continue
        max_h, max_w = _corner_pad_limits(corner, pad_top, pad_bot, pad_left, pad_right)
        if max_h <= 1 or max_w <= 1:
            continue
        nh = random.randint(1, max_h - 1)
        nw = random.randint(1, max_w - 1)
        trial = mask.copy()
        rs, cs = _corner_slices(trial, corner, nh, nw)
        trial[rs, cs] = False
        if is_contiguous(trial.astype(int), background=0, connectivity=4) and _has_interior_hole(trial):
            mask = trial
            applied += 1
        if applied >= extra_notches:
            break

    mask = _extend_mask(mask)
    mask = _crop_mask(mask)
    if mask is None:
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    if not _has_interior_hole(mask):
        return None

    holes = _interior_holes(mask, np.argwhere(mask))
    if not holes:
        return None
    if require_gap and _component_outline_gap_count(mask, holes) == 0:
        return None
    return mask


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


def _d5_only_main(grid: np.ndarray) -> np.ndarray:
    """Reference rendering for 'independent hole border but no hole-neighborhood red suppression'."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = HOLLOW_BODY_COLOR
            for hole in holes:
                hole_mask = _coords_mask(g.shape, hole)
                out[hole[:, 0], hole[:, 1]] = HOLE_FILL_COLOR
                out[_inner_perimeter(hole_mask)] = HOLE_INNER_OUTLINE_COLOR
                solid |= hole_mask

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])
    required_gap_hollow = random.choice(tuple(hollow_indices))

    components: list[np.ndarray] = []
    has_gap_component = False
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            need_gap = i == required_gap_hollow
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS), require_gap=need_gap)
                if mask is None or int(mask.sum()) < HOLLOW_MIN_AREA:
                    continue
                holes = _interior_holes(mask, np.argwhere(mask))
                gap_count = _component_outline_gap_count(mask, holes)
                if need_gap and gap_count == 0:
                    continue
                if gap_count > 0:
                    has_gap_component = True
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

    if len(components) != num_components or not has_gap_component:
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

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out == HOLE_INNER_OUTLINE_COLOR):
        return None
    if not np.any(out == HOLE_FILL_COLOR):
        return None
    if not np.any(out == HOLLOW_BODY_COLOR):
        return None
    if not np.any(out == OUTLINE_COLOR):
        return None

    d5_only = _d5_only_main(grid)
    if np.array_equal(out, d5_only):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

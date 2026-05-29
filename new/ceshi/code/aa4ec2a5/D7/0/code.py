# Variant of aa4ec2a5
# Dimension: D7 — 多步处理序列 (multi_step)
# Change: 两步序列：第一步检测洞并条件性重着色（对象变青色、洞变品红）；第二步对所有对象统一绘制红色轮廓... → 按对象面积从大到小依次处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis for this D7 variant
# -----------------------------------
# variant_spec:
#   preserved_input:
#     - grid is still a yellow background with 3-4 blue 4-connected components
#     - 1-2 components are hollow and contain enclosed 4-connected background holes
#   changed_output_process:
#     - components are no longer "all recolor first, all outline second"
#     - instead, components are processed one by one in descending area order
#     - on a component's turn, its own body is restored/recolored first, then its
#       8-neighborhood outline is painted red onto the current canvas
#     - outline painting is unconditional with respect to previous colors, so later
#       components can overwrite earlier red marks and earlier components can be cut
#       by later outlines
#   concrete_visual_effect:
#     - to make order visible, the generator creates at least one pair of components
#       that only touch diagonally; they stay distinct under 4-connectivity, but each
#       lies in the other's 8-neighborhood
#     - this makes the final red cells around the contact depend on which area is
#       larger, matching the required D7 sequential behavior
#   example_trace:
#     - original logic: two diagonal components remain blue/teal at their own cells
#       because outlines are written only on background
#     - new logic: if the larger component is handled first, it may briefly paint
#       red on the smaller component's corner cell; when the smaller component's turn
#       arrives, that cell is restored to its base color, and then the smaller
#       component's outline may paint red onto the larger component
#     - difference: final ownership at the diagonal interface depends on area order
#   complexity_calibration:
#     - areas are forced to be distinct, so the descending order is unambiguous from
#       visual size alone; no hidden tie-break based on color ids is needed
#     - only one designated diagonal-contact pair is required, which keeps the scene
#       readable while still making the sequential rule observable
#   implicit_assumptions_revised:
#     - original `out[outline & (out == bg)] = 2` relied on "outlines never need to
#       overwrite anything"; that is invalid here and must become unconditional
#     - original solid components were never rewritten because nobody could damage
#       them; here every component must rewrite its own body on its turn so earlier
#       outlines cannot permanently erase later components
#     - component detection must still come from the untouched input grid, not from
#       the progressively edited output canvas
#   placement_legality:
#     - components never overlap and are never orthogonally adjacent, so all blue
#       shapes remain separate 4-connected objects
#     - exactly the designated pair is allowed diagonal contact; all other pairs keep
#       Chebyshev distance >= 2 to avoid excessive clutter
#   observability_checks:
#     - the generator rejects duplicate areas and forces at least one diagonal-contact
#       pair involving the two largest components
#     - on such inputs, the new solver output differs from the original background-
#       only outline rule, so the D7 change is not degenerate


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
    Recolor hollow components and paint red outlines in descending area order.

    Input structure:
    - A uniform yellow background contains several blue 4-connected components.
    - Some components are hollow and enclose one or more 4-connected background holes.

    Rule:
    - Sort the blue components by area from largest to smallest.
    - Process them one at a time: restore/recolor the current component first, then
      draw its one-cell 8-neighborhood red outline onto the current canvas.
    - Hollow components become teal and their enclosed holes become magenta before
      that component's outline is drawn.

    Key insight: The solver still detects holes from the original input objects, but
    the outline step is now sequential and overwriting, so later smaller components
    can cut into earlier larger ones when their 8-neighborhoods interact.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    comps = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        comps.append(
            {
                "coords": coords,
                "mask": mask,
                "holes": holes,
                "area": int(coords.shape[0]),
                "bbox": bbox_coords(coords),
                "color": int(comp["color"]),
            }
        )

    comps.sort(key=lambda comp: (-comp["area"], comp["bbox"][0], comp["bbox"][1]))

    for comp in comps:
        mask = comp["mask"]
        holes = comp["holes"]
        solid = mask.copy()

        # Restore the component on its own turn, because previous outlines may have
        # temporarily overwritten some of its cells.
        out[mask] = 8 if holes else comp["color"]
        if holes:
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline] = 2

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

SMALL_SHAPE_PROB = 0.15
SMALL_SHAPE_SIZE_RANGE = (2, 4)

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

BORDER_MIN = 1
PLACEMENT_RETRIES = 350
GEN_OUTER_RETRIES = 400
MIN_AREA_GAP_TOP2 = 3


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
        if not bool(touches_edge):
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


def _placed_coords(mask: np.ndarray, r: int, c: int) -> np.ndarray:
    coords = np.argwhere(mask)
    placed = coords.copy()
    placed[:, 0] += int(r)
    placed[:, 1] += int(c)
    return placed


def _pair_stats(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[bool, bool, bool, int]:
    has_overlap = False
    has_orth = False
    has_diag = False
    min_cheb = 10**9

    for ra, ca in coords_a:
        for rb, cb in coords_b:
            dr = abs(int(ra) - int(rb))
            dc = abs(int(ca) - int(cb))
            cheb = max(dr, dc)
            if cheb < min_cheb:
                min_cheb = cheb
            if dr == 0 and dc == 0:
                has_overlap = True
            elif dr + dc == 1:
                has_orth = True
            elif dr == 1 and dc == 1:
                has_diag = True

    return has_overlap, has_orth, has_diag, int(min_cheb)


def _can_place_regular(
    coords: np.ndarray,
    existing: list[tuple[int, int, np.ndarray]],
) -> bool:
    for _, _, other_coords in existing:
        has_overlap, has_orth, _, min_cheb = _pair_stats(coords, other_coords)
        if has_overlap or has_orth or min_cheb < 2:
            return False
    return True


def _can_place_designated(
    coords: np.ndarray,
    existing: list[tuple[int, int, np.ndarray]],
    contact_with_idx: int,
) -> bool:
    saw_required_diag = False

    for idx, (_, _, other_coords) in enumerate(existing):
        has_overlap, has_orth, has_diag, min_cheb = _pair_stats(coords, other_coords)
        if has_overlap or has_orth:
            return False
        if idx == contact_with_idx:
            if not has_diag or min_cheb != 1:
                return False
            saw_required_diag = True
        elif min_cheb < 2:
            return False

    return saw_required_diag


def _place_components(H: int, W: int, components: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]] | None:
    order = sorted(range(len(components)), key=lambda i: -int(components[i].sum()))
    second_idx = order[1]

    placed: list[tuple[int, int, np.ndarray]] = []

    for slot, idx in enumerate(order):
        mask = components[idx]
        mh, mw = mask.shape
        max_r = H - mh - BORDER_MIN
        max_c = W - mw - BORDER_MIN
        if max_r < BORDER_MIN or max_c < BORDER_MIN:
            return None
        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max_r)
            c = random.randint(BORDER_MIN, max_c)
            coords = _placed_coords(mask, r, c)
            if slot == 0:
                placed.append((r, c, coords))
                placed_ok = True
                break
            if idx == second_idx:
                if not _can_place_designated(coords, placed, contact_with_idx=0):
                    continue
            else:
                if not _can_place_regular(coords, placed):
                    continue
            placed.append((r, c, coords))
            placed_ok = True
            break
        if not placed_ok:
            return None

    return [(r, c, components[idx]) for idx, (r, c, _coords) in zip(order, placed)]


def _diag_contact_exists(placed: list[tuple[int, int, np.ndarray]]) -> bool:
    coords_list = [_placed_coords(mask, r, c) for r, c, mask in placed]
    for i in range(len(coords_list)):
        for j in range(i + 1, len(coords_list)):
            _, has_orth, has_diag, _ = _pair_stats(coords_list[i], coords_list[j])
            if has_diag and not has_orth:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

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

    areas = sorted((int(mask.sum()) for mask in components), reverse=True)
    if len(set(areas)) != len(areas):
        return None
    if areas[0] - areas[1] < MIN_AREA_GAP_TOP2:
        return None

    placed = _place_components(H, W, components)
    if placed is None:
        return None
    if not _diag_contact_exists(placed):
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, _legacy_main(grid)):
        return None

    return grid


def _legacy_main(grid: np.ndarray) -> np.ndarray:
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
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

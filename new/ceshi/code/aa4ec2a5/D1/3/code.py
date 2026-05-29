# Variant of aa4ec2a5
# Dimension: D1 — 轮廓绘制方式 (transform)
# Change: 在每个蓝色对象周围绘制1像素宽的8邻域（Chebyshev距离=1）红色轮廓... → 轮廓仅绘制在背景侧

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for this D1 variant
# - Concrete behavior:
#   The original generator kept components at Chebyshev distance >= 3, so a
#   background-only outline and a naive "ring around the object" looked the same.
#   This variant makes the conditional drawing rule visible by allowing at least
#   one pair of 4-connected components to touch diagonally (Chebyshev distance 1)
#   while staying 4-disconnected. In the output, red appears only on cells that
#   were background in the input; the diagonally touching object pixels remain
#   blue/teal, so no red is inserted between the two objects.
# - Small traced example:
#   If two single blue cells sit at (r,c) and (r+1,c+1), the old isolated-case
#   logic would first mark every 8-neighbor position as candidate outline. Under
#   the variant, each object's candidate set still includes the other object's
#   cell, but that candidate is rejected because it was not background in the
#   original input. The visible difference is the missing red at the contact.
# - Complexity calibration:
#   Only D1 changes. Object colors, hole handling, hole recoloring, and 8-neighbor
#   geometry stay intact. The generator does not introduce many new regimes: it
#   only guarantees at least one close diagonal pair, with all other pairs kept
#   at Chebyshev distance >= 2 to avoid clutter and preserve easy parsing.
# - Implicit-assumption audit for main():
#   The original `out[outline & (out == bg)] = 2` was harmless when objects were
#   widely separated, but it encoded the background test through a mutated output
#   buffer. Once components may be adjacent, the intended rule is "draw only on
#   cells that were background in the input", so the variant makes that predicate
#   explicit with `bg_mask = (g == bg)`. Hole filling still happens before the
#   outline computation by expanding the component mask with its enclosed holes.
# - Placement legality:
#   The generator constructs one forced close pair that is diagonal-only:
#   overlap is forbidden, orthogonal adjacency is forbidden, and at least one
#   diagonal contact is required. Remaining components must stay outside each
#   other's 8-neighborhoods, so the layout is valid and solver-friendly.
# - Visual observability and cleanliness:
#   Every generated grid contains at least one blocked outline contact, measured
#   as a candidate outline cell occupied by a different component. That metric is
#   used as a rejection test so the variant does not collapse back to the isolated
#   original behavior. The rest of the spacing remains generous enough that red,
#   blue/teal, magenta, and yellow regions stay visually separable.


ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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


def _background_outline_mask(mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
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
    outline &= bg_mask
    return outline


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects and draw red borders only on the background side.

    Input structure:
    - A uniform background contains several blue 4-connected components.
    - Some components may enclose 4-connected background holes inside their bbox.
    - Distinct components may sit diagonally adjacent, so an 8-neighborhood border
      candidate can land on another object's cell instead of on background.

    Rule:
    - For every foreground component, consider its one-cell 8-neighborhood border.
    - Paint that border red (2) only where the corresponding input cell was background.
    - If a component encloses holes, recolor the component teal (8), recolor those
      holes magenta (6), then compute the outer border using the hole-filled mask.

    Key insight:
    - The border is defined geometrically from the component's 8-neighborhood, but
      it is rendered conditionally: occupied cells block the border, while enclosed
      holes count as part of the solid object when deciding where the outer border lies.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bg_mask = g == bg
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

        outline = _background_outline_mask(solid, bg_mask)
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

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

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
PLACEMENT_RETRIES = 300
PAIR_PLACEMENT_RETRIES = 180
SEPARATED_CHEBYSHEV_MIN = 2

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


def _absolute_cells(mask: np.ndarray, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return rows + r, cols + c


def _cell_set(rows: np.ndarray, cols: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(rows.tolist(), cols.tolist()))


def _contact_profile(
    rows_a: np.ndarray, cols_a: np.ndarray, rows_b: np.ndarray, cols_b: np.ndarray
) -> tuple[bool, bool, bool]:
    bset = _cell_set(rows_b, cols_b)
    overlap = False
    orth = False
    diag = False
    for ra, ca in zip(rows_a.tolist(), cols_a.tolist()):
        if (ra, ca) in bset:
            overlap = True
            break
        for dr, dc in ORTH_DIRS:
            if (ra + dr, ca + dc) in bset:
                orth = True
        for dr, dc in DIAG_DIRS:
            if (ra + dr, ca + dc) in bset:
                diag = True
        if orth and diag:
            continue
    return overlap, orth, diag


def _is_separated_from_placed(
    rows: np.ndarray, cols: np.ndarray, placed: list[dict[str, object]]
) -> bool:
    for item in placed:
        overlap, orth, diag = _contact_profile(
            rows, cols, item["rows"], item["cols"]  # type: ignore[arg-type]
        )
        if overlap or orth or diag:
            return False
    return True


def _can_place_close_pair(
    rows: np.ndarray, cols: np.ndarray, anchor: dict[str, object]
) -> bool:
    overlap, orth, diag = _contact_profile(
        rows, cols, anchor["rows"], anchor["cols"]  # type: ignore[arg-type]
    )
    return (not overlap) and (not orth) and diag


def _random_origin(H: int, W: int, mh: int, mw: int) -> tuple[int, int] | None:
    rmax = H - mh - BORDER_MIN
    cmax = W - mw - BORDER_MIN
    if rmax < BORDER_MIN or cmax < BORDER_MIN:
        return None
    return random.randint(BORDER_MIN, rmax), random.randint(BORDER_MIN, cmax)


def _append_placement(
    placed: list[dict[str, object]], idx: int, mask: np.ndarray, r: int, c: int
) -> None:
    rows, cols = _absolute_cells(mask, r, c)
    placed.append({"idx": idx, "mask": mask, "r": r, "c": c, "rows": rows, "cols": cols})


def _variant_signal_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    comp_masks: list[np.ndarray] = []
    for comp in comps:
        mask = _coords_mask(g.shape, comp["coords"])
        holes = _interior_holes(mask, comp["coords"])
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        comp_masks.append(solid)

    occupied = np.any(np.stack(comp_masks), axis=0)
    close_pairs = 0
    blocked_outline_cells = 0

    for i, mask in enumerate(comp_masks):
        candidate = np.zeros_like(mask, dtype=bool)
        rows, cols = np.where(mask)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = rows + dr
                cc = cols + dc
                ok = (0 <= rr) & (rr < g.shape[0]) & (0 <= cc) & (cc < g.shape[1])
                candidate[rr[ok], cc[ok]] = True
        candidate &= ~mask
        blocked_outline_cells += int(np.sum(candidate & occupied))

        rows_i, cols_i = np.where(mask)
        for j in range(i + 1, len(comp_masks)):
            rows_j, cols_j = np.where(comp_masks[j])
            overlap, orth, diag = _contact_profile(rows_i, cols_i, rows_j, cols_j)
            if not overlap and not orth and diag:
                close_pairs += 1

    return {
        "close_pairs": int(close_pairs),
        "blocked_outline_cells": int(blocked_outline_cells),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

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

    if len(components) != num_components:
        return None

    close_pair = tuple(sorted(range(num_components), key=lambda i: components[i].sum())[:2])

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[dict[str, object]] = []

    first_idx, second_idx = close_pair
    first_mask = components[first_idx]
    second_mask = components[second_idx]

    pair_ok = False
    for _ in range(PAIR_PLACEMENT_RETRIES):
        first_origin = _random_origin(H, W, first_mask.shape[0], first_mask.shape[1])
        if first_origin is None:
            return None
        r1, c1 = first_origin
        placed.clear()
        _append_placement(placed, first_idx, first_mask, r1, c1)

        for _ in range(PLACEMENT_RETRIES):
            second_origin = _random_origin(H, W, second_mask.shape[0], second_mask.shape[1])
            if second_origin is None:
                return None
            r2, c2 = second_origin
            rows2, cols2 = _absolute_cells(second_mask, r2, c2)
            if _can_place_close_pair(rows2, cols2, placed[0]):
                _append_placement(placed, second_idx, second_mask, r2, c2)
                pair_ok = True
                break
        if pair_ok:
            break

    if not pair_ok:
        return None

    remaining = [
        idx
        for idx in sorted(range(num_components), key=lambda i: -components[i].sum())
        if idx not in close_pair
    ]
    for idx in remaining:
        mask = components[idx]
        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            origin = _random_origin(H, W, mask.shape[0], mask.shape[1])
            if origin is None:
                return None
            r, c = origin
            rows, cols = _absolute_cells(mask, r, c)
            if _is_separated_from_placed(rows, cols, placed):
                _append_placement(placed, idx, mask, r, c)
                placed_ok = True
                break
        if not placed_ok:
            return None

    for item in placed:
        rows = item["rows"]  # type: ignore[assignment]
        cols = item["cols"]  # type: ignore[assignment]
        grid[rows, cols] = FG_COLOR

    metrics = _variant_signal_metrics(grid)
    if metrics["close_pairs"] < 1 or metrics["blocked_outline_cells"] < 1:
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

# Combo variant of aa4ec2a5 (index 21)
# combo_spec: D2/3+D16/2
# Name: 所有对象统一着色+只有满足条件的对象画轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - Base task: every blue component gets a red 8-neighborhood outline; only hollow
#      components also turn teal and paint their holes magenta.
#    - Combo task: the recolor rule and the outline rule are decoupled.
#      * Every component turns teal, whether or not it has a hole.
#      * Every enclosed hole still turns magenta.
#      * Only components with area >= 15 receive the red outline.
#    - Example trace:
#      * A 20-cell hollow object becomes teal with a magenta hole and gets a red outline.
#      * A 22-cell solid object also becomes teal and gets a red outline.
#      * An 8-cell hollow object becomes teal with a magenta hole but gets no outline.
#      The contrast shows that hole detection controls only magenta filling, while area
#      controls only the outline.
# b) Complexity calibration:
#    - The size split is intentionally wide: skipped objects are <= 12 cells and outlined
#      objects are >= 18 cells. This avoids ambiguous near-threshold counts.
#    - Each grid always contains a selected hollow, a selected solid, and a skipped hollow.
#      That prevents incorrect shortcuts such as "only hollow objects change" or
#      "only hole-free objects get outlines".
# c) Implicit assumptions audit for main():
#    - The original solver assumed recoloring and outlining were coupled. That no longer
#      holds, so recoloring must happen for every component before any size filter is used.
#    - Hole detection can no longer be skipped for small components, because a small hollow
#      object still needs its magenta interior even when it receives no outline.
#    - The outline still has to be computed from the object-plus-hole-filled solid mask;
#      otherwise hollow selected objects would get an inner red ring around the hole.
# d) Placement legality:
#    - Components remain 4-connected and are placed with Chebyshev distance >= 3 plus a
#      border margin >= 1. Therefore any selected component's 1-pixel outline stays
#      disjoint from other objects and can touch the canvas edge without going out of bounds.
# e) Visual observability:
#    - Every sample contains both outlined and unoutlined objects.
#    - Every sample also contains both hollow and solid objects.
#    - Because some outlined objects are solid and some unoutlined objects are hollow, a
#      human can infer that outline presence is independent from hole presence.
# f) Visual cleanliness:
#    - Only three output colors are introduced on objects/background adjacency: teal for all
#      object bodies, magenta only inside enclosed holes, and red only on selected outlines.
#      The spacing constraints prevent these markers from colliding.
# g) Change inferability:
#    - Teal is no longer a class label for "has a hole"; it is the uniform recolor for all
#      objects. Magenta remains the only direct evidence of holes.
#    - Red becomes the visible marker for the separate size-based predicate.
# 9) New assumptions audit:
#    - `main()` assumes area is measured on the original foreground component only, not on the
#      hole-filled solid mask. `generate_input()` keeps selected/skipped areas far apart so
#      that this choice is stable and observable.
#    - `generate_input()` guarantees at least one selected hollow, one selected solid, and
#      one skipped hollow in every grid, so the fused rule cannot collapse back to the base
#      task or to a pure hole-based classifier in the training examples.


OUTLINE_MIN_AREA = 15


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


def _component_stats(grid: np.ndarray, bg: int) -> list[tuple[int, bool]]:
    stats: list[tuple[int, bool]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        stats.append((int(len(coords)), bool(_interior_holes(mask, coords))))
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every object uniformly, but outline only the larger ones.

    Input structure:
    - A uniform background contains several blue 4-connected components.
    - Some components contain fully enclosed 4-connected background holes.
    - Component sizes differ noticeably, with both clearly small and clearly large objects.

    Rule:
    - Recolor every foreground component to teal (8), regardless of whether it has a hole.
    - For any enclosed hole, fill that hole with magenta (6).
    - Only if a component has at least 15 cells, draw a one-cell red (2) outline on the
      background cells in its 8-neighborhood, using the hole-filled shape for the outline.

    Key insight: Hole detection and outline selection are separate decisions: magenta marks
    enclosed cavities for every hollow object, while the red border appears only on objects
    that pass the area threshold.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        out[mask] = 8

        solid = mask.copy()
        for hole in holes:
            out[hole[:, 0], hole[:, 1]] = 6
            solid[hole[:, 0], hole[:, 1]] = True

        if int(len(coords)) < OUTLINE_MIN_AREA:
            continue

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (combo variant) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (17, 28)
GRID_W_RANGE = (17, 27)
BG_COLOR = 4
FG_COLOR = 1

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 60

SMALL_SHAPE_SIZE_RANGE = (1, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 120

UNSELECTED_MAX_AREA = 12
SELECTED_MIN_AREA = 18

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 500
GEN_OUTER_RETRIES = 400

COMPONENT_TEMPLATES = (
    ("large_hollow", "large_solid", "small_hollow"),
    ("large_hollow", "large_solid", "small_hollow", "small_solid"),
    ("large_hollow", "large_solid", "small_hollow", "large_solid"),
    ("large_hollow", "small_hollow", "large_solid", "small_solid"),
)


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


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int, extension_prob: float) -> np.ndarray:
    if random.random() >= extension_prob:
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
    if mh <= hh + 1 or mw <= hw + 1:
        return None

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


def _make_hollow_component(
    target_hole_area: int,
    *,
    pad_range: tuple[int, int],
    max_notches: int,
    extension_prob: float,
) -> np.ndarray | None:
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*pad_range)
    pad_bot = random.randint(*pad_range)
    pad_left = random.randint(*pad_range)
    pad_right = random.randint(*pad_range)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, max_notches)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        max_h = pad_top - 1 if corner[0] == "t" else pad_bot - 1
        max_w = pad_left - 1 if corner[1] == "l" else pad_right - 1
        if max_h <= 0 or max_w <= 0:
            continue
        nh = random.randint(1, max_h)
        nw = random.randint(1, max_w)
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w, extension_prob)
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


def _make_solid_in_area(min_area: int | None, max_area: int | None) -> np.ndarray | None:
    for _ in range(POLY_GEN_RETRIES):
        use_small = max_area is not None and random.random() < 0.4
        mask = _make_small_shape() if use_small else _make_polyomino()
        if mask is None:
            continue
        area = int(mask.sum())
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        return mask
    return None


def _make_hollow_in_area(
    min_area: int | None,
    max_area: int | None,
    *,
    hole_choices: list[int],
    pad_range: tuple[int, int],
    max_notches: int,
    extension_prob: float,
) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        mask = _make_hollow_component(
            random.choice(hole_choices),
            pad_range=pad_range,
            max_notches=max_notches,
            extension_prob=extension_prob,
        )
        if mask is None:
            continue
        area = int(mask.sum())
        if min_area is not None and area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        return mask
    return None


def _build_component(kind: str) -> np.ndarray | None:
    if kind == "large_solid":
        return _make_solid_in_area(SELECTED_MIN_AREA, None)
    if kind == "small_solid":
        return _make_solid_in_area(None, UNSELECTED_MAX_AREA)
    if kind == "large_hollow":
        return _make_hollow_in_area(
            SELECTED_MIN_AREA,
            None,
            hole_choices=HOLE_VALID_AREAS,
            pad_range=(2, 3),
            max_notches=HOLLOW_MAX_NOTCHES,
            extension_prob=0.7,
        )
    if kind == "small_hollow":
        return _make_hollow_in_area(
            None,
            UNSELECTED_MAX_AREA,
            hole_choices=[2, 3, 4],
            pad_range=(1, 2),
            max_notches=1,
            extension_prob=0.0,
        )
    raise ValueError(f"Unknown component kind: {kind}")


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    kinds = list(random.choice(COMPONENT_TEMPLATES))

    components: list[np.ndarray] = []
    for kind in kinds:
        mask = _build_component(kind)
        if mask is None:
            return None
        components.append(mask)

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(len(components)), key=lambda i: -int(components[i].sum()))

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

    if np.array_equal(out, grid):
        return None

    stats = _component_stats(grid, BG_COLOR)
    selected_hollow = sum(area >= OUTLINE_MIN_AREA and has_hole for area, has_hole in stats)
    selected_solid = sum(area >= OUTLINE_MIN_AREA and not has_hole for area, has_hole in stats)
    skipped_hollow = sum(area < OUTLINE_MIN_AREA and has_hole for area, has_hole in stats)
    skipped_solid = sum(area < OUTLINE_MIN_AREA and not has_hole for area, has_hole in stats)

    if selected_hollow < 1 or selected_solid < 1 or skipped_hollow < 1:
        return None
    if selected_hollow + selected_solid == len(stats):
        return None
    if skipped_hollow + skipped_solid == len(stats):
        return None

    # Guard against degeneration back to the original task: at least one component must
    # be skipped by the outline rule, and at least one solid must still be recolored.
    if skipped_hollow + skipped_solid < 1:
        return None
    if selected_solid < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

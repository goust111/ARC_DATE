# Combo variant of aa4ec2a5 (index 4)
# combo_spec: D2/2+D8/2
# Name: 反转着色条件+对象与洞颜色互换

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The input family stays the same: 3-4 blue 4-connected objects on a yellow background,
#      with 1-2 objects containing enclosed rectangular holes.
#    - The original task does:
#        hollow object body -> teal 8
#        hole interior      -> magenta 6
#        solid object body  -> keep blue 1
#        every object       -> red 8-neighborhood outline on background
#    - This combo changes both classification and color assignment:
#        no-hole object     -> recolor whole object to magenta 6
#        hole object body   -> keep blue 1
#        hole interior      -> recolor to teal 8
#      Example trace:
#        * Solid 3x3 block:
#            original -> blue block + red outline
#            combo    -> magenta block + red outline
#        * Hollow ring with a 2x2 hole:
#            original -> teal ring + magenta hole + red outline
#            combo    -> blue ring + teal hole + red outline
#      The outline stage is unchanged, but the semantic evidence is now split across two
#      opposite cues: missing holes trigger body recoloring, while present holes trigger an
#      interior color fill that uses the swapped color.
# b) Complexity calibration
#    - Keep the original geometry, spacing, and hole construction so the puzzle remains
#      recognizably tied to aa4ec2a5 rather than becoming a new shape task.
#    - To make the fusion visible from 2-4 examples, every generated grid still contains at
#      least one hollow object and at least one solid object, and we require a solid object
#      with area >= 4 plus a hollow hole with area >= 4. This avoids tiny one-cell evidence
#      that would make the color inversion or swap hard to notice.
# c) Implicit-assumption audit for main()
#    - Hole detection still relies on component-local bbox cropping; that remains valid
#      because the geometric definition of a hole is unchanged.
#    - The solver must preserve hollow object bodies as blue, so `out = g.copy()` is now a
#      functional requirement rather than a convenience.
#    - Outline computation still has to use a filled mask (`object U holes`) so teal holes do
#      not create red interior outlines. The assumption "holes count as part of the outlined
#      solid" survives the combo unchanged.
#    - Writing the red outline only onto background cells still assumes object spacing is
#      large enough that outlines do not overwrite non-background content. The generator keeps
#      the original Chebyshev >= 3 separation to guarantee this.
# d) Placement legality
#    - Components are generated and placed under the original border-clearance and pairwise
#      Chebyshev constraints, so no object or outline goes out of bounds or overlaps another.
#    - Hollow shapes are still verified as 4-connected after carving, and holes are still
#      fully enclosed background regions inside the component bbox.
# e) Visual observability
#    - A solver can infer the combo from a few pairs because each sample contains both
#      branches: some blue inputs become solid magenta objects, while others stay blue but
#      gain teal interiors. The contrast is local and directly tied to the presence of holes.
# f) Visual cleanliness
#    - The output uses only five colors with clear roles:
#        background 4, original/hollow body 1, outline 2, solid-marker 6, hole-marker 8.
#      There is no overdraw except the intended red outline on background cells.
# g) Color-change inferability
#    - Teal and magenta are not arbitrary global remaps. Magenta now marks absence of holes
#      at the object-body level, while teal marks the interior of present holes. The paired
#      structural cues make the swap learnable without any hidden ordering or numeric-color
#      reasoning.
# h) New assumptions introduced by the combo
#    - Because hollow object bodies stay blue, a sample containing only hollow objects would
#      look too close to the input except for interior fills; the generator therefore always
#      includes at least one solid object.
#    - Because the swapped hole color is the main evidence for hollow objects, a sample with
#      only tiny holes could make the swap ambiguous; the generator therefore requires at
#      least one hollow object whose hole area is >= 4.

import random
import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    candidate = parent / "solvers" / "arc_utils.py"
    if candidate.exists():
        sys.path.insert(0, str(candidate.parent))
        break

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


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
    Add red outlines around objects, but invert which objects get recolored and swap the
    hole/object highlight colors.

    Input structure:
    - A mostly uniform background contains one or more 4-connected blue foreground objects.
    - Some objects may enclose fully surrounded 4-connected background holes inside their bbox.

    Rule:
    - For every foreground component, draw a one-cell 8-neighborhood outline in red (2).
    - If a component has no enclosed holes, recolor the whole object to magenta (6).
    - If a component does have enclosed holes, keep the object blue (1) and fill each hole
      with teal (8) before outlining.

    Key insight: The geometry test is unchanged, but the semantics are doubly inverted:
    missing holes are what trigger body recoloring, while present holes receive the swapped
    interior color; holes still count as filled-in mass for the outline computation.
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
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 8
                solid[hole[:, 0], hole[:, 1]] = True
        else:
            out[mask] = 6

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
MIN_RECOLORED_SOLID_AREA = 4
MIN_VISIBLE_HOLE_AREA = 4

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

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

GEN_OUTER_RETRIES = 300


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    holes: list[np.ndarray] = []
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            holes.append(bg_reg)
    return holes


def _has_interior_hole(mask: np.ndarray) -> bool:
    return bool(_hole_components(mask))


def _max_hole_area(mask: np.ndarray) -> int:
    return max((len(hole) for hole in _hole_components(mask)), default=0)


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
            1,
            max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1),
        )
        nw = random.randint(
            1,
            max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1),
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

    solid_areas = [int(components[i].sum()) for i in range(num_components) if i not in hollow_indices]
    hollow_hole_areas = [_max_hole_area(components[i]) for i in hollow_indices]
    if max(solid_areas, default=0) < MIN_RECOLORED_SOLID_AREA:
        return None
    if max(hollow_hole_areas, default=0) < MIN_VISIBLE_HOLE_AREA:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

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

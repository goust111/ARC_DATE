# Combo variant of aa4ec2a5 (index 20)
# combo_spec: D4/2+D15/2
# Name: 双色交替轮廓+轮廓颜色与洞颜色互补

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
for _extra in (_ROOT / "solvers", _ROOT / "arc2_solver_select"):
    if _extra.exists() and str(_extra) not in sys.path:
        sys.path.append(str(_extra))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis (combo D4/2 + D15/2)
# =====================================
# a) concretization_and_trace:
#    - Base task geometry stays intact: each blue 4-connected component gets the same 1-pixel
#      outline on background cells in its 8-neighborhood, and hollow components still recolor
#      to teal with magenta holes before outlining.
#    - The fused change affects color assignment on that outline. Solid components now alternate
#      red/orange along a deterministic clockwise sweep. Hollow components alternate green/orange
#      along the same sweep, where green is the visible complement of the magenta hole color.
#    - Small trace: for a 2x2 solid block, the original output had a 12-cell all-red contour;
#      here it becomes red, orange, red, orange, ... around the block. For a 4x4 ring with a
#      2x2 hole, the original hollow output had teal body + magenta hole + all-red contour;
#      here it becomes teal body + magenta hole + green, orange, green, orange, ... contour.
#      The geometry is unchanged, but the hollow case now couples interior-hole color and contour.
# b) complexity_calibration:
#    - Only one extra contour color is introduced beyond the D15 single variant: green. Orange
#      remains the shared secondary alternation color, which keeps the palette small and gives
#      repeated evidence across objects. This avoids a brittle many-color lookup puzzle.
#    - The mode split is binary and structural: "has enclosed hole" decides whether the primary
#      alternating color is red or green. Because every generated grid has both hollow and solid
#      components, the distinction is learnable from a few examples.
# c) implicit_assumptions_review:
#    - `outline = _outline_mask(solid)` was sufficient in the source because every outline cell
#      shared one color; after fusion, we must impose a deterministic order before painting.
#    - `solid[hole] = True` still holds: the contour should wrap the filled silhouette, not trace
#      the cavity separately, so holes remain merged into the outline reference mask.
#    - `out[outline & (out == bg)] = 2` no longer works because each outline cell may receive a
#      different color; the fused rule needs per-cell painting with a component-dependent palette.
#    - The new logic assumes each component outline is processed independently. The original
#      Chebyshev spacing >= 3 still guarantees outlines do not collide, so that assumption remains
#      valid after fusion.
# d) placement_legality:
#    - The generator preserves the original size ranges, border clearance, connectivity checks,
#      and pairwise Chebyshev spacing. The 1-cell outline may touch the frame but never requires
#      pixels outside the grid, and component outlines still cannot overlap each other.
# e) visual_observability:
#    - Samples deliberately contain both a hollow component and at least one solid component.
#      Post-generation validation also requires visible red, green, and orange contour cells, so
#      a solver can compare "solid -> red/orange" against "hollow -> green/orange" directly.
#    - The clockwise ordering uses only visible geometry (outline position around the object); it
#      does not depend on hidden ids or color values that a human could not infer from examples.
# f) cleanliness:
#    - The fused output still has only three layers: object body, optional hole fill, and one
#      outer contour. No extra markers are stacked into the same region, so the contour remains
#      readable even when colors differ by object type.
# g) inferability_of_color_change:
#    - Magenta holes are the only trigger for the complementary green contour. Orange appears in
#      both modes as the shared alternating partner, which makes the rule parsimonious:
#      find the same alternation pattern first, then notice that hollow objects swap red for the
#      complement of the hole color.
#    - Quantitative guardrail: generated samples are accepted only if at least one solid contour
#      exposes both red and orange cells, and at least one hollow contour exposes both green and
#      orange cells. This prevents silent regression to the source task or to a one-mode dataset.


BG_COLOR = 4
FG_COLOR = 1
HOLLOW_BODY_COLOR = 8
HOLE_COLOR = 6
COMPLEMENT_OF_HOLE_COLOR = 3
ALT_SECONDARY_COLOR = 7
SOLID_OUTLINE_COLORS = (2, ALT_SECONDARY_COLOR)
HOLLOW_OUTLINE_COLORS = (COMPLEMENT_OF_HOLE_COLOR, ALT_SECONDARY_COLOR)

MIN_RED_OUTLINE_CELLS = 6
MIN_GREEN_OUTLINE_CELLS = 6
MIN_ORANGE_OUTLINE_CELLS = 10


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


def _clockwise_outline_order(solid: np.ndarray) -> list[tuple[int, int]]:
    outline = _outline_mask(solid)
    coords = np.argwhere(outline)
    if coords.size == 0:
        return []

    solid_coords = np.argwhere(solid)
    r0, c0, r1, c1 = bbox_coords(solid_coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    decorated: list[tuple[float, float, int, int]] = []
    for r, c in coords:
        dy = float(r) - center_r
        dx = float(c) - center_c
        angle = float(np.arctan2(dx, -dy)) % (2.0 * np.pi)
        radius = max(abs(dy), abs(dx))
        decorated.append((angle, -radius, int(r), int(c)))

    decorated.sort()
    ordered = [(r, c) for _, _, r, c in decorated]
    start = min(range(len(ordered)), key=lambda i: (ordered[i][0], ordered[i][1]))
    return ordered[start:] + ordered[:start]


def _outline_colors(has_holes: bool) -> tuple[int, int]:
    return HOLLOW_OUTLINE_COLORS if has_holes else SOLID_OUTLINE_COLORS


def _paint_alternating_outline(
    out: np.ndarray, bg: int, solid: np.ndarray, colors: tuple[int, int]
) -> None:
    for idx, (r, c) in enumerate(_clockwise_outline_order(solid)):
        if out[r, c] == bg:
            out[r, c] = colors[idx % 2]


def _component_records(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    records: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        records.append(
            {
                "mask": mask,
                "holes": holes,
                "solid": solid,
                "order": _clockwise_outline_order(solid),
            }
        )
    return records


def _validate_combo_behavior(grid: np.ndarray, out: np.ndarray) -> bool:
    red = 0
    green = 0
    orange = 0
    solid_ok = 0
    hollow_ok = 0

    for rec in _component_records(grid):
        holes = rec["holes"]
        mask = rec["mask"]
        order = rec["order"]
        has_holes = bool(holes)
        expected = _outline_colors(has_holes)
        counts = {expected[0]: 0, expected[1]: 0}

        if has_holes:
            if not np.all(out[mask] == HOLLOW_BODY_COLOR):
                return False
            for hole in holes:
                if not np.all(out[hole[:, 0], hole[:, 1]] == HOLE_COLOR):
                    return False
        else:
            if not np.all(out[mask] == FG_COLOR):
                return False

        for idx, (r, c) in enumerate(order):
            color = int(out[r, c])
            if color != expected[idx % 2]:
                return False
            counts[color] += 1

        if has_holes and counts[HOLLOW_OUTLINE_COLORS[0]] > 0 and counts[HOLLOW_OUTLINE_COLORS[1]] > 0:
            hollow_ok += 1
        if not has_holes and counts[SOLID_OUTLINE_COLORS[0]] > 0 and counts[SOLID_OUTLINE_COLORS[1]] > 0:
            solid_ok += 1

        red += counts.get(2, 0)
        green += counts.get(3, 0)
        orange += counts.get(7, 0)

    return (
        solid_ok >= 1
        and hollow_ok >= 1
        and red >= MIN_RED_OUTLINE_CELLS
        and green >= MIN_GREEN_OUTLINE_CELLS
        and orange >= MIN_ORANGE_OUTLINE_CELLS
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects and draw alternating outlines whose primary color depends on holes.

    Input structure:
    - A uniform yellow background contains several blue 4-connected foreground components.
    - Some components enclose one or more 4-connected background holes fully inside their bbox.

    Rule:
    - Build the same 1-cell 8-neighborhood outer outline for every component, using the filled
      silhouette that results after treating enclosed holes as part of the shape.
    - Order each outline in a deterministic clockwise sweep and color it with alternating pairs.
      Solid components use red (2), orange (7), red (2), orange (7), ...
    - If a component encloses holes, recolor the component to teal (8), fill its holes with
      magenta (6), and then use the complementary alternating pair green (3), orange (7), ...

    Key insight:
    - The contour geometry is still the original outer shell, but two cues now interact:
      outline pixels alternate by position along the contour, and hollow objects swap the
      primary contour color from red to the complement of their magenta hole fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for rec in _component_records(g):
        mask = rec["mask"]
        holes = rec["holes"]
        solid = rec["solid"]
        has_holes = bool(holes)

        if has_holes:
            out[mask] = HOLLOW_BODY_COLOR
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = HOLE_COLOR

        _paint_alternating_outline(out, bg, solid, _outline_colors(has_holes))

    return out


# --------------- generator (c) ---------------

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

    if len(components) != num_components:
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

    if not _validate_combo_behavior(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

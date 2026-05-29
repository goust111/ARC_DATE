# Combo variant of aa4ec2a5 (index 24)
# combo_spec: D4/0+D7/3+D15/0
# Name: 三步序列+间断点轮廓+按类别轮廓色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# combo_variant_spec:
#   preserved_input:
#     - Grid size stays within 15-28 rows x 15-27 cols on a uniform yellow background.
#     - There are still 3-4 blue 4-connected components, with 1-2 hollow objects and
#       the rest solid, and all placement / hole-shape constraints remain the original ones.
#   fused_change:
#     - The original two-stage rule "detect holes / recolor if hollow -> draw one shared red outline"
#       becomes a three-stage chain:
#         1) detect whether each component has enclosed holes,
#         2) recolor only hollow components (body teal, holes magenta),
#         3) choose the outline style from that class result.
#     - Hollow objects get a red dotted outline: compute the usual 8-neighborhood ring around
#       the object-with-holes-filled mask, then keep only alternating cells in an object-anchored
#       checkerboard pattern.
#     - Solid objects keep their blue body and get the full orange outline.
#   concrete_behavior_trace:
#     - Example solid case: a 2x2 blue block originally produced a full red ring. Here it stays blue,
#       and every background cell in that outer ring becomes orange, so the geometry is unchanged but
#       the class-dependent color is different.
#     - Example hollow case: a blue frame around a 2x2 cavity originally became teal with a magenta
#       hole and then received a full red ring. Here the first two steps are identical, but the last
#       step keeps only alternating red ring cells, so the outer border is visibly sparse.
#     - The critical difference from the original task is that the same hole test now controls both
#       the interior recoloring and the final border's color/style pair.
#   complexity_calibration:
#     - There are only two visible classes and two outline templates, which is enough to create a
#       real inference task without exploding the mapping space.
#     - The generator keeps at least one hollow object and at least two solid ones so both classes
#       repeat inside a sample, and it rejects cases where the hollow dotted border is too weak.
#     - Using the bbox top-left as the dotted-pattern anchor makes the spacing rule visible from each
#       object itself; it does not rely on hidden object ordering or metadata.
#   implicit_assumptions_review:
#     - Original assumption: a single `out[outline & (out == bg)] = 2` works because all objects share
#       one border color and style. This is false now, so the solver must branch on the hole result.
#     - Original assumption: holes should be merged into the mask before outlining hollow objects.
#       This still must hold; otherwise the new dotted border would leak into the cavity and create
#       an extra inner ring that contradicts the task story.
#     - Original assumption: outline collisions are harmless because every outline writes the same
#       color. This is no longer true once orange and red coexist, so the generator must preserve the
#       original Chebyshev >= 3 spacing that keeps one-pixel outlines disjoint.
#     - New assumption: the dotted subset should be stable and nonempty. The solver enforces this with
#       an object-anchored parity rule plus a fallback that keeps one outline cell if a degenerate tiny
#       ring ever lands entirely on the discarded parity.
#   placement_legality:
#     - The original border clearance >= 1 is still sufficient because both the solid orange outline
#       and the hollow dotted red outline are subsets of the same clipped one-pixel 8-neighborhood ring.
#     - Since the new hollow outline is sparser than the full ring and solids still use the full ring,
#       the original spacing constraint remains safe and prevents overlap or out-of-bounds writes.
#   visual_observability:
#     - A human can compare blue+orange solid objects against teal+magenta objects with red dotted
#       borders and infer that the visible cavity is the trigger for both the recoloring and the border
#       template change.
#     - The generator explicitly filters for samples that contain all six meaningful colors and enough
#       red dotted / orange solid cells for the contrast to be obvious in 2-4 examples.
#   cleanliness:
#     - Only hollow objects receive interior recoloring, and all outlines remain on background cells.
#       There is no extra overlay inside objects beyond the original magenta hole marking.
#     - The palette stays structured: yellow background, blue or teal bodies, magenta holes, orange
#       solid borders, and red dotted hollow borders.
#   generator_guarantees_for_new_main:
#     - At least one hollow component exists, so the dotted-red branch is always exercised.
#     - At least two solid components exist, so the orange-solid branch has repeated evidence.
#     - Visibility metrics reject samples where the hollow dotted border barely differs from the full
#       ring or where one of the class-specific outline colors would be absent.


RED_OUTLINE = 2
MAGENTA_HOLE = 6
ORANGE_OUTLINE = 7
TEAL_HOLLOW = 8
OUTLINE_KEEP_PARITY = 0


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


def _dotted_outline_mask(mask: np.ndarray, coords: np.ndarray) -> np.ndarray:
    outline = _outline_mask(mask)
    r0, c0, _r1, _c1 = bbox_coords(coords)
    rr, cc = np.indices(mask.shape)
    dotted = outline & ((((rr - int(r0)) + (cc - int(c0))) % 2) == OUTLINE_KEEP_PARITY)

    if outline.any() and not dotted.any():
        first_r, first_c = np.argwhere(outline)[0]
        dotted[first_r, first_c] = True
    return dotted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a three-stage class-aware border transform to each foreground object.

    Input structure:
    - A uniform yellow background contains 3-4 blue 4-connected foreground components.
    - Some components enclose 4-connected background holes strictly inside their bounding boxes, while
      the remaining components are solid.

    Rule:
    - First classify each component by whether it contains enclosed holes.
    - Then recolor every hollow component to teal (8) and recolor its hole cells to magenta (6),
      treating those hole cells as filled when computing the outside border.
    - Finally draw class-specific one-pixel 8-neighborhood outlines on background cells only:
      hollow components get a red (2) alternating dotted outline, while solid components get a full
      orange (7) outline.

    Key insight:
    - The hole test is not just for interior recoloring anymore; it also selects the border template.
      Hollow objects must be outlined from their hole-filled silhouette, and solid objects are the only
      ones that keep a continuous border.
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
            out[mask] = TEAL_HOLLOW
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA_HOLE
                solid[hole[:, 0], hole[:, 1]] = True
            outline = _dotted_outline_mask(solid, coords)
            outline_color = RED_OUTLINE
        else:
            outline = _outline_mask(solid)
            outline_color = ORANGE_OUTLINE

        out[outline & (out == bg)] = outline_color

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)
MIN_SOLID_COUNT = 2

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

MIN_RED_DOTTED_OUTLINE_CELLS = 8
MIN_ORANGE_SOLID_OUTLINE_CELLS = 12
MIN_HOLLOW_OUTLINE_CELL_DROP = 6


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


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    metrics = {
        "hollow_components": 0,
        "solid_components": 0,
        "hollow_full_outline_cells": 0,
        "red_dotted_outline_cells": 0,
        "orange_solid_outline_cells": 0,
    }

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True

        if holes:
            metrics["hollow_components"] += 1
            metrics["hollow_full_outline_cells"] += int(_outline_mask(solid).sum())
            metrics["red_dotted_outline_cells"] += int(_dotted_outline_mask(solid, coords).sum())
        else:
            metrics["solid_components"] += 1
            metrics["orange_solid_outline_cells"] += int(_outline_mask(solid).sum())

    return metrics


def _variant_visible(out: np.ndarray, metrics: dict[str, int]) -> bool:
    colors = {int(v) for v in out.flat}
    required = {
        BG_COLOR,
        FG_COLOR,
        RED_OUTLINE,
        ORANGE_OUTLINE,
        MAGENTA_HOLE,
        TEAL_HOLLOW,
    }
    if not required.issubset(colors):
        return False

    if metrics["hollow_components"] < 1 or metrics["solid_components"] < MIN_SOLID_COUNT:
        return False
    if metrics["red_dotted_outline_cells"] < MIN_RED_DOTTED_OUTLINE_CELLS:
        return False
    if metrics["orange_solid_outline_cells"] < MIN_ORANGE_SOLID_OUTLINE_CELLS:
        return False
    if (
        metrics["hollow_full_outline_cells"] - metrics["red_dotted_outline_cells"]
        < MIN_HOLLOW_OUTLINE_CELL_DROP
    ):
        return False

    if int(np.count_nonzero(out == RED_OUTLINE)) != metrics["red_dotted_outline_cells"]:
        return False
    if int(np.count_nonzero(out == ORANGE_OUTLINE)) != metrics["orange_solid_outline_cells"]:
        return False

    return True


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


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    max_hollow = min(HOLLOW_COUNT_RANGE[1], num_components - MIN_SOLID_COUNT)
    if max_hollow < HOLLOW_COUNT_RANGE[0]:
        return None
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], max_hollow)

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
        out = main(grid)
    except Exception:
        return None

    metrics = _variant_metrics(grid)
    if not _variant_visible(out, metrics):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

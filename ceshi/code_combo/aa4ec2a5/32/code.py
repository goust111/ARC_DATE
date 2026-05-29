# Combo variant of aa4ec2a5 (index 32)
# combo_spec: D1/2+D7/2
# Name: 内轮廓+先轮廓再检测洞

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# source_task: aa4ec2a5
# a) concretization_and_trace:
#    - Input still contains 3-4 blue 4-connected components on a yellow background, with two of them
#      constructed as hollow shells around a rectangular hole.
#    - The fused rule changes both the contour side and the operation order. First, every original blue
#      component gets an inner red contour: cells on the object side of the outer boundary turn red.
#      Hole cells are temporarily filled only for this contour test, so the first step never paints a
#      red ring around the hole itself.
#    - Then hole detection runs on the already modified grid and only on the remaining blue components.
#      This means the blue shell has been eroded before we ask whether it still encloses anything.
#    - Concrete trace:
#      original order on a hollow shell with padding (top,left,right,bottom) = (1,2,2,2):
#      detect hole first -> whole shell becomes teal, hole becomes magenta, then the outer teal edge
#      turns red; the object is still treated as hollow.
#      fused order -> outer boundary turns red first, so the one-cell top shell disappears from the
#      remaining blue mask; the old hole now connects to the bbox edge and is no longer a detected hole.
#      That object stays red+blue only, with no teal/magenta recoloring.
#    - A second hollow shell with all paddings >= 2 behaves differently: after the red shrink, the
#      surviving blue ring still encloses the hole, so that remaining blue component becomes teal and
#      its cavity becomes magenta. This contrast is the core behavior the samples must expose.
# b) complexity_calibration:
#    - To keep the sequence inversion visible, the generator always includes one "persistent" hollow
#      component whose hole survives step 1 and one "fragile" hollow component whose hole does not.
#    - We keep the rest of the canvas close to the source task: the same color roles, the same 4-connected
#      component family, and the same placement separation. This avoids introducing extra concepts while
#      still making the order reversal observable.
#    - Thin solids are still allowed, but the generator rejects grids with too few blue residual
#      components after the first step; otherwise the puzzle would collapse into "paint objects red".
# c) implicit_assumption_audit_for_main:
#    - The original solver assumed hole detection saw the untouched blue geometry. That is false here:
#      step 2 must inspect the post-outline grid, not the input grid.
#    - The original outline helper wrote onto background cells only. Here the contour is internal, so the
#      first step must overwrite object cells instead.
#    - The original hollow recolor covered the full shell before outlining. Here that would erase the
#      intended sequencing effect. The recolor step must happen after outlining and only on blue remnants.
#    - The first-step contour still needs temporary hole filling while computing the boundary; otherwise
#      hollow objects would gain an extra red rim around the hole and the fusion would stop looking like
#      a true "inside analogue" of the source outline.
#    - The second-step hole detector intentionally treats the remaining blue components as the objects of
#      interest. Red contour cells are not recolored and can break a once-hollow object into a non-hollow
#      blue residue, which is exactly the intended inversion effect.
# d) placement_legality:
#    - Components are placed with Chebyshev distance >= 3 and at least one-cell border clearance, as in
#      the source generator. Since red is drawn inward rather than outward, these constraints remain more
#      than sufficient to avoid overlap or ambiguity.
#    - Hollow masks are validated after construction and again after rendering via solver-side checks, so
#      no generated input relies on undefined geometry.
# e) visual_observability:
#    - The samples visibly separate three outcomes:
#      solid objects: red shrink only;
#      fragile hollow objects: originally hollow in the input, but no teal/magenta after shrink;
#      persistent hollow objects: red outer shrink plus later teal/magenta inside.
#    - This lets a human notice both axes of change: the contour moved inward, and hole marking depends
#      on the already-shrunk blue residue rather than the original blue shell.
# f) visual_cleanliness:
#    - The output uses structured layers only: yellow background, red inner contour, blue or teal
#      surviving cores, and magenta only for holes that survive step 1. No two objects touch, and each
#      region type keeps a stable visual role across examples.
#    - Generator validation rejects degenerate cases where the fusion output matches the "detect first,
#      then inner-outline" baseline, ensuring the combo does not collapse into a single-dimension variant.
# g) color_change_inference:
#    - Colors remain low-entropy and role-based: red marks the first-step inner boundary, teal marks blue
#      components that are still hollow after step 1, and magenta marks their surviving cavities.
#    - Because one originally hollow object is forced to lose its hole while another keeps it, the samples
#      provide direct structural evidence for when teal/magenta appear; the rule is not an arbitrary color
#      remapping.


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _local_holes(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    holes: list[np.ndarray] = []
    for hole in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == h - 1)
            | (hole[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            holes.append(hole.copy())
    return holes


def _has_interior_hole(mask: np.ndarray) -> bool:
    return bool(_local_holes(mask))


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in _local_holes(sub):
        shifted = hole.copy()
        shifted[:, 0] += int(r0)
        shifted[:, 1] += int(c0)
        holes.append(shifted)
    return holes


def _inner_outline_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        if not padded[r : r + 3, c : c + 3].all():
            outline[r, c] = True
    return outline


def _apply_inner_outline_to_mask(mask: np.ndarray) -> np.ndarray:
    holes = _local_holes(mask)
    solid = mask.copy()
    for hole in holes:
        solid[hole[:, 0], hole[:, 1]] = True
    return _inner_outline_mask(solid) & mask


def _post_outline_blue_mask(mask: np.ndarray) -> np.ndarray:
    return mask & ~_apply_inner_outline_to_mask(mask)


def _post_outline_component_stats(mask: np.ndarray) -> tuple[int, int, int]:
    residual = _post_outline_blue_mask(mask)
    blue_components = 0
    hollow_components = 0
    blue_cells = int(residual.sum())

    for comp in components_mask(residual, connectivity=4):
        comp_mask = np.zeros_like(residual, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        cropped = _crop_mask(comp_mask)
        if cropped is None:
            continue
        blue_components += 1
        if _has_interior_hole(cropped):
            hollow_components += 1

    return blue_components, hollow_components, blue_cells


def _apply_first_step(grid: np.ndarray, bg: int) -> np.ndarray:
    out = grid.copy()
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        if grid[coords[0, 0], coords[0, 1]] != FG_COLOR:
            continue
        mask = _coords_mask(grid.shape, coords)
        inner = _apply_inner_outline_to_mask(mask)
        out[inner] = RED_COLOR
    return out


def _detect_then_inner_outline(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        if g[coords[0, 0], coords[0, 1]] != FG_COLOR:
            continue
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = TEAL_COLOR
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA_COLOR
                solid[hole[:, 0], hole[:, 1]] = True

        inner = _inner_outline_mask(solid) & mask
        out[inner] = RED_COLOR

    return out


def _first_step_grid_metrics(grid: np.ndarray, bg: int) -> tuple[int, int]:
    step1 = _apply_first_step(grid, bg)
    blue_components = 0
    hollow_blue_components = 0

    for comp in find_components(step1, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        if step1[coords[0, 0], coords[0, 1]] != FG_COLOR:
            continue
        blue_components += 1
        mask = _coords_mask(step1.shape, coords)
        if _interior_holes(mask, coords):
            hollow_blue_components += 1

    return blue_components, hollow_blue_components


def _order_difference_cells(grid: np.ndarray, bg: int) -> int:
    fused = main(grid)
    baseline = _detect_then_inner_outline(grid, bg)
    return int(np.count_nonzero(fused != baseline))


def main(grid: np.ndarray) -> np.ndarray:
    """
    First shrink every blue object with a red inner contour, then detect surviving holes.

    Input structure:
    - A uniform yellow background contains several blue 4-connected components.
    - Some of those blue components are hollow shells with enclosed rectangular cavities.

    Rule:
    - Step 1: for every original blue component, turn its outer object-side boundary red (2).
      The boundary is the one-cell 8-neighborhood inner contour, computed after temporarily
      filling any original holes so the first step follows only the exterior perimeter.
    - Step 2: on that already outlined grid, inspect only the remaining blue components.
      Any blue component that still encloses an internal cavity is recolored teal (8), and
      the cells of that surviving cavity become magenta (6).

    Key insight: Hole detection happens after the blue shape has been eroded by the red inner
    contour, so an originally hollow object may lose its enclosure and never receive teal or
    magenta. Only holes that survive the first-step shrink are marked.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = _apply_first_step(g, bg)

    for comp in find_components(out, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        if out[coords[0, 0], coords[0, 1]] != FG_COLOR:
            continue
        mask = _coords_mask(out.shape, coords)
        holes = _interior_holes(mask, coords)
        if not holes:
            continue
        out[mask] = TEAL_COLOR
        for hole in holes:
            out[hole[:, 0], hole[:, 1]] = MAGENTA_COLOR

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1
RED_COLOR = 2
MAGENTA_COLOR = 6
TEAL_COLOR = 8

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT = 2

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.15
SMALL_SHAPE_SIZE_RANGE = (1, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_GEN_RETRIES = 80
HOLLOW_MIN_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

MIN_BLUE_COMPONENTS_AFTER_STEP1 = 2
MIN_POST_HOLLOW_COMPONENTS = 1
MIN_ORDER_DIFF_CELLS = 4

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


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return (
            np.ones((1, 2), dtype=bool)
            if random.random() < 0.5
            else np.ones((2, 1), dtype=bool)
        )
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


def _extend_mask(mask: np.ndarray) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    base_h, base_w = mask.shape
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


def _sample_pads(role: str) -> tuple[int, int, int, int]:
    if role == "persistent":
        return tuple(random.randint(2, 3) for _ in range(4))

    pads = [random.randint(2, 3) for _ in range(4)]
    pads[random.randrange(4)] = 1
    return tuple(pads)


def _apply_corner_notches(
    mask: np.ndarray,
    pad_top: int,
    pad_bot: int,
    pad_left: int,
    pad_right: int,
    role: str,
) -> np.ndarray:
    max_notches = HOLLOW_MAX_NOTCHES if role == "persistent" else 1
    num_notches = random.randint(0, max_notches)

    h, w = mask.shape
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        max_nh = (pad_top - 1) if corner[0] == "t" else (pad_bot - 1)
        max_nw = (pad_left - 1) if corner[1] == "l" else (pad_right - 1)
        if max_nh <= 0 or max_nw <= 0:
            continue
        nh = random.randint(1, max_nh)
        nw = random.randint(1, max_nw)
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, w - nw : w] = False
        elif corner == "bl":
            mask[h - nh : h, 0:nw] = False
        else:
            mask[h - nh : h, w - nw : w] = False

    return mask


def _make_hollow_component(role: str) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        dims = _rect_dims(random.choice(HOLE_VALID_AREAS))
        if dims is None:
            continue
        hh, hw = dims
        pad_top, pad_bot, pad_left, pad_right = _sample_pads(role)

        base_h = hh + pad_top + pad_bot
        base_w = hw + pad_left + pad_right
        mask = np.ones((base_h, base_w), dtype=bool)
        mask[pad_top : pad_top + hh, pad_left : pad_left + hw] = False

        mask = _apply_corner_notches(mask, pad_top, pad_bot, pad_left, pad_right, role)
        mask = _extend_mask(mask)
        mask = _crop_mask(mask)

        if mask is None:
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        if mask.sum() < HOLLOW_MIN_AREA or not _has_interior_hole(mask):
            continue

        blue_components, hollow_components, blue_cells = _post_outline_component_stats(mask)
        if blue_cells == 0 or blue_components == 0:
            continue
        if role == "persistent" and hollow_components >= 1:
            return mask
        if role == "fragile" and hollow_components == 0:
            return mask

    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    num_components = random.randint(*COMP_COUNT_RANGE)

    if num_components <= HOLLOW_COUNT:
        return None

    roles: list[str] = ["persistent", "fragile"]
    roles.extend(["solid"] * (num_components - HOLLOW_COUNT))
    random.shuffle(roles)

    components: list[np.ndarray] = []
    fragile_count = 0
    persistent_count = 0
    for role in roles:
        if role == "solid":
            if random.random() < SMALL_SHAPE_PROB:
                mask = _make_small_shape()
            else:
                mask = None
                for _ in range(POLY_GEN_RETRIES):
                    mask = _make_polyomino()
                    if mask is not None:
                        break
                if mask is None:
                    return None
        else:
            mask = _make_hollow_component(role)
            if mask is None:
                return None
            if role == "persistent":
                persistent_count += 1
            else:
                fragile_count += 1
        components.append(mask)

    if persistent_count != 1 or fragile_count != 1:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, height - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, width - mw - BORDER_MIN))
            if r + mh > height - BORDER_MIN or c + mw > width - BORDER_MIN:
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

    blue_components, post_hollow_components = _first_step_grid_metrics(grid, BG_COLOR)
    if blue_components < MIN_BLUE_COMPONENTS_AFTER_STEP1:
        return None
    if post_hollow_components < MIN_POST_HOLLOW_COMPONENTS:
        return None
    if post_hollow_components >= HOLLOW_COUNT:
        return None
    if _order_difference_cells(grid, BG_COLOR) < MIN_ORDER_DIFF_CELLS:
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

# Variant of aa4ec2a5
# Dimension: D16 — 处理的基数关系 (meta)
# Change: 每个对象独立产生一组轮廓（1:1映射），对象数量不限... → 一个对象产生多层轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the D16 variant
# -----------------------------------
# a) Concrete behavior:
#    - Input stays a yellow background with blue 4-connected components.
#    - Components now come in three visible classes: 0 holes, 1 hole, 2 holes.
#    - Output keeps the original interior recolor rule for hollow objects:
#      foreground turns teal and enclosed holes turn magenta.
#    - The changed behavior is in the outline stage. Instead of exactly one red
#      outline per object, the object produces 1 + (# interior holes) concentric
#      8-neighborhood rings. The fixed color order is red, then orange, then green.
#      So 0-hole objects get red only; 1-hole objects get red+orange; 2-hole objects
#      get red+orange+green.
#    - Small trace example:
#      original: a one-hole object -> fill object teal, hole magenta, draw one red ring.
#      variant:  a one-hole object -> fill object teal, hole magenta, draw a red ring
#      around the filled footprint, then treat that ring as part of the footprint and
#      draw a second orange ring outside it.
# b) Complexity calibration:
#    - The rule only has three cardinality buckets (0/1/2 holes), so humans can infer
#      it from a few examples.
#    - Every generated grid contains exactly one object from each bucket, which gives
#      repeated support for the mapping "hole count -> number of rings" inside each sample.
# c) Old assumptions audited:
#    - `if holes:` still safely means "apply teal/magenta recolor", because that rule
#      remains unchanged.
#    - The original single `outline = _outline_mask(solid)` assumed there is only one
#      layer. That is no longer valid. The variant therefore iterates the dilation:
#      ring_i = outline(current_footprint), then current_footprint |= ring_i.
#    - The original `out[outline & (out == bg)] = 2` was harmless because all outlines
#      shared one color. With multiple colors it would be wrong if rings from different
#      objects overlapped, because later colors could be blocked unpredictably. The
#      generator therefore guarantees pairwise separation large enough that ring zones
#      never overlap.
# d) Placement legality:
#    - A component with L layers needs L cells of border clearance so all rings fit.
#    - Two components with layer counts La and Lb must have Chebyshev distance at least
#      La + Lb + 1 between any foreground cells so their ring zones stay disjoint.
# e) Visual observability:
#    - Each sample shows the three classes 0/1/2 holes together, so the viewer can
#      compare one, two, and three rings side by side.
#    - The ring color order is fixed globally, not object-specific.
# f) Visual cleanliness:
#    - The generator avoids tiny solids and uses extra spacing because three-layer
#      halos become messy if they interleave.
# g) Color inference:
#    - Only three outline colors are used: red, orange, green. Their order is spatial
#      (inner to outer), so the palette change is tied to visible geometry instead of
#      hidden metadata.
# h) Degeneracy prevention:
#    - Inputs always include at least one one-hole object and one two-hole object, so the
#      new multi-layer behavior is exercised on every generated sample.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes components have at most two enclosed holes if full color fidelity
#      is desired; the generator enforces exactly 0/1/2.
#    - `main()` assumes different objects' rings do not overlap; placement enforces that.


RING_COLORS = (2, 7, 3)  # red, orange, green
BG_COLOR = 4
FG_COLOR = 1
OBJECT_WITH_HOLE_COLOR = 8
HOLE_FILL_COLOR = 6


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
    Draw one to three colored halo layers around each object based on its hole count.

    Input structure:
    - A yellow background contains several blue 4-connected foreground objects.
    - Each object may enclose 0, 1, or 2 background holes completely inside its bbox.

    Rule:
    - Every component gets 1 + (# enclosed holes) concentric 8-neighborhood outline layers.
    - The outline colors always progress outward as red (2), orange (7), then green (3).
    - If a component contains enclosed holes, recolor the object teal (8), recolor each
      enclosed hole magenta (6), and then generate the outline layers around the filled footprint.

    Key insight:
    - First count interior holes, then treat those holes as if they were filled when building
      the successive halo rings; the hole count controls how many ring expansions are drawn.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        footprint = mask.copy()
        if holes:
            out[mask] = OBJECT_WITH_HOLE_COLOR
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = HOLE_FILL_COLOR
                footprint[hole[:, 0], hole[:, 1]] = True

        grown = footprint.copy()
        layers = min(len(RING_COLORS), 1 + len(holes))
        for color in RING_COLORS[:layers]:
            ring = _outline_mask(grown)
            out[ring & (out == bg)] = color
            grown |= ring

    return out


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (21, 28)
GRID_W_RANGE = (21, 27)

# Exactly one component from each hole-count bucket for observability
COMPONENT_HOLE_COUNTS = (0, 1, 2)

# Polyomino construction (solid shapes)
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Hollow component construction
SINGLE_HOLE_AREAS = [2, 3, 4, 6, 8, 9, 12]
DOUBLE_HOLE_AREAS = [2, 3, 4, 6]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_GEN_RETRIES = 60
HOLLOW_MIN_AREA = 12

# Placement
PLACEMENT_RETRIES = 600

# Top-level generation
GEN_OUTER_RETRIES = 300


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _count_interior_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    count = 0
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            count += 1
    return count


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
    rows_a: np.ndarray,
    cols_a: np.ndarray,
    pr: int,
    pc: int,
    pm: np.ndarray,
    min_sep: int,
) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < min_sep:
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

    anchor_h, anchor_w = rects[0]
    for i in range(1, num_rects):
        rh, rw = rects[i]
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), min(canvas_h - rh, r0 + anchor_h + 2))
            nc = random.randint(max(0, c0 - rw + 1), min(canvas_w - rw, c0 + anchor_w + 2))
            if mask[nr : nr + rh, nc : nc + rw].any():
                mask[nr : nr + rh, nc : nc + rw] = True
                break

    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


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


def _cut_corner_notches(
    mask: np.ndarray,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
) -> None:
    corners = ["tl", "tr", "bl", "br"]
    random.shuffle(corners)
    for corner in corners[: random.randint(0, HOLLOW_MAX_NOTCHES)]:
        max_h = pad_top - 1 if corner[0] == "t" else pad_bottom - 1
        max_w = pad_left - 1 if corner[1] == "l" else pad_right - 1
        if max_h <= 0 or max_w <= 0:
            continue
        nh = random.randint(1, max_h)
        nw = random.randint(1, max_w)
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, mask.shape[1] - nw : mask.shape[1]] = False
        elif corner == "bl":
            mask[mask.shape[0] - nh : mask.shape[0], 0:nw] = False
        else:
            mask[mask.shape[0] - nh : mask.shape[0], mask.shape[1] - nw : mask.shape[1]] = False


def _finalize_hollow(mask: np.ndarray, expected_holes: int) -> np.ndarray | None:
    mask = _extend_mask(mask)
    mask = _crop_mask(mask)
    if mask is None:
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    if _count_interior_holes(mask) != expected_holes:
        return None
    if int(mask.sum()) < HOLLOW_MIN_AREA:
        return None
    return mask


def _make_single_hole_component() -> np.ndarray | None:
    dims = _rect_dims(random.choice(SINGLE_HOLE_AREAS))
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bottom = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    mask = np.ones((hh + pad_top + pad_bottom, hw + pad_left + pad_right), dtype=bool)
    _cut_corner_notches(mask, pad_top, pad_bottom, pad_left, pad_right)
    mask[pad_top : pad_top + hh, pad_left : pad_left + hw] = False
    return _finalize_hollow(mask, expected_holes=1)


def _make_double_hole_component() -> np.ndarray | None:
    dims1 = _rect_dims(random.choice(DOUBLE_HOLE_AREAS))
    dims2 = _rect_dims(random.choice(DOUBLE_HOLE_AREAS))
    if dims1 is None or dims2 is None:
        return None
    h1, w1 = dims1
    h2, w2 = dims2

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bottom = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)
    mid_pad = random.randint(*HOLLOW_PAD_RANGE)
    horizontal = random.random() < 0.5

    if horizontal:
        inner_h = max(h1, h2)
        mask = np.ones(
            (inner_h + pad_top + pad_bottom, w1 + w2 + mid_pad + pad_left + pad_right),
            dtype=bool,
        )
        _cut_corner_notches(mask, pad_top, pad_bottom, pad_left, pad_right)
        r1 = pad_top + random.randint(0, inner_h - h1)
        r2 = pad_top + random.randint(0, inner_h - h2)
        c1 = pad_left
        c2 = pad_left + w1 + mid_pad
        mask[r1 : r1 + h1, c1 : c1 + w1] = False
        mask[r2 : r2 + h2, c2 : c2 + w2] = False
    else:
        inner_w = max(w1, w2)
        mask = np.ones(
            (h1 + h2 + mid_pad + pad_top + pad_bottom, inner_w + pad_left + pad_right),
            dtype=bool,
        )
        _cut_corner_notches(mask, pad_top, pad_bottom, pad_left, pad_right)
        r1 = pad_top
        r2 = pad_top + h1 + mid_pad
        c1 = pad_left + random.randint(0, inner_w - w1)
        c2 = pad_left + random.randint(0, inner_w - w2)
        mask[r1 : r1 + h1, c1 : c1 + w1] = False
        mask[r2 : r2 + h2, c2 : c2 + w2] = False

    return _finalize_hollow(mask, expected_holes=2)


def _make_component(hole_count: int) -> np.ndarray | None:
    if hole_count == 0:
        for _ in range(POLY_GEN_RETRIES):
            mask = _make_polyomino()
            if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
                return mask
        return None
    if hole_count == 1:
        for _ in range(HOLLOW_GEN_RETRIES):
            mask = _make_single_hole_component()
            if mask is not None:
                return mask
        return None
    if hole_count == 2:
        for _ in range(HOLLOW_GEN_RETRIES):
            mask = _make_double_hole_component()
            if mask is not None:
                return mask
        return None
    raise ValueError(f"Unsupported hole count: {hole_count}")


def _component_layers(mask: np.ndarray) -> int:
    return min(len(RING_COLORS), 1 + _count_interior_holes(mask))


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    specs = []
    for hole_count in COMPONENT_HOLE_COUNTS:
        mask = _make_component(hole_count)
        if mask is None:
            return None
        specs.append({"mask": mask, "holes": hole_count, "layers": hole_count + 1})

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray, int]] = []

    order = sorted(range(len(specs)), key=lambda i: -int(specs[i]["mask"].sum()))
    for idx in order:
        mask = specs[idx]["mask"]
        layers = int(specs[idx]["layers"])
        mh, mw = mask.shape

        if H - mh - 2 * layers < 0 or W - mw - 2 * layers < 0:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(layers, H - mh - layers)
            c = random.randint(layers, W - mw - layers)
            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, pm, pl in placed:
                min_sep = layers + pl + 1
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= min_sep:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm, min_sep):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask, layers))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask, _layers in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    # Enforce that the new cardinality-controlled behavior is genuinely present.
    ring_colors_present = {int(v) for v in out.flatten() if int(v) in set(RING_COLORS)}
    if 7 not in ring_colors_present or 3 not in ring_colors_present:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

# Combo variant of aa4ec2a5 (index 5)
# combo_spec: D1/1+D16/1
# Name: 可变半径轮廓+多层轮廓由洞数量决定

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ---------------
# a) Concrete behavior and trace:
#    - Original task: each blue component gets exactly one red 8-neighborhood outline; hollow
#      components are recolored teal and their interior holes magenta before outlining.
#    - Fused task: each component now has two independent latent attributes.
#      1. Area -> outline width: area <= 5 uses radius 1, area 6-15 uses radius 2,
#         area >= 16 uses radius 3.
#      2. Hole count -> layer count: 0 holes -> 1 layer, 1 hole -> 2 layers,
#         2 holes -> 3 layers.
#    - Small trace:
#      a 4-cell solid object uses radius 1 and 1 layer, so it stays blue and gets one thin red ring.
#      a 12-cell one-hole object uses radius 2 and 2 layers, so it becomes teal with a magenta hole,
#      then gets a 2-cell-thick red band followed by a 2-cell-thick orange band.
#      a 20-cell two-hole object uses radius 3 and 3 layers, so it becomes teal with two magenta
#      holes, then gets 3-cell-thick red, orange, and green bands in sequence.
#    - The important fusion is that each successive layer reuses the same area-derived radius.
#      This means the total spatial footprint is radius * layers, not "radius plus one-pixel rings".
# b) Complexity calibration:
#    - Only three area buckets and three hole-count buckets are used, so the rule remains inferable.
#    - Every sample contains one small solid, one one-hole medium object, and one two-hole large
#      object. This keeps the layout physically realizable while still showing all three width
#      buckets and all three layer counts inside each pair.
#    - The two latent variables remain separately visible because width is read from the thickness
#      of each colored band, while hole count is read from how many band colors appear.
# c) Implicit assumption audit for `main()`:
#    - `_interior_holes(...)` still works because hole detection remains bbox-local 4-connected
#      background analysis; the fusion changes only how the detected holes affect rendering.
#    - Area must be read from the original input component, not the hole-filled footprint. Otherwise
#      hollow objects would accidentally use a larger radius than the input visually suggests.
#    - The original single-outline assumption is invalid. The solver must iteratively grow the filled
#      footprint, emitting one radius-thick band per layer color.
#    - `out[band & (out == bg)] = color` still assumes disjoint halo zones. The generator therefore
#      enforces spacing based on total expansion radius * layers.
# d) Placement legality:
#    - A component with radius r and layer count L needs border clearance r * L so its full halo fits.
#    - Two components with expansions e1 = r1 * L1 and e2 = r2 * L2 must have foreground Chebyshev
#      separation at least e1 + e2 + 1 so none of their colored bands overlap.
# e) Visual observability:
#    - Each sample shows radius-1/radius-2/radius-3 widths side by side, and also shows 1/2/3 layers.
#    - Even though the medium object also has more layers than the small one, the two rules are still
#      visible separately: one concerns per-layer thickness, the other concerns how many differently
#      colored shells are stacked.
# f) Visual cleanliness:
#    - The generator uses only three components and reserves large spacing, which keeps thick multi-band
#      halos readable instead of forming merged color fields.
# g) Color inference:
#    - Outline colors are fixed by depth: inner red, then orange, then green. Colors do not depend on
#      object identity. The only adaptive quantities are band width and number of bands.
# h) Degeneracy prevention:
#    - Samples always include orange and green, so the task never collapses to the original one-layer
#      behavior.
#    - Samples always include at least one radius-2 object and one radius-3 object, so the task never
#      collapses to the fixed-width multi-layer variant.
#    - Generator validation compares the fused output against "width-only" and "layers-only" ablations
#      and rejects any sample where they accidentally match.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes at most two interior holes if full red/orange/green fidelity is desired.
#      The generator enforces exact hole counts in {0, 1, 2}.
#    - `main()` assumes each component is 4-connected and that halo zones do not intersect.
#      The generator constructs and places components to satisfy both conditions.


SMALL_MAX_AREA = 5
MEDIUM_MAX_AREA = 15
RING_COLORS = (2, 7, 3)  # red, orange, green

BG_COLOR = 4
FG_COLOR = 1
OBJECT_WITH_HOLE_COLOR = 8
HOLE_FILL_COLOR = 6

# Canvas bounds. Widened because radius-3 objects with three layers need 9 cells of total halo
# clearance, which leaves too little legal placement space on the original canvas family.
GRID_H_RANGE = (42, 46)
GRID_W_RANGE = (42, 46)

AREA_BUCKETS: dict[str, tuple[int, int]] = {
    "small": (1, 5),
    "medium": (8, 15),
    "large": (16, 30),
}

COMPONENT_PATTERNS = (
    (("small", 0), ("medium", 1), ("large", 2)),
)

SMALL_SHAPE_OPTIONS: dict[int, list[np.ndarray]] = {
    1: [np.array([[1]], dtype=bool)],
    2: [np.array([[1, 1]], dtype=bool), np.array([[1], [1]], dtype=bool)],
    3: [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    ],
    5: [
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    ],
}

SINGLE_HOLE_AREAS = {
    "medium": [2, 3, 4, 6],
    "large": [4, 6, 8, 9, 12],
}
DOUBLE_HOLE_AREAS = {
    "medium": [2, 3, 4],
    "large": [2, 3, 4, 6],
}
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = {
    "medium": (1, 2),
    "large": (1, 3),
}
MID_PAD_RANGE = {
    "medium": (1, 2),
    "large": (1, 3),
}
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.45
HOLLOW_EXT_H_RANGE = (1, 3)
HOLLOW_EXT_SIDE_W_RANGE = (1, 2)
HOLLOW_GEN_RETRIES = 200
PLACEMENT_RETRIES = 1200
GEN_OUTER_RETRIES = 400


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _outline_radius(area: int) -> int:
    if area <= SMALL_MAX_AREA:
        return 1
    if area <= MEDIUM_MAX_AREA:
        return 2
    return 3


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


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


def _outline_band(mask: np.ndarray, radius: int) -> np.ndarray:
    h, w = mask.shape
    band = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            band[rr[ok], cc[ok]] = True
    band &= ~mask
    return band


def _render(
    grid: np.ndarray,
    *,
    use_area_radius: bool,
    use_hole_layers: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        area = int(comp["area"])
        radius = _outline_radius(area) if use_area_radius else 1
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        layers = 1 + len(holes) if use_hole_layers else 1
        layers = min(len(RING_COLORS), layers)

        footprint = mask.copy()
        if holes:
            out[mask] = OBJECT_WITH_HOLE_COLOR
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = HOLE_FILL_COLOR
                footprint[hole[:, 0], hole[:, 1]] = True

        grown = footprint.copy()
        for color in RING_COLORS[:layers]:
            band = _outline_band(grown, radius)
            out[band & (out == bg)] = color
            grown |= band

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw colored halo stacks whose thickness comes from area and whose depth comes from hole count.

    Input structure:
    - A uniform background contains several blue 4-connected foreground objects.
    - Objects may enclose 0, 1, or 2 background holes fully inside their bounding boxes.

    Rule:
    - Measure each object's original blue area and convert it to an outline radius:
      area <= 5 -> radius 1, area 6-15 -> radius 2, area > 15 -> radius 3.
    - Count the object's enclosed holes and convert that to the number of outline layers:
      0 holes -> 1 layer, 1 hole -> 2 layers, 2 holes -> 3 layers.
    - Hollow objects are recolored teal (8), their enclosed holes are recolored magenta (6),
      and then the filled footprint emits concentric radius-thick bands colored red (2),
      orange (7), and green (3) from inner to outer.

    Key insight:
    - Two separate measurements drive the same halo construction: area fixes how thick each
      band is, while hole count fixes how many such bands are stacked around the object.
    """
    return _render(grid, use_area_radius=True, use_hole_layers=True)


def _make_small_component() -> np.ndarray:
    area = random.randint(*AREA_BUCKETS["small"])
    return random.choice(SMALL_SHAPE_OPTIONS[area]).copy()


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


def _finalize_hollow(
    mask: np.ndarray,
    *,
    expected_holes: int,
    area_lo: int,
    area_hi: int,
) -> np.ndarray | None:
    mask = _extend_mask(mask)
    mask = _crop_mask(mask)
    if mask is None:
        return None
    if not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    if _count_interior_holes(mask) != expected_holes:
        return None
    area = int(mask.sum())
    if not (area_lo <= area <= area_hi):
        return None
    return mask


def _make_single_hole_component(bucket: str) -> np.ndarray | None:
    area_lo, area_hi = AREA_BUCKETS[bucket]
    pad_lo, pad_hi = HOLLOW_PAD_RANGE[bucket]
    for _ in range(HOLLOW_GEN_RETRIES):
        dims = _rect_dims(random.choice(SINGLE_HOLE_AREAS[bucket]))
        if dims is None:
            continue
        hh, hw = dims

        pad_top = random.randint(pad_lo, pad_hi)
        pad_bottom = random.randint(pad_lo, pad_hi)
        pad_left = random.randint(pad_lo, pad_hi)
        pad_right = random.randint(pad_lo, pad_hi)

        mask = np.ones((hh + pad_top + pad_bottom, hw + pad_left + pad_right), dtype=bool)
        _cut_corner_notches(mask, pad_top, pad_bottom, pad_left, pad_right)
        mask[pad_top : pad_top + hh, pad_left : pad_left + hw] = False

        finalized = _finalize_hollow(
            mask,
            expected_holes=1,
            area_lo=area_lo,
            area_hi=area_hi,
        )
        if finalized is not None:
            return finalized
    return None


def _make_double_hole_component(bucket: str) -> np.ndarray | None:
    area_lo, area_hi = AREA_BUCKETS[bucket]
    pad_lo, pad_hi = HOLLOW_PAD_RANGE[bucket]
    mid_lo, mid_hi = MID_PAD_RANGE[bucket]
    for _ in range(HOLLOW_GEN_RETRIES):
        dims1 = _rect_dims(random.choice(DOUBLE_HOLE_AREAS[bucket]))
        dims2 = _rect_dims(random.choice(DOUBLE_HOLE_AREAS[bucket]))
        if dims1 is None or dims2 is None:
            continue
        h1, w1 = dims1
        h2, w2 = dims2

        pad_top = random.randint(pad_lo, pad_hi)
        pad_bottom = random.randint(pad_lo, pad_hi)
        pad_left = random.randint(pad_lo, pad_hi)
        pad_right = random.randint(pad_lo, pad_hi)
        mid_pad = random.randint(mid_lo, mid_hi)
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

        finalized = _finalize_hollow(
            mask,
            expected_holes=2,
            area_lo=area_lo,
            area_hi=area_hi,
        )
        if finalized is not None:
            return finalized
    return None


def _make_component(bucket: str, hole_count: int) -> np.ndarray | None:
    if hole_count == 0:
        if bucket == "small":
            return _make_small_component()
        raise ValueError(f"Unsupported solid bucket: {bucket}")
    if hole_count == 1:
        return _make_single_hole_component(bucket)
    if hole_count == 2:
        return _make_double_hole_component(bucket)
    raise ValueError(f"Unsupported hole count: {hole_count}")


def _chebyshev_conflict(
    rows_a: np.ndarray,
    cols_a: np.ndarray,
    rows_b: np.ndarray,
    cols_b: np.ndarray,
    min_sep: int,
) -> bool:
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < min_sep:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    pattern = list(random.choice(COMPONENT_PATTERNS))
    random.shuffle(pattern)

    components: list[dict[str, object]] = []
    for bucket, hole_count in pattern:
        mask = _make_component(bucket, hole_count)
        if mask is None:
            return None
        area = int(mask.sum())
        radius = _outline_radius(area)
        layers = hole_count + 1
        expansion = radius * layers
        components.append(
            {
                "mask": mask,
                "bucket": bucket,
                "hole_count": hole_count,
                "area": area,
                "radius": radius,
                "layers": layers,
                "expansion": expansion,
            }
        )

    radii = {int(comp["radius"]) for comp in components}
    layers_seen = {int(comp["layers"]) for comp in components}
    if radii != {1, 2, 3} or layers_seen != {1, 2, 3}:
        return None

    min_h = max(int(comp["mask"].shape[0]) + 2 * int(comp["expansion"]) for comp in components)
    min_w = max(int(comp["mask"].shape[1]) + 2 * int(comp["expansion"]) for comp in components)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    H = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    W = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])
    grid = np.full((H, W), BG_COLOR, dtype=int)

    placed: list[dict[str, object]] = []
    order = sorted(
        range(len(components)),
        key=lambda i: -(
            (components[i]["mask"].shape[0] + 2 * int(components[i]["expansion"]))
            * (components[i]["mask"].shape[1] + 2 * int(components[i]["expansion"]))
        ),
    )

    for idx in order:
        comp = components[idx]
        mask = comp["mask"]
        expansion = int(comp["expansion"])
        mh, mw = mask.shape
        row_min = expansion
        row_max = H - mh - expansion
        col_min = expansion
        col_max = W - mw - expansion
        if row_max < row_min or col_max < col_min:
            return None

        rows_local, cols_local = np.where(mask)
        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(row_min, row_max)
            c = random.randint(col_min, col_max)
            rows_abs = rows_local + r
            cols_abs = cols_local + c

            conflict = False
            for prev in placed:
                prev_rows = prev["rows_abs"]
                prev_cols = prev["cols_abs"]
                prev_r = int(prev["r"])
                prev_c = int(prev["c"])
                prev_mask = prev["mask"]
                prev_expansion = int(prev["expansion"])
                min_sep = expansion + prev_expansion + 1

                row_gap = max(0, max(r - (prev_r + prev_mask.shape[0] - 1), prev_r - (r + mh - 1)))
                col_gap = max(0, max(c - (prev_c + prev_mask.shape[1] - 1), prev_c - (c + mw - 1)))
                if max(row_gap, col_gap) >= min_sep:
                    continue
                if _chebyshev_conflict(rows_abs, cols_abs, prev_rows, prev_cols, min_sep):
                    conflict = True
                    break

            if not conflict:
                placed.append(
                    {
                        "r": r,
                        "c": c,
                        "mask": mask,
                        "rows_abs": rows_abs,
                        "cols_abs": cols_abs,
                        "expansion": expansion,
                    }
                )
                placed_ok = True
                break

        if not placed_ok:
            return None

    for item in placed:
        grid[item["rows_abs"], item["cols_abs"]] = FG_COLOR

    combo_out = main(grid)
    width_only_out = _render(grid, use_area_radius=True, use_hole_layers=False)
    layers_only_out = _render(grid, use_area_radius=False, use_hole_layers=True)
    if np.array_equal(combo_out, width_only_out) or np.array_equal(combo_out, layers_only_out):
        return None

    ring_colors_present = {int(v) for v in combo_out.flatten() if int(v) in set(RING_COLORS)}
    if ring_colors_present != set(RING_COLORS):
        return None

    comps = find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True)
    stats = []
    for comp in comps:
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        stats.append((_outline_radius(int(comp["area"])), len(holes) + 1))
    if {r for r, _ in stats} != {1, 2, 3}:
        return None
    if {l for _, l in stats} != {1, 2, 3}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

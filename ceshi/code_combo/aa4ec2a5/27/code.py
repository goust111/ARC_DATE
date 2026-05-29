# Combo variant of aa4ec2a5 (index 27)
# combo_spec: D6/2+D7/0
# Name: 每个对象独立输出子网格+按面积排序处理

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis for combo D6/2 + D7/0
# --------------------------------------
# a) Concrete behavior trace
#    - Original aa4ec2a5 keeps the full input canvas. Each blue 4-connected object is
#      processed in place: hollow ones become teal with magenta holes, then every object
#      gets a 1-pixel red 8-neighborhood outline.
#    - D6/2 breaks that shared-canvas assumption. Each object is now rendered on its own
#      tight local canvas that contains just the object plus its outline, and those local
#      canvases are concatenated horizontally.
#    - D7/0 says objects are handled from largest area to smallest area. On a single shared
#      canvas that would matter via overwriting; after D6/2 separates objects, the visible
#      effect becomes the order of the packed subgrids rather than cross-object painting.
#    - Small trace:
#      input reading order might be `small solid`, then `large hollow`, then `medium solid`.
#      Original task would leave them where they are on the big yellow grid.
#      Fused task instead produces three cropped frames:
#      `large hollow with red rim | medium solid with red rim | small solid with red rim`.
#      The large object's hollow recolor still happens locally, but the global scene is now
#      a size-sorted catalogue rather than a spatial map.
#
# b) Complexity calibration
#    - If the generator allowed equal areas, the packed order could be ambiguous. It
#      therefore enforces pairwise distinct component areas and a minimum area gap.
#    - If input reading order already matched descending area in most cases, D7/0 would be
#      visually weak. The generator rejects such cases and keeps only samples where the
#      output order disagrees with row-major input order.
#    - Packed output uses a single consistent divider color and top-aligned frames, so the
#      new shape rule stays readable instead of turning into arbitrary montage noise.
#
# c) Hidden-assumption audit of the original `main()`
#    - `out = g.copy()` depended on "answer lives on the input canvas"; D6/2 invalidates
#      that, so the solver must build one local frame per component and then pack them.
#    - The original loop could draw directly onto one shared array because all objects
#      remained in their original positions. Here component coordinates are first converted
#      into local frame coordinates with a 1-cell halo margin.
#    - The D7 single-variant changed outline writes to unconditional overwrites. After D6/2
#      there is no longer any cross-object interaction, so unconditional overwrite is only
#      needed inside each local frame; in practice it is equivalent to background-only
#      writing because a frame contains exactly one decorated object.
#    - Hole detection must still use the untouched input component mask, not the packed
#      output, so `_interior_holes(...)` remains object-local and unchanged.
#
# d) Placement legality
#    - Input components still must be non-overlapping, in-bounds, and pairwise separated by
#      Chebyshev distance >= 3, keeping the original "distinct isolated objects" structure.
#    - Because output separation is handled by the solver, generator legality is only about
#      the input scene; packed subgrids cannot overlap because `main()` assigns each one its
#      own horizontal slice with explicit divider columns.
#
# e) Visual observability
#    - A human can see three repeated signals across examples:
#      1. output shape no longer matches the input canvas,
#      2. each output chunk is a tight crop around exactly one outlined object,
#      3. chunk order follows visible object size rather than input position.
#    - No ordering depends on raw color ids, component labels, or any non-visual metadata.
#
# f) Visual cleanliness
#    - Input keeps the original spacious layout; output removes clutter by isolating one
#      object per frame.
#    - Divider columns are uniform and full-height, so object boundaries in the packed strip
#      are easy to parse even when neighboring frames have different heights.
#    - The generator caps packed width/height so outputs stay compact enough to inspect.
#
# g) Color inferability
#    - Original color semantics remain intact inside each frame: blue = solid object,
#      teal = hollow object body, magenta = enclosed hole, red = outline, yellow = local
#      background.
#    - The only new color is the green divider column, used purely as a repeated structural
#      separator between packed frames.
#
# h) Variant-specific metrics used after testing
#    - `order_mismatch`: descending-area order differs from row-major input order.
#    - `shape_changed`: packed output shape differs from input shape.
#    - `packed_width` / `packed_height`: verify the D6/2 strip stays visually tidy.
#    - `area_gaps`: adjacent descending areas stay separated by at least the configured gap.
#
# i) New assumptions introduced by the fused solver, and generator support
#    - `main()` assumes components can be ranked visually by area; generator guarantees
#      unique areas with a minimum gap.
#    - `main()` assumes the packed strip is top-aligned with a fixed divider style;
#      generator always expects that exact renderer.
#    - `main()` does not need diagonal-contact interactions from the standalone D7/0
#      variant anymore; generator intentionally avoids relying on overlap effects and makes
#      the size-based order observable through input/output reordering instead.

SEP_COLOR = 3
SEP_WIDTH = 1


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


def _collect_components(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps: list[dict[str, object]] = []
    for idx, comp in enumerate(find_components(g, bg=bg, connectivity=4, monochromatic=True)):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        comps.append(
            {
                "id": int(idx),
                "coords": coords,
                "mask": mask,
                "holes": holes,
                "area": int(coords.shape[0]),
                "bbox": bbox_coords(coords),
                "color": int(comp["color"]),
            }
        )
    return g, bg, comps


def _area_sort_key(comp: dict[str, object]) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return (-int(comp["area"]), int(r0), int(c0))


def _reading_sort_key(comp: dict[str, object]) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return (int(r0), int(c0), -int(comp["area"]))


def _render_component_frame(comp: dict[str, object], bg: int) -> np.ndarray:
    coords = np.asarray(comp["coords"], dtype=int)
    holes = comp["holes"]
    r0, c0, r1, c1 = comp["bbox"]
    frame = np.full((r1 - r0 + 3, c1 - c0 + 3), int(bg), dtype=int)
    solid = np.zeros_like(frame, dtype=bool)

    body_rows = coords[:, 0] - int(r0) + 1
    body_cols = coords[:, 1] - int(c0) + 1
    body_color = 8 if holes else int(comp["color"])
    frame[body_rows, body_cols] = int(body_color)
    solid[body_rows, body_cols] = True

    for hole in holes:
        local_rows = hole[:, 0] - int(r0) + 1
        local_cols = hole[:, 1] - int(c0) + 1
        frame[local_rows, local_cols] = 6
        solid[local_rows, local_cols] = True

    frame[_outline_mask(solid)] = 2
    return frame


def _pack_frames(frames: list[np.ndarray], bg: int) -> np.ndarray:
    if not frames:
        return np.zeros((0, 0), dtype=int)

    out_h = max(int(frame.shape[0]) for frame in frames)
    out_w = sum(int(frame.shape[1]) for frame in frames) + SEP_WIDTH * (len(frames) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    cursor = 0
    for i, frame in enumerate(frames):
        fh, fw = frame.shape
        out[0:fh, cursor : cursor + fw] = frame
        cursor += fw
        if i != len(frames) - 1:
            out[:, cursor : cursor + SEP_WIDTH] = SEP_COLOR
            cursor += SEP_WIDTH

    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    _g, bg, comps = _collect_components(grid)
    area_sorted = sorted(comps, key=_area_sort_key)
    reading_sorted = sorted(comps, key=_reading_sort_key)
    frames = [_render_component_frame(comp, bg) for comp in area_sorted]

    packed_height = max(int(frame.shape[0]) for frame in frames)
    packed_width = sum(int(frame.shape[1]) for frame in frames) + SEP_WIDTH * (len(frames) - 1)
    areas = [int(comp["area"]) for comp in area_sorted]

    return {
        "frame_count": len(frames),
        "area_order_ids": [int(comp["id"]) for comp in area_sorted],
        "reading_order_ids": [int(comp["id"]) for comp in reading_sorted],
        "order_mismatch": [int(comp["id"]) for comp in area_sorted]
        != [int(comp["id"]) for comp in reading_sorted],
        "areas_desc": areas,
        "area_gaps": [areas[i] - areas[i + 1] for i in range(len(areas) - 1)],
        "packed_height": int(packed_height),
        "packed_width": int(packed_width),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Render each object into its own cropped outline frame, then pack the frames by area.

    Input structure:
    - A mostly uniform yellow background contains several blue 4-connected objects.
    - Some objects enclose 4-connected background holes inside their own bounding boxes.

    Rule:
    - Process each component independently: hollow components become teal with magenta
      holes, and every component gets a one-cell red outline on its 8-neighborhood.
    - Crop each decorated component to its own tight subgrid, so the red outline sits on
      the border of that local frame.
    - Sort the subgrids from largest component area to smallest, then concatenate them
      left-to-right with full-height green divider columns; shorter subgrids are top-aligned.

    Key insight: After D6/2 separates objects into independent subgrids, D7/0 no longer
    shows up as cross-object overwriting on one shared canvas. Its visible role is the
    descending-area order of the packed output strip.
    """
    _g, bg, comps = _collect_components(grid)
    if not comps:
        return np.asarray(grid, dtype=int).copy()

    comps.sort(key=_area_sort_key)
    frames = [_render_component_frame(comp, bg) for comp in comps]
    return _pack_frames(frames, bg)


# --------------- generator (combo D6/2 + D7/0) ---------------

GRID_H_RANGE = (15, 26)
GRID_W_RANGE = (15, 25)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 5)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 35

SMALL_SHAPE_PROB = 0.15
SMALL_SHAPE_SIZE_RANGE = (2, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.45
HOLLOW_EXT_H_RANGE = (1, 3)
HOLLOW_EXT_SIDE_W_RANGE = (1, 2)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 60
HOLLOW_MIN_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 450

MIN_AREA_GAP = 3
MAX_PACKED_WIDTH = 48
MAX_PACKED_HEIGHT = 16
GEN_OUTER_RETRIES = 500


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

    canvas_h = sum(rh for rh, _rw in rects) + POLY_CANVAS_PAD
    canvas_w = sum(rw for _rh, rw in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for rh, rw in rects[1:]:
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), min(canvas_h - rh, r0 + rects[0][0] + 2))
            nc = random.randint(max(0, c0 - rw + 1), min(canvas_w - rw, c0 + rects[0][1] + 2))
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
    if mh - hh - 1 < 1 or mw - hw - 1 < 1:
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
    coords = coords.copy()
    coords[:, 0] += int(r)
    coords[:, 1] += int(c)
    return coords


def _min_chebyshev(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    diffs = np.abs(coords_a[:, None, :] - coords_b[None, :, :])
    return int(np.max(diffs, axis=2).min())


def _place_components(H: int, W: int, components: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]] | None:
    placements: dict[int, tuple[int, int, np.ndarray]] = {}
    placed_coords: list[np.ndarray] = []

    order = sorted(range(len(components)), key=lambda i: -int(components[i].sum()))
    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        max_r = H - mh - BORDER_MIN
        max_c = W - mw - BORDER_MIN
        if max_r < BORDER_MIN or max_c < BORDER_MIN:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max_r)
            c = random.randint(BORDER_MIN, max_c)
            coords = _placed_coords(mask, r, c)
            if all(_min_chebyshev(coords, other) >= CHEBYSHEV_MIN for other in placed_coords):
                placements[idx] = (r, c, mask)
                placed_coords.append(coords)
                placed = True
                break

        if not placed:
            return None

    return [placements[i] for i in range(len(components))]


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
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
                    components.append(mask)
                    break
            else:
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
    if any(areas[i] - areas[i + 1] < MIN_AREA_GAP for i in range(len(areas) - 1)):
        return None

    placed = _place_components(H, W, components)
    if placed is None:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    metrics = _behavior_metrics(grid)
    if not bool(metrics["order_mismatch"]):
        return None
    if int(metrics["packed_width"]) > MAX_PACKED_WIDTH or int(metrics["packed_height"]) > MAX_PACKED_HEIGHT:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape == grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")

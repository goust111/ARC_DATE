# Variant of aa4ec2a5
# Dimension: D1 — 轮廓绘制方式 (transform)
# Change: 在每个蓝色对象周围绘制1像素宽的8邻域（Chebyshev距离=1）红色轮廓... → 可变半径轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ----------------
# variant_spec:
#   core_change:
#     - Keep the original hollow-object recoloring rule.
#     - Replace the fixed one-cell Chebyshev outline with a size-coded radius:
#       area 1-5 -> radius 1, area 6-15 -> radius 2, area >= 16 -> radius 3.
#   behavior_trace:
#     - Original flow: detect one blue component -> optionally fill its holes for outlining ->
#       draw only the radius-1 red ring.
#     - Variant flow: detect the same component -> count its original blue cells -> map that
#       count to radius 1/2/3 -> optionally fill holes for outlining -> draw the chosen ring.
#     - Example: a 4-cell solid keeps blue and gets a 1-cell halo; a 12-cell hollow becomes
#       teal with a magenta hole and gets a 2-cell halo; a 20-cell solid gets a 3-cell halo.
#   complexity_calibration:
#     - The generator always includes at least one small, one medium, and one large component
#       so the three outline widths are simultaneously visible inside a single training pair.
#     - Large components are capped to moderate areas so multiple radius-3 halos still fit on
#       the original canvas family without turning the output into clutter.
#   main_assumption_audit:
#     - `_interior_holes` still relies on bbox-local 4-connected background detection; the
#       variant does not change that geometry, so the helper remains valid.
#     - Radius must be computed from the original foreground area, not from the hole-filled
#       mask, otherwise hollow objects would silently encode a different parameter.
#     - Writing red only onto `out == bg` is still required: wide halos must not overwrite the
#       teal body or magenta hole of the same object.
#   placement_legality:
#     - If two components use radii r1 and r2, their foreground cells must be at Chebyshev
#       distance at least r1 + r2 + 1; otherwise the red halos can overlap.
#     - Border clearance must be at least the component radius so the intended full-width halo
#       is visible instead of being cropped by the canvas edge.
#   observability_and_cleanliness:
#     - Every sample mixes the three size buckets, giving repeated evidence for the mapping
#       "small -> thin halo, medium -> medium halo, large -> thick halo".
#     - Colors remain structurally simple: blue for solid inputs, teal/magenta only for hollow
#       outputs, and red only for outlines. No unrelated color rule is introduced.


SMALL_MAX_AREA = 5
MEDIUM_MAX_AREA = 15


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


def _outline_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
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
    Draw size-dependent red halos around objects, while preserving the original hole recoloring.

    Input structure:
    - A uniform background contains several blue 4-connected foreground objects.
    - Some objects are hollow: they enclose one or more 4-connected background holes inside their bbox.

    Rule:
    - Measure each object's original blue-cell area and convert it to an outline radius:
      area <= 5 -> radius 1, area 6-15 -> radius 2, area > 15 -> radius 3.
    - Hollow objects are recolored teal (8), and their enclosed holes are recolored magenta (6).
    - Draw the chosen red (2) Chebyshev-neighborhood outline on background cells only; holes
      count as filled when computing that outline.

    Key insight: The red outline is no longer always one cell thick; object size encodes the
    required halo width, while the original hollow-object treatment remains unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        area = int(comp["area"])
        radius = _outline_radius(area)
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid, radius)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator ---------------

# Canvas / grid bounds
GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 27)
BG_COLOR = 4
FG_COLOR = 1

# Area buckets used by the new transform
AREA_BUCKETS: dict[str, tuple[int, int]] = {
    "small": (1, 5),
    "medium": (6, 15),
    "large": (16, 28),
}

# Component counts
COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

# Solid component generation
SOLID_GEN_RETRIES = 80
SOLID_GROW_RETRIES = 20

# Hollow component construction
HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 120

# Placement
PLACEMENT_RETRIES = 700

# Top-level generation
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
    rows_a: np.ndarray,
    cols_a: np.ndarray,
    rows_b: np.ndarray,
    cols_b: np.ndarray,
    min_dist: int,
) -> bool:
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < min_dist:
            return True
    return False


def _make_exact_small_shape(area: int) -> np.ndarray:
    options: dict[int, list[np.ndarray]] = {
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
    return random.choice(options[area]).copy()


def _frontier_candidates(mask: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = mask.shape
    frontier: dict[tuple[int, int], int] = {}
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                score = 0
                for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                    if 0 <= ar < h and 0 <= ac < w and mask[ar, ac]:
                        score += 1
                frontier[(nr, nc)] = max(frontier.get((nr, nc), 0), score)
    return [(score, r, c) for (r, c), score in frontier.items()]


def _grow_connected_shape(target_area: int) -> np.ndarray | None:
    canvas = max(9, int(np.ceil(np.sqrt(target_area))) + 7)
    for _ in range(SOLID_GROW_RETRIES):
        mask = np.zeros((canvas, canvas), dtype=bool)
        center = canvas // 2
        mask[center, center] = True

        while int(mask.sum()) < target_area:
            frontier = _frontier_candidates(mask)
            if not frontier:
                break
            if random.random() < 0.75:
                best = max(score for score, _, _ in frontier)
                candidates = [(r, c) for score, r, c in frontier if score == best]
            else:
                candidates = [(r, c) for _, r, c in frontier]
            r, c = random.choice(candidates)
            mask[r, c] = True

            remaining = target_area - int(mask.sum())
            if remaining >= 2 and random.random() < 0.20:
                second_frontier = _frontier_candidates(mask)
                if second_frontier:
                    second_best = max(score for score, _, _ in second_frontier)
                    second_candidates = [
                        (rr, cc) for score, rr, cc in second_frontier if score >= second_best - 1
                    ]
                    rr, cc = random.choice(second_candidates)
                    mask[rr, cc] = True

        cropped = _crop_mask(mask)
        if cropped is None or int(cropped.sum()) != target_area:
            continue
        if target_area >= 6 and min(cropped.shape) == 1:
            continue
        if target_area >= 10 and cropped.shape[0] * cropped.shape[1] > target_area * 4:
            continue
        if is_contiguous(cropped.astype(int), background=0, connectivity=4):
            return cropped
    return None


def _make_solid_component(area_lo: int, area_hi: int) -> np.ndarray | None:
    for _ in range(SOLID_GEN_RETRIES):
        target_area = random.randint(area_lo, area_hi)
        if target_area <= SMALL_MAX_AREA:
            return _make_exact_small_shape(target_area)
        mask = _grow_connected_shape(target_area)
        if mask is not None:
            return mask
    return None


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


def _make_hollow_component_in_area(area_lo: int, area_hi: int) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
        if mask is not None and area_lo <= int(mask.sum()) <= area_hi:
            return mask
    return None


def _component_blue_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _component_bucket_plan(num_components: int) -> list[str]:
    buckets = ["small", "medium", "large"]
    if num_components == 4:
        buckets.append(random.choice(["small", "medium", "large"]))
    random.shuffle(buckets)
    return buckets


def _gen_attempt() -> np.ndarray | None:
    num_components = random.randint(*COMP_COUNT_RANGE)
    buckets = _component_bucket_plan(num_components)
    eligible_hollow = [i for i, bucket in enumerate(buckets) if bucket != "small"]
    max_hollow = min(HOLLOW_COUNT_RANGE[1], num_components - 1, len(eligible_hollow))
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], max_hollow)
    hollow_indices = set(random.sample(eligible_hollow, num_hollow))

    components: list[dict[str, object]] = []
    for i, bucket in enumerate(buckets):
        area_lo, area_hi = AREA_BUCKETS[bucket]
        if i in hollow_indices:
            mask = _make_hollow_component_in_area(area_lo, area_hi)
        else:
            mask = _make_solid_component(area_lo, area_hi)
        if mask is None:
            return None
        area = _component_blue_area(mask)
        radius = _outline_radius(area)
        components.append({"mask": mask, "area": area, "radius": radius, "bucket": bucket})

    min_h = max(int(comp["mask"].shape[0]) + 2 * int(comp["radius"]) for comp in components)
    min_w = max(int(comp["mask"].shape[1]) + 2 * int(comp["radius"]) for comp in components)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    H = random.randint(max(GRID_H_RANGE[0], min_h), GRID_H_RANGE[1])
    W = random.randint(max(GRID_W_RANGE[0], min_w), GRID_W_RANGE[1])

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[dict[str, object]] = []
    order = sorted(
        range(num_components),
        key=lambda i: -(
            (components[i]["mask"].shape[0] + 2 * int(components[i]["radius"]))
            * (components[i]["mask"].shape[1] + 2 * int(components[i]["radius"]))
        ),
    )

    for idx in order:
        mask = components[idx]["mask"]
        radius = int(components[idx]["radius"])
        mh, mw = mask.shape
        row_min = radius
        row_max = H - mh - radius
        col_min = radius
        col_max = W - mw - radius
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
                prev_mh, prev_mw = prev["mask"].shape
                required_gap = radius + int(prev["radius"]) + 1

                row_gap = max(0, max(r - (prev_r + prev_mh - 1), prev_r - (r + mh - 1)))
                col_gap = max(0, max(c - (prev_c + prev_mw - 1), prev_c - (c + mw - 1)))
                if max(row_gap, col_gap) >= required_gap:
                    continue
                if _chebyshev_conflict(rows_abs, cols_abs, prev_rows, prev_cols, required_gap):
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
                        "radius": radius,
                    }
                )
                placed_ok = True
                break

        if not placed_ok:
            return None

    for item in placed:
        grid[item["rows_abs"], item["cols_abs"]] = FG_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    comps = find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True)
    radii = sorted(_outline_radius(int(comp["area"])) for comp in comps)
    if not ({1, 2, 3} <= set(radii)):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
